"""Independently audit frozen RTE-risk artifacts; never fit or select a model."""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from datetime import datetime, timezone
import gzip
import json
import math
from pathlib import Path
import sys
from zoneinfo import ZoneInfo

import nbformat
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common import fingerprint, forecast_origin, local_day_grid
from src.rte_assisted import RTEAssistedRun
from src.rte_risk_lab import RiskLabRun


def number(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def utc(value):
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError(f"Timezone-naive artifact timestamp: {value}")
    return parsed.astimezone(timezone.utc)


def equal_number(left, right, *, tolerance=1e-7):
    a, b = number(left), number(right)
    return (math.isnan(a) and math.isnan(b)) or math.isclose(a, b, rel_tol=1e-10, abs_tol=tolerance)


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _raw_outcomes(root):
    outcomes = {}
    with (root/"data/france_electricity_hourly.csv").open() as stream:
        for row in csv.DictReader(stream):
            key = utc(row["datetime_utc"])
            _require(key not in outcomes, f"Duplicate raw national timestamp: {key}")
            outcomes[key] = (number(row["load_mw"]), number(row["temperature_c"]))
    return outcomes


def _cold(delivery, temperature, spec):
    return (delivery.astimezone(ZoneInfo("Europe/Paris")).hour in spec["cold_evening"]["hours_local"]
            and temperature < spec["cold_evening"]["temperature_below_c"])


def _provenance(run):
    run.verify_manifest()
    manifest = json.loads((run.path/"manifest.json").read_text())
    original = run.root/run.spec["reference_run"]/"manifest.json"
    _require(fingerprint(original) == manifest["original_manifest_sha256"], "Original manifest changed")
    base = json.loads((run.path/"base_forecast_provenance.json").read_text())
    for relative, expected in base["source_sha256"].items():
        _require(fingerprint(run.root/relative) == expected, f"Changed base/reference artifact: {relative}")
    assisted_path = Path(run.spec["rte_reference_run"])
    assisted = json.loads((run.path/"assisted_reference_snapshot.json").read_text())
    required_assisted = {str(assisted_path/name) for name in (
        "manifest.json", "frozen_selection.json", "artifact_verification.json",
        "forecasts/assisted_calibration.csv.gz", "forecasts/assisted_test.csv.gz")}
    _require(required_assisted.issubset(assisted["source_sha256"]), "Incomplete assisted-reference snapshot")
    for relative, expected in assisted["source_sha256"].items():
        _require(fingerprint(run.root/relative) == expected, f"Changed assisted-reference artifact: {relative}")
    RTEAssistedRun(run.root, run.root/assisted_path).verify_manifest()
    assisted_audit = json.loads((run.root/assisted_path/"artifact_verification.json").read_text())
    _require(assisted_audit["status"] == "passed", "Prior assisted-reference audit did not pass")
    _require(base["model_count"] == 23, "Original 23 base variants not represented")
    for record in base["quarter_selections"] + base["final_selection_checks"]:
        cutoff = utc(record["selection_cutoff"])
        for option in record["choices"]:
            if option["latest_label_available_at"] is not None:
                _require(utc(option["latest_label_available_at"]) < cutoff, "Base choice used an unreleased outcome")
        if record["choices"]:
            eligible = [c for c in record["choices"] if c["raw_availability"] >= base["raw_availability_minimum"]
                        and math.isfinite(c["MAE_MW"])]
            selected = min(eligible, key=lambda c: (c["MAE_MW"], c["trial"]))
            _require(selected["trial"] == record["selected_trial"], "Base selection is not released-history MAE minimum")
        else:
            _require(record["selected_trial"] == 0 and record["burnin_only"], "Unexpected first-quarter base selection")
    learned = json.loads((run.path/"learned_feature_fits.json").read_text())
    for record in learned:
        origin = utc(record["fit_origin"])
        _require(origin <= utc(record["first_prediction_origin"]), "Learned feature was fitted after prediction origin")
        _require(record["out_of_sample_bank"] and record["original_feature_cutoffs_verified"], "Missing learned-feature cutoff evidence")
        if record["training_rows"]:
            _require(utc(record["latest_label_available_at"]) < origin, "Learned fit used an unreleased label")
            _require(utc(record["latest_training_origin"]) < origin, "Learned fit used a future origin")
        if record["arm"] == "regional_only":
            _require(not any(c.startswith("country_") for c in record["predictor_columns"]), "Country history in regional-only learned predictors")
    caches = list((run.path/"learned_features").glob("*/manifest.json"))
    _require(bool(caches), "Missing sealed learned-feature cache")
    for path in caches:
        cache = json.loads(path.read_text())
        for name, expected in cache["artifact_sha256"].items():
            _require(fingerprint(path.parent/name) == expected, f"Altered learned artifact: {name}")
    return {"original_base_variants": base["model_count"], "learned_fit_events": len(learned),
            "reference_artifact_hashes_checked": len(base["source_sha256"]),
            "assisted_reference_artifact_hashes_checked": len(assisted["source_sha256"]),
            "assisted_reference_manifest_verified": True}


def _selection(run, frozen, outcomes):
    for name, expected in frozen["selection_inputs_sha256"].items():
        _require(fingerprint(run.path/name) == expected, f"Changed selection input: {name}")
    variants = json.loads((run.path/"variant_register.json").read_text())
    _require(set(variants) == set(frozen["selections"]), "Variant coverage differs from frozen selections")
    trials = pd.read_csv(run.path/"tables/tuning_trials.csv")
    folds = {f["name"] for f in run.spec["validation_folds"]}
    development_days = set()
    development_keys = set()
    for fold in run.spec["validation_folds"]:
        days = pd.date_range(fold["start"], fold["end"], freq="D", inclusive="left")
        _require(not development_days.intersection(days), "Overlapping validation folds")
        development_days.update(days)
        development_keys.update(t.to_pydatetime() for day in days for t in local_day_grid(day))
    cold_deliveries = [timestamp for timestamp in development_keys
                       if math.isfinite(outcomes[timestamp][0]) and _cold(timestamp, outcomes[timestamp][1], run.spec)]
    cold_days = {timestamp.astimezone(ZoneInfo("Europe/Paris")).date() for timestamp in cold_deliveries}
    _require(len(trials) == len(variants)*2*len(folds), "Incomplete bounded chronological search")
    _require(not trials.duplicated(["model", "trial", "fold"]).any(), "Duplicate configuration-fold results")
    for model, entry in frozen["selections"].items():
        candidates = []
        selected_group = None
        for trial, group in trials.loc[trials.model.eq(model)].groupby("trial"):
            _require(set(group.fold) == folds, f"Missing chronological fold: {model}/{trial}")
            loss = math.fsum(group.weighted_loss_sum)/math.fsum(group.weight_sum)
            # Every fold has full calendar-day forecast grids. Availability is
            # pooled by its forecast count, including missing-target intervals.
            sizes = group.n+group.missing_target_n
            availability = float(np.average(group.raw_availability, weights=sizes))
            if availability >= run.spec["selection_guardrails"]["raw_availability_min"]:
                candidates.append((loss, int(trial)))
            if int(trial) == entry["trial"]:
                selected_group = group
                _require(equal_number(loss, entry["metrics"]["weighted_MAE_MW"]), f"Selected pooled loss mismatch: {model}")
        _require(min(candidates)[1] == entry["trial"], f"Configuration was not chosen from development loss: {model}")
        _require(selected_group is not None, f"Missing selected configuration: {model}")
    bias = frozen["development_bias_reference"]
    guardrails = run.spec["selection_guardrails"]
    _require(bias["cold_n"] == len(cold_deliveries) and bias["cold_days"] == len(cold_days),
             "Development reference cold support differs from configured-fold raw outcomes")
    eligible = {k: v for k, v in frozen["selections"].items()
                if v["metrics"]["MAE_MW"] <= guardrails["overall_mae_ratio_to_bias_max"]*bias["MAE_MW"]
                and v["metrics"]["cold_MAE_MW"] <= guardrails["cold_mae_ratio_to_bias_max"]*bias["cold_MAE_MW"]
                and v["metrics"]["cold_n"] >= guardrails["development_cold_hours_min"]
                and v["metrics"]["cold_days"] >= guardrails["development_cold_days_min"]}
    winner = min(eligible, key=lambda k: eligible[k]["metrics"]["weighted_MAE_MW"]) if eligible else None
    _require(set(eligible) == set(frozen["passing_guardrails"]), "Frozen guardrail eligibility mismatch")
    _require(winner == frozen["selected_point"], "Point selection violated predeclared guardrails")
    diagnostic = min(frozen["selections"], key=lambda k: frozen["selections"][k]["metrics"]["weighted_MAE_MW"])
    _require(diagnostic == frozen["diagnostic_best_point"], "Diagnostic minimum mismatch")
    probability = pd.read_csv(run.path/"tables/probability_development.csv")
    _require(probability.loc[probability.WIS_MW.idxmin(), "model"] == frozen["selected_quantile"], "Probability selection did not minimize development WIS")
    return {"new_variants": len(variants), "development_configuration_folds": len(trials),
            "development_cold_hours": len(cold_deliveries), "development_cold_days": len(cold_days)}


def _sealed_forecasts(run, frozen):
    rows = origins = fit_events = failures = 0
    freeze = utc(frozen["frozen_utc"])
    seals = sorted((run.path/"forecasts").glob("lab__*.json"))
    _require(bool(seals), "No sealed experiment forecasts")
    for path in seals:
        metadata = json.loads(path.read_text())
        forecast = path.with_suffix(".csv.gz")
        fits = run.path/"fits"/path.name
        ledger = run.path/"ledgers"/(path.stem+".jsonl.gz")
        for artifact, field in ((forecast, "sha256"), (fits, "fits_sha256"), (ledger, "ledger_sha256")):
            _require(fingerprint(artifact) == metadata[field], f"Altered sealed artifact: {artifact.name}")
        frame = pd.read_csv(forecast, low_memory=False)
        _require(not {"actual_mw", "realized_temperature_c", "label_available_at", "load_mw"}.intersection(frame), "Evaluation outcomes in prediction artifact")
        _require(len(frame) == metadata["rows"], f"Forecast row count mismatch: {forecast.name}")
        _require(not frame.duplicated(["origin", "delivery_time"]).any(), f"Duplicate keys: {forecast.name}")
        for name in ("origin", "delivery_time", "fit_origin"):
            frame[name] = pd.to_datetime(frame[name], utc=True)
        _require(frame.fit_origin.le(frame.origin).all(), "Fit after forecast origin")
        expected_days = pd.date_range(metadata["start"], metadata["end"], freq="D", inclusive="left").strftime("%Y-%m-%d")
        _require(set(frame.local_day) == set(expected_days), f"Missing or extra forecast days: {forecast.name}")
        for day, group in frame.groupby("local_day", sort=False):
            _require(pd.DatetimeIndex(group.delivery_time).equals(local_day_grid(day)), f"Incorrect DST/delivery grid: {day}")
            _require(group.origin.eq(forecast_origin(day, run.contract)).all(), "Incorrect pre-noon forecast origin")
        _require("rte_at_origin_mw" in metadata["predictor_columns"], "RTE missing from assisted predictors")
        if metadata["variant"]["arm"] == "regional_only":
            _require(not any(c.startswith("country_") for c in metadata["predictor_columns"]), "Country history in regional-only fit")
            _require(set(frame.fallback_source).issubset({"model", "rte", "unavailable"}), "Regional-only fallback consumed national history")
        is_test = pd.Timestamp(metadata["end"]) > pd.Timestamp(run.spec["test"]["start"])
        if is_test:
            _require(utc(metadata["created_utc"]) >= freeze, "Test forecast preceded frozen selection")
        records = json.loads(fits.read_text())
        if metadata["fixed"]:
            _require(len(records) == 1 and frame.fit_origin.nunique() == 1, "Fixed-model comparator refitted")
        for record in records:
            fit_events += 1
            if record["status"] != "fit":
                failures += 1
                continue
            _require(utc(record["latest_label_available_at"]) < utc(record["origin"]), "Fit consumed unreleased outcome")
            _require(record["predictor_columns"] == metadata["predictor_columns"], "Fit predictor list disagrees with forecast seal")
            _require(record["target_mode"] == metadata["variant"]["mode"], "Direct/residual target mismatch")
        # Ordinary Python records validate ledger values without sharing the
        # forecasting or metric implementation.
        lookup = {(row.origin.to_pydatetime(), row.delivery_time.to_pydatetime()): row
                  for row in frame.itertuples(index=False)}
        count = 0
        seen = set()
        with gzip.open(ledger, "rt") as stream:
            for line in stream:
                event = json.loads(line)
                origins += 1
                if is_test:
                    _require(utc(event["saved_at_utc"]) >= freeze, "Test ledger preceded frozen selection")
                for prediction in event["predictions"]:
                    _require(not {"actual_mw", "realized_temperature_c", "label_available_at"}.intersection(prediction), "Outcome in pre-scoring ledger")
                    key = (utc(prediction["origin"]), utc(prediction["delivery_time"]))
                    _require(key not in seen, "Duplicate ledger prediction")
                    seen.add(key)
                    saved = lookup[key]
                    for name in ("prediction_mw", "raw_prediction_mw"):
                        _require(equal_number(prediction.get(name), getattr(saved, name), tolerance=1e-6), f"Ledger forecast value mismatch: {path.stem}")
                    count += 1
        _require(count == len(frame), "Ledger does not cover all forecasts")
        rows += len(frame)
    return {"sealed_forecast_files": len(seals), "forecast_rows_checked": rows,
            "daily_origin_ledgers_checked": origins, "fit_events_checked": fit_events, "failed_fit_events": failures}


def _point_arithmetic(run, outcomes):
    predictions = defaultdict(dict)
    with gzip.open(run.path/"forecasts/all_test.csv.gz", "rt") as stream:
        for row in csv.DictReader(stream):
            key = (utc(row["origin"]), utc(row["delivery_time"]))
            model = row["model"]
            _require(key not in predictions[model], "Duplicate combined test prediction")
            predictions[model][key] = number(row["prediction_mw"])
    expected_days = pd.date_range(run.spec["test"]["start"], run.spec["test"]["end"], freq="D", inclusive="left")
    expected = {(forecast_origin(day, run.contract).to_pydatetime(), timestamp.to_pydatetime())
                for day in expected_days for timestamp in local_day_grid(day)}
    _require(bool(predictions), "No combined test policies")
    for model, values in predictions.items():
        _require(set(values) == expected, f"Policy forecast keys do not cover the full declared delivery grid: {model}")
    common = {key for key in expected if all(math.isfinite(p[key]) for p in predictions.values())}
    missing_truth = {key for key in expected if not math.isfinite(outcomes[key[1]][0])}
    exclusions = expected-common
    # An unavailable regional-only forecast is legitimate when its sole declared
    # fallback (RTE) is also missing. A shared missing demand target must remain a
    # recorded grid interval, but need not have a finite forecast for every policy.
    # This dataset has no nonmissing demand target lost by common-pair matching.
    _require(exclusions.issubset(missing_truth), "Common matching excluded a nonmissing actual target")
    saved = pd.read_csv(run.path/"tables/test_scorecard.csv").set_index("model")
    _require(set(saved.index) == set(predictions), "Scorecard omitted or added a policy")
    unmatched = pd.read_csv(run.path/"tables/unmatched_scorecard.csv").set_index("model")
    raw_scores = pd.read_csv(run.path/"tables/raw_scorecard.csv").set_index("model")
    _require(set(unmatched.index) == set(predictions) == set(raw_scores.index), "Availability scorecards omitted a policy")
    metrics = {}
    for model, values in predictions.items():
        errors, squared, signed, weighted, weights, cold_errors, cold_days = [], [], [], [], [], [], set()
        missing = len(missing_truth)
        matched_missing = 0
        for key in sorted(common):
            delivery = key[1]
            y, temperature = outcomes[delivery]
            if not math.isfinite(y):
                matched_missing += 1
                continue
            residual = y-values[key]
            error = abs(residual)
            cold = _cold(delivery, temperature, run.spec)
            weight = run.spec["cold_weight"] if cold else 1
            errors.append(error); squared.append(residual**2); signed.append(-residual)
            weighted.append(weight*error); weights.append(weight)
            if cold:
                cold_errors.append(error)
                cold_days.add(delivery.astimezone(ZoneInfo("Europe/Paris")).date())
        result = {"MAE_MW": math.fsum(errors)/len(errors), "RMSE_MW": math.sqrt(math.fsum(squared)/len(squared)),
                  "weighted_MAE_MW": math.fsum(weighted)/math.fsum(weights),
                  "cold_MAE_MW": math.fsum(cold_errors)/len(cold_errors),
                  "scored": len(errors), "missing_targets": missing, "matched_missing_targets": matched_missing,
                  "cold_scored": len(cold_errors), "cold_days": len(cold_days)}
        for name in ("MAE_MW", "RMSE_MW", "weighted_MAE_MW", "cold_MAE_MW", "scored", "cold_scored", "cold_days"):
            _require(equal_number(result[name], saved.loc[model, name]), f"Independent {name} mismatch: {model}")
        _require(len(errors) == 8759 and missing == 1, f"Unexpected target exclusions: {model}")
        _require(equal_number(saved.loc[model, "missing_targets"], matched_missing), f"Matched-score missingness mismatch: {model}")
        for name, table in (("unmatched", unmatched), ("raw", raw_scores)):
            _require(equal_number(table.loc[model, "missing_targets"], missing), f"Full-grid missing targets hidden in {name} scorecard: {model}")
            _require(equal_number(table.loc[model, "eligible_targets"], len(expected)-missing), f"Eligible-target denominator mismatch in {name} scorecard: {model}")
        _require(len(cold_errors) == 66 and len(cold_days) == 24, f"Unexpected cold segment support: {model}")
        metrics[model] = result
    exclusion_records = [{"origin": origin.isoformat(), "delivery_time": delivery.isoformat(),
                          "actual_target_missing": (origin, delivery) in missing_truth,
                          "nonfinite_prediction_models": sorted(model for model, values in predictions.items()
                                                                 if not math.isfinite(values[(origin, delivery)]))}
                         for origin, delivery in sorted(exclusions)]
    return {"policies": len(metrics), "full_origin_delivery_pairs_per_policy": len(expected),
            "matched_origin_delivery_pairs": len(common), "matched_nonmissing_hours": 8759,
            "missing_targets_per_policy": missing, "nonfinite_common_pair_exclusions": len(exclusions),
            "common_pair_exclusion_records": exclusion_records,
            "exclusion_interpretation": "Every policy retains all delivery keys; nonfinite common-pair exclusions are missing-target intervals only. Full-grid availability tables retain the missing target count.",
            "cold_hours": 66, "cold_days": 24, "independent_point_scores": metrics}


def _interval_arithmetic(run, outcomes):
    protocol = json.loads((run.path/"tables/uncertainty_protocol.json").read_text())
    first_test = forecast_origin(run.spec["test"]["start"], run.contract).to_pydatetime()
    _require(utc(protocol["first_test_origin"]) == first_test, "Incorrect uncertainty evaluation cutoff")
    _require(utc(protocol["calibration_latest_available_at"]) < first_test, "Unreleased calibration outcome used")
    _require(utc(protocol["fixed_fit_origin"]) <= forecast_origin(run.spec["calibration"]["start"], run.contract),
             "Fixed predictor fitted after calibration began")
    _require(protocol["calibration_n"] >= 200, "Insufficient declared calibration support")
    saved = pd.read_csv(run.path/"tables/cold_interval_scorecard.csv").set_index(["model", "method"])
    checks = []
    files = [run.path/"forecasts/selected_empirical_intervals.csv.gz", run.path/"forecasts/probabilistic_test.csv.gz"]
    for path in files:
        _require(path.exists(), f"Missing probability artifact: {path.name}")
        groups = defaultdict(list)
        with gzip.open(path, "rt") as stream:
            for row in csv.DictReader(stream):
                _require(not {"actual_mw", "realized_temperature_c", "label_available_at"}.intersection(row),
                         "Evaluation outcome in probabilistic forecast artifact")
                delivery = utc(row["delivery_time"])
                y, temperature = outcomes[delivery]
                if math.isfinite(y) and _cold(delivery, temperature, run.spec):
                    groups[(row["model"], row["method"])].append((delivery, y, row))
        for key, rows in groups.items():
            score = saved.loc[key]
            record = {"model": key[0], "method": key[1]}
            for label, low, high in (("central90", "q05", "q95"), ("asymmetric90", "q075", "q975")):
                valid = [(d, y, number(r.get(low)), number(r.get(high))) for d, y, r in rows
                         if math.isfinite(number(r.get(low))) and math.isfinite(number(r.get(high)))]
                if not valid:
                    continue
                n = len(valid)
                _require(all(lo <= hi for _, _, lo, hi in valid), "Crossed interval endpoints")
                covered = sum(lo <= y <= hi for _, y, lo, hi in valid)
                upper = [y-hi for _, y, _, hi in valid if y > hi]
                lower = sum(y < lo for _, y, lo, _ in valid)
                expected = {"nobs": n, "coverage": covered/n, "upper_exceedance": len(upper)/n,
                            "lower_exceedance": lower/n, "upper_misses": len(upper),
                            "upper_miss_mean_MW": math.fsum(upper)/n,
                            "upper_miss_conditional_severity_MW": math.fsum(upper)/len(upper) if upper else float("nan"),
                            "width_MW": math.fsum(hi-lo for _, _, lo, hi in valid)/n}
                for name, value in expected.items():
                    _require(equal_number(value, score[label+"_"+name]), f"Independent interval mismatch: {key}/{label}/{name}")
                if label == "central90":
                    winkler = math.fsum(hi-lo+20*max(lo-y, 0)+20*max(y-hi, 0) for _, y, lo, hi in valid)/n
                    _require(equal_number(winkler, score["Winkler_central90_MW"]), f"Winkler mismatch: {key}")
                record[label] = expected
            checks.append(record)
    return {"independent_cold_interval_checks": checks, "calibration_outcome_cutoff_verified": True}


def _reporting_tables(run, frozen, outcomes, point_results):
    """Cross-check comparison coverage, arithmetic and calibration boundaries."""
    original = json.loads((run.root/run.spec["reference_run"]/"selected_parameters.json").read_text())
    assisted_frozen = json.loads((run.root/run.spec["rte_reference_run"]/"frozen_selection.json").read_text())
    assisted = assisted_frozen["selections"]
    expected = set(original) | {"seasonal_naive", "comparable_days", "rte_proxy", "rte_bias_proxy"} | set(assisted) | set(frozen["selections"])
    metrics = point_results["independent_point_scores"]
    _require(set(metrics) == expected, "Combined comparison does not cover all original, assisted and new policies")
    ablation = pd.read_csv(run.path/"tables/feature_ablation.csv")
    ablation_expected = {f"lab__lightgbm__{arm}__{tier}__w{weight}__residual"
                         for arm in run.spec["arms"] for tier in ("basic", "engineered", "stacked") for weight in (1, 5)}
    _require(set(ablation.model) == ablation_expected and len(ablation) == 18, "Feature ablation coverage mismatch")
    for row in ablation.itertuples():
        for metric in ("MAE_MW", "weighted_MAE_MW", "cold_MAE_MW"):
            _require(equal_number(getattr(row, metric), metrics[row.model][metric]), f"Ablation score mismatch: {row.model}/{metric}")
    hybrids = pd.read_csv(run.path/"tables/hybrid_control_comparison.csv")
    hybrid_expected = {model for model, entry in frozen["selections"].items() if entry["variant"].get("base")}
    _require(set(hybrids.model) == hybrid_expected and len(hybrids) == 23, "Hybrid/control coverage mismatch")
    for row in hybrids.to_dict("records"):
        variant = frozen["selections"][row["model"]]["variant"]
        control = f"lab__hybrid_control__{variant['arm']}"
        _require(row["control_model"] == control, "Hybrid compared against a different data/history control")
        for metric in ("MAE_MW", "weighted_MAE_MW", "cold_MAE_MW"):
            delta = metrics[control][metric]-metrics[row["model"]][metric]
            _require(equal_number(delta, row[metric+"_improvement_MW"]), f"Hybrid increment arithmetic mismatch: {row['model']}/{metric}")
    primary = frozen["selected_point"] or frozen["diagnostic_best_point"]
    previous = assisted_frozen["selected_complete_policy"]
    bootstrap = pd.read_csv(run.path/"tables/bootstrap.csv")
    prior_evidence = bootstrap.loc[bootstrap.model.eq(primary) & bootstrap.reference.eq(previous)]
    expected_comparisons = {(metric, block) for metric in ("MAE_MW", "weighted_MAE_MW", "cold_MAE_MW")
                            for block in run.spec["bootstrap_block_days"]}
    _require(set(zip(prior_evidence.metric, prior_evidence.block_days)) == expected_comparisons
             and len(prior_evidence) == len(expected_comparisons), "Missing previous-selected-policy bootstrap comparison")
    summary = json.loads((run.path/"result_summary.json").read_text())
    _require(summary["previous_development_selected_policy"] == previous, "Previous selected policy differs from its frozen selection")
    for metric in ("MAE_MW", "weighted_MAE_MW", "cold_MAE_MW"):
        _require(equal_number(summary["previous_selected_"+metric], metrics[previous][metric]), "Previous selected policy score mismatch")
        delta = metrics[previous][metric]-metrics[primary][metric]
        _require(equal_number(summary["loss_improvement_vs_previous_selected_MW"][metric], delta), "Previous-policy delta mismatch")
        for row in prior_evidence.loc[prior_evidence.metric.eq(metric)].itertuples():
            _require(equal_number(row.absolute_loss_improvement_MW, delta), "Previous-policy bootstrap point estimate mismatch")
    variant = frozen["selections"][primary]["variant"]
    if variant.get("base"):
        control = f"lab__hybrid_control__{variant['arm']}"
        evidence = bootstrap.loc[bootstrap.model.eq(primary) & bootstrap.reference.eq(control) & bootstrap.metric.eq("weighted_MAE_MW")]
        _require(set(evidence.block_days) == set(run.spec["bootstrap_block_days"]), "Missing selected-hybrid control bootstrap")
        ci = evidence.loc[evidence.block_days.eq(14)].iloc[0]
        if ci.absolute_low <= 0 <= ci.absolute_high:
            _require("incremental value from the base architecture is not established" in (run.path/"recommendation.md").read_text(),
                     "Recommendation overclaims architecture value despite control CI including zero")
    calibration = defaultdict(set)
    first, end = run.spec["calibration"]["start"], run.spec["calibration"]["end"]
    with gzip.open(run.path/"forecasts/all_calibration.csv.gz", "rt") as stream:
        for row in csv.DictReader(stream):
            delivery = utc(row["delivery_time"])
            day = delivery.astimezone(ZoneInfo("Europe/Paris")).strftime("%Y-%m-%d")
            _require(first <= day < end, "Original/reference calibration escaped the new calibration period")
            key = (utc(row["origin"]), delivery)
            _require(key not in calibration[row["model"]], "Duplicate calibration forecast pair")
            calibration[row["model"]].add(key)
    expected_calibration = {(forecast_origin(day, run.contract).to_pydatetime(), timestamp.to_pydatetime())
                            for day in pd.date_range(first, end, freq="D", inclusive="left") for timestamp in local_day_grid(day)}
    _require(set(calibration) == expected, "Calibration policy coverage differs from test")
    _require(all(keys == expected_calibration for keys in calibration.values()), "Calibration source omitted or added delivery intervals")
    cold_count = sum(math.isfinite(outcomes[t][0]) and _cold(t, outcomes[t][1], run.spec) for _, t in expected_calibration)
    support = pd.read_csv(run.path/"tables/calibration_support.csv")
    _require(set(support.model) == expected and support.cold_calibration_hours.eq(cold_count).all(), "Calibration cold-support table disagrees with raw data")
    learned = pd.read_csv(run.path/"tables/learned_feature_oos_diagnostics.csv")
    evening = learned.loc[learned.phase.eq("calibration") & learned.component.eq("cold") & learned.segment.eq("evening")]
    _require(len(evening) == len(run.spec["arms"]), "Missing evening cold-classifier calibration diagnostic")
    if cold_count == 0:
        _require(evening.event_count.eq(0).all() and evening.ROC_AUC.isna().all() and evening.PR_average_precision.isna().all(),
                 "Single-class evening calibration was assigned an AUC/AP value")
    mapping = pd.read_csv(run.path/"tables/region_column_mapping.csv")
    _require(mapping.feature_prefix.is_unique and mapping.source_region.is_unique and len(mapping) == 12,
             "Regional feature dictionary is not a one-to-one 12-region mapping")
    return {"report_comparison_policies": len(expected), "feature_ablation_variants": len(ablation),
            "previous_selected_policy_comparisons": len(prior_evidence),
            "hybrid_control_comparisons": len(hybrids), "calibration_pairs_per_policy": len(expected_calibration),
            "calibration_cold_hours": cold_count, "regional_columns_mapped": len(mapping)}


def _notebooks(root):
    results = []
    for path in sorted((root/"notebooks").glob("0[78]_*.ipynb")):
        notebook = nbformat.read(path, as_version=4)
        nbformat.validate(notebook)
        cells = [c for c in notebook.cells if c.cell_type == "code" and c.metadata.get("rte_risk_extension")]
        _require(bool(cells), f"Missing risk-extension cells: {path.name}")
        _require(all(c.execution_count is not None for c in cells), f"Unexecuted risk-extension cell: {path.name}")
        _require(not any(o.output_type == "error" for c in cells for o in c.get("outputs", [])), f"Notebook execution error: {path.name}")
        results.append({"notebook": path.name, "executed_extension_code_cells": len(cells)})
    _require(len(results) == 2, "Expected the two existing research notebooks")
    return {"notebooks": results}


def verify(run):
    frozen = json.loads((run.path/"frozen_selection.json").read_text())
    result = {"status": "passed", "checked_at_utc": datetime.now(timezone.utc).isoformat(),
              "run_dir": str(run.path), "errors": []}
    outcomes = _raw_outcomes(run.root)
    result.update(_provenance(run))
    result.update(_selection(run, frozen, outcomes))
    result.update(_sealed_forecasts(run, frozen))
    point_results = _point_arithmetic(run, outcomes)
    result.update(point_results)
    result.update(_interval_arithmetic(run, outcomes))
    result.update(_reporting_tables(run, frozen, outcomes, point_results))
    result.update(_notebooks(run.root))
    result["interpretation"] = "Independent artifact and arithmetic checks; retrospective assumed RTE availability; previously exposed test"
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path)
    args = parser.parse_args()
    run = RiskLabRun(ROOT, args.run_dir) if args.run_dir else RiskLabRun.latest(ROOT)
    try:
        result = verify(run)
    except Exception as exc:
        import traceback
        result = {"status": "failed", "errors": [f"{type(exc).__name__}: {exc}"], "traceback": traceback.format_exc()}
    (run.path/"artifact_verification.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())
