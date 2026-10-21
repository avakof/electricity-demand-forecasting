"""Matched scores and risk diagnostics for the frozen RTE feature laboratory.

This module never trains or selects a model. It consumes saved forecasts and the
already-frozen selection. Weighted errors are MW risk proxies, not price savings.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common import write_json
from src.metrics import daily_metrics, point_metrics
from src.research_scores import empirical_crps

KEYS = ["origin", "delivery_time"]
EMPIRICAL = "past_calibration_empirical_residual_distribution"
QUANTILES = {"q05": .05, "q075": .075, "q25": .25, "q50": .5,
             "q75": .75, "q90": .9, "q95": .95, "q975": .975}


def threshold_weighted_empirical_crps(actual, prediction, residuals, threshold):
    """Exact upper-tail twCRPS for an equally weighted empirical residual CDF.

    The weight is 1{x >= threshold}. CRPS after clipping both the outcome and
    every ensemble member below at threshold equals this weighted integral.
    Prefix sums evaluate that exact discrete-distribution score in O(n log n)
    preprocessing and O(log n) per forecast, without constructing a huge matrix.
    """
    r = np.sort(np.asarray(residuals, dtype=float))
    r = r[np.isfinite(r)]
    actual, prediction = np.asarray(actual, dtype=float), np.asarray(prediction, dtype=float)
    if not len(r):
        return np.full(len(actual), np.nan)
    n = len(r)
    prefix = np.r_[0., np.cumsum(r)]
    coefficients = 2*np.arange(1, n+1)-n-1
    weighted_prefix = np.r_[0., np.cumsum(coefficients*r)]
    floor = threshold-prediction
    y = np.maximum(actual-prediction, floor)
    k_floor = np.searchsorted(r, floor, side="right")
    k_y = np.searchsorted(r, y, side="right")
    total = prefix[-1]-prefix[k_floor]+floor*k_floor
    partial = prefix[k_y]-prefix[k_floor]+floor*k_floor
    mean_absolute = (y*(2*k_y-n)+total-2*partial)/n
    half_pairwise = (weighted_prefix[-1]-weighted_prefix[k_floor]+floor*k_floor*(k_floor-n))/(n*n)
    return mean_absolute-half_pairwise


def read_forecasts(path):
    frame = pd.read_csv(path, low_memory=False)
    for name in ["origin", "delivery_time", "fit_origin"]:
        if name in frame:
            frame[name] = pd.to_datetime(frame[name], utc=True, format="mixed")
    return frame


def _previous_development_selection(run):
    frozen = json.loads((Path(run.root)/run.spec["rte_reference_run"]/"frozen_selection.json").read_text())
    return frozen["selected_complete_policy"]


def _decorate(frame, labels, spec):
    forbidden = {"actual_mw", "realized_temperature_c", "label_available_at"}
    if forbidden.intersection(frame):
        raise AssertionError("Forecast artifacts must not contain evaluation outcomes")
    out = frame.merge(labels[["delivery_time", "actual_mw", "realized_temperature_c", "label_available_at"]],
                      on="delivery_time", how="left", validate="many_to_one")
    local = out.delivery_time.dt.tz_convert("Europe/Paris")
    out["hour"] = local.dt.hour
    out["day"] = local.dt.tz_localize(None).dt.normalize()
    out["month"] = local.dt.month
    out["season"] = out.month.map({12: "Winter", 1: "Winter", 2: "Winter", 3: "Spring", 4: "Spring", 5: "Spring",
                                  6: "Summer", 7: "Summer", 8: "Summer", 9: "Autumn", 10: "Autumn", 11: "Autumn"})
    out["winter_id"] = (local.dt.year + out.month.eq(12).astype(int)).where(out.month.isin([12, 1, 2]))
    out["cold_evening"] = out.hour.isin(spec["cold_evening"]["hours_local"]) & out.realized_temperature_c.lt(spec["cold_evening"]["temperature_below_c"])
    out["lead_group"] = pd.cut((out.delivery_time - out.origin).dt.total_seconds()/3600,
                               bins=[0, 24, 48, np.inf], labels=["<=24h", "24-48h", ">48h"])
    out["risk_weight"] = np.where(out.cold_evening, spec["cold_weight"], 1.)
    return out


def _point(frame, prediction="prediction_mw"):
    result = point_metrics(frame, prediction)
    good = frame.loc[frame.actual_mw.notna() & frame[prediction].notna()]
    err = good.actual_mw - good[prediction]
    weights = good.risk_weight
    cold = good.loc[good.cold_evening]
    cold_error = cold.actual_mw-cold[prediction]
    result.update(weighted_MAE_MW=float((weights*err.abs()).sum()/weights.sum()) if len(good) else np.nan,
                  cold_MAE_MW=float(cold_error.abs().mean()), cold_scored=len(cold),
                  cold_days=int(cold.day.nunique()), cold_DJF_winters=int(cold.winter_id.nunique()),
                  cold_underforecast_rate=float(cold_error.gt(0).mean()) if len(cold) else np.nan,
                  cold_shortage_mean_MW=float(cold_error.clip(lower=0).mean()),
                  cold_shortage_conditional_severity_MW=float(cold_error.loc[cold_error.gt(0)].mean()),
                  missing_temperature_hours=int(good.realized_temperature_c.isna().sum()))
    return result


def _pinball(y, q, tau):
    error = np.asarray(y)-np.asarray(q)
    return np.where(error >= 0, tau*error, (tau-1)*error)


def _interval_scores(frame, native_quantiles=True):
    """Directional interval errors, properly labeled scores, and support counts."""
    result = {"eligible_targets": int(frame.actual_mw.notna().sum()), "segment_days": int(frame.day.nunique()),
              "segment_DJF_winters": int(frame.winter_id.nunique())}
    for name, tau in (QUANTILES.items() if native_quantiles else []):
        if name not in frame:
            continue
        good = frame.dropna(subset=["actual_mw", name])
        result[f"pinball_{name}_MW"] = float(_pinball(good.actual_mw, good[name], tau).mean()) if len(good) else np.nan
    for label, lower, upper, nominal in (("central90", "q05", "q95", .9), ("asymmetric90", "q075", "q975", .9)):
        if not {lower, upper}.issubset(frame):
            continue
        good = frame.dropna(subset=["actual_mw", lower, upper])
        if (good[lower] > good[upper]).any():
            raise AssertionError(f"Crossed intervals in {label}")
        y, lo, hi = good.actual_mw, good[lower], good[upper]
        upper_error = y-hi
        upper_event = upper_error.gt(0)
        prefix = label+"_"
        result.update({prefix+"nobs": len(good), prefix+"days": int(good.day.nunique()),
                       prefix+"DJF_winters": int(good.winter_id.nunique()),
                       prefix+"availability": len(good)/result["eligible_targets"] if result["eligible_targets"] else np.nan,
                       prefix+"coverage": float((y.ge(lo) & y.le(hi)).mean()) if len(good) else np.nan,
                       prefix+"width_MW": float((hi-lo).mean()),
                       prefix+"lower_exceedance": float(y.lt(lo).mean()) if len(good) else np.nan,
                       prefix+"upper_exceedance": float(upper_event.mean()) if len(good) else np.nan,
                       prefix+"upper_misses": int(upper_event.sum()),
                       prefix+"upper_miss_mean_MW": float(upper_error.clip(lower=0).mean()),
                       prefix+"upper_miss_conditional_severity_MW": float(upper_error.loc[upper_event].mean())})
        if label == "central90":
            winkler = hi-lo+20*(lo-y).clip(lower=0)+20*(y-hi).clip(lower=0)
            result["Winkler_central90_MW"] = float(winkler.mean())
        else:
            result["asymmetric90_endpoint_pinball_sum_MW"] = float((_pinball(y, lo, .075)+_pinball(y, hi, .975)).mean()) if len(good) else np.nan
    if {"q05", "q25", "q50", "q75", "q95"}.issubset(frame):
        good = frame.dropna(subset=["actual_mw", "q05", "q25", "q50", "q75", "q95"])
        if len(good):
            y = good.actual_mw
            is90 = good.q95-good.q05+20*(good.q05-y).clip(lower=0)+20*(y-good.q95).clip(lower=0)
            is50 = good.q75-good.q25+4*(good.q25-y).clip(lower=0)+4*(y-good.q75).clip(lower=0)
            wis = (.5*(y-good.q50).abs()+.05*is90+.25*is50)/2.5
            result["WIS_central50_90_MW"] = float(wis.mean())
            result["weighted_WIS_central50_90_MW"] = float((wis*good.risk_weight).sum()/good.risk_weight.sum())
            result["WIS_nobs"] = len(good)
    if "CRPS_MW" in frame:
        good = frame.dropna(subset=["CRPS_MW"])
        result["CRPS_empirical_MW"] = float(good.CRPS_MW.mean())
        result["weighted_CRPS_empirical_MW"] = float((good.CRPS_MW*good.risk_weight).sum()/good.risk_weight.sum()) if len(good) else np.nan
    if "twCRPS_MW" in frame:
        good = frame.dropna(subset=["twCRPS_MW"])
        result["upper_tail_twCRPS_empirical_MW"] = float(good.twCRPS_MW.mean())
        result["upper_tail_threshold_MW"] = float(frame.upper_tail_threshold_MW.iloc[0])
    return result


def _probability_rows(frame, spec, model, method):
    rows = []
    native_quantiles = method not in {"fixed_engineered_split_conformal", "fixed_engineered_CQR"}
    for dimension in ("all", "cold_evening", "hour", "season", "lead_group"):
        groups = [("all", frame)] if dimension == "all" else frame.groupby(dimension, observed=True)
        for value, group in groups:
            rows.append({"model": model, "method": method, "segment_type": dimension,
                         "segment": str(value), "endpoint_semantics": "forecast_quantiles" if native_quantiles else "calibrated_coverage_bounds",
                         **_interval_scores(group, native_quantiles=native_quantiles)})
    return rows


def _bootstrap(frame, candidate, reference, spec):
    pairs = frame.loc[frame.model.isin([candidate, reference])].pivot(index=KEYS, columns="model", values="prediction_mw").dropna()
    info = frame.drop_duplicates(KEYS).set_index(KEYS)[["actual_mw", "risk_weight", "cold_evening", "day"]]
    pairs = pairs.join(info).dropna(subset=["actual_mw"])
    if not len(pairs):
        return []
    errors = {name: (pairs.actual_mw-pairs[name]).abs() for name in (candidate, reference)}
    rows = []
    for metric in ("MAE_MW", "weighted_MAE_MW", "cold_MAE_MW"):
        weights = pairs.risk_weight if metric == "weighted_MAE_MW" else pairs.cold_evening.astype(float) if metric == "cold_MAE_MW" else pd.Series(1., index=pairs.index)
        daily = pd.DataFrame({"day": pairs.day, "candidate_loss": errors[candidate]*weights,
                              "reference_loss": errors[reference]*weights, "weight": weights}).groupby("day").sum()
        values, days = daily.to_numpy(), daily.index
        total = values.sum(axis=0)
        for block in spec["bootstrap_block_days"]:
            windows = [np.arange(i, i+block) for i in range(len(daily)-block+1)
                       if days[i+block-1]-days[i] == pd.Timedelta(days=block-1)]
            row = {"model": candidate, "reference": reference, "metric": metric, "block_days": block,
                   "days": len(daily), "status": "ok" if windows and total[2] > 0 else "insufficient_support"}
            if windows and total[2] > 0:
                rng = np.random.default_rng(spec["seed"])
                samples = []
                for _ in range(spec["bootstrap_repetitions"]):
                    index = np.concatenate([windows[i] for i in rng.integers(0, len(windows), size=int(np.ceil(len(daily)/block)))])[:len(daily)]
                    c, r, w = values[index].sum(axis=0)
                    if w > 0 and r > 0:
                        samples.append([(r-c)/w, 100*(1-c/r)])
                row.update(absolute_loss_improvement_MW=(total[1]-total[0])/total[2],
                           relative_improvement_percent=100*(1-total[0]/total[1]) if total[1] else np.nan,
                           valid_bootstrap_samples=len(samples))
                if samples:
                    bounds = np.quantile(samples, [.025, .975], axis=0)
                    row.update(absolute_low=bounds[0, 0], absolute_high=bounds[1, 0],
                               relative_low=bounds[0, 1], relative_high=bounds[1, 1])
            rows.append(row)
    return rows


def _empirical_probabilities(calibration, test, labels, spec, roles):
    cutoff = test.origin.min()
    cal = calibration.merge(labels[["delivery_time", "actual_mw", "label_available_at"]], on="delivery_time", validate="many_to_one")
    cal = cal.loc[cal.label_available_at.lt(cutoff)]
    historical = labels.loc[labels.delivery_time.dt.tz_convert("Europe/Paris").dt.year.eq(2023), "actual_mw"].dropna()
    threshold = float(historical.quantile(.9)) if len(historical) else np.nan
    rows, retained = [], []
    for model, group in test.groupby("model", sort=True):
        past = cal.loc[cal.model.eq(model)]
        residuals = (past.actual_mw-past.prediction_mw).dropna().to_numpy()
        out = group.copy()
        out["method"], out["calibration_n"] = EMPIRICAL, len(residuals)
        for name, tau in QUANTILES.items():
            out[name] = out.prediction_mw + np.quantile(residuals, tau) if len(residuals) >= 200 else np.nan
        decorated = _decorate(out, labels, spec)
        decorated["CRPS_MW"] = empirical_crps(decorated.actual_mw-decorated.prediction_mw, residuals) if len(residuals) >= 200 else np.nan
        if model in roles and np.isfinite(threshold) and len(residuals) >= 200:
            decorated["twCRPS_MW"] = threshold_weighted_empirical_crps(decorated.actual_mw, decorated.prediction_mw, residuals, threshold)
            decorated["upper_tail_threshold_MW"] = threshold
        scores = _probability_rows(decorated, spec, model, EMPIRICAL)
        rows.extend({**row, "calibration_n": len(residuals)} for row in scores)
        if model in roles:
            retained.append(out)
    return pd.DataFrame(rows), pd.concat(retained, ignore_index=True) if retained else pd.DataFrame()


def _risk_scenarios(frame, spec):
    rows = []
    for model, group in frame.groupby("model", sort=True):
        good = group.dropna(subset=["actual_mw", "prediction_mw"])
        error = good.actual_mw-good.prediction_mw
        for weight in spec["risk_scenarios"]["cold_weights"]:
            weights = np.where(good.cold_evening, weight, 1.)
            for ratio in spec["risk_scenarios"]["shortage_surplus_ratios"]:
                loss = ratio*error.clip(lower=0)+(-error).clip(lower=0)
                rows.append({"model": model, "cold_weight": weight, "shortage_surplus_ratio": ratio,
                             "normalized_asymmetric_loss_MW": float(np.sum(weights*loss)/np.sum(weights)),
                             "scored": len(good), "interpretation": "Retrospective weighted-MW error proxy; unchanged point forecast; not euro savings or optimal procurement"})
    return pd.DataFrame(rows)


def _development_fit_diagnostics(run, frozen):
    """Compare the same fitted model's sampled train error with its later OOS error.

    The training diagnostic is the engine's last up-to-2000 training rows, not
    the full training score. A temporal gap can reflect shift as well as fitting;
    it is not a formal bias/variance decomposition.
    """
    trials = pd.read_csv(run.path/"tables/tuning_trials.csv")
    chosen = pd.concat([trials.loc[trials.model.eq(model) & trials.trial.eq(entry["trial"])]
                        for model, entry in frozen["selections"].items()], ignore_index=True)
    chosen.to_csv(run.path/"tables/selected_development_quarters.csv", index=False)
    gaps, fits = [], []
    for path in sorted((run.path/"forecasts").glob("lab__*.json")):
        meta = json.loads(path.read_text())
        variant, phase = meta.get("variant", {}), meta.get("phase", "")
        model = variant.get("model")
        if model not in frozen["selections"] or meta.get("q") is not None:
            continue
        selection = frozen["selections"][model]
        selected_phases = {f"{fold['name']}_trial{selection['trial']}" for fold in run.spec["validation_folds"]}
        if phase not in selected_phases:
            continue
        fit_path = run.path/"fits"/path.name
        records = json.loads(fit_path.read_text())
        predictions = read_forecasts(path.with_suffix(".csv.gz"))
        scored = predictions.merge(run.data.labels[["delivery_time", "actual_mw"]], on="delivery_time", validate="many_to_one")
        for record in records:
            origin = pd.Timestamp(record["origin"])
            out = scored.loc[scored.fit_origin.eq(origin)].dropna(subset=["actual_mw", "raw_prediction_mw"])
            mae = float((out.actual_mw-out.raw_prediction_mw).abs().mean())
            training_mae = record.get("sampled_train_MAE_MW", np.nan)
            gaps.append({"model": model, "fold": phase, "fit_origin": str(origin),
                         "training_rows": record.get("training_rows"), "sampled_training_rows": record.get("training_diagnostic_rows"),
                         "sampled_training_MAE_MW": training_mae, "following_OOS_MAE_MW": mae,
                         "OOS_minus_sampled_train_MAE_MW": mae-training_mae, "OOS_scored": len(out),
                         "interpretation": "Same fit; training sample is last up-to2000 rows; shift and fitting both affect gap; not formal bias-variance decomposition"})
            fits.append({"model": model, "fold": phase,
                         **{k: v for k, v in record.items() if k != "predictor_columns"}})
    pd.DataFrame(gaps).to_csv(run.path/"tables/development_fit_gaps.csv", index=False)
    pd.DataFrame(fits).to_csv(run.path/"tables/selected_development_fit_diagnostics.csv", index=False)


def _learned_feature_diagnostics(run):
    """Score causal first-stage features only; never select or recalibrate them."""
    from sklearn.metrics import average_precision_score, roc_auc_score

    names = list(run.data.regional_load.columns)
    pd.DataFrame([{"feature_prefix": f"region_{i:02d}", "source_region": name,
                   "mapping": "Actual regional pivot-column order used when constructing features"}
                  for i, name in enumerate(names)]).to_csv(run.path/"tables/region_column_mapping.csv", index=False)
    features = run.frame
    columns = [*KEYS, "local_day", "rte_at_origin_mw"]
    columns += [name for name in features if name.startswith("learned_")]
    scored = features[columns].merge(run.data.labels[["delivery_time", "actual_mw", "realized_temperature_c"]],
                                     on="delivery_time", how="left", validate="many_to_one")
    masks = {}
    development = pd.Series(False, index=scored.index)
    for fold in run.spec["validation_folds"]:
        development |= scored.local_day.ge(fold["start"]) & scored.local_day.lt(fold["end"])
    masks["development"] = development
    for phase in ("calibration", "test"):
        dates = run.spec[phase]
        masks[phase] = scored.local_day.ge(dates["start"]) & scored.local_day.lt(dates["end"])
    rows = []
    for phase, mask in masks.items():
        phase_data = scored.loc[mask]
        evening = phase_data.delivery_time.dt.tz_convert("Europe/Paris").dt.hour.isin(run.spec["cold_evening"]["hours_local"])
        for arm in run.spec["arms"]:
            for component, field, target, valid_target in (
                ("cold", f"learned_cold_probability__{arm}",
                 phase_data.realized_temperature_c.lt(5).astype(float), np.isfinite(phase_data.realized_temperature_c)),
                ("underprediction", f"learned_underprediction_probability__{arm}",
                 (phase_data.actual_mw-phase_data.rte_at_origin_mw).gt(1000).astype(float),
                 np.isfinite(phase_data.actual_mw) & np.isfinite(phase_data.rte_at_origin_mw)),
            ):
                for segment, eligible in (("all_hours", pd.Series(True, index=phase_data.index)), ("evening", evening)):
                    valid = eligible & valid_target & np.isfinite(phase_data[field])
                    y = target.loc[valid].to_numpy()
                    p = phase_data.loc[valid, field].to_numpy()
                    if ((p < 0) | (p > 1)).any():
                        raise AssertionError("Learned classifier probability outside[0,1]")
                    both = len(np.unique(y)) == 2
                    rows.append({"phase": phase, "arm": arm, "component": component, "segment": segment, "scored": len(y),
                                 "missing_prediction_or_label": int((eligible & ~valid).sum()),
                                 "event_count": int(y.sum()), "event_base_rate": float(y.mean()) if len(y) else np.nan,
                                 "mean_predicted_probability": float(p.mean()) if len(p) else np.nan,
                                 "Brier_score": float(np.mean((y-p)**2)) if len(y) else np.nan,
                                 "ROC_AUC": float(roc_auc_score(y, p)) if both else np.nan,
                                 "PR_average_precision": float(average_precision_score(y, p)) if both else np.nan,
                                 "ranking_metric_status": "both_classes_observed" if both else "undefined_single_or_no_class",
                                 "interpretation": "Chronological out-of-sample learned feature; scores are diagnostic only; no refitting or feature selection"})
            field = f"learned_ridge_demand_mw__{arm}"
            valid = np.isfinite(phase_data.actual_mw) & np.isfinite(phase_data[field])
            good = phase_data.loc[valid]
            error = good.actual_mw-good[field]
            rte_good = good.loc[np.isfinite(good.rte_at_origin_mw)]
            rows.append({"phase": phase, "arm": arm, "component": "ridge_demand", "segment": "all_hours", "scored": len(good),
                         "missing_prediction_or_label": int((~valid).sum()), "MAE_MW": float(error.abs().mean()),
                         "RMSE_MW": float(np.sqrt((error**2).mean())),
                         "matched_RTE_MAE_MW": float((rte_good.actual_mw-rte_good.rte_at_origin_mw).abs().mean()),
                         "matched_ridge_MAE_MW": float((rte_good.actual_mw-rte_good[field]).abs().mean()),
                         "matched_RTE_scored": len(rte_good),
                         "interpretation": "Chronological out-of-sample linear prediction used as a feature; diagnostic only"})
    pd.DataFrame(rows).to_csv(run.path/"tables/learned_feature_oos_diagnostics.csv", index=False)


def _quantile_decision_scenarios(empirical, native, labels, spec):
    """Same predeclared cost-ratio quantile policy for every compared distribution.

    q = shortage_cost / (shortage_cost + surplus_cost). This is a hypothetical
    weighted-MW loss comparison; neither realized cold nor prices change q.
    """
    columns = ["model", "method", *KEYS, "q50", "q75", "q90", "q95"]
    blocks = [empirical.loc[empirical.model.isin(["rte_proxy", "rte_bias_proxy"]), columns]]
    if native is not None:
        blocks.append(native.loc[native.method.eq("native_quantile_lightgbm"), columns])
    values = pd.concat(blocks, ignore_index=True)
    rows = []
    for ratio, q_name, tau in ((1, "q50", .5), (3, "q75", .75), (9, "q90", .9), (19, "q95", .95)):
        common = values.pivot(index=KEYS, columns=["model", "method"], values=q_name).dropna().index.to_frame(index=False)
        matched = values.merge(common, on=KEYS, validate="many_to_one")
        scored = _decorate(matched, labels, spec)
        for (model, method), group in scored.groupby(["model", "method"]):
            good = group.dropna(subset=["actual_mw", q_name])
            error = good.actual_mw-good[q_name]
            loss = ratio*error.clip(lower=0)+(-error).clip(lower=0)
            for weight in spec["risk_scenarios"]["cold_weights"]:
                weights = np.where(good.cold_evening, weight, 1.)
                rows.append({"model": model, "method": method, "shortage_surplus_ratio": ratio,
                             "decision_quantile": tau, "cold_weight": weight, "scored": len(good),
                             "cold_scored": int(good.cold_evening.sum()),
                             "normalized_asymmetric_loss_MW": float(np.sum(weights*loss)/np.sum(weights)) if len(good) else np.nan,
                             "policy": "Predeclared q=ratio/(1+ratio) on every interval; same matched observations and cost rule",
                             "interpretation": "Hypothetical weighted-MW decision-loss proxy; not realized euro savings; no future cold decision trigger"})
    return pd.DataFrame(rows)


def _ablation_and_hybrid_tables(run, scores, frozen):
    indexed = scores.set_index("model")
    primary = frozen.get("selected_point") or frozen["diagnostic_best_point"]
    hybrids, ablations = [], []
    for model, entry in frozen["selections"].items():
        variant = entry["variant"]
        if variant.get("base"):
            control = f"lab__hybrid_control__{variant['arm']}"
            if control not in indexed.index:
                raise AssertionError(f"Missing same-arm hybrid control: {control}")
            row = {"model": model, "base_model": variant["base"], "arm": variant["arm"],
                   "control_model": control, "selected_in_development": model == primary,
                   "matched_scored": int(indexed.loc[model, "scored"]),
                   "comparison": "Same engineered/learned features, burn-in restriction, tuning budget and fit policy; base forecast feature added",
                   "interpretation": "Descriptive preplanned ablation, not test-based model selection; individual hyperparameters were tuned in development"}
            for metric in ("MAE_MW", "weighted_MAE_MW", "cold_MAE_MW"):
                candidate, baseline = indexed.loc[model, metric], indexed.loc[control, metric]
                row[metric] = candidate
                row["control_"+metric] = baseline
                row[metric+"_improvement_MW"] = baseline-candidate
                row[metric+"_improvement_percent"] = 100*(1-candidate/baseline) if baseline else np.nan
            hybrids.append(row)
    for arm in run.spec["arms"]:
        for tier in ("basic", "engineered", "stacked"):
            for weight in (1, 5):
                model = f"lab__lightgbm__{arm}__{tier}__w{weight}__residual"
                if model not in indexed.index:
                    raise AssertionError(f"Missing predeclared feature ablation: {model}")
                entry = frozen["selections"][model]
                row = {"model": model, "arm": arm, "feature_tier": tier,
                       "training_cold_weight_cap": weight,
                       "training_weight_rule": "1" if weight == 1 else "1+4*predicted_cold_probability*known_evening_flag",
                       "evaluation_weight_rule": "5 on realized cold evenings;1 otherwise, identical for every variant",
                       "selected_configuration": json.dumps(entry["params"], sort_keys=True),
                       "interpretation": "Preplanned feature/training-weight ablation; each configuration selected on development only; previously inspected historical test"}
                row.update({metric: indexed.loc[model, metric] for metric in
                            ("MAE_MW", "weighted_MAE_MW", "cold_MAE_MW", "scored", "cold_scored", "raw_availability", "fallback_fraction")})
                ablations.append(row)
    pd.DataFrame(hybrids).to_csv(run.path/"tables/hybrid_control_comparison.csv", index=False)
    pd.DataFrame(ablations).to_csv(run.path/"tables/feature_ablation.csv", index=False)
    return pd.DataFrame(ablations)


def _plots(run, scores, probability, frozen):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "figure.facecolor": "white", "axes.facecolor": "white"})
    figures = run.path/"figures"
    figures.mkdir(exist_ok=True)
    primary = frozen.get("selected_point") or frozen["diagnostic_best_point"]
    previous = _previous_development_selection(run)
    roles = [("rte_proxy", "RTE", "#71818a"),
             ("rte_bias_proxy", "Rolling correction", "#449488"),
             (previous, "Prior RTE-assisted", "#cc8434"),
             (primary, "Selected point", "#176b87")]
    if frozen.get("selected_quantile"):
        roles.append((frozen["selected_quantile"], "Selected quantile", "#8064a2"))
    # Roles and display order come from frozen development selections. Test
    # losses determine positions only, never the highlighted comparison set.
    roles = list({model: (model, label, color) for model, label, color in roles}.values())
    indexed = scores.set_index("model")
    roles = [role for role in roles if role[0] in indexed.index]
    frame = indexed.loc[[m for m, _, _ in roles]].reset_index()
    labels = [label for _, label, _ in roles]
    colors = [color for _, _, color in roles]
    selected = {m for m, _, _ in roles}
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.8), sharey=True, constrained_layout=True)
    y = np.arange(len(frame))
    for ax, column, title in zip(axes, ["MAE_MW", "weighted_MAE_MW", "cold_MAE_MW"],
                                 ["All-hour MAE", "Cold evenings weighted ×5", "Cold-evening MAE"]):
        bars = ax.barh(y, frame[column], color=colors, height=.65)
        ax.bar_label(bars, labels=[f"{v:,.0f}" for v in frame[column]], padding=4, fontsize=9)
        ax.set_yticks(y, labels)
        ax.set_xlim(0, frame[column].max()*1.18)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("MW; lower is better")
        ax.grid(axis="x", alpha=.18)
        ax.set_axisbelow(True)
    axes[0].invert_yaxis()
    fig.suptitle("Frozen development selections: historical test comparison", fontsize=13)
    fig.savefig(figures/"risk_point_comparison.png", dpi=170)
    plt.close(fig)
    # Keep the complete policy range, then show a reference-defined zoom. Its
    # limits use RTE, rolling correction and the prior frozen assisted policy;
    # neither a test winner nor a test-selected cluster determines the bounds.
    references = indexed.loc[["rte_proxy", "rte_bias_proxy", previous]]
    xbounds = (.5*references.MAE_MW.min(), 1.15*references.MAE_MW.max())
    ybounds = (.5*references.cold_MAE_MW.min(), 1.15*references.cold_MAE_MW.max())
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.8), constrained_layout=True)
    ordinary = scores.loc[~scores.model.isin(selected)]
    for ax in axes:
        ax.scatter(ordinary.MAE_MW, ordinary.cold_MAE_MW, s=25, alpha=.4,
                   color="#78858b", label="Other evaluated policies")
        for model, label, color in roles:
            row = indexed.loc[model]
            ax.scatter(row.MAE_MW, row.cold_MAE_MW, s=100, marker="D", color=color,
                       edgecolor="white", linewidth=.8, label=label, zorder=4)
        ax.set(xlabel="All-hour MAE (MW)", ylabel="Cold-evening MAE (MW)")
        ax.grid(alpha=.18)
    axes[0].set_title("Full overview: every evaluated policy", fontsize=11)
    axes[1].set(xlim=xbounds, ylim=ybounds, title="Zoom defined by reference errors")
    from matplotlib.patches import Rectangle
    axes[0].add_patch(Rectangle((xbounds[0], ybounds[0]), xbounds[1]-xbounds[0],
                               ybounds[1]-ybounds[0], fill=False, linestyle="--",
                               linewidth=1.1, edgecolor="#176b87"))
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="outside lower center", ncol=3, fontsize=9, frameon=False)
    fig.suptitle("Overall and cold-evening accuracy: lower is better on both axes", fontsize=12)
    fig.savefig(figures/"risk_error_tradeoff.png", dpi=170)
    plt.close(fig)
    cold = probability.loc[probability.segment_type.eq("cold_evening") & probability.segment.astype(str).eq("True")]
    cold = cold.loc[cold.central90_nobs.gt(0)].copy() if "central90_nobs" in cold else pd.DataFrame()
    if len(cold):
        probability_model = frozen.get("selected_quantile")
        fixed_model = probability_model+"__fixed_engineered" if probability_model else None
        interval_roles = [
            ("rte_proxy", EMPIRICAL, "RTE · residual intervals", "#71818a"),
            ("rte_bias_proxy", EMPIRICAL, "Rolling correction · residuals", "#449488"),
            (primary, EMPIRICAL, "Selected point · residuals", "#176b87"),
            (probability_model, "native_quantile_lightgbm", "Quantile model · native", "#8064a2"),
            (fixed_model, "fixed_engineered_native_quantiles", "Fixed model · native", "#a996be"),
            (fixed_model, "fixed_engineered_split_conformal", "Fixed model · split conformal", "#c09a70"),
            (fixed_model, "fixed_engineered_CQR", "Fixed model · CQR", "#b45e50"),
        ]
        entries = []
        for model, method, label, color in interval_roles:
            match = cold.loc[cold.model.eq(model) & cold.method.eq(method)]
            if len(match) == 1:
                entries.append(dict(match.iloc[0], display=label, color=color))
        cold = pd.DataFrame(entries)
        if len(cold):
            fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.5), sharey=True, constrained_layout=True)
            y = np.arange(len(cold))
            for ax, column, title in zip(axes, ["central90_coverage", "central90_upper_exceedance"],
                                         ["Central 90% interval coverage", "Actual demand above upper bound"]):
                values = cold[column].to_numpy()*100
                bars = ax.barh(y, values, color=cold.color, height=.66)
                ax.bar_label(bars, labels=[f"{v:.1f}%" for v in values], padding=4, fontsize=9)
                ax.set_yticks(y, cold.display)
                ax.set_title(title, fontsize=11)
                ax.set_xlabel("Percent of cold-evening hours")
                ax.grid(axis="x", alpha=.18)
                ax.set_axisbelow(True)
                ax.axhline(2.5, color="#d4d9dc", linewidth=.8)
            axes[0].invert_yaxis()
            axes[0].set_xlim(0, 105)
            axes[1].set_xlim(0, max(12., cold.central90_upper_exceedance.max()*118))
            coverage_line = axes[0].axvline(90, color="#a23b32", linestyle="--", label="90% coverage reference")
            tail_line = axes[1].axvline(5, color="#657b83", linestyle="--", label="5% upper-tail reference")
            count = cold.central90_nobs.astype(int)
            days = cold.central90_days.astype(int)
            support = (f"{count.iloc[0]} hours across {days.iloc[0]} days" if count.nunique() == days.nunique() == 1
                       else f"{count.min()}–{count.max()} hours across {days.min()}–{days.max()} days")
            fig.suptitle(f"Cold evenings: {support}\nFixed comparators use a separate predictor; references are not coverage guarantees", fontsize=12)
            fig.legend([coverage_line, tail_line], [coverage_line.get_label(), tail_line.get_label()],
                       loc="outside lower center", ncol=2, fontsize=9, frameon=False)
            fig.savefig(figures/"risk_cold_intervals.png", dpi=170)
            plt.close(fig)
    ablation = pd.read_csv(run.path/"tables/feature_ablation.csv")
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 5.0), sharey=True, constrained_layout=True)
    tiers = ["basic", "engineered", "stacked"]
    arms = ["national", "regional_only", "combined"]
    names = ["National history", "Regional history only", "Combined histories"]
    rte = scores.set_index("model").loc["rte_proxy", "weighted_MAE_MW"]
    bias = scores.set_index("model").loc["rte_bias_proxy", "weighted_MAE_MW"]
    for ax, arm, name in zip(axes, arms, names):
        for weight, color, label in ((1, "#176b87", "Uniform training weights"),
                                     (5, "#b04b37", "Forecast-cold training weights (cap 5)")):
            part = ablation.loc[ablation.arm.eq(arm) & ablation.training_cold_weight_cap.eq(weight)].set_index("feature_tier").reindex(tiers)
            ax.plot(range(3), part.weighted_MAE_MW, marker="o", linewidth=2, color=color, label=label)
        ax.axhline(rte, color="#777777", linestyle="--", linewidth=1.1, label="RTE")
        ax.axhline(bias, color="#8a8a8a", linestyle=":", linewidth=1.3, label="Rolling RTE correction")
        ax.set_xticks(range(3), ["Basic", "Engineered", "+ Learned\nfeatures"])
        ax.set_title(name)
        ax.set_xlim(-.15, 2.15)
        ax.grid(axis="y", alpha=.2)
    axes[0].set_ylabel("Weighted MAE (MW); lower is better")
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="outside lower center", ncol=2, fontsize=9, frameon=False)
    fig.suptitle("LightGBM residual feature ablation\nSame evaluation weights: cold evenings ×5; historical test previously inspected", fontsize=12)
    fig.savefig(figures/"risk_feature_ablation.png", dpi=180)
    plt.close(fig)


def _recommendation(run, scores, probability, bootstrap, frozen):
    primary = frozen.get("selected_point") or frozen["diagnostic_best_point"]
    row = scores.set_index("model").loc[primary]
    rte, bias = (scores.set_index("model").loc[m] for m in ("rte_proxy", "rte_bias_proxy"))
    previous_name = _previous_development_selection(run)
    previous = scores.set_index("model").loc[previous_name]
    previous_delta = {metric: float(previous[metric]-row[metric])
                      for metric in ("MAE_MW", "weighted_MAE_MW", "cold_MAE_MW")}
    prior_ci = bootstrap.loc[bootstrap.model.eq(primary) & bootstrap.reference.eq(previous_name) &
                             bootstrap.metric.eq("weighted_MAE_MW") & bootstrap.block_days.eq(14)]
    prior_bounds = "not estimable"
    if len(prior_ci) and pd.notna(prior_ci.iloc[0].get("absolute_low")):
        prior_bounds = f"[{prior_ci.iloc[0].absolute_low:+,.1f}, {prior_ci.iloc[0].absolute_high:+,.1f}] MW"
    previous_text = (f"The earlier RTE-assisted study selected **{previous_name}** using development data. "
                     f"On these same observations it has MAE **{previous.MAE_MW:,.1f} MW** and cold-evening MAE "
                     f"**{previous.cold_MAE_MW:,.1f} MW**. Relative to that prior selection, the new candidate's "
                     f"loss improvements are **{previous_delta['MAE_MW']:+,.1f} MW** overall, "
                     f"**{previous_delta['cold_MAE_MW']:+,.1f} MW** on cold evenings, and "
                     f"**{previous_delta['weighted_MAE_MW']:+,.1f} MW** under the declared weighting "
                     f"(positive is improvement; negative is regression). The weighted-loss improvement's "
                     f"paired 14-day-block 95% interval is **{prior_bounds}**.")
    prior_cold_ci = bootstrap.loc[bootstrap.model.eq(primary) & bootstrap.reference.eq(previous_name) &
                                  bootstrap.metric.eq("cold_MAE_MW") & bootstrap.block_days.eq(14)]
    if len(prior_cold_ci) and pd.notna(prior_cold_ci.iloc[0].get("absolute_low")):
        lower, upper = prior_cold_ci.iloc[0].absolute_low, prior_cold_ci.iloc[0].absolute_high
        previous_text += (f" The cold-evening improvement's corresponding interval is "
                          f"**[{lower:+,.1f}, {upper:+,.1f}] MW**. ")
        if lower <= 0 <= upper:
            previous_text += "Additional cold-evening point improvement over the prior policy is not established."
    qualified = frozen.get("selected_point") is not None
    development = frozen["selections"][primary]["metrics"]
    variant = frozen["selections"][primary]["variant"]
    control_text = ""
    if variant.get("base"):
        control = f"lab__hybrid_control__{variant['arm']}"
        control_score = scores.set_index("model").loc[control]
        evidence = bootstrap.loc[bootstrap.model.eq(primary) & bootstrap.reference.eq(control) &
                                 bootstrap.metric.eq("weighted_MAE_MW") & bootstrap.block_days.eq(14)]
        control_text = (f"The matching hybrid control has weighted MAE **{control_score.weighted_MAE_MW:,.1f} MW**. "
                        f"Adding the **{variant['base']}** forecast changes weighted loss by "
                        f"**{control_score.weighted_MAE_MW-row.weighted_MAE_MW:+,.1f} MW** (positive is improvement). ")
        if len(evidence) and pd.notna(evidence.iloc[0].get("absolute_low")):
            lower, upper = evidence.iloc[0].absolute_low, evidence.iloc[0].absolute_high
            control_text += f"The paired 14-day-block 95% interval is **[{lower:+,.1f}, {upper:+,.1f}] MW**. "
            if lower <= 0 <= upper:
                control_text += "This interval includes zero: incremental value from the base architecture is not established."
            elif lower > 0:
                control_text += "This comparison supports incremental value in this historical experiment; prospective confirmation remains necessary."
            else:
                control_text += "The comparison does not support incremental improvement over the matched control."
        else:
            control_text += "The uncertainty interval is unavailable; incremental architecture value is unconfirmed."
    calibration_support = pd.read_csv(run.path/"tables/calibration_support.csv").set_index("model").loc[primary]
    cold_calibration_n = int(calibration_support.cold_calibration_hours)
    calibration_warning = ("Calibration contains **zero cold-evening outcomes**; these uncertainty methods were not calibrated on that critical segment."
                           if cold_calibration_n == 0 else f"Calibration contains {cold_calibration_n} cold-evening outcomes.")
    ci = bootstrap.loc[bootstrap.model.eq(primary) & bootstrap.reference.eq("rte_proxy") & bootstrap.metric.eq("weighted_MAE_MW") & bootstrap.block_days.eq(14)]
    bounds = "not estimable" if ci.empty or "relative_low" not in ci else f"[{ci.iloc[0].relative_low:.2f}, {ci.iloc[0].relative_high:.2f}]%"
    cold = probability.loc[probability.model.eq(primary) & probability.method.eq(EMPIRICAL) & probability.segment_type.eq("cold_evening") & probability.segment.eq("True")]
    cold_text = "The selected model's empirical interval diagnostic is unavailable."
    if len(cold):
        q = cold.iloc[0]
        severity = "not estimable because there were no upper misses" if pd.isna(q.central90_upper_miss_conditional_severity_MW) else f"{q.central90_upper_miss_conditional_severity_MW:,.0f} MW conditional mean severity"
        cold_text = (f"Its past-calibration empirical 90% interval covers {q.central90_coverage:.1%} of cold-evening outcomes; "
                     f"upper-tail misses occur in {q.central90_upper_exceedance:.1%}, with {severity}. "
                     "These are dependent historical observations, not a conditional-coverage guarantee.")
    probability_text = ""
    critical_probability = probability.loc[probability.segment_type.eq("cold_evening") & probability.segment.eq("True")]
    for method, label in (("native_quantile_lightgbm", "The separately selected native quantile model"),
                          ("fixed_engineered_CQR", "The fixed engineered CQR comparator")):
        method_rows = critical_probability.loc[critical_probability.method.eq(method)]
        if len(method_rows) == 1:
            q = method_rows.iloc[0]
            probability_text += (f"{label}'s central 90% interval covers **{q.central90_coverage:.1%}** "
                                 f"of cold-evening outcomes, with **{q.central90_upper_exceedance:.1%}** upper misses. ")
            if method == "native_quantile_lightgbm":
                probability_text += (f"Its asymmetric 90% interval covers **{q.asymmetric90_coverage:.1%}**, "
                                     f"with **{q.asymmetric90_upper_exceedance:.1%}** upper misses against a nominal 2.5% tail. ")
    probability_text += "These results do not establish adequate cold-evening interval calibration."
    text = f"""# RTE-assisted feature and cold-risk research recommendation

## Improvement over RTE

The development-frozen point {'selection' if qualified else 'diagnostic candidate'} is **{primary}**.
{'It passed the predeclared development guardrails.' if qualified else 'No candidate passed every predeclared development guardrail; this candidate is diagnostic and is not a qualified winner.'}
On **{int(row.scored):,} matched nonmissing hours**, its MAE is **{row.MAE_MW:,.1f} MW**,
versus **{rte.MAE_MW:,.1f} MW** for RTE and **{bias.MAE_MW:,.1f} MW** for rolling RTE bias correction.
The RTE-relative MAE improvement is **{100*(1-row.MAE_MW/rte.MAE_MW):.2f}%**.
With cold evenings weighted five times, normalized weighted MAE is **{row.weighted_MAE_MW:,.1f} MW**;
the RTE-relative improvement is **{100*(1-row.weighted_MAE_MW/rte.weighted_MAE_MW):.2f}%**,
with paired 14-day-block 95% interval **{bounds}**. Scores use exactly the same origin-delivery pairs.
{previous_text}
{control_text}

## Uncertainty and failures

Cold-evening MAE is **{row.cold_MAE_MW:,.1f} MW**, versus RTE's **{rte.cold_MAE_MW:,.1f} MW**.
This segment contains **{int(row.cold_scored)} hours, {int(row.cold_days)} days, and {int(row.cold_DJF_winters)} DJF winters**.
The declared chronological development selection had **{int(development.get('cold_n', 0))} cold-evening hours across {int(development.get('cold_days', 0))} days**;
that limited support constrains claims about persistent winter superiority.
{cold_text}
{probability_text}
The probability tables report central and asymmetric intervals, pinball loss, central-interval WIS,
upper-tail severity, and exact CRPS only for a specified empirical residual distribution.
A sparse quantile grid is not labeled exact CRPS. Genuine peaks and missing-target exclusions are retained.
The quantile arm was selected using **unweighted overall development WIS**, without a cold-interval
selection guardrail; it is not established as optimal for cold evenings. The fixed engineered-only
comparators retain forecast-cold training weights while omitting learned predictors from their inputs.
Their native endpoints and calibrated bounds require empirical coverage assessment; nominal conditional
quantiles and cold-conditional coverage are not assumed.
Raw forecast availability is **{row.raw_availability:.2%}** and fallback frequency is **{row.fallback_fraction:.2%}**.
Worst-day causes remain unproven without archived weather forecasts and input-vintage evidence.

## Recommendation and risks

Retain the frozen point model as a candidate for the next unseen research evaluation.
The interval methods do not yet support reliable cold-evening reserve-risk quantification.
This is a retrospective research comparison. Use the supplied **rte_forecast_d1_mw** under the explicitly
assumed pre-noon availability rule; no next-but-one-day product or 05:00 rule enters this experiment.
The 11:00 Europe/Paris origin leaves a pre-auction buffer. Original issue vintages remain unverified.
All 13 original families are represented through clearly labeled base-model-plus-RTE-feature hybrids;
that does not imply native exogenous support in every neural or statistical architecture.
The national, regional-history-only, and combined feature arms share national consumption as evaluation truth.
Q1 2024 supplies causal stacking burn-in; the declared subsequent chronological folds select parameters and policies.
Calibration runs from **{run.spec['calibration']['start']}** up to but excluding **{run.spec['calibration']['end']}**.
All reference methods use that same calibration period. Its seasonal and cold-support limitations are recorded in the calibration-support table.
{calibration_warning}
The 2025–26 evaluation period was previously inspected, so prospective confirmation remains necessary.
Cold weights and shortage/surplus ratios are risk scenarios in weighted MW, not euro savings.
No production or procurement implementation is included. National demand remains a customer-portfolio proxy.
"""
    (run.path/"recommendation.md").write_text(text)
    summary = {"selected_point": frozen.get("selected_point"), "reported_candidate": primary,
               "qualified_in_development": qualified, "selected_quantile": frozen.get("selected_quantile"),
               "policies": len(scores), "matched_scored_hours": int(row.scored),
               "candidate_MAE_MW": row.MAE_MW, "RTE_MAE_MW": rte.MAE_MW, "bias_correction_MAE_MW": bias.MAE_MW,
               "candidate_weighted_MAE_MW": row.weighted_MAE_MW, "RTE_weighted_MAE_MW": rte.weighted_MAE_MW,
               "candidate_cold_MAE_MW": row.cold_MAE_MW, "cold_hours": int(row.cold_scored),
               "cold_days": int(row.cold_days), "cold_DJF_winters": int(row.cold_DJF_winters),
               "previous_development_selected_policy": previous_name,
               "previous_selected_MAE_MW": float(previous.MAE_MW),
               "previous_selected_weighted_MAE_MW": float(previous.weighted_MAE_MW),
               "previous_selected_cold_MAE_MW": float(previous.cold_MAE_MW),
               "loss_improvement_vs_previous_selected_MW": previous_delta,
               "weighted_improvement_vs_previous_14day_CI_MW": [float(prior_ci.iloc[0].absolute_low), float(prior_ci.iloc[0].absolute_high)]
                   if len(prior_ci) and pd.notna(prior_ci.iloc[0].get("absolute_low")) else None,
               "test_previously_explored": True, "operational_vintages_verified": False}
    write_json(run.path/"result_summary.json", summary)


def report(run):
    """Read frozen predictions, score every policy, and write review artifacts."""
    run.verify_manifest()
    frozen = json.loads((run.path/"frozen_selection.json").read_text())
    root = Path(run.root)
    labels, spec = run.data.labels, run.spec
    (run.path/"tables").mkdir(exist_ok=True)
    combined = {}
    for phase in ("calibration", "test"):
        sources = [root/spec["reference_run"]/"forecasts"/f"all_{phase}.csv.gz",
                   root/spec["rte_reference_run"]/"forecasts"/f"assisted_{phase}.csv.gz",
                   run.path/"forecasts"/f"lab_{phase}.csv.gz"]
        parts = [read_forecasts(path) for path in sources]
        lower = pd.Timestamp(spec[phase]["start"], tz="Europe/Paris").tz_convert("UTC")
        upper = pd.Timestamp(spec[phase]["end"], tz="Europe/Paris").tz_convert("UTC")
        parts = [part.loc[part.delivery_time.ge(lower) & part.delivery_time.lt(upper)].copy() for part in parts]
        common_cols = ["model", *KEYS, "prediction_mw", "raw_prediction_mw", "fallback_used"]
        frame = pd.concat([p[common_cols] for p in parts], ignore_index=True)
        if frame.duplicated(["model", *KEYS]).any():
            raise AssertionError("Duplicate or colliding policy forecast keys")
        combined[phase] = frame
        frame.to_csv(run.path/"forecasts"/f"all_{phase}.csv.gz", index=False)
    test, calibration = combined["test"], combined["calibration"]
    pivot = test.pivot(index=KEYS, columns="model", values="prediction_mw")
    common_keys = pivot.dropna().reset_index()[KEYS]
    matched = test.merge(common_keys, on=KEYS, validate="many_to_one")
    decorated = _decorate(matched, labels, spec)
    full = _decorate(test, labels, spec)
    records = []
    for model, group in decorated.groupby("model", sort=True):
        original = test.loc[test.model.eq(model)]
        records.append({"model": model, **_point(group), "raw_availability": float(original.raw_prediction_mw.notna().mean()),
                        "fallback_fraction": float(original.fallback_used.astype(bool).mean())})
    scores = pd.DataFrame(records)
    for reference, label in (("rte_proxy", "RTE"), ("rte_bias_proxy", "bias_correction")):
        ref = scores.set_index("model").loc[reference]
        for metric in ("MAE_MW", "weighted_MAE_MW", "cold_MAE_MW"):
            scores[f"{metric}_improvement_vs_{label}_MW"] = ref[metric]-scores[metric]
            scores[f"{metric}_improvement_vs_{label}_percent"] = 100*(1-scores[metric]/ref[metric]) if ref[metric] else np.nan
    scores = scores.sort_values("weighted_MAE_MW")
    scores.to_csv(run.path/"tables/test_scorecard.csv", index=False)
    _development_fit_diagnostics(run, frozen)
    _learned_feature_diagnostics(run)
    _ablation_and_hybrid_tables(run, scores, frozen)
    pd.DataFrame([{"model": m, **_point(g)} for m, g in full.groupby("model")]).to_csv(run.path/"tables/unmatched_scorecard.csv", index=False)
    pd.DataFrame([{"model": m, **_point(g, "raw_prediction_mw")} for m, g in full.groupby("model")]).to_csv(run.path/"tables/raw_scorecard.csv", index=False)
    segments = []
    for dimension in ("hour", "season", "lead_group", "cold_evening"):
        for (model, value), group in decorated.groupby(["model", dimension], observed=True):
            segments.append({"model": model, "segment_type": dimension, "segment": str(value), **_point(group)})
    pd.DataFrame(segments).to_csv(run.path/"tables/test_segments.csv", index=False)
    primary = frozen.get("selected_point") or frozen["diagnostic_best_point"]
    roles = {primary, "rte_proxy", "rte_bias_proxy"}
    if frozen.get("selected_quantile"):
        roles.add(frozen["selected_quantile"])
    comparisons = [(candidate, reference) for candidate in sorted(roles-{ "rte_proxy", "rte_bias_proxy"})
                   for reference in ("rte_proxy", "rte_bias_proxy")]
    previous_name = _previous_development_selection(run)
    _previous = scores.loc[scores.model.eq(previous_name)]
    if len(_previous) != 1:
        raise AssertionError("Prior development-selected complete policy missing from matched comparison")
    comparisons.append((primary, previous_name))
    primary_variant = frozen["selections"][primary]["variant"]
    if primary_variant.get("base"):
        comparisons.append((primary, f"lab__hybrid_control__{primary_variant['arm']}"))
    bootstrap = pd.DataFrame([row for candidate, reference in comparisons
                              for row in _bootstrap(decorated, candidate, reference, spec)])
    bootstrap.to_csv(run.path/"tables/bootstrap.csv", index=False)
    daily = daily_metrics(matched, labels)
    daily.to_csv(run.path/"tables/daily_metrics.csv", index=False)
    worst = daily.loc[daily.model.eq(primary)].nlargest(10, "absolute_daily_energy_error_MWh").copy()
    worst["cause"] = "Unexplained; calendar/weather/shift/input attribution requires separate evidence"
    worst.to_csv(run.path/"tables/worst_days.csv", index=False)
    _risk_scenarios(decorated, spec).to_csv(run.path/"tables/asymmetric_risk_scenarios.csv", index=False)
    probability, retained = _empirical_probabilities(calibration, test, labels, spec, roles)
    retained.to_csv(run.path/"forecasts/selected_empirical_intervals.csv.gz", index=False)
    native_path = run.path/"forecasts/probabilistic_test.csv.gz"
    native = None
    if native_path.exists():
        native = read_forecasts(native_path)
        native_rows = []
        for (model, method), group in native.groupby(["model", "method"]):
            if group.duplicated(KEYS).any():
                raise AssertionError("Duplicate model/method probabilistic forecasts")
            native_rows.extend(_probability_rows(_decorate(group, labels, spec), spec, model, method))
        probability = pd.concat([probability, pd.DataFrame(native_rows)], ignore_index=True)
    _quantile_decision_scenarios(retained, native, labels, spec).to_csv(run.path/"tables/quantile_decision_risk_scenarios.csv", index=False)
    cal_support = _decorate(calibration, labels, spec)
    first_origin = test.origin.min()
    cal_support = cal_support.loc[cal_support.label_available_at.lt(first_origin) & cal_support.actual_mw.notna()]
    support = cal_support.groupby("model").agg(calibration_hours=("actual_mw", "size"),
        calibration_days=("day", "nunique"), cold_calibration_hours=("cold_evening", "sum"),
        calibration_first_delivery=("delivery_time", "min"), calibration_last_delivery=("delivery_time", "max"),
        calibration_latest_outcome_available=("label_available_at", "max"))
    support.to_csv(run.path/"tables/calibration_support.csv")
    probability.to_csv(run.path/"tables/uncertainty_segments.csv", index=False)
    probability.loc[probability.segment_type.eq("all")].to_csv(run.path/"tables/uncertainty_scorecard.csv", index=False)
    probability.loc[probability.segment_type.eq("cold_evening") & probability.segment.eq("True")].to_csv(run.path/"tables/cold_interval_scorecard.csv", index=False)
    _plots(run, scores, probability, frozen)
    _recommendation(run, scores, probability, bootstrap, frozen)
    write_json(run.path/"tables/reporting_specification.json", {
        "policy_count": test.model.nunique(), "common_origin_delivery_pairs": len(common_keys),
        "weighted_MAE": "sum(weight*abs(error))/sum(weight); weight=5 for realized cold evenings, else1",
        "central_WIS": "[0.5*absolute_median_error+0.05*IS90+0.25*IS50]/2.5",
        "asymmetric90": "q075 to q975; nominal lower7.5% and upper2.5% exceedances; endpoint pinball reported",
        "CRPS": "Exact for specified empirical past-calibration residual distribution only; no exact sparse-grid CRPS",
        "upper_tail_twCRPS": "Exact CRPS of outcome and empirical forecast members clipped below at a fixed 2023 national-demand 90th percentile; threshold-weight function1{x>=threshold}",
        "conditional_severity_zero_events": "NaN; conditioning event not observed",
        "uncertainty_scope": "Empirical residual scorecards cover all policies; native/conformal methods retain their own frozen predictor labels",
        "selection": "Reporting consumes frozen selections and does not select from test scores"})
    return scores, probability, bootstrap


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir")
    args = parser.parse_args()
    from src.rte_risk_lab import RiskLabRun
    run = RiskLabRun(ROOT, Path(args.run_dir)) if args.run_dir else RiskLabRun.latest(ROOT)
    report(run)


if __name__ == "__main__":
    main()
