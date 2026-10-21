from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import gzip
import hashlib
import json
import shutil
import uuid

import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline

from .common import TZ, fingerprint, write_json
from .features import CALENDAR, HISTORY, PAST_WEATHER, FORECAST_WEATHER, ORACLE_WEATHER, RTE, prediction_fallback


@dataclass(frozen=True)
class Experiment:
    name: str
    hypothesis: str
    groups: tuple = ("calendar", "history")
    window_days: int | None = 365
    retrain_days: int = 28
    max_leaf_nodes: int = 15
    max_iter: int = 70
    correction: bool = False
    oracle: bool = False
    status: str = "retrospective"


def experiments():
    return [
        Experiment("calendar", "Calendar alone explains recurring demand profiles.", groups=("calendar",)),
        Experiment("history_365", "Available demand history improves calendar-only forecasts."),
        Experiment("history_730", "A two-year window improves seasonal coverage.", window_days=730),
        Experiment("history_expanding", "Longer history outweighs structural drift.", window_days=None),
        Experiment("history_weekly_fit", "More frequent retraining tracks changes.", retrain_days=7),
        Experiment("history_more_leaves", "Modestly higher capacity captures interactions.", max_leaf_nodes=31),
        Experiment("history_past_weather", "Temperature already observed at cutoff adds information.", groups=("calendar", "history", "past_weather")),
        Experiment("rte_correction_proxy", "Calendar/history predict residual RTE errors beyond trailing bias.", groups=("calendar", "history", "rte"), correction=True),
        Experiment("weather_archive", "Authentic archived weather improves demand forecasts.", groups=("calendar", "history", "forecast_weather"), status="gated_no_verified_archive"),
        Experiment("oracle_weather_sensitivity", "Recorded future weather measures sensitivity, not deployable skill.", groups=("calendar", "history", "oracle_weather"), oracle=True, status="oracle_sensitivity_only"),
    ]


def feature_columns(experiment):
    groups = {"calendar": CALENDAR, "history": HISTORY, "past_weather": PAST_WEATHER, "forecast_weather": FORECAST_WEATHER, "rte": RTE, "oracle_weather": ORACLE_WEATHER}
    return list(dict.fromkeys(column for group in experiment.groups for column in groups[group]))


def new_model(experiment, seed=42, quantile=None):
    kwargs = {"loss": "absolute_error" if quantile is None else "quantile", "learning_rate": .08, "max_iter": experiment.max_iter, "max_leaf_nodes": experiment.max_leaf_nodes, "max_bins": 64, "min_samples_leaf": 40, "l2_regularization": 1., "early_stopping": False, "random_state": seed}
    if quantile is not None:
        kwargs["quantile"] = quantile
    # Imputation parameters are learned only from each training slice.
    return make_pipeline(SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True), HistGradientBoostingRegressor(**kwargs))


def fit_at_origin(features, labels, origin, experiment, contract, quantile=None, demand_archive=None):
    train = features.loc[features.origin < origin].merge(labels[["delivery_time", "actual_mw", "label_available_at"]], on="delivery_time", validate="many_to_one")
    if demand_archive is not None:
        versions = demand_archive.loc[(demand_archive.available_at < origin) & (demand_archive.valid_time + pd.Timedelta(hours=1) <= origin)].sort_values("available_at").drop_duplicates("valid_time", keep="last").set_index("valid_time")
        train["actual_mw"] = versions.value.reindex(train.delivery_time).to_numpy()
        train["label_available_at"] = versions.available_at.reindex(train.delivery_time).to_numpy()
        train["label_available_at"] = pd.to_datetime(train.label_available_at, utc=True)
    train = train.loc[train.label_available_at.lt(origin) & train.actual_mw.notna()]
    if experiment.window_days is not None:
        train = train.loc[train.delivery_time >= origin-pd.Timedelta(days=experiment.window_days)]
    if experiment.correction:
        train = train.loc[train.rte_at_origin_mw.notna()]
    if train.empty or (train.delivery_time.max()-train.delivery_time.min()) < pd.Timedelta(days=contract["minimum_training_days"]-7):
        return None, {"fit_origin": origin, "training_rows": len(train), "status": "insufficient_training"}
    columns = feature_columns(experiment)
    if train.max_history_available_at.dropna().ge(train.loc[train.max_history_available_at.notna(), "origin"]).any():
        raise AssertionError("Historical feature availability violates an origin cutoff.")
    y = train.actual_mw-train.rte_at_origin_mw if experiment.correction else train.actual_mw
    model = new_model(experiment, contract["random_seed"], quantile)
    model.fit(train[columns], y)
    return model, {"fit_origin": origin, "training_rows": len(train), "latest_label_available_at": train.label_available_at.max(), "status": "fit", "quantile": quantile}


def archive_predictions(root, name, predictions, contract, origin_ledger=None):
    path = root / "reports/forecasts" / f"{name}.csv.gz"
    path.parent.mkdir(parents=True, exist_ok=True)
    # This function must run BEFORE evaluation joins any corresponding outcomes.
    if "actual_mw" in predictions:
        raise AssertionError("Forecast archives must not contain evaluation outcomes.")
    archive_dir=path.parent/"immutable"
    archive_dir.mkdir(exist_ok=True)
    if path.exists():
        previous=archive_dir/f"{name}_previous_{fingerprint(path)[:16]}.csv.gz"
        if not previous.exists(): shutil.copy2(path,previous)
    payload=predictions.to_csv(index=False).encode()
    digest=hashlib.sha256(payload).hexdigest()
    immutable=archive_dir/f"{name}_{digest}.csv.gz"
    if not immutable.exists():
        with immutable.open("xb") as stream: stream.write(gzip.compress(payload,mtime=0))
    temporary=path.with_suffix(".tmp")
    temporary.write_bytes(immutable.read_bytes())
    temporary.replace(path)
    manifest={"forecast_sha256": fingerprint(path),"csv_payload_sha256":digest,"immutable_archive":str(immutable.relative_to(root)), "contract_sha256": fingerprint(root / "config/forecast_contract.json"), "rows": len(predictions), "predictions_archived_before_scoring": True, "regime": "retrospective_assumed_availability", "test_previously_explored": contract["test_previously_explored"]}
    if origin_ledger is not None:
        manifest.update({"origin_ledger":str(origin_ledger.relative_to(root)),"origin_ledger_sha256":fingerprint(origin_ledger),"each_origin_saved_before_next_origin_fit":True})
    write_json(path.with_suffix(".manifest.json"),manifest)
    write_json(immutable.with_suffix(".manifest.json"),manifest)
    return path


def baseline_replay(features, start, end, root, contract, phase="validation"):
    selected = features.loc[features.local_day.ge(start) & features.local_day.lt(end)].copy()
    outputs = []
    definitions = {"seasonal_naive": selected.week_lag_1, "comparable_days": selected.comparable_mean, "rte_proxy": selected.rte_at_origin_mw, "rte_bias_proxy": selected.rte_at_origin_mw + selected.rte_bias_recent_mw}
    for name, raw in definitions.items():
        prediction, fallback = prediction_fallback(selected, raw, rte_assisted=name.startswith("rte"))
        out = selected[["origin", "delivery_time", "local_day"]].copy()
        out["model"] = name
        out["raw_prediction_mw"] = raw
        out["prediction_mw"] = prediction
        out["fallback_source"] = fallback.where(raw.isna(), "baseline")
        out["fallback_used"] = raw.isna() & prediction.notna()
        out["regime"] = "retrospective_assumed_availability"
        outputs.append(out)
    result = pd.concat(outputs, ignore_index=True)
    archive_predictions(root, f"baselines_{phase}", result, contract)
    return result


def model_replay(features, labels, start, end, experiment, root, contract, phase="validation", quantiles=False, demand_archive=None):
    if experiment.status.startswith("gated"):
        raise RuntimeError("Authentic weather model is gated until provenance/archives are supplied.")
    evaluation = features.loc[features.local_day.ge(start) & features.local_day.lt(end)]
    ledger_dir=root/"reports/forecasts/origin_events"
    ledger_dir.mkdir(parents=True,exist_ok=True)
    ledger_path=ledger_dir/f"{experiment.name}_{phase}_{uuid.uuid4().hex}.jsonl.gz"
    fitted = None; quantile_models = {}; last_fit = None; logs = []; outputs = []
    for origin, day in evaluation.groupby("origin", sort=True):
        if last_fit is None or origin-last_fit >= pd.Timedelta(days=experiment.retrain_days):
            fitted, log = fit_at_origin(features, labels, origin, experiment, contract, demand_archive=demand_archive)
            logs.append(log); last_fit = origin
            if quantiles:
                quantile_models = {}
                for q in contract["quantiles"]:
                    quantile_models[q], qlog = fit_at_origin(features, labels, origin, experiment, contract, quantile=q, demand_archive=demand_archive)
                    logs.append(qlog)
        raw = np.full(len(day), np.nan)
        if fitted is not None:
            raw = fitted.predict(day[feature_columns(experiment)])
            if experiment.correction:
                raw += day.rte_at_origin_mw.to_numpy()
            # Invalid/nonpositive model values invoke the declared fallback.
            raw[~np.isfinite(raw) | (raw <= 0)] = np.nan
            if "history" in experiment.groups:
                raw[day.history_fraction_24h.to_numpy() < .5] = np.nan
            if "forecast_weather" in experiment.groups:
                raw[day.weather_forecast_c.isna().to_numpy()] = np.nan
        prediction, fallback = prediction_fallback(day, raw, experiment.correction)
        out = day[["origin", "delivery_time", "local_day"]].copy()
        out["model"] = experiment.name
        out["raw_prediction_mw"] = raw
        out["prediction_mw"] = prediction.to_numpy()
        out["fallback_source"] = fallback.to_numpy()
        out["fallback_used"] = np.isnan(raw) & prediction.notna().to_numpy()
        out["fit_origin"] = last_fit
        out["regime"] = "oracle_future_weather_sensitivity" if experiment.oracle else "retrospective_assumed_availability"
        if quantiles:
            for q, model in quantile_models.items():
                values = model.predict(day[feature_columns(experiment)]) if model is not None else np.full(len(day), np.nan)
                if experiment.correction:
                    values += day.rte_at_origin_mw.to_numpy()
                values[np.isnan(raw)] = np.nan
                out[f"q{int(q*100):02d}"] = values
            # Quantile crossing is repaired explicitly by sorting predicted bounds.
            names = ["q05", "q50", "q95"]
            out["quantile_crossing_repaired"] = (out.q05 > out.q50) | (out.q50 > out.q95)
            out[names] = np.sort(out[names].to_numpy(), axis=1)
        # Close the compressed append after EACH origin, before the next loop
        # can fit using outcomes from this forecast day. The ledger has a unique
        # run ID and is never overwritten, including during notebook reruns.
        event={"origin":origin.isoformat(),"model":experiment.name,"phase":phase,"predictions":json.loads(out.to_json(orient="records",date_format="iso"))}
        with gzip.open(ledger_path,"at",encoding="utf-8") as stream:
            stream.write(json.dumps(event)+"\n")
        outputs.append(out)
    result = pd.concat(outputs, ignore_index=True)
    archive_predictions(root, f"{experiment.name}_{phase}", result, contract,origin_ledger=ledger_path)
    pd.DataFrame(logs).to_csv(root / "reports/models" / f"{experiment.name}_{phase}_fits.csv", index=False)
    joblib.dump({"point_pipeline":fitted,"quantile_pipelines":quantile_models,"feature_columns":feature_columns(experiment),"experiment":asdict(experiment),"last_fit_origin":last_fit,"contract_sha256":fingerprint(root/"config/forecast_contract.json")},root/"reports/models"/f"{experiment.name}_{phase}_pipelines.joblib")
    return result


def select_candidate(scores, segments, registry, contract):
    allowed = {e.name for e in registry if not e.oracle and not e.status.startswith("gated")}
    scores = scores.loc[scores.model.isin(allowed)].copy()
    cold = segments.loc[(segments.segment_type == "cold_evening") & (segments.segment == "True")].set_index("model")
    rte_cold = cold.loc["rte_proxy", "MAE_MW"] if "rte_proxy" in cold.index else np.nan
    scores["cold_MAE_ratio_to_RTE"] = scores.model.map(cold.MAE_MW / rte_cold)
    scores["passes_guardrails"] = scores.availability.ge(contract["guardrails"]["minimum_policy_availability"]) & scores.cold_MAE_ratio_to_RTE.le(contract["guardrails"]["maximum_cold_MAE_ratio_to_RTE"])
    passed = scores.loc[scores.passes_guardrails]
    complexity = {"calendar": 0, "history_365": 1, "history_730": 2, "history_expanding": 3, "history_past_weather": 4, "history_more_leaves": 5, "history_weekly_fit": 6, "rte_correction_proxy": 7}
    if len(passed):
        near_best = passed.loc[passed.MAE_MW <= passed.MAE_MW.min()*(1+contract["simplicity_tolerance_percent"]/100)].copy()
        near_best["complexity_rank"] = near_best.model.map(complexity).fillna(99)
        winner = near_best.sort_values(["complexity_rank", "MAE_MW", "model"]).iloc[0].model
    else:
        winner = None
    return winner, scores


def stress_inputs(features, scenario):
    changed = features.copy()
    if scenario == "missing_demand_lags":
        changed[HISTORY] = np.nan
    elif scenario == "missing_rte":
        changed[RTE] = np.nan
    elif scenario == "stale_weather":
        changed[FORECAST_WEATHER] = np.nan
        changed["weather_run_status"] = "stale"
    elif scenario == "rapid_demand_change":
        columns = [c for c in HISTORY if c != "history_fraction_24h"]
        changed[columns] = changed[columns] * 1.2
    else:
        raise ValueError(scenario)
    return changed
