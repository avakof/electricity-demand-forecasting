"""Fixed, chronological first-stage predictions for the RTE feature laboratory.

Every bank value is a prediction from a model fitted before that value's
forecast origin. There are no fitted training predictions in the bank. These
models are deliberately fixed rather than selected using development/test
scores. National future temperature is a released training label for the cold
classifier, including in the regional-only arm; it is never an input feature.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from .common import fingerprint
from .features import HISTORY, PAST_WEATHER


ARMS = ("national", "regional_only", "combined")
KEYS = ["origin", "delivery_time"]
SPECIFICATION = {
    "version": "chronological-learned-features-v1",
    "window_days": 365,
    "minimum_training_days": 90,
    "refit": "First available origin in each local calendar month",
    "ridge_alpha": 100.0,
    "logistic_C": 0.1,
    "logistic_max_iter": 300,
    "numeric_thread_limit": 1,
    "cold_below_c": 5.0,
    "underprediction_above_mw": 1000.0,
    "temperature_release": "Explicit temperature_available_at when supplied; otherwise the same assumed interval-end plus publication delay as demand",
    "minimum_history_policy": "Missing predictions until the next scheduled fit with at least 90 days of released history",
    "one_class_policy": "Empirical training event frequency; no artificial certainty from a failed classifier",
}


def learned_columns(arm):
    if arm not in ARMS:
        raise ValueError(f"Unknown learned-feature arm: {arm}")
    return [f"learned_ridge_demand_mw__{arm}",
            f"learned_cold_probability__{arm}",
            f"learned_underprediction_probability__{arm}"]


def _utc(series, name):
    if not isinstance(series.dtype, pd.DatetimeTZDtype):
        raise ValueError(f"{name} must contain timezone-aware timestamps")
    return series.dt.tz_convert("UTC")


def _numeric(frame):
    return frame.astype(float).replace([np.inf, -np.inf], np.nan)


def _pipeline(estimator):
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("scaler", StandardScaler()),
        ("estimator", estimator),
    ])


def _column_map(data, arms):
    result = {}
    for arm in arms:
        columns = (data.engineered_columns(arm) if hasattr(data, "engineered_columns")
                   else data.columns(arm, tier="engineered"))
        columns = list(dict.fromkeys(columns))
        forbidden = {"actual_mw", "realized_temperature_c", "load_mw", "temperature_c",
                     "label_available_at", "temperature_available_at", *KEYS}
        if forbidden.intersection(columns) or any(c.startswith(("learned_", "oracle_")) for c in columns):
            raise AssertionError("Outcome, timestamp, or recursive learned feature in first-stage predictors")
        if arm == "regional_only" and (any(c.startswith("country_") for c in columns)
                or set(columns).intersection(HISTORY+PAST_WEATHER+["rte_bias_recent_mw"])):
            raise AssertionError("National observed-history predictor in regional-only arm")
        if not columns or set(columns)-set(data.features):
            raise ValueError(f"Missing engineered predictors for {arm}")
        if "rte_at_origin_mw" not in columns:
            raise ValueError("Every learned-feature arm must include the supplied RTE D1 feature")
        result[arm] = columns
    return result


def _frame_hash(frame):
    digest = hashlib.sha256()
    digest.update(json.dumps([(c, str(frame[c].dtype)) for c in frame], sort_keys=True).encode())
    digest.update(pd.util.hash_pandas_object(frame, index=False).to_numpy().tobytes())
    return digest.hexdigest()


def _cache_inputs(data, features, labels, column_map):
    source_dir = Path(__file__).parent
    sources = ("rte_learned_features.py", "rte_feature_lab.py", "research_data.py", "features.py", "common.py", "data.py")
    raw = {}
    if hasattr(data, "root"):
        for name in ("france_electricity_hourly.csv", "france_regional_electricity_hourly.csv"):
            path = Path(data.root)/"data"/name
            if path.exists():
                raw[name] = fingerprint(path)
    return {
        "specification": SPECIFICATION,
        "timezone": data.contract.get("timezone", "Europe/Paris"),
        "columns": column_map,
        "feature_sha256": _frame_hash(features),
        "labels_sha256": _frame_hash(labels),
        "source_sha256": {name: fingerprint(source_dir/name) for name in sources if (source_dir/name).exists()},
        "raw_sha256": raw,
    }


def _fit_component(training, prediction, columns, target, released_at, origin, *, classification):
    """Return honest predictions and auditable metadata, including one-class fits."""
    valid = target.notna() & np.isfinite(target) & released_at.notna() & released_at.lt(origin)
    train = training.loc[valid]
    y = target.loc[valid].astype(float)
    released = released_at.loc[valid]
    if len(train):
        if not train.origin.lt(origin).all() or not released.lt(origin).all():
            raise AssertionError("Training origin or label not strictly before learned-feature fit")
        known = train.max_history_available_at.notna()
        if not train.loc[known, "max_history_available_at"].lt(train.loc[known, "origin"]).all():
            raise AssertionError("Training feature observation not released before its original origin")
    span = (train.delivery_time.max()-train.delivery_time.min()).total_seconds()/86400 if len(train) else 0.0
    record = {
        "training_rows": len(train), "training_span_days": span,
        "latest_label_available_at": str(released.max()) if len(train) else None,
        "earliest_training_origin": str(train.origin.min()) if len(train) else None,
        "latest_training_origin": str(train.origin.max()) if len(train) else None,
        "original_feature_cutoffs_verified": True,
        "predictor_columns": columns,
    }
    if span < SPECIFICATION["minimum_training_days"]:
        record["status"] = "insufficient_history"
        return np.full(len(prediction), np.nan), record
    if classification and y.nunique() == 1:
        # This is the observed frequency in the eligible training window. The
        # ledger distinguishes it from an estimated logistic probability.
        prior = float(y.mean())
        record.update(status="empirical_one_class", event_probability=prior,
                      positive_training_rows=int(y.sum()), convergence_warning=False)
        return np.full(len(prediction), prior), record
    estimator = (LogisticRegression(C=SPECIFICATION["logistic_C"], solver="lbfgs",
                    max_iter=SPECIFICATION["logistic_max_iter"], random_state=42)
                 if classification else Ridge(alpha=SPECIFICATION["ridge_alpha"]))
    model = _pipeline(estimator)
    with threadpool_limits(limits=1), warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(_numeric(train[columns]), y)
        values = (model.predict_proba(_numeric(prediction[columns]))[:, 1] if classification
                  else model.predict(_numeric(prediction[columns])))
    record.update(status="fit", convergence_warning=any(issubclass(w.category, ConvergenceWarning) for w in captured))
    if classification:
        record["positive_training_rows"] = int(y.sum())
    return np.asarray(values, dtype=float), record


def build_learned_feature_bank(data, cache_dir=None, arms=ARMS):
    """Return origin/delivery keys plus three chronological predictors per arm.

    The optional cache is addressed by source, raw-data, engineered-feature,
    label and fixed-specification hashes. ``result.attrs['fit_ledger']`` records
    every monthly component fit. A temperature label without its own release
    timestamp uses the supplied demand release assumption explicitly; that
    assumption does not establish historical publication vintages.

    The input data object and its predictor columns are not modified. All rows
    retain their historical origin, including 23/25-hour daylight-saving days.
    Missing warm-up predictions remain missing for downstream fold-fitted
    imputation; no future fit is used to backfill them.
    """
    arms = tuple(arms)
    if not arms or len(set(arms)) != len(arms) or set(arms)-set(ARMS):
        raise ValueError("arms must be a nonempty unique subset of the declared arms")
    column_map = _column_map(data, arms)
    all_columns = list(dict.fromkeys(c for values in column_map.values() for c in values))
    features = data.features[KEYS+["max_history_available_at"]+all_columns].copy()
    for name in KEYS+["max_history_available_at"]:
        features[name] = _utc(features[name], name)
    if features[KEYS].isna().any().any() or features.duplicated(KEYS).any():
        raise AssertionError("Missing or duplicate forecast keys in learned-feature source")
    known = features.max_history_available_at.notna()
    if not features.loc[known, "max_history_available_at"].lt(features.loc[known, "origin"]).all():
        raise AssertionError("Feature observation not released before its original origin")
    features = features.sort_values(KEYS).reset_index(drop=True)
    label_columns = ["delivery_time", "actual_mw", "label_available_at", "realized_temperature_c"]
    if "temperature_available_at" in data.labels:
        label_columns.append("temperature_available_at")
    labels = data.labels[label_columns].copy()
    for name in ("delivery_time", "label_available_at", "temperature_available_at"):
        if name in labels:
            labels[name] = _utc(labels[name], name)
    if labels.delivery_time.isna().any() or labels.delivery_time.duplicated().any():
        raise AssertionError("Missing or duplicate delivery labels in learned-feature source")
    if "temperature_available_at" not in labels:
        labels["temperature_available_at"] = labels.label_available_at
    interval_end = labels.delivery_time+pd.Timedelta(hours=data.contract.get("interval_duration_hours", 1))
    for name in ("label_available_at", "temperature_available_at"):
        known = labels[name].notna()
        if labels.loc[known, name].lt(interval_end.loc[known]).any():
            raise AssertionError(f"{name} precedes the observed delivery interval end")
    labels = labels.sort_values("delivery_time").reset_index(drop=True)
    inputs = _cache_inputs(data, features, labels, column_map)
    signature = hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest()
    cache = Path(cache_dir)/signature if cache_dir is not None else None
    if cache is not None and (cache/"manifest.json").exists():
        manifest = json.loads((cache/"manifest.json").read_text())
        if manifest["signature"] != signature or manifest["inputs"] != inputs:
            raise RuntimeError("Learned-feature cache identity mismatch")
        for name in ("features.parquet", "fit_ledger.json"):
            if not (cache/name).exists() or fingerprint(cache/name) != manifest["artifact_sha256"][name]:
                raise RuntimeError(f"Altered learned-feature cache: {name}")
        result = pd.read_parquet(cache/"features.parquet")
        result.attrs.update(fit_ledger=json.loads((cache/"fit_ledger.json").read_text()),
                            cache_signature=signature, cache_path=str(cache), specification=SPECIFICATION.copy())
        return result
    merged = features.merge(labels, on="delivery_time", how="left", validate="many_to_one")
    result = features[KEYS].copy()
    for arm in arms:
        for column in learned_columns(arm):
            result[column] = np.nan
    months = features.origin.dt.tz_convert(data.contract.get("timezone", "Europe/Paris")).dt.strftime("%Y-%m")
    ledger = []
    for month, prediction in features.groupby(months, sort=True):
        origin = prediction.origin.min()
        training = merged.loc[merged.origin.lt(origin)
            & merged.delivery_time.ge(origin-pd.Timedelta(days=SPECIFICATION["window_days"]))]
        # Temperature and demand release masks are evaluated independently.
        # Unreleased outcomes can exist in this table but cannot enter a fit.
        cold = training.realized_temperature_c.lt(SPECIFICATION["cold_below_c"]).astype(float)
        cold = cold.where(np.isfinite(training.realized_temperature_c))
        residual = training.actual_mw-training.rte_at_origin_mw
        under = residual.gt(SPECIFICATION["underprediction_above_mw"]).astype(float).where(np.isfinite(residual))
        tasks = (("ridge_demand", training.actual_mw, training.label_available_at, False),
                 ("cold", cold, training.temperature_available_at, True),
                 ("underprediction", under, training.label_available_at, True))
        for arm in arms:
            for column, (component, target, released, classification) in zip(learned_columns(arm), tasks):
                values, record = _fit_component(training, prediction, column_map[arm], target, released,
                                                origin, classification=classification)
                result.loc[prediction.index, column] = values
                ledger.append({"arm": arm, "component": component, "feature": column,
                    "fit_origin": str(origin), "month": month,
                    "first_prediction_origin": str(prediction.origin.min()),
                    "last_prediction_origin": str(prediction.origin.max()),
                    "prediction_rows": len(prediction), "out_of_sample_bank": True, **record})
    if cache is not None:
        cache.mkdir(parents=True, exist_ok=True)
        result.to_parquet(cache/"features.parquet", index=False)
        (cache/"fit_ledger.json").write_text(json.dumps(ledger, indent=2, allow_nan=False)+"\n")
        manifest = {"signature": signature, "inputs": inputs,
                    "artifact_sha256": {name: fingerprint(cache/name) for name in ("features.parquet", "fit_ledger.json")}}
        (cache/"manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False)+"\n")
    result.attrs.update(fit_ledger=ledger, cache_signature=signature,
                        cache_path=str(cache) if cache is not None else None, specification=SPECIFICATION.copy())
    return result


def predicted_cold_evening_weights(features, bank, arm, multiplier=3.0):
    """Point-loss training weights from original-origin cold predictions only.

    Missing warm-up probabilities receive neutral weight one. These weights
    emphasize forecast cold evenings without looking at realized future cold
    conditions. They are not a settlement-cost estimate and should not be used
    as outcome-dependent weighting of a nominal quantile distribution.
    """
    if not np.isfinite(multiplier) or multiplier < 1:
        raise ValueError("Cold-evening weight multiplier must be at least one")
    column = learned_columns(arm)[1]
    inputs = features[KEYS+["cal_evening"]].merge(bank[KEYS+[column]], on=KEYS, how="left", validate="one_to_one")
    if not inputs.cal_evening.isin([0, 1]).all():
        raise AssertionError("Evening flag must contain only zero or one")
    probability = inputs[column]
    if (probability.dropna().lt(0) | probability.dropna().gt(1)).any():
        raise AssertionError("Cold probability outside [0, 1]")
    values = 1+(multiplier-1)*inputs.cal_evening.to_numpy()*probability.fillna(0).to_numpy()
    return pd.Series(values, index=features.index, name="forecast_cold_evening_weight")
