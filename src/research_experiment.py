"""Daily forecast-origin replay, chronological selection and research artifacts.

Every prediction is written before its outcomes are joined for scoring. The
framework is intentionally a research runner, not a serving/deployment system.
"""
from __future__ import annotations

import gc
import fcntl
import gzip
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time
import traceback
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from .common import fingerprint, forecast_origin, write_json
from .features import prediction_fallback
from .metrics import point_metrics
from .research_data import calendar_exog, load_research_contract, prepare_research_data
from .research_models import MODEL_REGISTRY, make_statistical, make_tabular, parameter_grid

POINT_FAMILIES = list(MODEL_REGISTRY)
NEURAL_FAMILIES = ["nbeats", "nhits", "tft", "gru", "patchtst", "dlinear"]
BASELINES = ["seasonal_naive", "comparable_days", "rte_proxy", "rte_bias_proxy"]


def _json(value):
    return json.dumps(value, default=str, sort_keys=True)


def _read_predictions(path):
    frame = pd.read_csv(path)
    for name in ("origin", "delivery_time", "fit_origin"):
        if name in frame:
            frame[name] = pd.to_datetime(frame[name], utc=True)
    return frame


class ResearchRun:
    def __init__(self, root, run_dir, profile="compact"):
        self.root = Path(root)
        self.path = Path(run_dir)
        self.contract = load_research_contract(root)
        self.profile = profile
        self._data = None
        self.path.mkdir(parents=True, exist_ok=True)
        for name in ("forecasts", "ledgers", "fits", "tables", "figures", "models"):
            (self.path/name).mkdir(exist_ok=True)

    @classmethod
    def create(cls, root, profile="compact"):
        root = Path(root)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        obj = cls(root, root/"reports/research_v2"/stamp, profile)
        write_json(root/".runtime/latest_research_run.json", {"run_dir": str(obj.path), "profile": profile})
        obj.write_manifest()
        return obj

    @classmethod
    def latest(cls, root):
        state = json.loads((Path(root)/".runtime/latest_research_run.json").read_text())
        return cls(root, state["run_dir"], state["profile"])

    @property
    def data(self):
        if self._data is None:
            self._data = prepare_research_data(self.root)
        return self._data

    def write_manifest(self):
        files = [self.root/"config/research_experiment.json"] + sorted((self.root/"src").glob("research_*.py"))
        files += [self.root/"src"/name for name in ("features.py", "common.py", "data.py", "metrics.py", "uncertainty.py")]
        files += [self.root/"requirements-research-neural.txt"]
        write_json(self.path/"manifest.json", {
            "run_directory": str(self.path), "profile": self.profile,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "python": sys.version, "platform": platform.platform(),
            "contract": self.contract,
            "source_sha256": {str(p.relative_to(self.root)): fingerprint(p) for p in files},
            "raw_sha256": {name: fingerprint(self.root/"data"/name) for name in
                           ("france_electricity_hourly.csv", "france_regional_electricity_hourly.csv")},
            "production_scope": False, "predictions_include_outcomes": False})

    def _event(self, path, value):
        with gzip.open(path, "at", encoding="utf-8") as stream:
            stream.write(_json(value)+"\n")

    def _record_trial(self, record):
        """Allow independent model families to train in separate CPU processes."""
        path = self.path/"tables/tuning_trials.csv"
        with (self.path/".selection.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            previous = pd.read_csv(path) if path.exists() else pd.DataFrame()
            updated = pd.concat([previous, pd.DataFrame([record])], ignore_index=True)
            updated = updated.drop_duplicates(["model", "trial", "fold"], keep="last")
            temporary = path.with_suffix(f".tmp.{os.getpid()}")
            updated.to_csv(temporary, index=False)
            temporary.replace(path)

    def _record_selection(self, key, entry):
        path = self.path/"selected_parameters.json"
        with (self.path/".selection.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            previous = json.loads(path.read_text()) if path.exists() else {}
            previous[key] = entry
            temporary = path.with_suffix(f".tmp.{os.getpid()}")
            write_json(temporary, previous)
            temporary.replace(path)
        return previous

    def verify_manifest(self):
        """Never resume a run after changing its raw inputs or predictor code."""
        manifest = json.loads((self.path/"manifest.json").read_text())
        for name, expected in {**manifest["source_sha256"], **{f"data/{k}": v for k, v in manifest["raw_sha256"].items()}}.items():
            if fingerprint(self.root/name) != expected:
                raise RuntimeError(f"Research inputs/code changed: {name}. Start a NEW run.")
        return fingerprint(self.path/"manifest.json")

    def replay(self, family, params, start, end, *, arm="national", phase="validation",
               tag=None, seed=42, fixed=False, quantiles=False, full_history=False,
               rte_correction=False):
        """Fit on released labels, forecast every origin, archive before scoring.

        Fixed means fixed fitted parameters, not frozen observation history.
        Statistical state filtering and neural context updates remain allowed.
        """
        tag = tag or f"{family}__{arm}"
        manifest_hash = self.verify_manifest()
        signature = hashlib.sha256(_json([family, params, start, end, arm, seed, fixed,
                                         quantiles, full_history, rte_correction, manifest_hash]).encode()).hexdigest()[:12]
        stem = f"{tag}__{phase}__{signature}"
        outpath = self.path/"forecasts"/f"{stem}.csv.gz"
        complete = self.path/"forecasts"/f"{stem}.json"
        if complete.exists() and outpath.exists():
            meta = json.loads(complete.read_text())
            ledger_path = self.path/"ledgers"/f"{stem}.jsonl.gz"
            if fingerprint(outpath) == meta["sha256"] and ledger_path.exists() and fingerprint(ledger_path) == meta["ledger_sha256"]:
                return _read_predictions(outpath)
        data = prepare_research_data(self.root, "2019-01-02") if full_history else self.data
        evaluation = data.phase(start, end)
        if evaluation.empty:
            raise ValueError(f"No delivery intervals in {start} through {end}")
        columns = data.columns(arm, rte_correction)
        estimator = None
        qmodels = {}
        fit_origin = None
        last_month = None
        outputs = []
        fit_records = []
        ledger = self.path/"ledgers"/f"{stem}.jsonl.gz"
        if ledger.exists():
            ledger.unlink()  # only an incomplete run is restarted
        fit_ledger = self.path/"fits"/f"{stem}.jsonl.gz"
        if fit_ledger.exists():
            fit_ledger.unlink()
        kind = "neural" if family in NEURAL_FAMILIES else MODEL_REGISTRY.get(family, {}).get("kind", "baseline")
        window_days = params.get("window_days")
        refit_days = params.get("refit_days")
        model_params = {k: v for k, v in params.items() if k not in ("window_days", "refit_days")}
        print(f"REPLAY {tag} {phase} {start}..{end} config={signature}", flush=True)
        for origin, day in evaluation.groupby("origin", sort=True):
            month = origin.tz_convert(self.contract["timezone"]).strftime("%Y-%m")
            due = month != last_month if refit_days is None else (fit_origin is None or origin-fit_origin >= pd.Timedelta(days=refit_days))
            need_fit = family not in BASELINES and (fit_origin is None or (not fixed and due))
            if need_fit:
                if estimator is not None and hasattr(estimator, "close"):
                    estimator.close()
                estimator = None
                qmodels = {}
                started = time.perf_counter()
                record = {"model": tag, "phase": phase, "origin": origin,
                          "params": model_params, "arm": arm, "seed": seed,
                          "fixed_parameters": fixed, "status": "failed", "metadata": {}}
                try:
                    if kind == "tabular":
                        training = data.training_rows(origin, window_days)
                        span = (training.delivery_time.max()-training.delivery_time.min()).total_seconds()/86400
                        if span < self.contract["minimum_training_days"]:
                            raise ValueError(f"Insufficient training span {span:.1f} days")
                        if rte_correction:
                            training = training.loc[training.rte_at_origin_mw.notna()]
                        target = training.actual_mw-(training.rte_at_origin_mw if rte_correction else 0)
                        estimator = make_tabular(family, model_params, seed=seed).fit(training[columns], target)
                        # Same-regime train error is diagnostic, not out-of-sample performance.
                        record.update(training_rows=len(training), latest_label_available_at=training.label_available_at.max(),
                                      train_MAE_MW=float(np.mean(np.abs(target-estimator.predict(training[columns])))),
                                      metadata=estimator.metadata_)
                        if quantiles:
                            for q in self.contract["quantiles"]:
                                if q == .5 and family == "lightgbm_quantile":
                                    qmodels[q] = estimator
                                else:
                                    qmodels[q] = make_tabular(family, model_params, quantile=q, seed=seed).fit(training[columns], target)
                    elif kind == "statistical":
                        hist = data.history(origin, window_days)
                        estimator = make_statistical(family, model_params).fit(
                            hist.load_mw, calendar_exog(hist.index) if family == "sarimax" else None)
                        record.update(training_rows=len(hist), latest_label_available_at=hist.label_available_at_assumed.max(), metadata=estimator.metadata_)
                    else:
                        from .research_neural import NeuralAdapter
                        panel = data.panel(origin, arm, window_days)
                        estimator = NeuralAdapter(family, model_params, seed=seed).fit(panel, validation_size=168)
                        record.update(training_rows=len(panel), latest_label_available_at=data.history(origin).label_available_at_assumed.max(),
                                      metadata=getattr(estimator, "metadata_", getattr(estimator, "fit_metadata", {})))
                    assert pd.Timestamp(record["latest_label_available_at"]) < origin
                    record["status"] = "fit"
                except Exception as exc:
                    estimator = None
                    record.update(error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc(limit=5))
                    print(f"FIT FAILED {tag}: {record['error']}", flush=True)
                record["elapsed_seconds"] = time.perf_counter()-started
                fit_records.append(record)
                self._event(self.path/"fits"/f"{stem}.jsonl.gz", record)
                fit_origin, last_month = origin, month
                gc.collect()
            raw = np.full(len(day), np.nan)
            predicted_q = {}
            prediction_error = None
            try:
                if family in BASELINES:
                    raw = {"seasonal_naive": day.week_lag_1,
                           "comparable_days": day.comparable_mean,
                           "rte_proxy": day.rte_at_origin_mw,
                           "rte_bias_proxy": day.rte_at_origin_mw+day.rte_bias_recent_mw}[family].to_numpy()
                elif estimator is not None and kind == "tabular":
                    raw = estimator.predict(day[columns])
                    predicted_q = {q: m.predict(day[columns]) for q, m in qmodels.items()}
                    if rte_correction:
                        raw += day.rte_at_origin_mw.to_numpy()
                        predicted_q = {q: p+day.rte_at_origin_mw.to_numpy() for q, p in predicted_q.items()}
                elif estimator is not None and kind == "statistical":
                    hist = data.history(origin)
                    if hist.index[-1] > estimator.last_timestamp_:
                        new = hist.loc[hist.index > estimator.last_timestamp_]
                        estimator.update(new.load_mw, calendar_exog(new.index) if family == "sarimax" else None)
                    future_index = pd.date_range(estimator.last_timestamp_+pd.Timedelta(hours=1), day.delivery_time.max(), freq="h")
                    forecast = estimator.forecast(len(future_index), calendar_exog(future_index) if family == "sarimax" else None)
                    raw = pd.Series(forecast, index=future_index).reindex(day.delivery_time).to_numpy()
                elif estimator is not None:
                    context_days = int(np.ceil((model_params.get("input_size", 168)+48)/24))+2
                    context = data.panel(origin, arm, window_days=context_days)
                    forecast = estimator.predict(context)
                    national = forecast.loc[forecast.unique_id.eq("country")].copy()
                    times = pd.to_datetime(national.ds, utc=True)
                    series = pd.Series(national.prediction_mw.to_numpy(), index=times)
                    if not day.delivery_time.isin(series.index).all():
                        raise ValueError("Neural horizon does not cover every delivery interval")
                    raw = series.reindex(day.delivery_time).to_numpy()
            except Exception as exc:
                prediction_error = f"{type(exc).__name__}: {exc}"
            raw = np.array(raw, dtype=float, copy=True)
            raw[~np.isfinite(raw) | (raw <= 0)] = np.nan
            prediction, source = prediction_fallback(day, raw, rte_assisted=rte_correction or family.startswith("rte"))
            out = day[["origin", "delivery_time", "local_day"]].copy()
            out["model"] = tag
            out["family"] = family
            out["arm"] = arm
            out["prediction_mw"] = prediction.to_numpy()
            out["raw_prediction_mw"] = raw
            out["fallback_used"] = np.isnan(raw)
            out["fallback_source"] = source.to_numpy()
            out["fit_origin"] = fit_origin
            out["prediction_error"] = prediction_error
            out["regime"] = self.contract["status"]
            if predicted_q:
                names = [f"q{int(q*100):02d}" for q in sorted(predicted_q)]
                array = np.column_stack([predicted_q[q] for q in sorted(predicted_q)])
                array[np.isnan(raw), :] = np.nan
                out["quantile_crossing_repaired"] = (np.diff(array, axis=1)<0).any(axis=1)
                out[names] = np.sort(array, axis=1)
            assert "actual_mw" not in out
            self._event(ledger, {"origin": origin, "predictions": json.loads(out.to_json(orient="records", date_format="iso"))})
            outputs.append(out)
        if estimator is not None and hasattr(estimator, "close"):
            estimator.close()
        result = pd.concat(outputs, ignore_index=True)
        result.to_csv(outpath, index=False, compression="gzip")
        pd.DataFrame(fit_records).to_json(self.path/"fits"/f"{stem}.json", orient="records", indent=2, date_format="iso", default_handler=str)
        write_json(complete, {"sha256": fingerprint(outpath), "rows": len(result), "ledger_sha256": fingerprint(ledger),
                              "predictions_archived_before_scoring": True, "signature": signature,
                              "family": family, "params": params, "arm": arm, "phase": phase})
        return result

    def tune(self, families, arms=("national", "regional")):
        """All hyperparameters selected on 2024 delivery-day replay only."""
        selection_path = self.path/"selected_parameters.json"
        selected = json.loads(selection_path.read_text()) if selection_path.exists() else {}
        records_path = self.path/"tables/tuning_trials.csv"
        records = pd.read_csv(records_path).to_dict("records") if records_path.exists() else []
        for family in families:
            applicable = ("national",) if MODEL_REGISTRY.get(family, {}).get("kind") == "statistical" else arms
            if family in NEURAL_FAMILIES:
                from .research_neural import neural_configurations
                configurations = neural_configurations(family)
            else:
                configurations = parameter_grid(family, "fast" if self.profile == "compact" else "full")
            for arm in applicable:
                key = f"{family}__{arm}"
                if key in selected:
                    continue
                candidates = []
                for trial, params in enumerate(configurations):
                    all_predictions = []
                    for fold in self.contract["validation_folds"]:
                        predictions = self.replay(family, params, fold["start"], fold["end"], arm=arm,
                                                  tag=key, phase=f"{fold['name']}_trial{trial}")
                        joined = predictions.merge(self.data.labels[["delivery_time", "actual_mw"]], on="delivery_time", validate="many_to_one")
                        metrics = point_metrics(joined)
                        raw = point_metrics(joined, "raw_prediction_mw")
                        record = {"model": key, "family": family, "arm": arm, "trial": trial, "fold": fold["name"],
                                  "params_json": _json(params), **metrics, "raw_availability": raw["availability"],
                                  "raw_MAE_MW": raw["MAE_MW"], "fallback_fraction": float(predictions.fallback_used.mean())}
                        records.append(record)
                        all_predictions.append(joined)
                        self._record_trial(record)
                    joined = pd.concat(all_predictions, ignore_index=True)
                    metrics = point_metrics(joined)
                    raw = point_metrics(joined, "raw_prediction_mw")
                    candidates.append({"trial": trial, "params": params, "MAE_MW": metrics["MAE_MW"],
                                       "raw_availability": raw["availability"]})
                eligible = [c for c in candidates if c["raw_availability"] >= .95 and np.isfinite(c["MAE_MW"])]
                if eligible:
                    best = min(eligible, key=lambda c: c["MAE_MW"])
                    selected[key] = {"family": family, "arm": arm, "status": "selected", **best}
                else:
                    selected[key] = {"family": family, "arm": arm, "status": "no_eligible_configuration", "candidates": candidates}
                selected = self._record_selection(key, selected[key])
                print(f"SELECTED {key}: {_json(selected[key])}", flush=True)
        return pd.read_csv(records_path)

    def baselines(self, start, end, phase):
        return pd.concat([self.replay(name, {}, start, end, tag=name, phase=phase) for name in BASELINES], ignore_index=True)

    def freeze(self):
        self.verify_manifest()
        selections = json.loads((self.path/"selected_parameters.json").read_text())
        expected = set(POINT_FAMILIES+NEURAL_FAMILIES)
        observed = {v["family"] for v in selections.values()}
        if not expected.issubset(observed):
            raise RuntimeError(f"Cannot freeze before every family is investigated: {sorted(expected-observed)}")
        expected_keys = {f"{family}__{arm}" for family in expected for arm in
                         (("national",) if MODEL_REGISTRY.get(family, {}).get("kind") == "statistical" else ("national", "regional"))}
        if set(selections) != expected_keys:
            raise RuntimeError(f"Missing intended experiment arms: {sorted(expected_keys-set(selections))}")
        trials = pd.read_csv(self.path/"tables/tuning_trials.csv")
        for key in expected_keys:
            per_trial = trials.loc[trials.model.eq(key)].groupby("trial").fold.nunique()
            if len(per_trial) < 2 or not per_trial.eq(4).all():
                raise RuntimeError(f"Missing validation folds for {key}")
        eligible = {key: value for key, value in selections.items() if value["status"] == "selected"}
        primary = min(eligible, key=lambda key: eligible[key]["MAE_MW"]) if eligible else None
        payload = {"selections": selections, "contract_sha256": fingerprint(self.root/"config/research_experiment.json"),
                   "selection_period": "2024 only", "test_previously_explored": True,
                   "primary_candidate_from_validation": primary,
                   "freeze_time_utc": datetime.now(timezone.utc).isoformat()}
        path = self.path/"frozen_selection.json"
        if not path.exists():
            write_json(path, payload)
        else:
            previous = json.loads(path.read_text())
            assert previous["selections"] == selections, "Frozen selection changed: start a NEW research run"
            assert previous["contract_sha256"] == payload["contract_sha256"]
        return selections

    def final_replay(self):
        selected = self.freeze()
        phases = {"calibration": self.contract["calibration"], "test": self.contract["test"]}
        outputs = {}
        for phase, dates in phases.items():
            predictions = [self.baselines(dates["start"], dates["end"], phase)]
            for key, entry in selected.items():
                if entry["status"] != "selected":
                    continue
                predictions.append(self.replay(entry["family"], entry["params"], dates["start"], dates["end"],
                                               arm=entry["arm"], tag=key, phase=phase,
                                               quantiles=entry["family"] == "lightgbm_quantile"))
            outputs[phase] = pd.concat(predictions, ignore_index=True)
            outputs[phase].to_csv(self.path/"forecasts"/f"all_{phase}.csv.gz", index=False)
        return outputs
