"""Chronological RTE-assisted extension; preserves the original frozen experiment.

Direct demand prediction with RTE predictors and residual learning are distinct
predeclared variants. Both receive identical predictors and fallback rules.
Only tabular adapters are supported; no neural architecture is substituted.
"""
from __future__ import annotations

from datetime import datetime, timezone
import gc
import gzip
import hashlib
import json
from pathlib import Path
import time
import traceback

import numpy as np
import pandas as pd

from .common import fingerprint, forecast_origin, write_json
from .features import prediction_fallback
from .metrics import point_metrics
from .research_experiment import ResearchRun, _read_predictions
from .research_models import make_tabular, parameter_grid


def model_key(family, arm, mode):
    if mode not in ("direct", "residual"):
        raise ValueError(f"Unknown RTE-assisted mode: {mode}")
    return f"{family}__{arm}__rte_{mode}"


def training_target(training, mode):
    if mode == "direct":
        return training.actual_mw
    if mode == "residual":
        return training.actual_mw-training.rte_at_origin_mw
    raise ValueError(mode)


def demand_prediction(prediction, day, mode):
    result = np.asarray(prediction, dtype=float).copy()
    if mode == "residual":
        result += day.rte_at_origin_mw.to_numpy()
    elif mode != "direct":
        raise ValueError(mode)
    return result


class RTEAssistedRun(ResearchRun):
    def __init__(self, root, run_dir):
        super().__init__(root, run_dir)
        self.spec = json.loads((self.root/"config/rte_assisted_experiment.json").read_text())
        self.reference = self.root/self.spec["reference_run"]

    @classmethod
    def create(cls, root):
        root = Path(root)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        run = cls(root, root/"reports/rte_assisted"/stamp)
        run.write_manifest()
        write_json(root/".runtime/latest_rte_assisted_run.json", {"run_dir": str(run.path)})
        return run

    @classmethod
    def latest(cls, root):
        root = Path(root)
        pointer = root/".runtime/latest_rte_assisted_run.json"
        if not pointer.exists():
            return cls.create(root)
        run = cls(root, json.loads(pointer.read_text())["run_dir"])
        run.verify_manifest()
        return run

    def write_manifest(self):
        reference = json.loads((self.reference/"manifest.json").read_text())
        if reference["contract"] != self.contract:
            raise RuntimeError("Reference forecasting contract differs")
        for name, digest in reference["source_sha256"].items():
            if fingerprint(self.root/name) != digest:
                raise RuntimeError(f"Original predictor changed: {name}")
        for name, digest in reference["raw_sha256"].items():
            if fingerprint(self.root/"data"/name) != digest:
                raise RuntimeError(f"Original data changed: {name}")
        own = ["src/rte_assisted.py", "config/rte_assisted_experiment.json"]
        dependencies = ["manifest.json", "selected_parameters.json", "development_policy_selection.json",
                        "tables/development_baselines.csv", "forecasts/all_test.csv.gz", "forecasts/all_calibration.csv.gz"]
        write_json(self.path/"manifest.json", {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "contract": self.contract, "extension_spec": self.spec,
            "source_sha256": {**reference["source_sha256"], **{n: fingerprint(self.root/n) for n in own}},
            "raw_sha256": reference["raw_sha256"],
            "reference_sha256": {n: fingerprint(self.reference/n) for n in dependencies},
            "test_previously_explored": True, "production_scope": False,
        })

    def verify_manifest(self):
        digest = super().verify_manifest()
        manifest = json.loads((self.path/"manifest.json").read_text())
        for name, expected in manifest["reference_sha256"].items():
            if fingerprint(self.reference/name) != expected:
                raise RuntimeError(f"Original reference artifact changed: {name}")
        return digest

    def replay_variant(self, family, params, start, end, *, arm, mode,
                       phase="validation", fixed=False, quantiles=False):
        if family not in self.spec["families"] or arm not in self.spec["arms"]:
            raise ValueError("Unsupported RTE-assisted family/arm")
        key = model_key(family, arm, mode)
        digest = self.verify_manifest()
        signature = hashlib.sha256(json.dumps([key, params, start, end, fixed, quantiles, digest],
                                                sort_keys=True).encode()).hexdigest()[:12]
        stem = f"{key}__{phase}__{signature}"
        path = self.path/"forecasts"/f"{stem}.csv.gz"
        seal = path.with_suffix("").with_suffix(".json")
        ledger = self.path/"ledgers"/f"{stem}.jsonl.gz"
        fits_path = self.path/"fits"/f"{stem}.json"
        if seal.exists():
            meta = json.loads(seal.read_text())
            if path.exists() and ledger.exists() and fits_path.exists() and all((
                fingerprint(path) == meta["sha256"], fingerprint(ledger) == meta["ledger_sha256"],
                fingerprint(fits_path) == meta["fits_sha256"],
            )):
                return _read_predictions(path)
            raise RuntimeError(f"Sealed forecast cache was altered: {stem}")
        if ledger.exists():
            ledger.unlink()  # only unsealed incomplete replays restart
        data = self.data
        evaluation = data.phase(start, end)
        if evaluation.empty:
            raise ValueError("Empty delivery period")
        columns = data.columns(arm, rte_assisted=True)
        assert {"rte_at_origin_mw", "rte_bias_recent_mw"}.issubset(columns)
        estimator = None
        quantile_models = {}
        last_month = None
        fit_origin = None
        records, outputs = [], []
        print(f"REPLAY {key} {phase} {start}..{end}", flush=True)
        for origin, day in evaluation.groupby("origin", sort=True):
            month = origin.tz_convert(self.contract["timezone"]).strftime("%Y-%m")
            if fit_origin is None or (not fixed and month != last_month):
                estimator, quantile_models = None, {}
                timer = time.perf_counter()
                record = {"model": key, "family": family, "arm": arm, "mode": mode,
                          "origin": str(origin), "phase": phase, "params": params,
                          "predictor_columns": columns, "fixed_parameters": fixed, "status": "failed"}
                try:
                    train = data.training_rows(origin)
                    train = train.loc[train.rte_at_origin_mw.notna()]
                    span = (train.delivery_time.max()-train.delivery_time.min()).total_seconds()/86400
                    if span < self.contract["minimum_training_days"]:
                        raise ValueError("Insufficient chronological training history")
                    if not train.label_available_at.lt(origin).all():
                        raise AssertionError("Training label released after fit cutoff")
                    valid = train.max_history_available_at.notna()
                    if not (train.loc[valid, "max_history_available_at"] < train.loc[valid, "origin"]).all():
                        raise AssertionError("Feature observation released after its original origin")
                    target = training_target(train, mode)
                    estimator = make_tabular(family, params, seed=self.contract["random_seed"]).fit(train[columns], target)
                    record.update(training_rows=len(train),
                                  latest_label_available_at=str(train.label_available_at.max()),
                                  train_MAE_MW=float(np.mean(np.abs(target-estimator.predict(train[columns])))),
                                  metadata=estimator.metadata_)
                    if quantiles:
                        for q in self.contract["quantiles"]:
                            quantile_models[q] = estimator if q == .5 else make_tabular(
                                family, params, quantile=q, seed=self.contract["random_seed"]).fit(train[columns], target)
                    record["status"] = "fit"
                except Exception as exc:
                    estimator, quantile_models = None, {}
                    record.update(error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc(limit=4))
                    print(f"FIT FAILED {key}: {record['error']}", flush=True)
                record["elapsed_seconds"] = time.perf_counter()-timer
                records.append(record)
                fit_origin, last_month = origin, month
                gc.collect()
            raw = np.full(len(day), np.nan)
            qs = {}
            prediction_error = None
            try:
                if estimator is not None:
                    raw = demand_prediction(estimator.predict(day[columns]), day, mode)
                    qs = {q: demand_prediction(m.predict(day[columns]), day, mode)
                          for q, m in quantile_models.items()}
            except Exception as exc:
                prediction_error = f"{type(exc).__name__}: {exc}"
            invalid = ~np.isfinite(raw) | (raw <= 0) | day.rte_at_origin_mw.isna().to_numpy()
            raw[invalid] = np.nan
            prediction, source = prediction_fallback(day, raw, rte_assisted=True)
            out = day[["origin", "delivery_time", "local_day"]].copy()
            out["model"], out["family"], out["arm"], out["mode"] = key, family, arm, mode
            out["prediction_mw"], out["raw_prediction_mw"] = prediction.to_numpy(), raw
            out["fallback_used"], out["fallback_source"] = source.ne("model").to_numpy(), source.to_numpy()
            out["fit_origin"], out["prediction_error"] = fit_origin, prediction_error
            out["regime"] = "retrospective_assumed_availability"
            if qs:
                matrix = np.column_stack([qs[q] for q in self.contract["quantiles"]])
                crossing = np.any(np.diff(matrix, axis=1) < 0, axis=1)
                matrix = np.sort(matrix, axis=1)
                matrix[invalid] = np.nan
                for i, q in enumerate(self.contract["quantiles"]):
                    out[f"q{int(q*100):02d}"] = matrix[:, i]
                out["quantile_crossing_repaired"] = crossing
            if "actual_mw" in out:
                raise AssertionError("Outcomes in prediction ledger")
            self._event(ledger, {"origin": str(origin), "predictions": json.loads(out.to_json(orient="records", date_format="iso"))})
            outputs.append(out)
        result = pd.concat(outputs, ignore_index=True)
        result.to_csv(path, index=False, compression="gzip")
        write_json(fits_path, records)
        write_json(seal, {"sha256": fingerprint(path), "ledger_sha256": fingerprint(ledger),
                          "fits_sha256": fingerprint(fits_path), "rows": len(result),
                          "model": key, "family": family, "arm": arm, "mode": mode,
                          "phase": phase, "params": params, "predictor_columns": columns,
                          "fixed_parameters": fixed, "native_quantiles": quantiles,
                          "predictions_archived_before_scoring": True})
        return result

    def tune_assisted(self):
        selected_path = self.path/"selected_parameters.json"
        selected = json.loads(selected_path.read_text()) if selected_path.exists() else {}
        for family in self.spec["families"]:
            for arm in self.spec["arms"]:
                for mode in self.spec["modes"]:
                    key = model_key(family, arm, mode)
                    if key in selected:
                        continue
                    choices = []
                    for trial, params in enumerate(parameter_grid(family, "fast")):
                        predictions = []
                        for fold in self.contract["validation_folds"]:
                            p = self.replay_variant(family, params, fold["start"], fold["end"], arm=arm,
                                                    mode=mode, phase=f"{fold['name']}_trial{trial}")
                            joined = p.merge(self.data.labels[["delivery_time", "actual_mw"]], on="delivery_time", validate="many_to_one")
                            record = {"model": key, "family": family, "arm": arm, "mode": mode,
                                      "trial": trial, "fold": fold["name"], "params_json": json.dumps(params, sort_keys=True),
                                      **point_metrics(joined), "raw_availability": p.raw_prediction_mw.notna().mean(),
                                      "fallback_fraction": p.fallback_used.mean()}
                            self._record_trial(record)
                            predictions.append(joined)
                        joined = pd.concat(predictions, ignore_index=True)
                        choices.append({"trial": trial, "params": params, **point_metrics(joined),
                                        "raw_availability": joined.raw_prediction_mw.notna().mean()})
                    eligible = [x for x in choices if x["raw_availability"] >= .99 and np.isfinite(x["MAE_MW"])]
                    if not eligible:
                        raise RuntimeError(f"No eligible assisted configuration: {key}")
                    best = min(eligible, key=lambda x: (x["MAE_MW"], x["trial"]))
                    selected[key] = {"family": family, "arm": arm, "mode": mode, "status": "selected", **best}
                    selected = self._record_selection(key, selected[key])
                    print(f"SELECTED {key} {best['MAE_MW']:.3f} MW", flush=True)
        return pd.read_csv(self.path/"tables/tuning_trials.csv")

    def freeze_assisted(self):
        self.verify_manifest()
        path = self.path/"frozen_selection.json"
        if path.exists():
            return json.loads(path.read_text())
        if any((self.path/"forecasts").glob("*__test__*")):
            raise RuntimeError("Cannot first freeze selection after test replay")
        selected = json.loads((self.path/"selected_parameters.json").read_text())
        expected = {model_key(f, a, m) for f in self.spec["families"] for a in self.spec["arms"] for m in self.spec["modes"]}
        if set(selected) != expected:
            raise RuntimeError("All assisted variants must finish development first")
        trials = pd.read_csv(self.path/"tables/tuning_trials.csv")
        for key in expected:
            folds = trials.loc[trials.model.eq(key)].groupby("trial").fold.nunique()
            if len(folds) != 2 or not folds.eq(4).all():
                raise RuntimeError(f"Missing configuration/quarter: {key}")
        old = json.loads((self.reference/"development_policy_selection.json").read_text())["entries"]
        alternatives = old + [{"model": k, "MAE_MW": v["MAE_MW"], "kind": "rte_assisted_model"}
                              for k, v in selected.items()]
        result = {"frozen_utc": datetime.now(timezone.utc).isoformat(),
                  "selection_period": "2024 only", "test_previously_explored": True,
                  "selection_rule": self.spec["selection"], "selections": selected,
                  "selected_assisted_model": min(selected, key=lambda k: selected[k]["MAE_MW"]),
                  "selected_complete_policy": min(alternatives, key=lambda x: x["MAE_MW"])["model"],
                  "alternatives": alternatives,
                  "selection_inputs_sha256": {n: fingerprint(self.path/n) for n in
                                               ("selected_parameters.json", "tables/tuning_trials.csv")}}
        write_json(path, result)
        return result

    def replay_final_assisted(self):
        frozen = self.freeze_assisted()
        outputs = {}
        for phase in ("calibration", "test"):
            dates = self.contract[phase]
            parts = []
            for key, entry in frozen["selections"].items():
                parts.append(self.replay_variant(entry["family"], entry["params"], dates["start"], dates["end"],
                    arm=entry["arm"], mode=entry["mode"], phase=phase,
                    quantiles=entry["family"] == "lightgbm_quantile"))
            out = pd.concat(parts, ignore_index=True)
            out.to_csv(self.path/"forecasts"/f"assisted_{phase}.csv.gz", index=False, compression="gzip")
            outputs[phase] = out
        return outputs

    def fixed_conformal_replay(self):
        frozen = self.freeze_assisted()
        entry = frozen["selections"][frozen["selected_assisted_model"]]
        cal, test = self.contract["calibration"], self.contract["test"]
        return self.replay_variant(entry["family"], entry["params"], cal["start"], test["end"],
            arm=entry["arm"], mode=entry["mode"], phase="fixed_calibration_and_test", fixed=True)
