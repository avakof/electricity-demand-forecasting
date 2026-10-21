"""Causal, already-out-of-sample base forecasts for RTE feature stacking.

The original benchmark contains both candidate configurations in every 2024
quarter. Selecting its full-year winner inside that same year's stacker would
leak model-selection information. This bank instead starts with trial 0, then
selects each later quarter using only earlier-quarter outcomes released before
that quarter's first forecast origin. January 2025 freezes the full-2024 winner,
which must match the original run before its calibration/test forecasts are used.

This is an explicitly two-stage hybrid feature bank. It does not claim that
N-BEATS, PatchTST, ARIMA, or any other base model gained native exogenous inputs.
No fitting, raw-data mutation, or source-artifact mutation occurs here.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


REFERENCE_RUN = "reports/research_v2/20261020T164443776583Z"
KEYS = ["origin", "delivery_time"]
RAW_AVAILABILITY_MINIMUM = 0.95


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _utc(values):
    return pd.to_datetime(values, utc=True, format="mixed")


def _labels(frame):
    columns = ["delivery_time", "actual_mw", "label_available_at"]
    result = frame[columns].copy()
    result["delivery_time"] = _utc(result.delivery_time)
    result["label_available_at"] = _utc(result.label_available_at)
    if result.delivery_time.duplicated().any():
        raise ValueError("Duplicate outcome timestamps in base selection labels")
    return result


def select_released_trial(trial_predictions, labels, cutoff, minimum_availability=RAW_AVAILABILITY_MINIMUM):
    """Select from externally supplied OOS trials using strictly released labels.

    The caller supplies only previous-quarter predictions. Rows at or after the
    selection origin are additionally excluded. Availability follows the original
    benchmark: raw predictions on nonmissing eligible evaluation targets.
    """
    cutoff = pd.Timestamp(cutoff)
    if cutoff.tzinfo is None:
        raise ValueError("Selection cutoff must be timezone-aware")
    outcome = _labels(labels)
    choices = []
    for trial, predictions in sorted(trial_predictions.items()):
        if predictions.duplicated(KEYS).any():
            raise ValueError("Duplicate OOS forecast keys")
        frame = predictions.copy()
        frame["origin"] = _utc(frame.origin)
        frame["delivery_time"] = _utc(frame.delivery_time)
        frame = frame.loc[frame.origin.lt(cutoff)].merge(outcome, on="delivery_time", validate="many_to_one")
        frame = frame.loc[frame.label_available_at.lt(cutoff) & frame.actual_mw.notna()]
        raw_good = frame.raw_prediction_mw.notna()
        good = frame.prediction_mw.notna()
        availability = float(raw_good.mean()) if len(frame) else np.nan
        mae = float((frame.loc[good, "actual_mw"] - frame.loc[good, "prediction_mw"]).abs().mean())
        choices.append({"trial": int(trial), "MAE_MW": mae, "raw_availability": availability,
                        "eligible_targets": len(frame), "scored": int(good.sum()),
                        "latest_label_available_at": str(frame.label_available_at.max()) if len(frame) else None})
    eligible = [c for c in choices if c["raw_availability"] >= minimum_availability and np.isfinite(c["MAE_MW"])]
    if not eligible:
        raise ValueError("No causal base configuration meets raw-availability and finite-loss requirements")
    selected = min(eligible, key=lambda item: (item["MAE_MW"], item["trial"]))
    return selected["trial"], choices


def causal_quarter_stream(quarter_trials, labels):
    """Return one variant's 2024 OOS stream and its causal selection ledger.

    ``quarter_trials`` is an ordered sequence of (quarter_name, {trial: frame}).
    Q1 is fixed-trial burn-in, suitable for training later stacker folds only.
    """
    past = {}
    pieces, records = [], []
    previous_end = None
    for number, (quarter, trials) in enumerate(quarter_trials):
        if set(trials) != {0, 1}:
            raise ValueError("Both original predeclared configurations are required")
        for frame in trials.values():
            _check_oos(frame)
        grids = [frame[KEYS].sort_values(KEYS).reset_index(drop=True) for frame in trials.values()]
        if not grids[0].equals(grids[1]):
            raise ValueError("Base configurations have different origin/delivery grids")
        cutoff = _utc(trials[0].origin).min()
        if previous_end is not None and cutoff <= previous_end:
            raise ValueError("Quarter streams must be strictly chronological")
        if number == 0:
            trial, choices = 0, []
        else:
            trial, choices = select_released_trial({key: pd.concat(value, ignore_index=True)
                                                   for key, value in past.items()}, labels, cutoff)
        chosen = trials[trial].copy()
        chosen["selected_base_trial"] = trial
        chosen["base_selection_cutoff"] = cutoff
        pieces.append(chosen)
        records.append({"quarter": quarter, "selection_cutoff": str(cutoff),
                        "selected_trial": trial, "policy": "fixed_trial0_burnin" if number == 0 else "released_prior_quarters_MAE",
                        "choices": choices, "burnin_only": number == 0})
        for key, frame in trials.items():
            past.setdefault(key, []).append(frame)
        previous_end = _utc(trials[0].origin).max()
    return pd.concat(pieces, ignore_index=True), records


def _check_oos(frame):
    forbidden = {"actual_mw", "load_mw", "realized_temperature_c", "target", "y"}
    if forbidden.intersection(frame.columns):
        raise ValueError("Outcome columns are forbidden in a base prediction artifact")
    required = set(KEYS + ["prediction_mw", "raw_prediction_mw", "fit_origin"])
    if not required.issubset(frame):
        raise ValueError(f"Missing OOS provenance columns: {sorted(required.difference(frame.columns))}")
    if frame.duplicated(KEYS).any():
        raise ValueError("Duplicate OOS forecast keys")
    origins, deliveries, fits = (_utc(frame[name]) for name in ["origin", "delivery_time", "fit_origin"])
    if origins.isna().any() or deliveries.isna().any() or fits.isna().any():
        raise ValueError("Missing OOS timestamps")
    if not (fits.le(origins) & origins.lt(deliveries)).all():
        raise ValueError("In-sample or post-origin-fitted forecast cannot become a meta-feature")
    for column in ("prediction_mw", "raw_prediction_mw"):
        values = pd.to_numeric(frame[column], errors="raise")
        if np.isinf(values).any():
            raise ValueError("Infinite base predictions")


def _verify_reference(root, reference):
    manifest_path = reference / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    provenance = {str(manifest_path.relative_to(root)): _sha256(manifest_path)}
    for relative, digest in manifest["source_sha256"].items():
        if _sha256(root / relative) != digest:
            raise RuntimeError(f"Original frozen source changed: {relative}")
    for relative, digest in manifest["raw_sha256"].items():
        if _sha256(root / "data" / relative) != digest:
            raise RuntimeError(f"Original frozen raw dataset changed: {relative}")
    provenance.update(manifest["source_sha256"])
    provenance.update({"data/" + key: value for key, value in manifest["raw_sha256"].items()})
    return manifest, provenance


def _read_sealed(root, reference, model, phase, provenance):
    seals = list((reference / "forecasts").glob(f"{model}__{phase}__*.json"))
    if len(seals) != 1:
        raise RuntimeError(f"Expected exactly one original artifact for {model}/{phase}; found {len(seals)}")
    seal = seals[0]
    metadata = json.loads(seal.read_text())
    csv = seal.with_suffix(".csv.gz")
    ledger = reference / "ledgers" / (seal.stem + ".jsonl.gz")
    fits = reference / "fits" / seal.name
    if metadata.get("predictions_archived_before_scoring") is not True:
        raise RuntimeError(f"Original artifact lacks OOS ledger claim: {seal.name}")
    for path, expected in ((csv, metadata["sha256"]), (ledger, metadata["ledger_sha256"])):
        if _sha256(path) != expected:
            raise RuntimeError(f"Original prediction artifact hash changed: {path.name}")
    fit_records = json.loads(fits.read_text())
    for record in fit_records:
        if record["status"] == "fit":
            if not pd.Timestamp(record["latest_label_available_at"]) < pd.Timestamp(record["origin"]):
                raise RuntimeError(f"Base fit consumed an unreleased outcome: {fits.name}")
    frame = pd.read_csv(csv)
    for name in ["origin", "delivery_time", "fit_origin"]:
        frame[name] = _utc(frame[name])
    _check_oos(frame)
    if len(frame) != metadata["rows"] or not frame.model.eq(model).all():
        raise RuntimeError(f"Original artifact identity mismatch: {csv.name}")
    for path in (seal, csv, ledger, fits):
        provenance[str(path.relative_to(root))] = _sha256(path)
    return frame, metadata


def load_causal_base_bank(root, labels):
    """Return (wide origin-delivery base features, selection ledger).

    The ledger is a JSON-serializable dictionary containing quarterly selections,
    final-selection agreement, every referenced artifact hash, and arm semantics.
    No observed outcomes enter the returned wide feature table.
    """
    root = Path(root).resolve()
    reference = root / REFERENCE_RUN
    manifest, provenance = _verify_reference(root, reference)
    selected_path = reference / "selected_parameters.json"
    selected = json.loads(selected_path.read_text())
    provenance[str(selected_path.relative_to(root))] = _sha256(selected_path)
    if len(selected) != 23 or any(value["status"] != "selected" for value in selected.values()):
        raise RuntimeError("Expected all 23 originally selected family/arm variants")
    labels = _labels(labels)
    pieces, selections, final_checks = [], [], []
    for model, chosen_original in sorted(selected.items()):
        quarters, all_trials, params_by_trial = [], {0: [], 1: []}, {}
        for quarter in manifest["contract"]["validation_folds"]:
            frames = {}
            for trial in (0, 1):
                frame, metadata = _read_sealed(root, reference, model, f"{quarter['name']}_trial{trial}", provenance)
                frames[trial] = frame
                all_trials[trial].append(frame)
                prior = params_by_trial.setdefault(trial, metadata["params"])
                if prior != metadata["params"]:
                    raise RuntimeError("Base configuration changed between quarters")
            quarters.append((quarter["name"], frames))
        development, records = causal_quarter_stream(quarters, labels)
        selections.extend({"model": model, **record} for record in records)
        calibration, calibration_meta = _read_sealed(root, reference, model, "calibration", provenance)
        final_cutoff = calibration.origin.min()
        trial, choices = select_released_trial({key: pd.concat(value, ignore_index=True)
                                               for key, value in all_trials.items()}, labels, final_cutoff)
        if trial != int(chosen_original["trial"]) or params_by_trial[trial] != chosen_original["params"]:
            raise RuntimeError(f"Causal January selection disagrees with original final selection: {model}")
        if calibration_meta["params"] != params_by_trial[trial]:
            raise RuntimeError(f"Calibration configuration disagrees with frozen base choice: {model}")
        test, test_meta = _read_sealed(root, reference, model, "test", provenance)
        if test_meta["params"] != params_by_trial[trial]:
            raise RuntimeError(f"Test configuration disagrees with frozen base choice: {model}")
        final_checks.append({"model": model, "selection_cutoff": str(final_cutoff), "selected_trial": trial,
                             "matches_original": True, "choices": choices})
        stream = pd.concat([development, calibration, test], ignore_index=True)
        if stream.duplicated(KEYS).any():
            raise RuntimeError(f"Development/calibration/test origin pairs overlap: {model}")
        prefix = f"base__{model}"
        output = stream[KEYS].copy()
        output[prefix + "__prediction"] = stream.prediction_mw.to_numpy()
        output[prefix + "__missing"] = stream.prediction_mw.isna().astype(float).to_numpy()
        # Raw availability records actual base failure even when its declared
        # fallback supplies the complete-policy prediction used above.
        output[prefix + "__fallback"] = stream.raw_prediction_mw.isna().astype(float).to_numpy()
        pieces.append(output.set_index(KEYS))
    expected = pieces[0].index
    if any(not part.index.equals(expected) for part in pieces[1:]):
        raise RuntimeError("Base variants do not share identical origin-delivery pairs")
    wide = pd.concat(pieces, axis=1).reset_index().sort_values(KEYS).reset_index(drop=True)
    ledger = {"reference_run": REFERENCE_RUN, "reference_manifest_created_utc": manifest["created_utc"],
              "policy": "Q1 fixed trial0 burn-in; later 2024 quarters choose released past-quarter MAE; freeze all-2024 choice in January 2025",
              "meta_validation_start": "2024-04-01", "raw_availability_minimum": RAW_AVAILABILITY_MINIMUM,
              "model_count": len(selected), "rows": len(wide), "quarter_selections": selections,
              "final_selection_checks": final_checks, "source_sha256": provenance,
              "feature_scope": "Previously out-of-sample base forecasts for an explicitly two-stage hybrid; no native exogenous architecture claim",
              "regional_arm_semantics": "Original regional base models use national plus regional information, never regional-only",
              "previous_test_exposed": True, "fresh_confirmation": False}
    return wide, ledger
