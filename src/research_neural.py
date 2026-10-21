"""Official NeuralForecast adapters for the research benchmark.

The six architectures are Nixtla implementations, pinned to NeuralForecast 1.7.7
for this Intel Mac's PyTorch 2.2.2. No fallback architecture is substituted.
The forecast-origin engine, not this adapter, is responsible for publication
delays, eligible history, delivery-day selection, and origin-based outer scoring.

All variants here deliberately use target history only. Global panel fitting is
shared-parameter univariate learning, not explicit cross-region multivariate
learning. Each call predicts ``h`` consecutive UTC hours after the context end;
the engine must select tomorrow's 23/24/25 actual delivery intervals.

Missing training observations remain on the grid with ``available_mask=0`` and
are excluded from normalization statistics/loss. Predictor placeholders use only
the previous observed value within that series; leading gaps use zero. This is
explicit causal input imputation, with no backward fill or target interpolation.
The caller's evaluation truth remains unchanged and missing targets stay excluded.

Upstream sources (versioned):
https://github.com/Nixtla/neuralforecast/tree/v1.7.7/neuralforecast/models
https://github.com/Nixtla/neuralforecast/blob/v1.7.7/neuralforecast/core.py
https://github.com/Nixtla/neuralforecast/blob/v1.7.7/settings.ini
"""
from __future__ import annotations

import gc
import importlib.metadata
import logging
import os
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import pandas as pd


FAMILIES = ("nbeats", "nhits", "tft", "gru", "patchtst", "dlinear")
PINNED_NEURALFORECAST = "1.7.7"


def _backend():
    """Load project-local dependencies without modifying the scientific venv."""
    os.environ.setdefault("NIXTLA_ID_AS_COL", "1")
    dependency_dir = Path(__file__).resolve().parents[1] / ".research-deps"
    if dependency_dir.is_dir() and str(dependency_dir) not in sys.path:
        sys.path.insert(0, str(dependency_dir))
    try:
        import torch
        from neuralforecast import NeuralForecast
        from neuralforecast import models
        from neuralforecast.losses.pytorch import MAE
    except ImportError as exc:
        raise ImportError(
            "The pinned research neural stack is unavailable. Install "
            "requirements-research-neural.txt into .research-deps using the "
            "documented --no-deps command, with the existing torch==2.2.2. "
            "No substitute model has been run."
        ) from exc
    actual = importlib.metadata.version("neuralforecast")
    if actual != PINNED_NEURALFORECAST:
        raise RuntimeError(
            f"Expected neuralforecast=={PINNED_NEURALFORECAST}; found {actual}. "
            "Revalidate model APIs, masks and reproducibility before changing this pin."
        )
    torch.set_num_threads(2)
    return torch, NeuralForecast, models, MAE


def backend_versions() -> dict[str, str]:
    """Return dependency provenance, raising rather than faking availability."""
    torch, *_ = _backend()
    names = ("neuralforecast", "pytorch-lightning", "torchmetrics", "coreforecast",
             "utilsforecast", "ray", "numpy", "pandas")
    return {"torch": torch.__version__, **{
        name: importlib.metadata.version(name) for name in names
    }}


def neural_configurations(family: str) -> list[dict[str, Any]]:
    """Two predeclared, compact CPU configurations; validation selects the winner.

    These are deliberately bounded research budgets, not claims of optimal tuning.
    The training loss spans the 48-step bridge; external selection scores only
    the requested delivery intervals. Inner early stopping uses a chronological
    training tail and never the outer fold's delivery outcomes.
    """
    family = family.lower()
    if family not in FAMILIES:
        raise ValueError(f"Unknown neural family: {family}")
    return [
        {"h": 48, "input_size": 168, "max_steps": 20,
         "learning_rate": 1e-3, "val_check_steps": 10},
        {"h": 48, "input_size": 336, "max_steps": 40,
         "learning_rate": 5e-4, "val_check_steps": 10},
    ]


def _panel(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"unique_id", "ds", "y"}
    if not required.issubset(frame):
        raise ValueError(f"Panel must contain {sorted(required)}")
    result = frame[["unique_id", "ds", "y"]].copy()
    result["unique_id"] = result["unique_id"].astype(str).astype(object)
    parsed = pd.to_datetime(result["ds"], utc=True)
    # Pin nanosecond precision for compatibility with the older Nixtla helpers.
    result["ds"] = parsed.dt.tz_localize(None).astype("datetime64[ns]")
    if result["ds"].isna().any():
        raise ValueError("Panel timestamps cannot be missing")
    if result.duplicated(["unique_id", "ds"]).any():
        raise ValueError("Duplicate panel series/timestamp keys")
    result = result.sort_values(["unique_id", "ds"]).reset_index(drop=True)
    for uid, group in result.groupby("unique_id", sort=False):
        if len(group) < 2 or not group["ds"].diff().iloc[1:].eq(pd.Timedelta(hours=1)).all():
            raise ValueError(f"Series {uid!r} must retain a complete regular hourly grid")
    values = pd.to_numeric(result["y"], errors="raise").to_numpy(dtype=np.float32)
    observed = np.isfinite(values)
    if "available_mask" in frame:
        mask_frame = frame[["unique_id", "ds", "available_mask"]].copy()
        mask_frame["unique_id"] = mask_frame["unique_id"].astype(str).astype(object)
        mask_frame["ds"] = pd.to_datetime(mask_frame["ds"], utc=True).dt.tz_localize(None).astype("datetime64[ns]")
        supplied = result[["unique_id", "ds"]].merge(
            mask_frame, on=["unique_id", "ds"], validate="one_to_one"
        )["available_mask"].to_numpy()
        if not np.isin(supplied, [0, 1]).all():
            raise ValueError("available_mask must contain only 0 or 1")
        observed &= supplied.astype(bool)
    result["available_mask"] = observed.astype(np.float32)
    # Several official architectures ignore the input mask. Give them a causal
    # stale-value placeholder instead of an artificial zero-demand spike. Missing
    # labels still carry mask=0, so these placeholders never become target truth.
    result["y"] = np.where(observed, values, np.nan).astype(np.float32)
    result["y"] = result.groupby("unique_id", sort=False)["y"].ffill().fillna(0).astype(np.float32)
    if not result.groupby("unique_id")["available_mask"].sum().gt(0).all():
        raise ValueError("Every series needs at least one observed training/context value")
    return result


class NeuralAdapter:
    """Real target-history-only forecasting model, with auditable fit metadata."""

    def __init__(self, family: str, params: dict[str, Any] | None = None, seed: int = 42):
        self.family = family.lower()
        if self.family not in FAMILIES:
            raise ValueError(f"Unknown neural family: {family}; expected {FAMILIES}")
        self.params = dict(params or {})
        if any(self.params.get(key) for key in ("futr_exog_list", "hist_exog_list", "stat_exog_list")):
            raise ValueError("This benchmark adapter deliberately supports target history only")
        self.seed = int(seed)
        self.h = int(self.params.get("h", 48))
        self.input_size = int(self.params.get("input_size", 168))
        self.forecaster = None
        self.fit_metadata: dict[str, Any] = {}

    def _model(self):
        _, _, models, MAE = _backend()
        common = dict(
            h=self.h, input_size=self.input_size,
            loss=MAE(), valid_loss=MAE(), learning_rate=1e-3,
            max_steps=20, val_check_steps=10, early_stop_patience_steps=3,
            scaler_type="robust", random_seed=self.seed,
            batch_size=16, valid_batch_size=16, num_workers_loader=0,
            accelerator="cpu", devices=1, logger=False, enable_checkpointing=False,
            enable_progress_bar=False, enable_model_summary=False,
            deterministic=True, num_sanity_val_steps=0,
        )
        defaults = {
            "nbeats": dict(stack_types=["identity", "trend", "seasonality"],
                           n_blocks=[1, 1, 1], mlp_units=[[32, 32]] * 3),
            "nhits": dict(n_blocks=[1, 1, 1], mlp_units=[[32, 32]] * 3,
                          n_pool_kernel_size=[4, 2, 1], n_freq_downsample=[4, 2, 1]),
            "tft": dict(hidden_size=16, n_head=2, dropout=0.1),
            "gru": dict(encoder_hidden_size=32, encoder_n_layers=1,
                        decoder_hidden_size=32, decoder_layers=1, context_size=8,
                        inference_input_size=self.input_size),
            "patchtst": dict(hidden_size=16, n_heads=2, encoder_layers=1,
                             linear_hidden_size=32, patch_len=24, stride=12,
                             dropout=0.1, fc_dropout=0.1),
            "dlinear": dict(moving_avg_window=25),
        }
        common.update(defaults[self.family])
        if self.family != "gru":
            common.update(windows_batch_size=32, inference_windows_batch_size=32,
                          step_size=24, start_padding_enabled=False)
        common.update(self.params)
        # Never permit accelerator defaults or an alias to conceal the architecture.
        common.update(accelerator="cpu", devices=1, random_seed=self.seed)
        names = {"nbeats": "NBEATS", "nhits": "NHITS", "tft": "TFT", "gru": "GRU",
                 "patchtst": "PatchTST", "dlinear": "DLinear"}
        return getattr(models, names[self.family])(**common)

    def fit(self, panel: pd.DataFrame, validation_size: int = 168) -> "NeuralAdapter":
        _, NeuralForecast, _, _ = _backend()
        cleaned = _panel(panel)
        validation_size = int(validation_size)
        if validation_size < self.h:
            raise ValueError("Chronological early-stopping tail must be at least h hours")
        smallest = int(cleaned.groupby("unique_id").size().min())
        if smallest < self.input_size + self.h + validation_size:
            raise ValueError("Insufficient history for context + horizon + validation tail")
        for name in ("pytorch_lightning", "lightning", "lightning_fabric",
                     "pytorch_lightning.utilities.rank_zero"):
            logging.getLogger(name).setLevel(logging.ERROR)
        model = self._model()
        self.forecaster = NeuralForecast(models=[model], freq="h")
        started = time.perf_counter()
        self.forecaster.fit(df=cleaned, val_size=validation_size, verbose=False)
        trained = self.forecaster.models[0]
        self.fit_metadata = {
            "family": self.family, "implementation": "neuralforecast",
            "version": PINNED_NEURALFORECAST, "seed": self.seed,
            "fit_seconds": time.perf_counter() - started,
            "training_rows": len(cleaned), "series_count": cleaned["unique_id"].nunique(),
            "masked_training_rows": int(cleaned["available_mask"].eq(0).sum()),
            "validation_tail_hours": validation_size,
            "training_start": str(cleaned["ds"].min()),
            "training_end": str(cleaned["ds"].max()),
            "train_trajectory": [(int(s), float(v)) for s, v in getattr(trained, "train_trajectories", [])],
            "validation_trajectory": [(int(s), float(v)) for s, v in getattr(trained, "valid_trajectories", [])],
            "train_trajectory_units": "per-window robust-scaled target units",
            "validation_trajectory_units": "MW (Nixtla validation inverse-transforms targets)",
            "training_objective": f"MAE on {self.h}-step bridge; outer selection uses delivery intervals",
            "feature_scope": "target history only; per-window robust scaling",
            "missing_input_policy": "causal forward-fill predictor placeholders within series; leading gaps zero; available_mask=0 excludes missing target loss/statistics; some architectures do not consume input masks explicitly",
        }
        return self

    def predict(self, context: pd.DataFrame, future: pd.DataFrame | None = None) -> pd.DataFrame:
        if self.forecaster is None:
            raise RuntimeError("Fit the adapter before prediction")
        cleaned = _panel(context)
        if cleaned.groupby("unique_id").size().min() < self.input_size:
            raise ValueError("Prediction context is shorter than input_size")
        # There are no exogenous features. A future table, if supplied, is a request
        # grid only; it must exactly match the model's next-h-hour output.
        output = self.forecaster.predict(df=cleaned, verbose=False)
        if "unique_id" not in output.columns:
            output = output.reset_index()
        prediction_columns = [name for name in output if name not in ("unique_id", "ds")]
        if len(prediction_columns) != 1:
            raise RuntimeError(f"Expected one point forecast column, got {prediction_columns}")
        output = output.rename(columns={prediction_columns[0]: "prediction_mw"})
        output["ds"] = pd.to_datetime(output["ds"]).astype("datetime64[ns]")
        if not np.isfinite(output["prediction_mw"]).all():
            raise RuntimeError("Neural model produced nonfinite predictions")
        if future is not None:
            requested = future[["unique_id", "ds"]].copy()
            requested["unique_id"] = requested["unique_id"].astype(str).astype(object)
            requested["ds"] = pd.to_datetime(requested["ds"], utc=True).dt.tz_localize(None).astype("datetime64[ns]")
            checked = requested.merge(output, on=["unique_id", "ds"], how="outer", indicator=True,
                                      validate="one_to_one")
            if not checked["_merge"].eq("both").all():
                raise ValueError("Requested future grid does not equal the next h UTC hours")
        return output[["unique_id", "ds", "prediction_mw"]]

    def close(self) -> None:
        """Release fitted weights before the next CPU search trial."""
        self.forecaster = None
        gc.collect()
