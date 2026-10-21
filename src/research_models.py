"""Small, auditable model adapters for the research notebooks.

This module does not split data, select hyperparameters, or know evaluation
outcomes. The replay engine must supply only information released before each
forecast origin. In particular, ``fit`` never performs a random validation split.

The GAM is a penalized additive B-spline regression, implemented with sklearn.
It is not advertised as pyGAM: smoothing is Ridge's alpha and must be selected
with external chronological validation. No cross-feature interactions are added.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
import time
import warnings
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler
from threadpoolctl import threadpool_limits


MODEL_REGISTRY = {
    "arima": {"kind": "statistical", "hypothesis": "A small nonseasonal autoregressive model is a useful history-only reference.", "objective": "Gaussian likelihood; conditional mean", "uses_exog": False},
    "sarima": {"kind": "statistical", "hypothesis": "Explicit daily seasonality improves on nonseasonal autoregression.", "objective": "Gaussian likelihood; conditional mean", "uses_exog": False},
    "sarimax": {"kind": "statistical", "hypothesis": "Available calendar and historical regressors improve seasonal autoregression.", "objective": "Gaussian likelihood; conditional mean", "uses_exog": True},
    "hist_gradient_boosting": {"kind": "tabular", "hypothesis": "Regularized nonlinear interactions improve on recurring profiles.", "objective": "absolute error; conditional median", "uses_exog": True},
    "lightgbm": {"kind": "tabular", "hypothesis": "Leaf-wise boosting improves accuracy or efficiency over histogram boosting.", "objective": "absolute error; conditional median", "uses_exog": True},
    "lightgbm_quantile": {"kind": "tabular", "hypothesis": "Conditional quantiles describe asymmetric demand uncertainty.", "objective": "pinball loss at the declared quantile", "uses_exog": True},
    "gam": {"kind": "tabular", "hypothesis": "Penalized additive smooth effects can compete with more complex interactions.", "objective": "penalized squared error; conditional mean", "uses_exog": True, "implementation": "sklearn independent B-splines + Ridge; not pyGAM"},
}


_FAST_GRIDS = {
    # Explicit recent-window research policies make the compact benchmark
    # feasible on the Intel CPU. Optimizer caps are 40/60 iterations, with
    # convergence reported explicitly; hitting a cap does not prove underfit.
    # The replay engine removes window_days before
    # constructing the adapter. These are not claims of optimal ARIMA orders.
    "arima": [{"order": [2, 1, 0], "maxiter": 40, "window_days": 56}, {"order": [3, 1, 1], "maxiter": 60, "window_days": 84}],
    "sarima": [{"order": [1, 0, 0], "seasonal_order": [1, 0, 0, 24], "maxiter": 40, "window_days": 56}, {"order": [2, 0, 0], "seasonal_order": [0, 0, 1, 24], "maxiter": 60, "window_days": 84}],
    # SARIMAX uses an explicit standardized-likelihood numerical tolerance.
    # Its forecast stability was checked without evaluation outcomes; the
    # numerical stopping status remains reported for every individual fit.
    "sarimax": [{"order": [1, 0, 0], "seasonal_order": [1, 0, 0, 24], "maxiter": 40, "window_days": 56, "pgtol": 1e-3, "factr": 1e8}, {"order": [2, 0, 0], "seasonal_order": [0, 0, 1, 24], "maxiter": 60, "window_days": 84, "pgtol": 1e-3, "factr": 1e8}],
    "hist_gradient_boosting": [{"max_iter": 80, "max_leaf_nodes": 15, "min_samples_leaf": 48, "l2_regularization": 2.0}, {"max_iter": 120, "max_leaf_nodes": 31, "min_samples_leaf": 72, "l2_regularization": 5.0}],
    "lightgbm": [{"n_estimators": 100, "num_leaves": 15, "min_child_samples": 48, "reg_lambda": 2.0}, {"n_estimators": 150, "num_leaves": 31, "min_child_samples": 72, "reg_lambda": 5.0}],
    "lightgbm_quantile": [{"n_estimators": 100, "num_leaves": 15, "min_child_samples": 72, "reg_lambda": 2.0}, {"n_estimators": 150, "num_leaves": 31, "min_child_samples": 96, "reg_lambda": 5.0}],
    "gam": [{"n_knots": 4, "degree": 3, "alpha": 20.0}, {"n_knots": 6, "degree": 3, "alpha": 100.0}],
}


def parameter_grid(family: str, profile: str = "fast") -> list[dict[str, Any]]:
    """Return bounded, deterministic candidates; never use test-set information.

    ``fast`` gives two configurations per family. Statistical fast candidates
    include explicit 56/84-day windows and iteration caps for bounded CPU cost;
    these do not establish optimally tuned statistical performance.
    ``full`` is a modest expansion
    (8 statistical and 12 tabular configurations), not an exhaustive search.
    """
    if family not in _FAST_GRIDS:
        raise ValueError(f"Unknown model family: {family}")
    if profile not in {"fast", "full"}:
        raise ValueError("profile must be 'fast' or 'full'")
    if profile == "fast":
        return deepcopy(_FAST_GRIDS[family])
    if family in {"arima", "sarima", "sarimax"}:
        grid = []
        for p in (1, 2):
            for q in (0, 1):
                for d in (0, 1):
                    params = {"order": [p, d, q], "maxiter": 60, "window_days": 365}
                    if family != "arima":
                        params["seasonal_order"] = [1, 0, 0, 24]
                    grid.append(params)
        return grid
    grid = []
    for base in _FAST_GRIDS[family]:
        for scale in (0.5, 1.0, 2.0):
            for rate in (0.04, 0.08):
                params = deepcopy(base)
                if family == "gam":
                    params["alpha"] *= scale
                    params["n_knots"] += int(rate == 0.08)
                else:
                    key = "l2_regularization" if family == "hist_gradient_boosting" else "reg_lambda"
                    params[key] *= scale
                    params["learning_rate"] = rate
                grid.append(params)
    return grid


def _numeric_frame(X: Any, columns: list | None = None) -> pd.DataFrame:
    if isinstance(X, pd.DataFrame):
        frame = X.copy()
        if frame.columns.duplicated().any():
            raise ValueError("Duplicate feature names are not supported")
        if columns is not None:
            missing = set(columns).difference(frame.columns)
            if missing:
                raise ValueError(f"Missing fitted feature columns: {sorted(missing)}")
            frame = frame.loc[:, columns]
    else:
        array = np.asarray(X, dtype=float)
        if array.ndim != 2:
            raise ValueError("X must be a two-dimensional numeric feature matrix")
        frame = pd.DataFrame(array, columns=columns)
    if frame.shape[1] == 0:
        raise ValueError("At least one feature is required")
    frame = frame.astype(float)
    return frame.replace([np.inf, -np.inf], np.nan)


@dataclass
class TabularAdapter:
    family: str
    params: dict[str, Any] = field(default_factory=dict)
    quantile: float | None = None
    seed: int = 42

    def fit(self, X: Any, y: Any) -> "TabularAdapter":
        frame = _numeric_frame(X)
        target = np.asarray(y, dtype=float).reshape(-1)
        if len(frame) != len(target) or not np.isfinite(target).all():
            raise ValueError("Training targets must be finite, aligned, and explicitly filtered by the caller")
        self.feature_names_in_ = list(frame.columns)
        options = deepcopy(self.params)
        self.thread_limit_ = int(options.pop("n_jobs", 1))
        imputer = SimpleImputer(strategy="median", keep_empty_features=True)
        started = time.perf_counter()
        if self.family == "hist_gradient_boosting":
            defaults = dict(loss="absolute_error", learning_rate=0.06, max_iter=120,
                            max_leaf_nodes=15, min_samples_leaf=48,
                            l2_regularization=2.0, early_stopping=False,
                            random_state=self.seed)
            defaults.update(options)
            if defaults.get("early_stopping") is not False:
                raise ValueError("Random internal early stopping is forbidden; use external chronological validation")
            if self.quantile is not None:
                defaults.update(loss="quantile", quantile=self.quantile)
            self.estimator_ = Pipeline([("imputer", imputer), ("model", HistGradientBoostingRegressor(**defaults))])
        elif self.family in {"lightgbm", "lightgbm_quantile"}:
            try:
                from lightgbm import LGBMRegressor
            except ImportError as exc:
                raise ImportError("Install the research LightGBM dependency in the selected isolated environment") from exc
            defaults = dict(objective="regression_l1", learning_rate=0.06,
                            n_estimators=150, num_leaves=15, min_child_samples=48,
                            reg_lambda=2.0, random_state=self.seed, n_jobs=1,
                            verbosity=-1, deterministic=True, force_col_wise=True)
            defaults.update(options)
            if self.quantile is not None:
                defaults.update(objective="quantile", alpha=self.quantile)
            self.estimator_ = Pipeline([("imputer", imputer), ("model", LGBMRegressor(**defaults))])
        elif self.family == "gam":
            allowed = {"alpha", "n_knots", "degree", "knots", "extrapolation"}
            unknown = set(options).difference(allowed)
            if unknown:
                raise ValueError(f"Unknown additive spline parameters: {sorted(unknown)}")
            spline = SplineTransformer(n_knots=options.get("n_knots", 5),
                                       degree=options.get("degree", 3),
                                       knots=options.get("knots", "quantile"),
                                       extrapolation=options.get("extrapolation", "linear"),
                                       include_bias=False)
            self.estimator_ = Pipeline([("imputer", imputer), ("splines", spline),
                                        ("scale", StandardScaler()),
                                        ("model", Ridge(alpha=options.get("alpha", 20.0)))])
        else:
            raise ValueError(f"Unknown tabular family: {self.family}")
        with threadpool_limits(limits=self.thread_limit_), warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            self.estimator_.fit(frame, target)
        self.metadata_ = {"family": self.family, "training_rows": len(frame),
                          "feature_count": frame.shape[1], "fit_seconds": time.perf_counter() - started,
                          "warnings": list(dict.fromkeys(str(w.message) for w in caught)),
                          "quantile": self.quantile, "seed": self.seed,
                          "implementation": MODEL_REGISTRY[self.family].get("implementation", self.estimator_["model"].__class__.__name__),
                          "objective": (f"pinball loss q={self.quantile}" if self.quantile is not None else MODEL_REGISTRY[self.family]["objective"])}
        return self

    def predict(self, X: Any) -> np.ndarray:
        if not hasattr(self, "estimator_"):
            raise RuntimeError("Call fit before predict")
        with threadpool_limits(limits=self.thread_limit_):
            return np.asarray(self.estimator_.predict(_numeric_frame(X, self.feature_names_in_)), dtype=float)


def make_tabular(family: str, params: dict | None = None, quantile: float | None = None, seed: int = 42) -> TabularAdapter:
    if family not in {"hist_gradient_boosting", "lightgbm", "lightgbm_quantile", "gam"}:
        raise ValueError(f"Unknown tabular family: {family}")
    if quantile is not None and not 0 < quantile < 1:
        raise ValueError("quantile must lie strictly between zero and one")
    if family == "lightgbm_quantile" and quantile is None:
        quantile = 0.5
    if family == "gam" and quantile is not None:
        raise ValueError("This penalized least-squares GAM does not fit quantile loss")
    return TabularAdapter(family, deepcopy(params or {}), quantile, seed)


def _hourly_series(series: pd.Series) -> pd.Series:
    if not isinstance(series, pd.Series) or not isinstance(series.index, pd.DatetimeIndex):
        raise TypeError("series must be a pandas Series with a DatetimeIndex")
    if series.empty or not series.index.is_unique or not series.index.is_monotonic_increasing:
        raise ValueError("series must be nonempty with unique, ascending timestamps")
    if series.index.tz is None:
        raise ValueError("A timezone-aware index is required; UTC avoids ambiguous DST hours")
    values = series.astype(float).copy()
    values.index = values.index.tz_convert("UTC")
    if ((values.index - values.index[0]).total_seconds() % 3600 != 0).any():
        raise ValueError("series timestamps must share one hourly grid")
    grid = pd.date_range(values.index[0], values.index[-1], freq="h")
    return values.reindex(grid).replace([np.inf, -np.inf], np.nan).rename("load")


@dataclass
class StatisticalAdapter:
    family: str
    params: dict[str, Any] = field(default_factory=dict)

    def _exog(self, exog: pd.DataFrame | None, index: pd.DatetimeIndex, fit: bool = False) -> np.ndarray | None:
        if self.family != "sarimax":
            if exog is not None:
                raise ValueError(f"{self.family} is history-only; use sarimax for regressors")
            return None
        if fit:
            self.exog_columns_ = [] if exog is None else list(exog.columns)
            if not self.exog_columns_:
                self.exog_transformer_ = None
                return None
            frame = _numeric_frame(exog.reindex(index))
            self.exog_transformer_ = Pipeline([("imputer", SimpleImputer(strategy="median", keep_empty_features=True)), ("scale", StandardScaler())])
            return self.exog_transformer_.fit_transform(frame)
        if not self.exog_columns_:
            if exog is not None and exog.shape[1]:
                raise ValueError("Model was fitted without exogenous regressors")
            return None
        if exog is None:
            raise ValueError("Future/update exogenous regressors are required")
        if not isinstance(exog.index, pd.DatetimeIndex) or not index.isin(exog.index).all():
            raise ValueError("Exogenous regressors must cover every required hourly timestamp")
        frame = _numeric_frame(exog.reindex(index), self.exog_columns_)
        return self.exog_transformer_.transform(frame)

    @staticmethod
    def _conserve_memory(model):
        # low_memory=True removes the public predicted_state needed by extend.
        # Selective conservation avoids smoothing/filter/gain storage while
        # retaining the prediction state and covariance required for exact
        # fixed-parameter chronological state updates.
        from statsmodels.tsa.statespace.kalman_filter import MEMORY_CONSERVE, MEMORY_NO_PREDICTED
        model.ssm.set_conserve_memory(MEMORY_CONSERVE & ~MEMORY_NO_PREDICTED)

    def fit(self, series: pd.Series, exog: pd.DataFrame | None = None) -> "StatisticalAdapter":
        values = _hourly_series(series)
        if values.notna().sum() < 48:
            raise ValueError("At least 48 observed hourly training targets are required")
        self.location_ = float(values.mean())
        self.scale_ = max(float(values.std()), 1.0)
        normalized = (values - self.location_) / self.scale_
        transformed_exog = self._exog(exog, values.index, fit=True)
        options = deepcopy(self.params)
        order = tuple(options.pop("order", (2, 1, 0)))
        seasonal = tuple(options.pop("seasonal_order", (1, 0, 0, 24)))
        maxiter = int(options.pop("maxiter", 40))
        optimizer_options = {name: options.pop(name) for name in ("pgtol", "factr") if name in options}
        allowed = {"trend", "enforce_stationarity", "enforce_invertibility"}
        if set(options).difference(allowed):
            raise ValueError(f"Unknown statistical parameters: {sorted(set(options).difference(allowed))}")
        # Disable costly covariance estimation; this experiment calibrates its own
        # out-of-sample uncertainty rather than using Gaussian model intervals.
        started = time.perf_counter()
        with threadpool_limits(limits=1), warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            if self.family == "arima":
                from statsmodels.tsa.arima.model import ARIMA
                model = ARIMA(normalized, order=order, **options)
                self._conserve_memory(model)
                self.result_ = model.fit(method_kwargs={"maxiter": maxiter, **optimizer_options}, cov_type="none")
            else:
                from statsmodels.tsa.statespace.sarimax import SARIMAX
                model = SARIMAX(normalized, exog=transformed_exog, order=order,
                                seasonal_order=seasonal, **options)
                self._conserve_memory(model)
                self.result_ = model.fit(maxiter=maxiter, disp=False, cov_type="none", **optimizer_options)
        self.last_timestamp_ = values.index[-1]
        mle = getattr(self.result_, "mle_retvals", {}) or {}
        self.metadata_ = {"family": self.family, "training_rows": len(values),
                          "observed_training_rows": int(values.notna().sum()),
                          "missing_training_rows": int(values.isna().sum()),
                          "training_start": str(values.index[0]), "training_end": str(values.index[-1]),
                          "fit_seconds": time.perf_counter() - started,
                          "warnings": list(dict.fromkeys(str(w.message) for w in caught)),
                          "converged": bool(mle.get("converged", True)),
                          "optimizer_iterations": int(mle.get("iterations", 0)),
                          "optimizer_options": {"maxiter": maxiter, **optimizer_options},
                          "optimizer_gradient_max_abs": float(np.max(np.abs(mle.get("gopt", [np.nan])))),
                          "optimizer_cap_does_not_prove_underfit": True,
                          "state_updates": 0, "parameter_refits": 1,
                          "memory_policy": "conserve smoothing/filter storage; retain predicted states for extend",
                          "order": list(order), "seasonal_order": ([0, 0, 0, 0] if self.family == "arima" else list(seasonal)),
                          "objective": MODEL_REGISTRY[self.family]["objective"]}
        return self

    def forecast(self, steps: int, exog: pd.DataFrame | None = None) -> np.ndarray:
        if not hasattr(self, "result_"):
            raise RuntimeError("Call fit before forecast")
        if not isinstance(steps, (int, np.integer)) or steps < 1:
            raise ValueError("steps must be a positive integer")
        index = pd.date_range(self.last_timestamp_ + pd.Timedelta(hours=1), periods=steps, freq="h")
        transformed = self._exog(exog, index)
        with threadpool_limits(limits=1):
            predicted = self.result_.forecast(steps=steps, exog=transformed)
        return np.asarray(predicted, dtype=float) * self.scale_ + self.location_

    def update(self, series: pd.Series, exog: pd.DataFrame | None = None) -> "StatisticalAdapter":
        """Filter newly released observations while keeping fitted parameters fixed.

        Accepts either only new observations or a larger history prefix. Missing
        intermediate hours are represented as NaN, never removed/interpolated.
        Overlapping timestamps are ignored; revised-vintage refitting is a
        separate policy and must be explicit in the replay engine.
        """
        if not hasattr(self, "result_"):
            raise RuntimeError("Call fit before update")
        incoming = _hourly_series(series)
        new = incoming.loc[incoming.index > self.last_timestamp_]
        if new.empty:
            return self
        index = pd.date_range(self.last_timestamp_ + pd.Timedelta(hours=1), new.index[-1], freq="h")
        new = new.reindex(index)
        transformed = self._exog(exog, index)
        normalized = ((new - self.location_) / self.scale_).rename("load")
        # extend performs state filtering only and keeps storage bounded. Because
        # every update starts immediately after last_timestamp_, no observation
        # is consumed twice and the hourly state transition remains intact.
        with threadpool_limits(limits=1):
            self.result_ = self.result_.extend(normalized, exog=transformed)
        self.last_timestamp_ = index[-1]
        self.metadata_["state_updates"] += 1
        self.metadata_["latest_state_timestamp"] = str(index[-1])
        return self


def make_statistical(family: str, params: dict | None = None) -> StatisticalAdapter:
    if family not in {"arima", "sarima", "sarimax"}:
        raise ValueError(f"Unknown statistical family: {family}")
    return StatisticalAdapter(family, deepcopy(params or {}))
