"""Shared evaluation module for the electricity demand forecasting project.

Every model notebook (naive baseline, SARIMAX, LightGBM, probabilistic intervals,
final comparison, ...) imports from here instead of redefining its own data
loading, train/test splits, or metrics. That's the whole point of this module:
if each notebook rolled its own splits or metric formulas, results across models
would not be a fair comparison. By sharing `load_data`, `walk_forward_folds`,
the metric functions, and `save_result`, every `results/<model>.json` file is
directly comparable to every other one.

Validation standard implemented here: walk-forward, quarterly, expanding (or
optionally sliding) window over the test year 2025-07-01 -> 2026-06-30, split
into 4 quarterly folds. See readme.md for the full rationale.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

# (train_end, test_start, test_end), all inclusive, in Europe/Paris local time.
_FOLD_BOUNDARIES = [
    ("2025-06-30 23:00", "2025-07-01 00:00", "2025-09-30 23:00"),
    ("2025-09-30 23:00", "2025-10-01 00:00", "2025-12-31 23:00"),
    ("2025-12-31 23:00", "2026-01-01 00:00", "2026-03-31 23:00"),
    ("2026-03-31 23:00", "2026-04-01 00:00", "2026-06-30 23:00"),
]


def load_data(data_dir="../data"):
    """Load the hourly France electricity CSV, convert to Europe/Paris local time.

    Parses `datetime_utc`, converts to `Europe/Paris`, sorts chronologically, names
    the index `datetime_local`, and drops the 7 rows with NaN load/RTE forecast at
    the autumn DST transition each year.
    """
    data_dir = Path(data_dir)
    raw = pd.read_csv(data_dir / "france_electricity_hourly.csv")
    raw["datetime_utc"] = pd.to_datetime(raw["datetime_utc"], utc=True)

    df = raw.set_index("datetime_utc").tz_convert("Europe/Paris").sort_index()
    df.index.name = "datetime_local"

    df = df.dropna(subset=["load_mw", "rte_forecast_d1_mw"])
    return df


def mae(y, y_hat):
    y, y_hat = np.asarray(y), np.asarray(y_hat)
    return float(np.mean(np.abs(y - y_hat)))


def rmse(y, y_hat):
    y, y_hat = np.asarray(y), np.asarray(y_hat)
    return float(np.sqrt(np.mean((y - y_hat) ** 2)))


def mape(y, y_hat):
    y, y_hat = np.asarray(y), np.asarray(y_hat)
    return float(np.mean(np.abs((y - y_hat) / y)) * 100)


def score(y, y_hat):
    """Align two series, drop unpaired/NaN rows, return MAE/RMSE/MAPE dict."""
    paired = pd.concat([y, y_hat], axis=1).dropna()
    return {
        "MAE_MW": mae(paired.iloc[:, 0], paired.iloc[:, 1]),
        "RMSE_MW": rmse(paired.iloc[:, 0], paired.iloc[:, 1]),
        "MAPE_pct": mape(paired.iloc[:, 0], paired.iloc[:, 1]),
    }


def walk_forward_folds(df, n_folds=4, window="expanding", window_size=None):
    """Yield (fold_idx, train, test) over the 4 fixed quarterly folds.

    Test region: 2025-07-01 -> 2026-06-30, split into 4 quarterly folds:
      fold 1: train <= 2025-06-30 23:00, test = Jul-Sep 2025
      fold 2: train <= 2025-09-30 23:00, test = Oct-Dec 2025
      fold 3: train <= 2025-12-31 23:00, test = Jan-Mar 2026
      fold 4: train <= 2026-03-31 23:00, test = Apr-Jun 2026

    window="expanding": train is everything in `df` from its start up to the
    fold's train-end timestamp (inclusive).
    window="sliding": train is only the last `window_size` (a pandas.Timedelta,
    or a number of hours) ending at the same train-end timestamp.
    """
    if n_folds != 4:
        raise ValueError("this project's walk-forward scheme is fixed to 4 quarterly folds")
    if window not in ("expanding", "sliding"):
        raise ValueError("window must be 'expanding' or 'sliding'")
    if window == "sliding" and window_size is None:
        raise ValueError("window_size is required when window='sliding'")

    if window == "sliding" and not isinstance(window_size, pd.Timedelta):
        window_size = pd.Timedelta(hours=window_size)

    for fold_idx, (train_end, test_start, test_end) in enumerate(_FOLD_BOUNDARIES, start=1):
        if window == "expanding":
            train = df.loc[:train_end]
        else:
            train_end_ts = df.loc[:train_end].index.max()
            train_start_ts = train_end_ts - window_size + pd.Timedelta(hours=1)
            train = df[(df.index >= train_start_ts) & (df.index <= train_end_ts)]

        test = df.loc[test_start:test_end]
        yield fold_idx, train, test


def save_result(name, description, validation, fold_results, results_dir="../results"):
    """Save per-fold metrics plus their mean/std to results/<name>.json.

    `fold_results` is a list of dicts, one per fold, each shaped like:
        {"fold": 1, "train_start": ..., "train_end": ..., "test_start": ...,
         "test_end": ..., "metrics": {"MAE_MW": ..., "RMSE_MW": ..., "MAPE_pct": ...}}
    """
    metric_names = ["MAE_MW", "RMSE_MW", "MAPE_pct"]
    metric_values = {m: [fr["metrics"][m] for fr in fold_results] for m in metric_names}

    mean_metrics = {m: float(np.mean(v)) for m, v in metric_values.items()}
    std_metrics = {m: float(np.std(v)) for m, v in metric_values.items()}

    payload = {
        "model": name,
        "description": description,
        "validation": validation,
        "folds": fold_results,
        "mean_metrics": mean_metrics,
        "std_metrics": std_metrics,
    }

    results_dir = Path(results_dir)
    results_dir.mkdir(exist_ok=True)
    path = results_dir / f"{name}.json"
    path.write_text(json.dumps(payload, indent=2, default=str))
    print(f"saved {path}")
