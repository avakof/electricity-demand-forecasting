# Timeseries ProjectFrench Electricity Demand Forecasting

Forecast next-day electricity demand and compare performance
against RTE's public day-ahead forecast.

## Project contents

- notebooks/: exploration and analysis
- data/: provided datasets

## Role

You are a data scientist at a European energy utility: build, evaluate honestly and defend a day-ahead forecasting system for hourly electricity demand, and turn it into a one-page recommendation that a head of trading can act on.

## DATA

https://drive.google.com/drive/folders/1GY9IOY_7zdZpAm5xfbzSkFa-DunAA3Ag?usp=sharing

## CONTEXT

Your utility supplies electricity to households and businesses and buys the corresponding energy on the day-ahead market: every day D, before the auction closes at 12:00 CET, the trading desk must know the demand of each hour of day D+1. Every MWh of forecast error is settled later at imbalance prices, and the evening peaks of cold days are where errors are most expensive. The same forecasts feed reserve planning and hedging decisions.

In this project the utility's demand is represented by French national consumption (the provided dataset). Your manager asks three questions

1. Can we do better than simply using RTE's public day-ahead forecast, and by how much, honestly measured?
2. How uncertain is the forecast, and when does it fail?
3. What should we deploy, and what are the risks ?

## DELIVERABLE: 
Source Code on Github or Gitlab with clear code and git history. + 15 minutes PowerPoint Presentation (10 minutes + 5 minutes Q&As). Group of 2 people needed 

## Project plan / build order

This is not everything, we can also add other data and be more creative with the models we use. 

| Notebook                                       | Model                                                                                            | Depends on   | Status            | Owner |
| ---------------------------------------------- | ------------------------------------------------------------------------------------------------ | ------------ | ----------------- | ----- |
| `01_data_audit.ipynb`                          | Data audit                                                                                       | -            | done              | Ali   |
| `02_jg_baseline_model.ipynb`                   | Naive (lag-24, lag-168) + RTE benchmark                                                          | -            | done              | Julia |
| `03_sarimax.ipynb`                             | SARIMAX (weekly-seasonal, temperature exogenous)                                                 | -            | not started       | TBD   |
| `04_lightgbm.ipynb`                            | LightGBM (lags &gt;=24h, calendar, temperature)                                                  | -            | not started       | TBD   |
| `05_intervals.ipynb`                           | Probabilistic intervals wrapping the best point model                                            | 03 and/or 04 | not started       | TBD   |
| `06_final_comparison.ipynb`                    | Loads every file in `results/`, writes the recommendation                                        | all above    | not started       | TBD   |
| (later/stretch) deep model, e.g. N-BEATS/NHiTS | Possibly a global model over the regional series (`data/france_regional_electricity_hourly.csv`) | 03, 04       | not yet scheduled | TBD   |

The deep-model row is deliberately beyond this course's own stated scope (the course's guidance is that N-BEATS/NHiTS/TFT-style models earn their keep with many related series and long histories — a single national series doesn't strictly need them). We're doing it anyway since there's time budget left, most likely framed as a global model over the 12 regional series rather than the single national one, since that's the setting where deep/global models actually have an edge. Not scheduled in detail yet — needs its own discussion once 03-06 are done.

## Validation standard

Every model is scored with **walk-forward, quarterly, expanding-window validation** over the same test year used by the baseline (2025-07-01 -> 2026-06-30), split into 4 quarterly folds:

- fold 1: train <= 2025-06-30 23:00, test = Jul-Sep 2025
- fold 2: train <= 2025-09-30 23:00, test = Oct-Dec 2025
- fold 3: train <= 2025-12-31 23:00, test = Jan-Mar 2026
- fold 4: train <= 2026-03-31 23:00, test = Apr-Jun 2026

Why: a single holdout hides whether a model's score is a fluke of one period. Walk-forward with the same 4 folds for every model is the only way the models in `results/` are actually comparable to each other.

**SARIMAX-specific note**: a state-space MLE fit on the full expanding history (fold 1 alone is ~57,000 hourly rows) with a weekly seasonal period (`s=168`) can be very slow to refit from scratch each fold. `scripts/evaluation.py`'s `walk_forward_folds(..., window="sliding", window_size=...)` supports a sliding/truncated training window (e.g. the last 2 years) as a tractable alternative — SARIMAX should use that. LightGBM has no such constraint and should use the full expanding history (`window="expanding"`).

## Metrics standard

- **Headline**: MAE (MW) — robust, easy to communicate.
- **Also report**: RMSE (MW, penalizes large misses — relevant since evening peaks of cold days are the most expensive failure mode) and MAPE (%, safe here since load is never near zero).
- **Always relative, not just absolute**: report the % MAE improvement vs. the naive baseline and vs. the RTE benchmark for every model — that's the actual question the manager is asking.
- **Probabilistic models** (step 5, `05_intervals.ipynb`) additionally need coverage (does the nominal X% interval actually contain X% of outcomes), interval width, and pinball loss / Winkler score — point MAE alone isn't enough once you're forecasting a distribution.

## Shared evaluation module

`scripts/evaluation.py` is the shared module every model notebook should import from, so splits/metrics/results-saving stay identical across models:

- `load_data(data_dir="../data")` — loads the CSV, converts to `Europe/Paris` local time, sorts, drops the 7 DST-transition NaN rows.
- `walk_forward_folds(df, n_folds=4, window="expanding"|"sliding", window_size=None)` — yields `(fold_idx, train, test)` for the 4 quarterly folds above.
- `mae(y, y_hat)`, `rmse(y, y_hat)`, `mape(y, y_hat)` — the metric functions.
- `score(y, y_hat)` — aligns/dropna's a pair of series, returns `{"MAE_MW", "RMSE_MW", "MAPE_pct"}`.
- `save_result(name, description, validation, fold_results, results_dir="../results")` — writes `results/<name>.json` with per-fold metrics plus their mean/std.

Every model notebook should import these rather than redefining its own splits or metric formulas — see `02_jg_baseline_model.ipynb` for the pattern.

## `results/<model_name>.json` schema

```json
{
  "model": "<name>",
  "description": "<description>",
  "validation": "<validation>",
  "folds": [
    {
      "fold": 1,
      "train_start": "...",
      "train_end": "...",
      "test_start": "...",
      "test_end": "...",
      "metrics": {"MAE_MW": ..., "RMSE_MW": ..., "MAPE_pct": ...}
    }
  ],
  "mean_metrics": {"MAE_MW": ..., "RMSE_MW": ..., "MAPE_pct": ...},
  "std_metrics": {"MAE_MW": ..., "RMSE_MW": ..., "MAPE_pct": ...}
}
```

Save with `save_result(...)` from `scripts/evaluation.py` so every model's file has this exact shape — `06_final_comparison.ipynb` will just load every file in `results/` and assume it.

## Known issue to fix (not fixed here)

`01_data_audit.ipynb` currently loads data via a hardcoded absolute path specific to one machine (`/Users/alivakof/Desktop/Time Series/Project/data/...`), so it won't run as-is on another machine. Whenever convenient, it'd be worth switching it to the shared `load_data()` from `scripts/evaluation.py` (now that it exists) instead of its own inline loading — same result, just portable and consistent with the rest of the project.
