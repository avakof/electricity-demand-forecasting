# French electricity demand: reproducible next-day forecasting

Read and run the six notebooks in order. Every notebook contains executable analysis cells; shared, tested logic lives in `src/`. The original conversational notebook is preserved at `notebooks/legacy/01_data_audit_original.ipynb`.

**Evidence limits:** the provided CSVs have no trustworthy issuance/publication/revision archives or hourly-conversion lineage. Current model comparisons are explicitly retrospective. Recorded future weather is a separate, ineligible oracle sensitivity experiment. Earlier EDA already inspected the full history, so the chronological test is not a pristine holdout. These limitations cannot be repaired by a new split label.

| Notebook | Main outputs |
|---|---|
| `01_data_audit.ipynb` | Forecast contract, locked source identities, dictionary, availability register, audited dataset, source reconciliation, limitations and operational gate |
| `02_development_eda.ipynb` | Development-only calendar/holiday/temperature/delayed-temperature/structural-change investigation and hypothesis register |
| `03_backtest_and_baselines.ipynb` | Integrity tests, origin snapshots, four baseline policies, common-observation and availability scorecards |
| `04_model_comparison.ipynb` | Bounded ablations, chronological validation, rolling/expanding windows, retraining/capacity comparisons, input stress tests, frozen candidate, separate oracle sensitivity |
| `05_uncertainty_and_failures.ipynb` | Frozen test predictions, quantile/empirical intervals, trajectory scenarios, segment scores, paired multi-day bootstrap, failure catalogue |
| `06_business_and_deployment.ipynb` | Manager report, measured skill versus risk proxies, freshness/deployment assessment, monitoring and risk register |

## Run

Use Python 3.11 or newer. A clean project environment is recommended:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python scripts/execute_notebooks.py
```

`requirements-validated.txt` records the direct-package versions used for validation. The runner creates a **project-local** temporary kernel specification; it does not install a global Jupyter kernel. Notebooks also work interactively with a kernel containing the requirements. Keep their execution order. The long validation cell fits a bounded model comparison; it can take several minutes on a laptop.

To rerun a later notebook after its upstream artifacts exist:

```bash
python scripts/execute_notebooks.py --from-notebook 5 --to-notebook 6
```

`scripts/build_notebooks.py` is the authoring utility. Running it regenerates the six notebooks and clears their saved outputs. It never replaces the preserved legacy copy. The execution runner saves outputs and surfaces any failing cell.

## Forecast contract

- Origin: 11:00 Europe/Paris on D−1. Delivery: all intervals of the following local calendar day (23/24/25 hours).
- Primary metric: MAE in MW. Point model objective: conditional median, distinct from a mean forecast.
- Validation: July 2023–June 2025, reported as two chronological annual folds.
- Previously explored test: July 2025–June 2026. Fresh prospective evidence remains necessary.
- Assumed observation availability: interval end plus two hours. Finalized observations used historically are a documented vintage approximation.
- Scheduled online refitting and calibration may consume earlier evaluation-period observations **only after** their assumed availability. Feature selection, hyperparameters and policy choice use validation only.
- Missing evaluation targets remain missing. Peaks are retained. Incomplete days are counted and excluded only where whole-day metrics require complete trajectories.

Contract/source hashes are frozen by notebook 01 in `config/evaluation_lock.json`. A mismatch fails rather than silently replacing the experiment. To revise the research specification, preserve the old config, forecasts and reports under a separate run directory, explicitly remove/recreate the lock, and treat it as a new experiment. `config/frozen_candidate.json` is chosen using validation before test replay.

## Architecture

```text
config/       # Contract, source metadata, evaluation lock, frozen selection
data/
  *.csv       # Original provided CSVs, never rewritten
  raw/        # Source hashes and EMPTY archive schemas; no invented vintages
  processed/  # Audited grid and labels, preserving missingness
  origin_snapshots/ # Available-information feature store
notebooks/    # Six executable stages and preserved legacy notebook
src/          # Data, calendars, availability, features, replay, scoring, uncertainty
tests/        # Behavioral availability, leakage, DST and reproducibility checks
reports/
  audit/      # Dictionary, register, missingness, overlap, limitations
  forecasts/  # Predictions archived before evaluation joins
  scorecards/ # Raw/common-policy availability, fold and segment metrics
  models/     # Fit origins and latest training-label availability
  uncertainty/ # Quantile, calibration and whole-day trajectory reports
```

## Authentic archives and the operational gate

The three CSV files under `data/raw/` contain **headers only**. They are interface schemas, not sample forecasts. To support an authentic replay, supply original demand observations/revisions, RTE forecast runs and weather runs with UTC `valid_time`, `issued_at` (forecasts), `available_at`, `run_id` (forecasts), and numeric `value`. Observation archives also record `revision_status`. Publication time must not precede forecast issuance. Reject duplicate version keys. Prove interval/aggregation lineage and set the evidence flags only after verification in a new research run.

`select_forecast_run` selects one sufficiently fresh run available at the origin; missing intervals remain missing. `build_origin_features` supports authentic archives and as-of demand versions. `fit_at_origin` accepts an observation-vintage archive and uses the version available before fitting. No operational experiment runs on placeholder schemas or on provided D1 values disguised as issued forecasts.

Adding genuine archives requires checking origin coverage and seasonal coverage again; a shorter archive means a shorter authentic test, not permission to substitute realized weather. Run the gate and the same replay/scorecard functions on that archive-supported population in a separately versioned experiment. Do not combine its scores with the CSV proxy regime.

## Interpretation

Use `reports/final_recommendation.md` for the manager-facing assessment. Read paired block confidence intervals alongside point differences and segment counts. The bootstrap retains actual paired-loss sums/counts on partial days, matching headline hourly MAE eligibility; whole-day energy and peak metrics require complete days. Hourly quantile intervals are not simultaneous daily-trajectory intervals. Shortage/surplus energy proxies are not euro savings. Procurement evaluation needs prices, applicable settlement rules, instruments and a shared explicit decision policy. National consumption is not a validated utility customer portfolio.
