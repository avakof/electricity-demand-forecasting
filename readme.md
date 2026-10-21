# French electricity demand forecasting

Research comparison of day-ahead hourly national demand forecasts against the supplied RTE forecast. French national consumption represents the utility's demand proxy; this project does not implement production serving, scheduling, live monitoring or procurement integration.

The final recommendation answers three questions: improvement over RTE, uncertainty and failures, and the evidence supporting a research recommendation and its limitations.

## Architecture and code documentation

- [Interactive architecture overview](graphify-out/architecture.html): notebook stages, versioned experiments, predictor views, uncertainty methods and evidence boundaries.
- [Full relationship graph](graphify-out/graph.html) and [machine-readable graph](graphify-out/graph.json): source-level dependencies with retained relationship direction and source locations.
- [Code and notebook reference](graphify-out/CODE_DOCUMENTATION.md): current module APIs, notebook imports, artifact lineage and regeneration instructions.
- [Graph extraction audit](graphify-out/GRAPH_REPORT.md): coverage, relationship provenance, graph health and limitations.
- [Complete manager report](reports/manager_report_20261021/complete_report.md): results, supporting figures, uncertainty and the research recommendation.

The map distinguishes the original six-notebook investigation, the legacy baseline notebook, and the later research extensions in notebooks 07 and 08. The three versioned research protocols retain separate selections and results. Generated ledgers, fitted-model shards, plot exports and bundled report-app infrastructure are grouped to keep the architecture readable. The graph is a source/documentation map; it does not certify historical input availability or runtime behavior.

## Data

Provided sources: [project dataset folder](https://drive.google.com/drive/folders/1GY9IOY_7zdZpAm5xfbzSkFa-DunAA3Ag?usp=sharing).

- `data/france_electricity_hourly.csv`: national demand, RTE forecast and temperature, 2019–June 2026.
- `data/france_regional_electricity_hourly.csv`: twelve regional demand/temperature series, 2023–June 2026; no regional RTE forecasts.
- `data/raw/`: archive schemas. The supplied forecast-vintage archives currently contain no observations.

Keep raw observations unchanged. Preserve the hourly timestamp grid and report missing targets; do not interpolate evaluation truth or delete genuine demand peaks.

## Notebooks

| Notebook | Purpose |
| --- | --- |
| `01_data_audit.ipynb` | Data dictionary, availability, missingness, overlap and reconciliation |
| `02_development_eda.ipynb` | Development-period modeling questions and hypotheses |
| `02a_jg_baseline_model.ipynb` | Legacy baseline investigation; its evaluation protocol is not the new comparison protocol |
| `03_backtest_and_baselines.ipynb` | Existing retrospective origin-aware replay and baselines |
| `04_model_comparison.ipynb` | Existing compact candidate comparison |
| `05_uncertainty_and_failures.ipynb` | Existing interval calibration, matched uncertainty and failure analysis |
| `06_business_and_deployment.ipynb` | Research scorecard, business interpretation and limitations; filename retained for existing links |
| `07_statistical_and_boosting_models.ipynb` | Statistical/boosting experiments, RTE-assisted extensions, expanded causal features and cold-risk model selection |
| `08_deep_learning_and_uncertainty.ipynb` | Neural comparison, RTE-assisted evaluation, quantiles, calibration and cold-risk diagnostics |
| `09_jg_diagnostics.ipynb` | Residual (ACF/PACF, Ljung-Box), calibration/reliability, and feature-importance diagnostics, built from already-materialized research-pipeline artifacts |
| `10_jg_final_comparison.ipynb` | Independent recomputation of baseline and research-pipeline metrics against one trusted actuals series |

Reusable research logic belongs in `src/`; notebooks explain and orchestrate it. `scripts/build_notebooks.py` authors the earlier six-notebook workflow. `scripts/build_research_notebooks.py` authors the two new research notebooks; `scripts/execute_research_notebooks.py` executes them and preserves their outputs.

## Latest experiment: RTE features and cold-evening risk

The expanded study is saved separately in [reports/rte_risk/20261020T213640021260Z](reports/rte_risk/20261020T213640021260Z/). Its [frozen specification](config/rte_risk_experiment.json), [development selection](reports/rte_risk/20261020T213640021260Z/frozen_selection.json), and [read-only design audit](reports/rte_risk/20261020T213640021260Z/design_audit.md) define the interpretation of its results. Existing notebook 07 and 08 sections preserve the earlier studies; appended sections explain and execute this extension.

Use `rte_forecast_d1_mw` as an **assumed pre-cutoff input** at 11:00 `Europe/Paris` on D to forecast every actual hourly interval of D+1. Original RTE issue vintages remain unverified. No next-but-one-day forecast or its publication schedule is used. The origin is before local noon in both seasons; in summer it is 10:00 fixed CET, so it is also safely before a literal 12:00 CET deadline.

All three predictor views forecast the same **national consumption target**:

- **National:** national demand/weather/error histories, known calendar and RTE D1.
- **Regional-only:** regional demand/weather histories, known calendar and national RTE D1; national demand/weather/error histories are excluded.
- **Combined:** both historical predictor groups, known calendar and RTE D1.

Regional demand is not substituted for national evaluation truth. There are no regional RTE forecasts. Regional temperature aggregates are explicitly unweighted means, requiring complete regional composition.

Features include exact origin-relative lags, publication-gated delivery-relative lags, rolling means/dispersion/ranges/counts, exponentially weighted averages, changes and interactions, holiday/calendar indicators, released historical RTE errors, and the assumed-known delivery-day RTE profile. A delivery-minus-24-hour value is masked when unavailable at the origin. Moving averages of observations are distinguished from ARIMA's latent moving-average innovations. Future realized weather is excluded. Ridge demand predictions and cold/large-underprediction probabilities become extra features only through earlier monthly fits with released labels and training-only preprocessing.

The **55 new variants** comprise 30 standalone tabular variants across four families and three predictor views, **23 explicitly labeled hybrids**, and two hybrid controls. Each hybrid trains a LightGBM RTE-residual combiner using an original candidate's causal out-of-sample forecast plus the new predictors. All 13 original families are represented this way; their base architectures were not all retrained with native RTE inputs or new cold-weighted objectives. LightGBM feature, training-weight and direct-versus-residual alternatives test distinct changes. Two configurations per variant across four chronological folds give **440 configuration-fold comparisons**; probability comparisons are additional.

The expanded protocol differs from the historical studies below:

| Stage | Delivery dates and purpose |
| --- | --- |
| Base stacking burn-in | Q1 2024; later base configuration choices use released earlier-quarter outcomes |
| Development | Q2–Q4 2024 and 2 January–31 March 2025; 102 realized cold-evening hours across 41 days |
| Calibration | 2 April–29 June 2025; **zero realized cold-evening observations** |
| Retrospective evaluation | 1 July 2025–30 June 2026; already inspected in earlier studies |

New point models refit at delivery-quarter boundaries; learned feature generators follow their fixed monthly schedule. Reference policies retain their original fitting schedules. All training labels and observed-history features must be released before their respective cutoffs. Missing evaluation targets remain missing, and genuine peaks remain included.

Selection minimizes **normalized weighted MAE**, with weight five for realized cold evenings and one otherwise. Cold evenings are local hours 18:00–20:00 with realized temperature below 5 °C. Ordinary MAE must be no more than 1.02 times simple RTE bias correction, cold-evening MAE must not exceed that reference, and development support must reach 60 cold hours across 20 days. Training compares neutral weights with weights based on **previously predicted** cold probability; realized future cold labels are used for retrospective scoring, not prediction inputs. The point selection is `lab__lightgbm__combined__stacked__w1__residual`: its neutral training weights won the cold-weighted development selection. Extra training weights are tested, not assumed beneficial.

The quantile selection is `lab__lightgbm_quantile__combined__stacked__w5__residual`, chosen by **overall development WIS** among the three quantile-model views using point-selected parameters. It is not selected as the best cold-tail uncertainty model. Evaluation includes pinball loss, central and asymmetric 90% intervals, WIS/Winkler scores, empirical-distribution CRPS, coverage, widths and directional exceedances. Fixed engineered-model split-conformal/CQR comparators remain separate from adaptive models. Their retained historical training weights and the absence of cold calibration observations limit conditional-quantile and winter-coverage interpretations. No temporal or conditional coverage guarantee is established.

The [saved result summary](reports/rte_risk/20261020T213640021260Z/result_summary.json) and [recommendation](reports/rte_risk/20261020T213640021260Z/recommendation.md) report **98 policies on 8,759 matched nonmissing hours**, with one missing target excluded without interpolation. The point model was selected on development data before the new test replay; it was not chosen by sorting test scores.

| Policy | Overall MAE (MW) | Cold-weighted MAE (MW) | Cold-evening MAE (MW) |
| --- | ---: | ---: | ---: |
| Supplied RTE D1 | 1,438.98 | 1,467.91 | 2,427.77 |
| Rolling RTE bias correction | 1,111.05 | 1,132.51 | 1,844.66 |
| Prior development-selected RTE-assisted model | 806.76 | 830.32 | 1,611.93 |
| Expanded development-selected point model | **719.63** | **744.75** | **1,578.02** |

The expanded selection improves overall MAE by **49.99% versus RTE** and **10.80% versus the prior RTE-assisted selection**. Its cold-weighted improvement versus RTE is **49.26%**, with a paired 14-day-block 95% interval of **44.10%–54.28%**. Cold-evening MAE improves by **35.00% versus RTE**, but only **2.10% versus the prior selection**. That latter improvement is **33.92 MW**, with a paired 14-day-block interval of **−219.58 to +238.29 MW**: an incremental cold-evening gain over the prior model is not established. The critical segment contains **66 hours across 24 days and one represented DJF winter**. See the full [matched scorecard](reports/rte_risk/20261020T213640021260Z/tables/test_scorecard.csv) and [paired comparisons](reports/rte_risk/20261020T213640021260Z/tables/bootstrap.csv).

Cold-peak uncertainty remains inadequate. The selected point model's empirical 90% interval covers **43.94%** of cold-evening observations. The separately selected quantile model covers **53.03%** with its central 90% interval and **59.09%** with its asymmetric 90% interval. The separate fixed engineered-model CQR central 90% interval covers **63.64%**. These are different forecast/calibration policies, not interchangeable intervals for one point model. See the [cold-interval scorecard](reports/rte_risk/20261020T213640021260Z/tables/cold_interval_scorecard.csv). The point gains therefore support a retrospective research finding, while operational validation, adequate winter coverage, financial savings and general architecture superiority remain unestablished.

Verification is complete: **123 tests passed**, one optional test skipped; the artifact audit checked **1,927,344 forecast rows**, **80,306 daily-origin ledgers**, and **851 fit events** with zero execution failures. Independent review passed **505 numerical comparisons**. Both appended notebook sections executed successfully, and final figures and HTML previews were visually inspected. See [artifact verification](reports/rte_risk/20261020T213640021260Z/artifact_verification.json), [independent review](reports/rte_risk/20261020T213640021260Z/independent_result_review.md), and the [one-page recommendation](reports/rte_risk/20261020T213640021260Z/one_page_recommendation.md).

## Earlier RTE-assisted comparison

The [first RTE-assisted extension](reports/rte_assisted/20261020T203527821283Z/recommendation.md) selected `lightgbm_quantile__regional__rte_residual`. On 8,759 matched test observations, MAE was **806.76 MW**, compared with **1,438.98 MW** for supplied RTE and **1,111.05 MW** for rolling RTE bias correction: improvements of **43.94%** and **27.39%**, respectively. Its earlier “regional” arm used national plus regional histories; it was not a regional-only predictor view.

Its past-calibration empirical 90% interval covered only **51.52%** of the 66 cold-evening hours, with **45.45%** upper exceedances. The point improvement therefore did not establish adequate cold-peak uncertainty. This study used its own 2024 development and January–June 2025 calibration protocol; its interval results must not be conflated with the expanded study's shorter calibration period. See the [saved summary](reports/rte_assisted/20261020T203527821283Z/result_summary.json) and [scorecard](reports/rte_assisted/20261020T203527821283Z/tables/test_scorecard.csv).

## Original research_v2 comparison protocol

The new notebooks use a versioned specification and origin-aware research engine. Their results are saved under `reports/research_v2/<run_id>/`, separate from earlier `results/` and `reports/scorecards/` artifacts. Compare only artifacts sharing the same specification, data identity, eligible origins and delivery intervals.

- Origin: 11:00 `Europe/Paris` on day D, before forecasting every hourly interval of D+1. Local days contain 23, 24 or 25 intervals around DST transitions.
- Publication delay and interval-start labels are declared assumptions. Under the current assumptions the last eligible observed interval starts at 07:00 on D; a sequence model must bridge the rest of D before reaching D+1.
- Match national-only and regional-assisted histories from 2023. Evaluate 2019-onward national history separately as a history-length sensitivity.
- Tune on quarterly chronological folds within 2024, replaying complete daily origins. Use an inner chronological training tail for early stopping.
- Calibration delivery dates: 2 January–29 June 2025. June 30 is a label-availability boundary buffer.
- Locked historical comparison: 1 July 2025–30 June 2026. This period has already been inspected; results are retrospective, not an untouched confirmation test.
- Fit preprocessing on eligible training data only. Publication availability, complete origin groups and forecast horizons must be checked explicitly; a random split or a fixed row gap does not establish them.
- Unknown future demand and realized delivery-day temperature are excluded from ordinary predictors. Authentic archived forecast weather is required for a weather-assisted operational interpretation. The provided RTE column is a retrospective proxy because original issue versions are unverified.

The older `scripts/evaluation.py` and baseline JSON files use a different quarterly protocol. They remain historical artifacts, not authoritative inputs to the new comparison. In particular, dropping missing timestamp rows or using an unrestricted `lag_24` does not reproduce the new information cutoff.

## Original research_v2 models and uncertainty

Notebook 07 compares ARIMA, SARIMA, SARIMAX, histogram gradient boosting, LightGBM, quantile LightGBM and penalized GAM. Notebook 08 compares N-BEATS, NHiTS, TFT, GRU, PatchTST and DLinear. Weekly persistence, recent comparable-day demand, supplied RTE and past-error RTE correction remain references.

Split conformal prediction is a calibration method. CRPS is a distributional score, not a model. Strict split-conformal experiments keep fitted predictor parameters fixed after calibration begins; adaptive/refitted variants are labeled separately. Time-series dependence prevents an automatic exchangeability guarantee.

## Original research_v2 evaluation

Primary metric: pooled national **MAE in MW**. Also report RMSE, signed bias, WAPE, daily energy/peak errors, critical-segment counts, runtime, failures and forecast availability. RTE-relative improvement uses exactly matched origin–delivery observations. Paired day-block resampling quantifies uncertainty without treating hourly errors as independent.

Probabilistic evaluation includes pinball loss, coverage, interval width, upper/lower exceedances and interval score/WIS. Use CRPS only for an actual predictive distribution, samples or a declared quantile approximation. Separate marginal hourly intervals do not define daily energy or peak intervals.

## Original research_v2 executed comparison

The [versioned manager recommendation](reports/research_v2/20261020T164443776583Z/manager_recommendation.md) distinguishes the best complete policy from the best independent model, both selected on 2024 data. On 8,759 matched observations from July 2025–June 2026, the preselected RTE bias correction achieved MAE **1,111.0 MW**, versus **1,439.0 MW** for the RTE proxy: **22.79% improvement** (paired 14-day-block 95% interval **18.99%–27.80%**). The independently selected national GAM achieved **1,865.7 MW**. One missing evaluation target was excluded without interpolation.

The bias-corrected policy's empirical 90% interval covered **90.47% overall**, but only **69.7%** of **66 realized cold-evening hours across 24 days and one represented DJF winter**. Its upper bound was exceeded **27.3%** of the time in that segment, with **1,350 MW** mean severity when exceeded. This fails the cold-evening upper-tail diagnostic; the overall accuracy gain does not establish adequate peak-risk protection. Conditional peak intervals were not selected because the 2024 selection segment had only 33 proxy-cold observations over 11 days, below the declared support minimum.

All 13 model families completed the bounded search: 23 national/regional variants, 184 configuration-quarter validation combinations and four reference policies. Inspect the [point scorecard](reports/research_v2/20261020T164443776583Z/tables/test_scorecard.csv), [cold-evening scorecard](reports/research_v2/20261020T164443776583Z/tables/peak_risk_scorecard.csv), and [failure cases](reports/research_v2/20261020T164443776583Z/failure_cases.md). These remain retrospective results with assumed availability and unverified original forecast vintages; no operational superiority or euro savings is established.

## Independent comparison check

`10_jg_final_comparison.ipynb` recomputes MAE/RMSE/MAPE for the baseline and research-pipeline models using one trusted actuals series (`scripts/evaluation.py`'s `load_data`/`mae`/`rmse`/`mape`), rather than reading claimed scores from any report file. Inputs: `results/*.json` (baseline/RTE, 4-fold quarterly walk-forward), `reports/forecasts/all_test.csv.gz` (proxy models), and `reports/manager_report_20261021/diagnostics/matched_hourly.csv` (prior and selected research candidates), over the test year 2025-07-01 to 2026-06-30.

Realized demand in `matched_hourly.csv` matches `load_data()`'s actuals exactly for all 8,759 matched hours. The research pipeline's selected candidate recomputes to 719.63 MW MAE, matching `one_page_recommendation.md`/`complete_report.md` and contradicting the superseded `reports/final_recommendation.md` figure (832.05 MW, which independently matches an earlier candidate, not the selected one). The baseline/RTE numbers use 4-fold quarterly walk-forward means; every other score in this notebook is a single continuous-window score over the same test year — the two protocols are not identical, so the comparison is same-year/same-actuals/same-metrics but not same-validation-protocol. The notebook also reports cold-evening-only MAE (66 hours) alongside the overall figure for both the selected candidate and RTE.

## Scope and reproducibility

The earlier workflow's research recommendation is in `reports/final_recommendation.md`; each later study writes its own versioned outputs. Model winners must be determined from executed results, not assumed from architecture names. Original research_v2 and RTE-assisted artifacts remain historical comparisons with their own frozen protocols.

The expanded RTE-risk experiment uses `src/rte_feature_lab.py`, `src/rte_learned_features.py`, `src/rte_base_replay.py`, `src/rte_risk_lab.py`, and `src/rte_risk_uncertainty.py`. Completed sealed replays can be reused without fitting them again. With the scientific Python environment, resume its computation or verify completed artifacts explicitly:

```bash
python scripts/run_rte_risk_experiment.py --run-dir reports/rte_risk/20261020T213640021260Z --stage all
python scripts/check_rte_risk_artifacts.py --run-dir reports/rte_risk/20261020T213640021260Z
```

The checker requires final reports and executed notebook extension cells. `scripts/extend_rte_risk_notebooks.py` maintains those appended sections. Changed predictive sources, raw data or specifications require a new experiment; do not edit a frozen run to improve a reported test score.

Run the integrity checks with the project's scientific Python environment:

```bash
python -m unittest discover -s tests -v
```

The local Intel Mac has 8 GB RAM and no supported GPU. Check compatibility and measure a small fit before allocating neural tuning budgets. Record skipped or failed configurations explicitly rather than representing them as trained results. Do not mix fallback-only runs with successful model fits.

The original research_v2 compact search uses two configurations per family, four 2024 validation quarters, daily forecast origins and both data arms where supported. Statistical fits use recent 56/84-day windows; neural fits use explicit 20/40-step optimization caps. These are bounded experiments, not exhaustive optimization. Statistical convergence and neural learning curves are retained. The GAM is an explicit penalized additive B-spline regression using sklearn, rather than pyGAM.

Neural scores describe the executed configurations, not general architecture rankings. N-BEATS, NHiTS, TFT, PatchTST and DLinear use a fixed 24-hour training-window stride with no left padding. With the matched panel starting at local midnight, training forecast blocks begin at 23:00 UTC (00:00 local in winter, 01:00 in summer), whereas the replay forecast bridge begins at 08:00 local. GRU uses hourly recurrent windows and does not share this exact restriction. This training-origin mismatch, the small optimization budgets and the history-only neural inputs limit comparisons with tabular models using engineered calendar/history features. Poor validation scores do not by themselves prove overfitting or undertraining, or establish that a neural architecture is inferior. A future development experiment should use hourly training anchors or explicitly match the information cutoff with a DST policy, then reassess training convergence before drawing architecture conclusions.

Recorded wall times include host memory contention, suspended workers and a user-requested pause. They are not a clean hardware benchmark. The original run's [execution notes](reports/research_v2/20261020T164443776583Z/execution_notes.md) preserve the timing and neural-window limitations; predictor sources and recorded results were not changed to address them.

The compatible Intel neural stack is pinned to NeuralForecast 1.7.7 and PyTorch 2.2.2. Install its helper packages into a project-local directory without upgrading the scientific environment:

```bash
python -m pip install --target .research-deps --no-deps -r requirements-research-neural.txt
python scripts/execute_research_notebooks.py
python scripts/check_research_artifacts.py
```

Use the tested scientific interpreter and dependencies in `requirements-research.txt`. The original notebook 07 section creates a research_v2 run; notebook 08 resumes it. Later extension sections maintain separate run identities. Rebuilding whole notebooks clears recorded outputs, so do not regenerate them merely to view results. Main predictor changes or altered raw inputs require a fresh run. Fixed-predictor conformal comparators remain separate from refitted point models; marginal hourly intervals do not establish daily-trajectory coverage.

Financial savings require prices, applicable settlement rules and an explicit common decision policy. Demand forecast metrics alone establish neither euro savings nor customer-portfolio performance.
