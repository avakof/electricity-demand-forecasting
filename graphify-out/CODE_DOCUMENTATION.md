# Project architecture and code reference

Generated from current source definitions, notebook code cells and saved experiment evidence. This documentation does not run the forecasting code or certify historical issue-time availability.

Start with [the interactive architecture](architecture.html), [the full relationship graph](graph.html), [extraction audit](GRAPH_REPORT.md), and [verification record](verification.json).

## Research lanes and ownership

| Lane | Entry points | Implementation | Evidence |
|---|---|---|---|
| Original investigation | Notebooks 01–06; `scripts/build_notebooks.py` | `data`, `eda`, `features`, `backtest`, `metrics`, `uncertainty`, `research_risks` | Original contract, scorecards and recommendation |
| Legacy baseline branch | `02_jg_baseline_model.ipynb` | `scripts/evaluation.py` | Earlier protocol; not a required step in the current comparison |
| Broad research comparison | Notebooks 07/08; `ResearchRun` | `research_data`, `research_experiment`, `research_models`, `research_neural`, `research_scores` | `config/research_experiment.json`; versioned `reports/research_v2/` |
| RTE-assisted extension | Appended 07/08 sections; `RTEAssistedRun(ResearchRun)` | `rte_assisted`, `scripts/rte_assisted_reporting.py` | `config/rte_assisted_experiment.json`; `reports/rte_assisted/` |
| Expanded features and cold risk | Appended 07/08 sections; independent `RiskLabRun` | `rte_feature_lab`, `rte_learned_features`, `rte_base_replay`, `rte_risk_lab`, `rte_risk_uncertainty` | `config/rte_risk_experiment.json`; `reports/rte_risk/` |
| Manager report | Frozen results → diagnostics → figures/snapshot → report | `reports/manager_report_20261021/` scripts and authored report content | Complete written and interactive reports; no model selection or training |

## Model and information boundaries

- Every expanded predictor view forecasts national consumption. The regional-only view uses regional histories plus national RTE D1 and calendar; it has no regional RTE product.
- Origin is 11:00 Europe/Paris on D for all actual hourly intervals of D+1. Observation publication/revision delays and original RTE issue vintages remain assumptions. No next-but-one-day product is used.
- National/regional observed histories feed cutoff-gated lags, rolling statistics, calendar and RTE-profile features. Future actual weather is excluded. Complete regional composition is required for sums and unweighted temperature means.
- Earlier monthly fits create Ridge demand predictions, cold probabilities and probabilities of RTE underprediction above 1,000 MW. Learned features and training weights use released information.
- The selected point model is `lab__lightgbm__combined__stacked__w1__residual`: RTE plus a LightGBM correction with engineered and learned predictors. Its archived base-model field is null; it is not an ensemble of all original architectures.
- The 23 hybrids each add one causal archived base forecast to a LightGBM residual combiner. They do not establish native RTE-input retraining of every base architecture.
- The selected quantile policy is separate from the point selection. Split conformal and CQR calibrate intervals; CRPS/WIS/pinball/Winkler evaluate distributions or quantiles. They are not additional model architectures.
- Expanded point fits are quarterly; learned feature generators are monthly. Older reference policies retain their own schedules. The historical evaluation was previously inspected; cold calibration has zero qualifying hours.
- Deleted `src/pipeline.py` and monitoring-specification files are not current components. Scikit-learn preprocessing pipelines are unrelated to serving or procurement integration.

Evidence for these distinctions: [refresh audit](refresh_audit.md), [project README](../readme.md), [expanded specification](../config/rte_risk_experiment.json), [manager report](../reports/manager_report_20261021/complete_report.md).

## Notebook import and call map

Cell numbers below are one-based positions in the saved notebook, including Markdown cells. Calls are source observations, not proof that a cell ran.

### 01_data_audit.ipynb

[Open notebook](../notebooks/01_data_audit.ipynb)

Internal imports: [common](../src/common.py), [data](../src/data.py).

| Imported callable | Cell | Line within cell |
|---|---:|---:|
| `load_contract` | 2 | 11 |
| `prepare_directories` | 2 | 12 |
| `plotting_style` | 2 | 13 |

### 02_development_eda.ipynb

[Open notebook](../notebooks/02_development_eda.ipynb)

Internal imports: [common](../src/common.py), [data](../src/data.py), [eda](../src/eda.py).

| Imported callable | Cell | Line within cell |
|---|---:|---:|
| `load_contract` | 2 | 11 |
| `prepare_directories` | 2 | 12 |
| `plotting_style` | 2 | 13 |
| `read_audited` | 3 | 3 |
| `development_frame` | 3 | 4 |

### 02_jg_baseline_model.ipynb

[Open notebook](../notebooks/02_jg_baseline_model.ipynb)

Internal imports: [evaluation](../scripts/evaluation.py).


### 03_backtest_and_baselines.ipynb

[Open notebook](../notebooks/03_backtest_and_baselines.ipynb)

Internal imports: [backtest](../src/backtest.py), [common](../src/common.py), [data](../src/data.py), [features](../src/features.py), [metrics](../src/metrics.py).

| Imported callable | Cell | Line within cell |
|---|---:|---:|
| `load_contract` | 2 | 11 |
| `prepare_directories` | 2 | 12 |
| `plotting_style` | 2 | 13 |
| `read_audited` | 6 | 3 |
| `build_supervised` | 6 | 4 |
| `availability_gate` | 6 | 11 |
| `baseline_replay` | 8 | 5 |
| `scorecard` | 8 | 6 |
| `scorecard` | 8 | 7 |
| `matched_scorecard` | 8 | 8 |

### 04_model_comparison.ipynb

[Open notebook](../notebooks/04_model_comparison.ipynb)

Internal imports: [backtest](../src/backtest.py), [common](../src/common.py), [data](../src/data.py), [features](../src/features.py), [metrics](../src/metrics.py), [storage](../src/storage.py).

| Imported callable | Cell | Line within cell |
|---|---:|---:|
| `load_contract` | 2 | 11 |
| `prepare_directories` | 2 | 12 |
| `plotting_style` | 2 | 13 |
| `read_features` | 3 | 5 |
| `read_labels` | 3 | 6 |
| `experiments` | 3 | 7 |
| `prediction_fallback` | 9 | 11 |
| `read_audited` | 11 | 2 |

### 05_uncertainty_and_failures.ipynb

[Open notebook](../notebooks/05_uncertainty_and_failures.ipynb)

Internal imports: [backtest](../src/backtest.py), [common](../src/common.py), [metrics](../src/metrics.py), [storage](../src/storage.py), [uncertainty](../src/uncertainty.py).

| Imported callable | Cell | Line within cell |
|---|---:|---:|
| `load_contract` | 2 | 11 |
| `prepare_directories` | 2 | 12 |
| `plotting_style` | 2 | 13 |
| `read_features` | 3 | 5 |
| `read_labels` | 3 | 6 |
| `Experiment` | 3 | 9 |

### 06_business_and_deployment.ipynb

[Open notebook](../notebooks/06_business_and_deployment.ipynb)

Internal imports: [common](../src/common.py), [research_risks](../src/research_risks.py).

| Imported callable | Cell | Line within cell |
|---|---:|---:|
| `load_contract` | 2 | 11 |
| `prepare_directories` | 2 | 12 |
| `plotting_style` | 2 | 13 |

### 07_statistical_and_boosting_models.ipynb

[Open notebook](../notebooks/07_statistical_and_boosting_models.ipynb)

Internal imports: [common](../src/common.py), [research_data](../src/research_data.py), [research_experiment](../src/research_experiment.py), [research_models](../src/research_models.py), [rte_assisted](../src/rte_assisted.py), [rte_risk_lab](../src/rte_risk_lab.py).

| Imported callable | Cell | Line within cell |
|---|---:|---:|
| `plotting_style` | 2 | 18 |
| `data_summary` | 6 | 2 |
| `data_summary` | 6 | 7 |
| `parameter_grid` | 10 | 3 |

### 08_deep_learning_and_uncertainty.ipynb

[Open notebook](../notebooks/08_deep_learning_and_uncertainty.ipynb)

Internal imports: [development_fit_diagnostics](../scripts/development_fit_diagnostics.py), [freeze_research_policy](../scripts/freeze_research_policy.py), [research_comparison_plots](../scripts/research_comparison_plots.py), [research_failure_cases](../scripts/research_failure_cases.py), [rte_assisted_reporting](../scripts/rte_assisted_reporting.py), [rte_risk_reporting](../scripts/rte_risk_reporting.py), [write_manager_recommendation](../scripts/write_manager_recommendation.py), [common](../src/common.py), [metrics](../src/metrics.py), [research_experiment](../src/research_experiment.py), [research_extensions](../src/research_extensions.py), [research_neural](../src/research_neural.py), [research_peak_risk](../src/research_peak_risk.py), [research_scores](../src/research_scores.py), [rte_assisted](../src/rte_assisted.py), [rte_risk_lab](../src/rte_risk_lab.py), [rte_risk_uncertainty](../src/rte_risk_uncertainty.py).

| Imported callable | Cell | Line within cell |
|---|---:|---:|
| `plotting_style` | 2 | 19 |
| `backend_versions` | 3 | 6 |
| `write_json` | 3 | 9 |
| `forecast_origin` | 5 | 7 |
| `local_day_grid` | 5 | 9 |
| `scorecard` | 7 | 10 |
| `development_sensitivities` | 9 | 4 |
| `stress_inputs` | 9 | 6 |
| `freeze_peak_spec` | 9 | 3 |
| `build_diagnostics` | 11 | 26 |
| `freeze_policy` | 13 | 2 |
| `select_peak_intervals` | 13 | 6 |
| `strict_conformal` | 15 | 2 |
| `build_comparison` | 15 | 7 |
| `analyze_peak_risk` | 15 | 9 |
| `predeclared_plot_roles` | 15 | 14 |
| `plot_cold_evening` | 15 | 15 |
| `plot_overall_comparison` | 17 | 2 |
| `analyze_failures` | 17 | 11 |
| `write_manager_recommendation` | 19 | 5 |
| `score_extension` | 21 | 12 |
| `uncertainty_replay` | 27 | 9 |
| `report_rte_risk` | 27 | 10 |

## Python API index

Signatures and line numbers are extracted from definitions. An absent docstring remains absent; this index does not invent function behavior. Private helpers are listed because they are part of the research implementation.

### reports/manager_report_20261021/diagnostics/build_diagnostics.py

[Source](../reports/manager_report_20261021/diagnostics/build_diagnostics.py)

Descriptive failure diagnostics from frozen forecasts; no model fitting.

- [`sha(path)`](../reports/manager_report_20261021/diagnostics/build_diagnostics.py#L26) — function, line 26.
- [`write_json(name, obj)`](../reports/manager_report_20261021/diagnostics/build_diagnostics.py#L30) — function, line 30.
- [`metrics(frame, role)`](../reports/manager_report_20261021/diagnostics/build_diagnostics.py#L34) — function, line 34.
- [`grouped(frame, roles, column)`](../reports/manager_report_20261021/diagnostics/build_diagnostics.py#L46) — function, line 46.
- [`clean(value)`](../reports/manager_report_20261021/diagnostics/build_diagnostics.py#L52) — function, line 52.
- [`main()`](../reports/manager_report_20261021/diagnostics/build_diagnostics.py#L64) — function, line 64.

### reports/manager_report_20261021/prepare_manager_snapshot.py

[Source](../reports/manager_report_20261021/prepare_manager_snapshot.py)

Build the local report snapshot from frozen, reviewed research outputs only.

- [`query(name, frame, source_files, components, definition)`](../reports/manager_report_20261021/prepare_manager_snapshot.py#L15) — function, line 15.

### reports/manager_report_20261021/render_research_figures.py

[Source](../reports/manager_report_20261021/render_research_figures.py)

Export report figures from frozen research scores; never fits or selects models.

- [`read(name)`](../reports/manager_report_20261021/render_research_figures.py#L27) — function, line 27.
- [`save(fig, stem, note)`](../reports/manager_report_20261021/render_research_figures.py#L32) — function, line 32.

### scripts/build_notebooks.py

[Source](../scripts/build_notebooks.py)

Author the notebook workflow with nbformat. Never execute the legacy notebook.

- [`md(text)`](../scripts/build_notebooks.py#L16) — function, line 16.
- [`code(text)`](../scripts/build_notebooks.py#L17) — function, line 17.
- [`save(name, cells)`](../scripts/build_notebooks.py#L40) — function, line 40.

### scripts/build_research_notebooks.py

[Source](../scripts/build_research_notebooks.py)

Build exactly two research notebooks; does not execute experiments.

- [`markdown(text)`](../scripts/build_research_notebooks.py#L8) — function, line 8.
- [`code(text)`](../scripts/build_research_notebooks.py#L12) — function, line 12.
- [`save(name, cells)`](../scripts/build_research_notebooks.py#L38) — function, line 38.
- [`main()`](../scripts/build_research_notebooks.py#L46) — function, line 46.

### scripts/check_research_artifacts.py

[Source](../scripts/check_research_artifacts.py)

Read-only validation of executed research-v2 artifacts.

- [`ArtifactChecker`](../scripts/check_research_artifacts.py#L36) — class, line 36.
- [`main(argv=None)`](../scripts/check_research_artifacts.py#L501) — function, line 501.

Methods on `ArtifactChecker`:

- [`__init__(self, root: Path, run: Path, allow_incomplete=False)`](../scripts/check_research_artifacts.py#L37)
- [`require(self, condition, message)`](../scripts/check_research_artifacts.py#L47)
- [`unfinished(self, message)`](../scripts/check_research_artifacts.py#L52)
- [`read_json(self, path, required=True)`](../scripts/check_research_artifacts.py#L55)
- [`manifest(self)`](../scripts/check_research_artifacts.py#L66)
- [`check_frame(self, frame, name)`](../scripts/check_research_artifacts.py#L89)
- [`forecasts_and_ledgers(self)`](../scripts/check_research_artifacts.py#L119)
- [`fit_records(self)`](../scripts/check_research_artifacts.py#L196)
- [`frozen_selection(self)`](../scripts/check_research_artifacts.py#L223)
- [`notebooks(self)`](../scripts/check_research_artifacts.py#L261)
- [`scorecards(self)`](../scripts/check_research_artifacts.py#L285)
- [`development_policy(self)`](../scripts/check_research_artifacts.py#L335)
- [`peak_extension(self)`](../scripts/check_research_artifacts.py#L417)
- [`check(self)`](../scripts/check_research_artifacts.py#L487)

### scripts/check_rte_assisted_artifacts.py

[Source](../scripts/check_rte_assisted_artifacts.py)

Verify extension provenance and recompute headline losses independently.

- [`number(value)`](../scripts/check_rte_assisted_artifacts.py#L26) — function, line 26.
- [`utc(value)`](../scripts/check_rte_assisted_artifacts.py#L33) — function, line 33.
- [`verify(run)`](../scripts/check_rte_assisted_artifacts.py#L37) — function, line 37.

### scripts/check_rte_risk_artifacts.py

[Source](../scripts/check_rte_risk_artifacts.py)

Independently audit frozen RTE-risk artifacts; never fit or select a model.

- [`number(value)`](../scripts/check_rte_risk_artifacts.py#L28) — function, line 28.
- [`utc(value)`](../scripts/check_rte_risk_artifacts.py#L35) — function, line 35.
- [`equal_number(left, right, *, tolerance=1e-07)`](../scripts/check_rte_risk_artifacts.py#L42) — function, line 42.
- [`_require(condition, message)`](../scripts/check_rte_risk_artifacts.py#L47) — function, line 47.
- [`_raw_outcomes(root)`](../scripts/check_rte_risk_artifacts.py#L52) — function, line 52.
- [`_cold(delivery, temperature, spec)`](../scripts/check_rte_risk_artifacts.py#L62) — function, line 62.
- [`_provenance(run)`](../scripts/check_rte_risk_artifacts.py#L67) — function, line 67.
- [`_selection(run, frozen, outcomes)`](../scripts/check_rte_risk_artifacts.py#L121) — function, line 121.
- [`_sealed_forecasts(run, frozen)`](../scripts/check_rte_risk_artifacts.py#L177) — function, line 177.
- [`_point_arithmetic(run, outcomes)`](../scripts/check_rte_risk_artifacts.py#L246) — function, line 246.
- [`_interval_arithmetic(run, outcomes)`](../scripts/check_rte_risk_artifacts.py#L320) — function, line 320.
- [`_reporting_tables(run, frozen, outcomes, point_results)`](../scripts/check_rte_risk_artifacts.py#L370) — function, line 370.
  Cross-check comparison coverage, arithmetic and calibration boundaries.
- [`_notebooks(root)`](../scripts/check_rte_risk_artifacts.py#L452) — function, line 452.
- [`verify(run)`](../scripts/check_rte_risk_artifacts.py#L466) — function, line 466.
- [`main()`](../scripts/check_rte_risk_artifacts.py#L483) — function, line 483.

### scripts/development_fit_diagnostics.py

[Source](../scripts/development_fit_diagnostics.py)

Reporting-only diagnostics from saved development forecasts and fit records.

- [`_labels(root: Path, start: pd.Timestamp, end: pd.Timestamp)`](../scripts/development_fit_diagnostics.py#L23) — function, line 23.
- [`_functional(metadata: dict, family: str)`](../scripts/development_fit_diagnostics.py#L46) — function, line 46.
- [`_metrics(frame: pd.DataFrame)`](../scripts/development_fit_diagnostics.py#L62) — function, line 62.
- [`_plot_gaps(gaps: pd.DataFrame, path: Path)`](../scripts/development_fit_diagnostics.py#L76) — function, line 76.
- [`_plot_quarters(summary: pd.DataFrame, path: Path)`](../scripts/development_fit_diagnostics.py#L115) — function, line 115.
- [`build_diagnostics(run_dir: str | Path, root: str | Path)`](../scripts/development_fit_diagnostics.py#L137) — function, line 137.
  Compare saved SAME-FIT tabular train MAE with subsequent raw OOS MAE.
- [`main()`](../scripts/development_fit_diagnostics.py#L277) — function, line 277.

### scripts/evaluation.py

[Source](../scripts/evaluation.py)

Shared evaluation module for the electricity demand forecasting project.

- [`load_data(data_dir='../data')`](../scripts/evaluation.py#L31) — function, line 31.
  Load the hourly France electricity CSV, convert to Europe/Paris local time.
- [`mae(y, y_hat)`](../scripts/evaluation.py#L49) — function, line 49.
- [`rmse(y, y_hat)`](../scripts/evaluation.py#L54) — function, line 54.
- [`mape(y, y_hat)`](../scripts/evaluation.py#L59) — function, line 59.
- [`score(y, y_hat)`](../scripts/evaluation.py#L64) — function, line 64.
  Align two series, drop unpaired/NaN rows, return MAE/RMSE/MAPE dict.
- [`walk_forward_folds(df, n_folds=4, window='expanding', window_size=None)`](../scripts/evaluation.py#L74) — function, line 74.
  Yield (fold_idx, train, test) over the 4 fixed quarterly folds.
- [`save_result(name, description, validation, fold_results, results_dir='../results')`](../scripts/evaluation.py#L110) — function, line 110.
  Save per-fold metrics plus their mean/std to results/<name>.json.

### scripts/execute_notebooks.py

[Source](../scripts/execute_notebooks.py)

Run the six notebooks in order with this interpreter; no global kernel install.

- [`main()`](../scripts/execute_notebooks.py#L15) — function, line 15.

### scripts/execute_research_notebooks.py

[Source](../scripts/execute_research_notebooks.py)

Execute the two research notebooks with the current Python, saving diagnostics.

- [`main()`](../scripts/execute_research_notebooks.py#L15) — function, line 15.

### scripts/extend_rte_assisted_notebooks.py

[Source](../scripts/extend_rte_assisted_notebooks.py)

Append and execute the RTE extension without clearing original notebook outputs.

- [`md(source)`](../scripts/extend_rte_assisted_notebooks.py#L20) — function, line 20.
- [`code(source)`](../scripts/extend_rte_assisted_notebooks.py#L24) — function, line 24.
- [`cells07()`](../scripts/extend_rte_assisted_notebooks.py#L29) — function, line 29.
- [`cells08()`](../scripts/extend_rte_assisted_notebooks.py#L68) — function, line 68.
- [`prepare()`](../scripts/extend_rte_assisted_notebooks.py#L113) — function, line 113.
- [`execute(number)`](../scripts/extend_rte_assisted_notebooks.py#L141) — function, line 141.

### scripts/extend_rte_risk_notebooks.py

[Source](../scripts/extend_rte_risk_notebooks.py)

Append independently runnable feature/risk sections to the two research notebooks.

- [`md(text)`](../scripts/extend_rte_risk_notebooks.py#L20) — function, line 20.
- [`code(text)`](../scripts/extend_rte_risk_notebooks.py#L24) — function, line 24.
- [`cells07()`](../scripts/extend_rte_risk_notebooks.py#L29) — function, line 29.
- [`cells08()`](../scripts/extend_rte_risk_notebooks.py#L87) — function, line 87.
- [`prepare()`](../scripts/extend_rte_risk_notebooks.py#L155) — function, line 155.
- [`execute(number)`](../scripts/extend_rte_risk_notebooks.py#L173) — function, line 173.

### scripts/finalize_research_presentation.py

[Source](../scripts/finalize_research_presentation.py)

Finalize reporting after notebook 08 passes; never execute model code.

- [`template_08_cells(path: Path)`](../scripts/finalize_research_presentation.py#L33) — function, line 33.
  Evaluate only the literal cell-construction list for save(notebook08).
- [`assert_executed(notebook, name)`](../scripts/finalize_research_presentation.py#L78) — function, line 78.
- [`_protected_snapshot(notebook)`](../scripts/finalize_research_presentation.py#L93) — function, line 93.
- [`mirror_markdown(notebook, expected)`](../scripts/finalize_research_presentation.py#L99) — function, line 99.
  Return changed indices, while asserting executable content is identical.
- [`execution_limitations(notes: str)`](../scripts/finalize_research_presentation.py#L119) — function, line 119.
  Extract the existing interpretation and timing caveats, without new claims.
- [`complete_policy_distinction(policy, scores=None)`](../scripts/finalize_research_presentation.py#L134) — function, line 134.
  Describe two pretest choices; never select anything using test losses.
- [`_atomic_text(path, text)`](../scripts/finalize_research_presentation.py#L164) — function, line 164.
- [`main(argv=None)`](../scripts/finalize_research_presentation.py#L170) — function, line 170.

### scripts/freeze_research_policy.py

[Source](../scripts/freeze_research_policy.py)

Freeze the best complete policy using 2024 results, including baselines.

- [`freeze_policy(run: Path)`](../scripts/freeze_research_policy.py#L19) — function, line 19.

### scripts/precompute_research_final.py

[Source](../scripts/precompute_research_final.py)

Replay disjoint frozen model groups in bounded CPU worker processes.

- [`main()`](../scripts/precompute_research_final.py#L26) — function, line 26.

### scripts/refresh_rte_assisted_figures.py

[Source](../scripts/refresh_rte_assisted_figures.py)

Refresh one presentation cell from archived results; never refit models.


### scripts/research_comparison_plots.py

[Source](../scripts/research_comparison_plots.py)

Reporting-only plots with roles chosen from pre-test selection records.

- [`_name(model: str)`](../scripts/research_comparison_plots.py#L18) — function, line 18.
- [`predeclared_plot_roles(run_dir: str | Path)`](../scripts/research_comparison_plots.py#L24) — function, line 24.
  Read selection seals only; never select using a comparison-period score.
- [`_number(value)`](../scripts/research_comparison_plots.py#L51) — function, line 51.
- [`_count(value)`](../scripts/research_comparison_plots.py#L59) — function, line 59.
- [`_figure(size)`](../scripts/research_comparison_plots.py#L64) — function, line 64.
- [`_point(frame, spec, coverage_column)`](../scripts/research_comparison_plots.py#L72) — function, line 72.
- [`_background(axis, frame, y_column)`](../scripts/research_comparison_plots.py#L86) — function, line 86.
- [`_annotate(axis, row, spec, y_column, position, extra='')`](../scripts/research_comparison_plots.py#L99) — function, line 99.
- [`plot_cold_evening(scorecard: pd.DataFrame, roles: list[dict], output: str | Path)`](../scripts/research_comparison_plots.py#L112) — function, line 112.
  Highlight frozen role/method pairs and available 14-day block intervals.
- [`plot_overall_comparison(point_scores: pd.DataFrame, peak_scorecard: pd.DataFrame, roles: list[dict], output: str | Path)`](../scripts/research_comparison_plots.py#L156) — function, line 156.
  Tall point leaderboard and overall coverage using the SAME frozen roles.

### scripts/research_failure_cases.py

[Source](../scripts/research_failure_cases.py)

Reporting-only failure cases for the development-frozen complete policy.

- [`_grid(day, timezone)`](../scripts/research_failure_cases.py#L23) — function, line 23.
- [`_load_forecasts(path)`](../scripts/research_failure_cases.py#L29) — function, line 29.
- [`_measured_days(frame, country, timezone, evening_hours, cold_threshold, holidays)`](../scripts/research_failure_cases.py#L36) — function, line 36.
  Measure errors and contemporaneous observations; never infer causes.
- [`_select_cases(daily)`](../scripts/research_failure_cases.py#L97) — function, line 97.
  Select distinct diagnostic days; explicitly retain rank when criteria overlap.
- [`_plot_case(frame, case, destination, policy, timezone, evening_hours, cold_threshold)`](../scripts/research_failure_cases.py#L138) — function, line 138.
- [`analyze_failures(run_dir, root=ROOT)`](../scripts/research_failure_cases.py#L191) — function, line 191.
  Create measured tables/figures after final scores, without changing policy.
- [`main(argv=None)`](../scripts/research_failure_cases.py#L270) — function, line 270.

### scripts/rte_assisted_reporting.py

[Source](../scripts/rte_assisted_reporting.py)

Matched retrospective scores for the separately frozen RTE-assisted study.

- [`read_forecasts(path)`](../scripts/rte_assisted_reporting.py#L26) — function, line 26.
- [`conformal_comparator(run)`](../scripts/rte_assisted_reporting.py#L34) — function, line 34.
- [`development_scores(run)`](../scripts/rte_assisted_reporting.py#L59) — function, line 59.
- [`score_extension(run, outputs=None)`](../scripts/rte_assisted_reporting.py#L96) — function, line 96.
- [`draw_figures(run, scores, uncertainty, cold)`](../scripts/rte_assisted_reporting.py#L189) — function, line 189.
- [`write_report(run, scores, uncertainty, cold, bootstrap)`](../scripts/rte_assisted_reporting.py#L248) — function, line 248.

### scripts/rte_risk_reporting.py

[Source](../scripts/rte_risk_reporting.py)

Matched scores and risk diagnostics for the frozen RTE feature laboratory.

- [`threshold_weighted_empirical_crps(actual, prediction, residuals, threshold)`](../scripts/rte_risk_reporting.py#L30) — function, line 30.
  Exact upper-tail twCRPS for an equally weighted empirical residual CDF.
- [`read_forecasts(path)`](../scripts/rte_risk_reporting.py#L58) — function, line 58.
- [`_previous_development_selection(run)`](../scripts/rte_risk_reporting.py#L66) — function, line 66.
- [`_decorate(frame, labels, spec)`](../scripts/rte_risk_reporting.py#L71) — function, line 71.
- [`_point(frame, prediction='prediction_mw')`](../scripts/rte_risk_reporting.py#L91) — function, line 91.
- [`_pinball(y, q, tau)`](../scripts/rte_risk_reporting.py#L108) — function, line 108.
- [`_interval_scores(frame, native_quantiles=True)`](../scripts/rte_risk_reporting.py#L113) — function, line 113.
  Directional interval errors, properly labeled scores, and support counts.
- [`_probability_rows(frame, spec, model, method)`](../scripts/rte_risk_reporting.py#L168) — function, line 168.
- [`_bootstrap(frame, candidate, reference, spec)`](../scripts/rte_risk_reporting.py#L180) — function, line 180.
- [`_empirical_probabilities(calibration, test, labels, spec, roles)`](../scripts/rte_risk_reporting.py#L218) — function, line 218.
- [`_risk_scenarios(frame, spec)`](../scripts/rte_risk_reporting.py#L244) — function, line 244.
- [`_development_fit_diagnostics(run, frozen)`](../scripts/rte_risk_reporting.py#L259) — function, line 259.
  Compare the same fitted model's sampled train error with its later OOS error.
- [`_learned_feature_diagnostics(run)`](../scripts/rte_risk_reporting.py#L301) — function, line 301.
  Score causal first-stage features only; never select or recalibrate them.
- [`_quantile_decision_scenarios(empirical, native, labels, spec)`](../scripts/rte_risk_reporting.py#L365) — function, line 365.
  Same predeclared cost-ratio quantile policy for every compared distribution.
- [`_ablation_and_hybrid_tables(run, scores, frozen)`](../scripts/rte_risk_reporting.py#L396) — function, line 396.
- [`_plots(run, scores, probability, frozen)`](../scripts/rte_risk_reporting.py#L439) — function, line 439.
- [`_recommendation(run, scores, probability, bootstrap, frozen)`](../scripts/rte_risk_reporting.py#L582) — function, line 582.
- [`report(run)`](../scripts/rte_risk_reporting.py#L732) — function, line 732.
  Read frozen predictions, score every policy, and write review artifacts.
- [`main()`](../scripts/rte_risk_reporting.py#L845) — function, line 845.

### scripts/run_rte_risk_experiment.py

[Source](../scripts/run_rte_risk_experiment.py)

Sequential, resumable research execution; each sealed replay is reusable.

- [`state(run, stage, status='running')`](../scripts/run_rte_risk_experiment.py#L18) — function, line 18.
- [`main()`](../scripts/run_rte_risk_experiment.py#L25) — function, line 25.

### scripts/validate_artifacts.py

[Source](../scripts/validate_artifacts.py)

Validate executed notebooks/artifact invariants and render portable previews.

- [`main()`](../scripts/validate_artifacts.py#L17) — function, line 17.

### scripts/write_manager_recommendation.py

[Source](../scripts/write_manager_recommendation.py)

Write a <=550-word manager summary AFTER notebook 08 has completed.

- [`_json(path: Path)`](../scripts/write_manager_recommendation.py#L23) — function, line 23.
- [`_csv(path: Path, columns: set[str])`](../scripts/write_manager_recommendation.py#L29) — function, line 29.
- [`_one(frame: pd.DataFrame, **keys)`](../scripts/write_manager_recommendation.py#L39) — function, line 39.
- [`_pct(value)`](../scripts/write_manager_recommendation.py#L48) — function, line 48.
- [`_mw(value)`](../scripts/write_manager_recommendation.py#L53) — function, line 53.
- [`_rate_ci(row)`](../scripts/write_manager_recommendation.py#L60) — function, line 60.
- [`_selected_peak_method(selection, model)`](../scripts/write_manager_recommendation.py#L65) — function, line 65.
- [`_diagnostics(run_dir)`](../scripts/write_manager_recommendation.py#L74) — function, line 74.
- [`_stress_note(run_dir)`](../scripts/write_manager_recommendation.py#L89) — function, line 89.
- [`write_manager_recommendation(run_dir: Path)`](../scripts/write_manager_recommendation.py#L107) — function, line 107.
- [`main()`](../scripts/write_manager_recommendation.py#L219) — function, line 219.

### src/__init__.py

[Source](../src/__init__.py)

Auditable next-day electricity forecasting components.


### src/backtest.py

[Source](../src/backtest.py)

- [`Experiment`](../src/backtest.py#L23) — class, line 23.
- [`experiments()`](../src/backtest.py#L36) — function, line 36.
- [`feature_columns(experiment)`](../src/backtest.py#L51) — function, line 51.
- [`new_model(experiment, seed=42, quantile=None)`](../src/backtest.py#L56) — function, line 56.
- [`fit_at_origin(features, labels, origin, experiment, contract, quantile=None, demand_archive=None)`](../src/backtest.py#L64) — function, line 64.
- [`archive_predictions(root, name, predictions, contract, origin_ledger=None)`](../src/backtest.py#L87) — function, line 87.
- [`baseline_replay(features, start, end, root, contract, phase='validation')`](../src/backtest.py#L114) — function, line 114.
- [`model_replay(features, labels, start, end, experiment, root, contract, phase='validation', quantiles=False, demand_archive=None)`](../src/backtest.py#L133) — function, line 133.
- [`select_candidate(scores, segments, registry, contract)`](../src/backtest.py#L195) — function, line 195.
- [`stress_inputs(features, scenario)`](../src/backtest.py#L213) — function, line 213.

### src/common.py

[Source](../src/common.py)

- [`project_root(start=None)`](../src/common.py#L13) — function, line 13.
- [`load_contract(root=None)`](../src/common.py#L21) — function, line 21.
- [`fingerprint(path)`](../src/common.py#L26) — function, line 26.
- [`write_json(path, value)`](../src/common.py#L30) — function, line 30.
- [`prepare_directories(root)`](../src/common.py#L36) — function, line 36.
- [`local_day_grid(day)`](../src/common.py#L41) — function, line 41.
- [`forecast_origin(day, contract)`](../src/common.py#L50) — function, line 50.
- [`holiday_dates(year)`](../src/common.py#L58) — function, line 58.
  Metropolitan national public holidays; excludes regional and school holidays.
- [`calendar_features(index)`](../src/common.py#L79) — function, line 79.
- [`plotting_style()`](../src/common.py#L100) — function, line 100.

### src/data.py

[Source](../src/data.py)

- [`load_inputs(root)`](../src/data.py#L12) — function, line 12.
- [`availability_gate(root, contract=None)`](../src/data.py#L29) — function, line 29.
- [`field_register(country, regional)`](../src/data.py#L50) — function, line 50.
- [`run_audit(root, contract=None)`](../src/data.py#L60) — function, line 60.
- [`read_audited(root)`](../src/data.py#L124) — function, line 124.
- [`load_archive(path, forecast=True)`](../src/data.py#L131) — function, line 131.
- [`select_forecast_run(archive, origin, deliveries, max_age_hours)`](../src/data.py#L150) — function, line 150.
  Select one latest accessible, sufficiently fresh run for the entire day.

### src/eda.py

[Source](../src/eda.py)

- [`development_frame(data, contract)`](../src/eda.py#L9) — function, line 9.
- [`temperature_summary(sample, width=2)`](../src/eda.py#L16) — function, line 16.
- [`hypothesis_register()`](../src/eda.py#L32) — function, line 32.

### src/features.py

[Source](../src/features.py)

- [`asof_history(data, origin, contract, demand_archive=None)`](../src/features.py#L17) — function, line 17.
  Only observed intervals ending before cutoff, published before cutoff.
- [`build_origin_features(data, day, contract, rte_mode='none', rte_archive=None, weather_archive=None, demand_archive=None, oracle_weather=False)`](../src/features.py#L31) — function, line 31.
- [`build_supervised(data, contract, rte_mode='proxy', **kwargs)`](../src/features.py#L96) — function, line 96.
- [`prediction_fallback(features, raw, rte_assisted=False)`](../src/features.py#L117) — function, line 117.

### src/metrics.py

[Source](../src/metrics.py)

- [`point_metrics(frame, prediction='prediction_mw')`](../src/metrics.py#L9) — function, line 9.
- [`scorecard(predictions, labels, prediction='prediction_mw')`](../src/metrics.py#L16) — function, line 16.
- [`matched_scorecard(predictions, labels, prediction='prediction_mw')`](../src/metrics.py#L21) — function, line 21.
- [`segment_scorecard(predictions, labels, contract)`](../src/metrics.py#L28) — function, line 28.
- [`daily_metrics(predictions, labels)`](../src/metrics.py#L46) — function, line 46.
- [`paired_block_bootstrap(predictions, labels, candidate, reference='rte_proxy', block_days=(7, 14, 28), repetitions=500, seed=42)`](../src/metrics.py#L61) — function, line 61.
  Paired moving calendar-day blocks; never bridge missing-day gaps.

### src/research_data.py

[Source](../src/research_data.py)

Research-v2 data views. UTC grid preserved; availability precedes every origin.

- [`load_research_contract(root)`](../src/research_data.py#L23) — function, line 23.
- [`latest_eligible_time(origin, contract)`](../src/research_data.py#L27) — function, line 27.
- [`calendar_exog(index)`](../src/research_data.py#L32) — function, line 32.
  Known calendar only, including the unobserved bridge for SARIMAX.
- [`ResearchData`](../src/research_data.py#L44) — class, line 44.
- [`prepare_research_data(root, history_start=None, use_cache=True)`](../src/research_data.py#L100) — function, line 100.
- [`data_summary(data)`](../src/research_data.py#L153) — function, line 153.

Methods on `ResearchData`:

- [`region_columns(self)`](../src/research_data.py#L54)
- [`columns(self, arm='national', rte_assisted=False)`](../src/research_data.py#L57)
- [`training_rows(self, origin, window_days=None)`](../src/research_data.py#L65)
- [`history(self, origin, window_days=None)`](../src/research_data.py#L79)
- [`panel(self, origin, arm='national', window_days=None)`](../src/research_data.py#L86)
- [`phase(self, start, end)`](../src/research_data.py#L96)

### src/research_experiment.py

[Source](../src/research_experiment.py)

Daily forecast-origin replay, chronological selection and research artifacts.

- [`_json(value)`](../src/research_experiment.py#L35) — function, line 35.
- [`_read_predictions(path)`](../src/research_experiment.py#L39) — function, line 39.
- [`ResearchRun`](../src/research_experiment.py#L47) — class, line 47.

Methods on `ResearchRun`:

- [`__init__(self, root, run_dir, profile='compact')`](../src/research_experiment.py#L48)
- [`create(cls, root, profile='compact')`](../src/research_experiment.py#L59)
- [`latest(cls, root)`](../src/research_experiment.py#L68)
- [`data(self)`](../src/research_experiment.py#L73)
- [`write_manifest(self)`](../src/research_experiment.py#L78)
- [`_event(self, path, value)`](../src/research_experiment.py#L92)
- [`_record_trial(self, record)`](../src/research_experiment.py#L96)
- [`_record_selection(self, key, entry)`](../src/research_experiment.py#L108)
- [`verify_manifest(self)`](../src/research_experiment.py#L119)
- [`replay(self, family, params, start, end, *, arm='national', phase='validation', tag=None, seed=42, fixed=False, quantiles=False, full_history=False, rte_correction=False)`](../src/research_experiment.py#L127)
- [`tune(self, families, arms=('national', 'regional'))`](../src/research_experiment.py#L292)
- [`baselines(self, start, end, phase)`](../src/research_experiment.py#L339)
- [`freeze(self)`](../src/research_experiment.py#L342)
- [`final_replay(self)`](../src/research_experiment.py#L373)

### src/research_extensions.py

[Source](../src/research_extensions.py)

Pre-test development sensitivities and input stress tests; research only.

- [`development_sensitivities(run)`](../src/research_extensions.py#L14) — function, line 14.
- [`stress_inputs(run)`](../src/research_extensions.py#L55) — function, line 55.
  Freeze one development fit, perturb only predictor inputs, count fallback.

### src/research_models.py

[Source](../src/research_models.py)

Small, auditable model adapters for the research notebooks.

- [`parameter_grid(family: str, profile: str='fast')`](../src/research_models.py#L59) — function, line 59.
  Return bounded, deterministic candidates; never use test-set information.
- [`_numeric_frame(X: Any, columns: list | None=None)`](../src/research_models.py#L100) — function, line 100.
- [`TabularAdapter`](../src/research_models.py#L122) — class, line 122.
- [`make_tabular(family: str, params: dict | None=None, quantile: float | None=None, seed: int=42)`](../src/research_models.py#L195) — function, line 195.
- [`_hourly_series(series: pd.Series)`](../src/research_models.py#L207) — function, line 207.
- [`StatisticalAdapter`](../src/research_models.py#L223) — class, line 223.
- [`make_statistical(family: str, params: dict | None=None)`](../src/research_models.py#L351) — function, line 351.

Methods on `TabularAdapter`:

- [`fit(self, X: Any, y: Any)`](../src/research_models.py#L128)
- [`predict(self, X: Any)`](../src/research_models.py#L188)

Methods on `StatisticalAdapter`:

- [`_exog(self, exog: pd.DataFrame | None, index: pd.DatetimeIndex, fit: bool=False)`](../src/research_models.py#L227)
- [`_conserve_memory(model)`](../src/research_models.py#L252)
- [`fit(self, series: pd.Series, exog: pd.DataFrame | None=None)`](../src/research_models.py#L260)
- [`forecast(self, steps: int, exog: pd.DataFrame | None=None)`](../src/research_models.py#L311)
- [`update(self, series: pd.Series, exog: pd.DataFrame | None=None)`](../src/research_models.py#L322)

### src/research_neural.py

[Source](../src/research_neural.py)

Official NeuralForecast adapters for the research benchmark.

- [`_backend()`](../src/research_neural.py#L43) — function, line 43.
  Load project-local dependencies without modifying the scientific venv.
- [`backend_versions()`](../src/research_neural.py#L71) — function, line 71.
  Return dependency provenance, raising rather than faking availability.
- [`neural_configurations(family: str)`](../src/research_neural.py#L81) — function, line 81.
  Two predeclared, compact CPU configurations; validation selects the winner.
- [`_panel(frame: pd.DataFrame)`](../src/research_neural.py#L100) — function, line 100.
- [`NeuralAdapter`](../src/research_neural.py#L140) — class, line 140.
  Real target-history-only forecasting model, with auditable fit metadata.

Methods on `NeuralAdapter`:

- [`__init__(self, family: str, params: dict[str, Any] | None=None, seed: int=42)`](../src/research_neural.py#L143)
- [`_model(self)`](../src/research_neural.py#L156)
- [`fit(self, panel: pd.DataFrame, validation_size: int=168)`](../src/research_neural.py#L193)
- [`predict(self, context: pd.DataFrame, future: pd.DataFrame | None=None)`](../src/research_neural.py#L229)
- [`close(self)`](../src/research_neural.py#L257)

### src/research_peak_risk.py

[Source](../src/research_peak_risk.py)

Predeclared peak-risk analysis; all monetary-looking objectives are MW proxies.

- [`_write_json(path, payload)`](../src/research_peak_risk.py#L43) — function, line 43.
- [`_test_exists(run)`](../src/research_peak_risk.py#L47) — function, line 47.
- [`freeze_peak_spec(run)`](../src/research_peak_risk.py#L54) — function, line 54.
  Freeze this extension's code and definitions without reading test outcomes.
- [`_read(path)`](../src/research_peak_risk.py#L71) — function, line 71.
- [`_decorate(frame, run)`](../src/research_peak_risk.py#L79) — function, line 79.
- [`_choose_pool(pool, cold_proxy, evening, spec=SPEC)`](../src/research_peak_risk.py#L95) — function, line 95.
  Select only by origin proxy/hour. Pool requires residual_mw, local_day, evening, precutoff_coldrisk_evening, and optionally winter_id columns.
- [`_intervals(target, pool, conditional)`](../src/research_peak_risk.py#L113) — function, line 113.
- [`_summary(frame)`](../src/research_peak_risk.py#L126) — function, line 126.
- [`_upper_bootstrap(full, segment, width)`](../src/research_peak_risk.py#L152) — function, line 152.
  Resample contiguous calendar-day blocks, retaining days with zero segment count.
- [`_validation_forecasts(run, key, entry)`](../src/research_peak_risk.py#L175) — function, line 175.
- [`select_peak_intervals(run)`](../src/research_peak_risk.py#L194) — function, line 194.
  Save the validation-only interval choice BEFORE the final replay.
- [`_cost_sensitivity(distributions)`](../src/research_peak_risk.py#L244) — function, line 244.
- [`analyze_peak_risk(run, outputs)`](../src/research_peak_risk.py#L274) — function, line 274.
  Apply the already selected interval policy; final outcomes score only.

### src/research_risks.py

[Source](../src/research_risks.py)

Limitations of the retrospective research evidence, without deployment logic.

- [`risk_register()`](../src/research_risks.py#L7) — function, line 7.
  Describe evidence limitations and the research needed to resolve them.

### src/research_scores.py

[Source](../src/research_scores.py)

Matched point scores, honest empirical distributions and frozen conformal.

- [`empirical_crps(errors, residuals)`](../src/research_scores.py#L18) — function, line 18.
  Exact CRPS of prediction + the equally weighted empirical residual CDF.
- [`conformal_radius(scores, alpha=0.1)`](../src/research_scores.py#L33) — function, line 33.
- [`interval_summary(frame)`](../src/research_scores.py#L44) — function, line 44.
- [`strict_conformal(run, selected)`](../src/research_scores.py#L60) — function, line 60.
  One continuous replay guarantees the SAME fitted predictor in both blocks.
- [`uncertainty_comparison(run, calibration, test, conformal)`](../src/research_scores.py#L94) — function, line 94.
- [`build_comparison(run, outputs, conformal)`](../src/research_scores.py#L138) — function, line 138.
- [`write_recommendation(run, scores, uncertainty, bootstrap, fits)`](../src/research_scores.py#L182) — function, line 182.

### src/rte_assisted.py

[Source](../src/rte_assisted.py)

Chronological RTE-assisted extension; preserves the original frozen experiment.

- [`model_key(family, arm, mode)`](../src/rte_assisted.py#L28) — function, line 28.
- [`training_target(training, mode)`](../src/rte_assisted.py#L34) — function, line 34.
- [`demand_prediction(prediction, day, mode)`](../src/rte_assisted.py#L42) — function, line 42.
- [`RTEAssistedRun`](../src/rte_assisted.py#L51) — class, line 51.

Methods on `RTEAssistedRun`:

- [`__init__(self, root, run_dir)`](../src/rte_assisted.py#L52)
- [`create(cls, root)`](../src/rte_assisted.py#L58)
- [`latest(cls, root)`](../src/rte_assisted.py#L67)
- [`write_manifest(self)`](../src/rte_assisted.py#L76)
- [`verify_manifest(self)`](../src/rte_assisted.py#L98)
- [`replay_variant(self, family, params, start, end, *, arm, mode, phase='validation', fixed=False, quantiles=False)`](../src/rte_assisted.py#L106)
- [`tune_assisted(self)`](../src/rte_assisted.py#L221)
- [`freeze_assisted(self)`](../src/rte_assisted.py#L255)
- [`replay_final_assisted(self)`](../src/rte_assisted.py#L285)
- [`fixed_conformal_replay(self)`](../src/rte_assisted.py#L300)

### src/rte_base_replay.py

[Source](../src/rte_base_replay.py)

Causal, already-out-of-sample base forecasts for RTE feature stacking.

- [`_sha256(path)`](../src/rte_base_replay.py#L29) — function, line 29.
- [`_utc(values)`](../src/rte_base_replay.py#L37) — function, line 37.
- [`_labels(frame)`](../src/rte_base_replay.py#L41) — function, line 41.
- [`select_released_trial(trial_predictions, labels, cutoff, minimum_availability=RAW_AVAILABILITY_MINIMUM)`](../src/rte_base_replay.py#L51) — function, line 51.
  Select from externally supplied OOS trials using strictly released labels.
- [`causal_quarter_stream(quarter_trials, labels)`](../src/rte_base_replay.py#L85) — function, line 85.
  Return one variant's 2024 OOS stream and its causal selection ledger.
- [`_check_oos(frame)`](../src/rte_base_replay.py#L123) — function, line 123.
- [`_verify_reference(root, reference)`](../src/rte_base_replay.py#L143) — function, line 143.
- [`_read_sealed(root, reference, model, phase, provenance)`](../src/rte_base_replay.py#L158) — function, line 158.
- [`load_causal_base_bank(root, labels)`](../src/rte_base_replay.py#L188) — function, line 188.
  Return (wide origin-delivery base features, selection ledger).

### src/rte_feature_lab.py

[Source](../src/rte_feature_lab.py)

Causal feature laboratory for the RTE D1-assisted retrospective experiment.

- [`_availability(index, contract)`](../src/rte_feature_lab.py#L33) — function, line 33.
- [`_eligible_national(national, origin, contract)`](../src/rte_feature_lab.py#L38) — function, line 38.
- [`_series_features(series, prefix, origin, deliveries, contract, *, full=True)`](../src/rte_feature_lab.py#L47) — function, line 47.
  All origin lags refer to an exact UTC hour, never the last nonmissing row.
- [`_calendar_additions(deliveries)`](../src/rte_feature_lab.py#L90) — function, line 90.
- [`build_lab_origin_features(national, regional_load, regional_temperature, day, contract)`](../src/rte_feature_lab.py#L115) — function, line 115.
  Build one complete 23/24/25-hour target day with a fixed pre-noon origin.
- [`FeatureLabData`](../src/rte_feature_lab.py#L193) — class, line 193.
  ResearchData-compatible view; country outcome remains identical in all arms.
- [`feature_register(data)`](../src/rte_feature_lab.py#L241) — function, line 241.
  Machine-readable availability and transformation register for every predictor.
- [`prepare_feature_lab(root, history_start=None, use_cache=True)`](../src/rte_feature_lab.py#L287) — function, line 287.
  Build and hash-cache expanded features without modifying original artifacts.

Methods on `FeatureLabData`:

- [`columns(self, arm='national', rte_assisted=True, tier='engineered')`](../src/rte_feature_lab.py#L196)
- [`feature_register(self)`](../src/rte_feature_lab.py#L237)

### src/rte_learned_features.py

[Source](../src/rte_learned_features.py)

Fixed, chronological first-stage predictions for the RTE feature laboratory.

- [`learned_columns(arm)`](../src/rte_learned_features.py#L48) — function, line 48.
- [`_utc(series, name)`](../src/rte_learned_features.py#L56) — function, line 56.
- [`_numeric(frame)`](../src/rte_learned_features.py#L62) — function, line 62.
- [`_pipeline(estimator)`](../src/rte_learned_features.py#L66) — function, line 66.
- [`_column_map(data, arms)`](../src/rte_learned_features.py#L74) — function, line 74.
- [`_frame_hash(frame)`](../src/rte_learned_features.py#L95) — function, line 95.
- [`_cache_inputs(data, features, labels, column_map)`](../src/rte_learned_features.py#L102) — function, line 102.
- [`_fit_component(training, prediction, columns, target, released_at, origin, *, classification)`](../src/rte_learned_features.py#L122) — function, line 122.
  Return honest predictions and auditable metadata, including one-class fits.
- [`build_learned_feature_bank(data, cache_dir=None, arms=ARMS)`](../src/rte_learned_features.py#L168) — function, line 168.
  Return origin/delivery keys plus three chronological predictors per arm.
- [`predicted_cold_evening_weights(features, bank, arm, multiplier=3.0)`](../src/rte_learned_features.py#L269) — function, line 269.
  Point-loss training weights from original-origin cold predictions only.

### src/rte_risk_lab.py

[Source](../src/rte_risk_lab.py)

Frozen, causal RTE feature/risk experiment. Research only; no deployment.

- [`read_frame(path)`](../src/rte_risk_lab.py#L29) — function, line 29.
- [`cold_mask(frame)`](../src/rte_risk_lab.py#L37) — function, line 37.
- [`point_scores(predictions, labels)`](../src/rte_risk_lab.py#L42) — function, line 42.
- [`candidate_variants(base_names)`](../src/rte_risk_lab.py#L59) — function, line 59.
- [`configurations(family)`](../src/rte_risk_lab.py#L84) — function, line 84.
- [`WeightedAdapter`](../src/rte_risk_lab.py#L96) — class, line 96.
- [`RiskLabRun`](../src/rte_risk_lab.py#L133) — class, line 133.

Methods on `WeightedAdapter`:

- [`__init__(self, family, params, q=None)`](../src/rte_risk_lab.py#L97)
- [`fit(self, X, y, weights)`](../src/rte_risk_lab.py#L100)
- [`predict(self, X)`](../src/rte_risk_lab.py#L128)

Methods on `RiskLabRun`:

- [`__init__(self, root, path)`](../src/rte_risk_lab.py#L134)
- [`create(cls, root)`](../src/rte_risk_lab.py#L144)
- [`latest(cls, root)`](../src/rte_risk_lab.py#L153)
- [`write_manifest(self)`](../src/rte_risk_lab.py#L156)
- [`verify_manifest(self)`](../src/rte_risk_lab.py#L168)
- [`data(self)`](../src/rte_risk_lab.py#L178)
- [`frame(self)`](../src/rte_risk_lab.py#L184)
- [`columns(self, variant)`](../src/rte_risk_lab.py#L200)
- [`replay(self, variant, params, start, end, phase, q=None, fixed=False)`](../src/rte_risk_lab.py#L212)
- [`tune(self)`](../src/rte_risk_lab.py#L309)
- [`freeze(self)`](../src/rte_risk_lab.py#L336)
- [`evaluate(self)`](../src/rte_risk_lab.py#L387)

### src/rte_risk_uncertainty.py

[Source](../src/rte_risk_uncertainty.py)

Quantile grid and fixed-predictor calibration for the RTE risk study.

- [`pinball(y, prediction, q)`](../src/rte_risk_uncertainty.py#L14) — function, line 14.
- [`interval_score(y, lo, hi, alpha)`](../src/rte_risk_uncertainty.py#L19) — function, line 19.
- [`quantile_scores(predictions, labels)`](../src/rte_risk_uncertainty.py#L24) — function, line 24.
- [`assemble_quantiles(run, entry, start, end, phase, fixed=False)`](../src/rte_risk_uncertainty.py#L35) — function, line 35.
- [`signed_quantile(scores, alpha)`](../src/rte_risk_uncertainty.py#L56) — function, line 56.
- [`uncertainty_replay(run)`](../src/rte_risk_uncertainty.py#L63) — function, line 63.

### src/storage.py

[Source](../src/storage.py)

- [`read_features(path)`](../src/storage.py#L4) — function, line 4.
- [`read_labels(path)`](../src/storage.py#L12) — function, line 12.
- [`read_predictions(path)`](../src/storage.py#L19) — function, line 19.

### src/uncertainty.py

[Source](../src/uncertainty.py)

- [`empirical_intervals(predictions, historical_predictions, labels, contract)`](../src/uncertainty.py#L9) — function, line 9.
  Rolling signed out-of-sample errors available strictly before each origin.
- [`interval_metrics(frame)`](../src/uncertainty.py#L31) — function, line 31.
- [`interval_scorecard(predictions, labels, contract)`](../src/uncertainty.py#L42) — function, line 42.
- [`trajectory_intervals(predictions, historical_predictions, labels, contract, scenarios=300)`](../src/uncertainty.py#L57) — function, line 57.
  Resample complete daily residual vectors; match 23/24/25-hour length.

### tests/__init__.py

[Source](../tests/__init__.py)


### tests/test_development_fit_diagnostics.py

[Source](../tests/test_development_fit_diagnostics.py)

Check same-fit aggregation and the reporting script's development-only gate.

- [`DevelopmentFitDiagnosticsTests`](../tests/test_development_fit_diagnostics.py#L13) — class, line 13.

Methods on `DevelopmentFitDiagnosticsTests`:

- [`fixture(self, root)`](../tests/test_development_fit_diagnostics.py#L14)
- [`test_same_fit_raw_gap_counts_and_development_only_read(self)`](../tests/test_development_fit_diagnostics.py#L59)
- [`test_absent_unscaled_training_score_is_explicitly_unavailable(self)`](../tests/test_development_fit_diagnostics.py#L82)
- [`test_out_of_fold_prediction_cache_is_rejected(self)`](../tests/test_development_fit_diagnostics.py#L93)

### tests/test_integrity.py

[Source](../tests/test_integrity.py)

Behavioral checks: availability, DST, leakage, replay and calibration.

- [`ForecastIntegrityTests`](../tests/test_integrity.py#L16) — class, line 16.

Methods on `ForecastIntegrityTests`:

- [`setUpClass(cls)`](../tests/test_integrity.py#L18)
- [`test_dst_delivery_counts(self)`](../tests/test_integrity.py#L23)
- [`test_origin_remains_pre_noon_across_dst(self)`](../tests/test_integrity.py#L29)
- [`test_future_observations_cannot_change_features_or_baseline(self)`](../tests/test_integrity.py#L34)
- [`test_delayed_publication_cannot_change_features(self)`](../tests/test_integrity.py#L45)
- [`test_archive_uses_only_accessible_issue_and_publication(self)`](../tests/test_integrity.py#L52)
- [`test_stale_and_partial_archive(self)`](../tests/test_integrity.py#L61)
- [`test_no_silent_target_interpolation(self)`](../tests/test_integrity.py#L70)
- [`test_missing_source_interval_does_not_remove_delivery_interval(self)`](../tests/test_integrity.py#L76)
- [`test_fallback_is_a_complete_policy(self)`](../tests/test_integrity.py#L84)
- [`test_repeated_source_hour_policy(self)`](../tests/test_integrity.py#L89)
- [`test_fallback_for_nonexistent_source_hour(self)`](../tests/test_integrity.py#L97)
- [`test_quantile_calibration_does_not_use_future_labels(self)`](../tests/test_integrity.py#L104)
- [`test_archive_rejects_duplicate_versions(self)`](../tests/test_integrity.py#L115)
- [`test_calendar_holidays(self)`](../tests/test_integrity.py#L122)
- [`test_trained_forecast_and_preprocessor_ignore_future_data(self)`](../tests/test_integrity.py#L125)
- [`test_paired_bootstrap_preserves_matched_loss_ratio(self)`](../tests/test_integrity.py#L144)
- [`test_replay_is_reproducible(self)`](../tests/test_integrity.py#L162)
- [`test_grouped_origins_do_not_cross_validation_boundary(self)`](../tests/test_integrity.py#L167)

### tests/test_research_integrity.py

[Source](../tests/test_research_integrity.py)

Behavioral checks for research-v2 chronology, panel views and feature causality.

- [`ResearchIntegrityTests`](../tests/test_research_integrity.py#L19) — class, line 19.

Methods on `ResearchIntegrityTests`:

- [`setUpClass(cls)`](../tests/test_research_integrity.py#L21)
- [`synthetic_data(self)`](../tests/test_research_integrity.py#L24)
- [`test_latest_eligible_time_uses_strict_publication_boundary(self)`](../tests/test_research_integrity.py#L53)
- [`test_sequence_bridge_covers_complete_dst_delivery_days(self)`](../tests/test_research_integrity.py#L62)
- [`test_known_calendar_covers_bridge_without_outcomes(self)`](../tests/test_research_integrity.py#L78)
- [`test_calibration_starts_after_last_validation_label_release(self)`](../tests/test_research_integrity.py#L87)
- [`test_calibration_end_buffer_precedes_first_test_origin(self)`](../tests/test_research_integrity.py#L94)
- [`test_training_rows_exclude_equal_future_and_missing_labels(self)`](../tests/test_research_integrity.py#L104)
- [`test_future_label_values_cannot_change_training_rows(self)`](../tests/test_research_integrity.py#L111)
- [`test_selected_features_exclude_targets_and_future_weather(self)`](../tests/test_research_integrity.py#L118)
- [`test_panel_retains_missing_values_and_common_utc_cutoff(self)`](../tests/test_research_integrity.py#L128)
- [`test_prepare_features_ignore_unreleased_national_and_regional_values(self)`](../tests/test_research_integrity.py#L143)

### tests/test_research_models.py

[Source](../tests/test_research_models.py)

Behavioral checks for fixed-parameter updates and training-only transforms.

- [`ResearchModelTests`](../tests/test_research_models.py#L10) — class, line 10.

Methods on `ResearchModelTests`:

- [`setUp(self)`](../tests/test_research_models.py#L11)
- [`test_additive_transform_is_fitted_only_on_training_data(self)`](../tests/test_research_models.py#L18)
- [`test_missing_labels_are_not_silently_imputed(self)`](../tests/test_research_models.py#L28)
- [`test_statistical_update_keeps_parameters_and_hourly_grid(self)`](../tests/test_research_models.py#L34)
- [`test_exogenous_future_must_cover_complete_bridge(self)`](../tests/test_research_models.py#L47)
- [`test_random_early_stopping_is_rejected(self)`](../tests/test_research_models.py#L55)
- [`test_quantile_models_have_explicit_level(self)`](../tests/test_research_models.py#L59)

### tests/test_research_neural.py

[Source](../tests/test_research_neural.py)

Neural-panel causality checks; optional real-backend reproducibility test.

- [`NeuralPanelTests`](../tests/test_research_neural.py#L20) — class, line 20.
- [`NeuralBackendIntegrationTests`](../tests/test_research_neural.py#L125) — class, line 125.

Methods on `NeuralPanelTests`:

- [`example(self)`](../tests/test_research_neural.py#L21)
- [`test_missing_targets_keep_grid_and_zero_loss_mask(self)`](../tests/test_research_neural.py#L28)
- [`test_predictor_fill_is_causal_with_leading_zero_only(self)`](../tests/test_research_neural.py#L37)
- [`test_fill_never_crosses_series_boundary(self)`](../tests/test_research_neural.py#L48)
- [`test_explicitly_masked_values_do_not_enter_predictor_fill(self)`](../tests/test_research_neural.py#L52)
- [`test_unsorted_panel_mask_stays_attached_to_its_timestamp(self)`](../tests/test_research_neural.py#L60)
- [`test_timestamps_preserve_utc_in_naive_nanosecond_format(self)`](../tests/test_research_neural.py#L67)
- [`test_missing_timestamp_row_is_rejected_instead_of_compressed(self)`](../tests/test_research_neural.py#L74)
- [`test_duplicate_timestamp_key_is_rejected(self)`](../tests/test_research_neural.py#L78)
- [`test_all_missing_series_is_rejected(self)`](../tests/test_research_neural.py#L83)
- [`test_invalid_loss_mask_is_rejected(self)`](../tests/test_research_neural.py#L89)
- [`test_unreleased_future_cannot_change_cutoff_panel_or_input_fill(self)`](../tests/test_research_neural.py#L96)
- [`test_48_hour_output_grid_covers_both_dst_delivery_days(self)`](../tests/test_research_neural.py#L112)

Methods on `NeuralBackendIntegrationTests`:

- [`test_real_dlinear_reproducibility_masked_values_and_cpu_policy(self)`](../tests/test_research_neural.py#L126)

### tests/test_research_peak_risk.py

[Source](../tests/test_research_peak_risk.py)

Behavioral checks for the frozen cold-evening uncertainty extension.

- [`PeakRiskTests`](../tests/test_research_peak_risk.py#L20) — class, line 20.

Methods on `PeakRiskTests`:

- [`setUp(self)`](../tests/test_research_peak_risk.py#L21)
- [`pool(self, days=50, cold_days=0)`](../tests/test_research_peak_risk.py#L24)
- [`test_sparse_cold_pool_explicitly_falls_back_to_evening(self)`](../tests/test_research_peak_risk.py#L34)
- [`test_cold_pool_requires_both_observations_and_distinct_days(self)`](../tests/test_research_peak_risk.py#L43)
- [`test_evening_pool_falls_back_overall_and_insufficient_is_explicit(self)`](../tests/test_research_peak_risk.py#L55)
- [`test_peak_summary_counts_and_upper_exceedance_severity(self)`](../tests/test_research_peak_risk.py#L64)
- [`test_realized_weather_changes_diagnostics_but_not_origin_proxy(self)`](../tests/test_research_peak_risk.py#L81)
- [`test_cost_ratios_are_labeled_proxies_and_policy_is_shared(self)`](../tests/test_research_peak_risk.py#L104)
- [`final_fixture(self, path)`](../tests/test_research_peak_risk.py#L128)
- [`test_final_calibration_is_strict_and_ignores_june30_and_test_outcomes(self)`](../tests/test_research_peak_risk.py#L166)
- [`test_cannot_first_freeze_or_select_after_test_replay_starts(self)`](../tests/test_research_peak_risk.py#L193)
- [`test_freeze_is_reusable_without_touching_outcomes_and_rejects_drift(self)`](../tests/test_research_peak_risk.py#L208)

### tests/test_research_run.py

[Source](../tests/test_research_run.py)

Cross-process selection persistence and immutable-input replay-cache checks.

- [`_selection_writer(root, run_path, family, start)`](../tests/test_research_run.py#L18) — function, line 18.
  Top-level target supports macOS spawn rather than relying on fork state.
- [`_PhaseData`](../tests/test_research_run.py#L35) — class, line 35.
- [`ResearchRunTests`](../tests/test_research_run.py#L47) — class, line 47.

Methods on `_PhaseData`:

- [`__init__(self, features)`](../tests/test_research_run.py#L36)
- [`phase(self, start, end)`](../tests/test_research_run.py#L39)
- [`columns(self, *args)`](../tests/test_research_run.py#L43)

Methods on `ResearchRunTests`:

- [`fixture(self, directory)`](../tests/test_research_run.py#L48)
- [`cache_fixture(self, directory)`](../tests/test_research_run.py#L65)
- [`test_disjoint_process_writes_preserve_all_trials_and_selections(self)`](../tests/test_research_run.py#L78)
- [`test_record_trial_replaces_one_key_without_losing_other_trials(self)`](../tests/test_research_run.py#L120)
- [`test_latest_pointer_and_manifest_are_consistent(self)`](../tests/test_research_run.py#L132)
- [`test_manifest_rejects_changed_raw_source(self)`](../tests/test_research_run.py#L144)
- [`test_manifest_rejects_changed_feature_code_and_contract(self)`](../tests/test_research_run.py#L151)
- [`test_valid_cache_skips_replay_but_corrupt_ledger_forces_replay(self)`](../tests/test_research_run.py#L160)
- [`test_corrupt_cached_forecast_forces_replay(self)`](../tests/test_research_run.py#L178)
- [`test_cache_cannot_bypass_manifest_change(self)`](../tests/test_research_run.py#L190)

### tests/test_research_scores.py

[Source](../tests/test_research_scores.py)

Independent numeric and cutoff checks for research uncertainty methods.

- [`ResearchScoreTests`](../tests/test_research_scores.py#L19) — class, line 19.

Methods on `ResearchScoreTests`:

- [`test_empirical_crps_matches_brute_pairwise_definition(self)`](../tests/test_research_scores.py#L20)
- [`test_empirical_crps_degenerate_and_empty_distributions(self)`](../tests/test_research_scores.py#L28)
- [`test_conformal_finite_sample_ceil_not_interpolated_quantile(self)`](../tests/test_research_scores.py#L33)
- [`test_conformal_reports_unbounded_for_insufficient_scores(self)`](../tests/test_research_scores.py#L39)
- [`test_conformal_rejects_invalid_probability_or_negative_abs_scores(self)`](../tests/test_research_scores.py#L44)
- [`test_interval_score_and_wis_known_values(self)`](../tests/test_research_scores.py#L51)
- [`test_unbounded_interval_score_is_infinite_not_nan(self)`](../tests/test_research_scores.py#L65)
- [`test_strict_conformal_single_fit_and_strict_label_availability(self)`](../tests/test_research_scores.py#L74)
- [`test_strict_conformal_rejects_parameter_refitting(self)`](../tests/test_research_scores.py#L125)
- [`test_strict_conformal_rejects_fallback_only_predictor(self)`](../tests/test_research_scores.py#L136)

### tests/test_rte_assisted.py

[Source](../tests/test_rte_assisted.py)

RTE-feature/residual distinction, cutoff rules, fallbacks and replay caches.

- [`ConstantEstimator`](../tests/test_rte_assisted.py#L18) — class, line 18.
- [`SyntheticData`](../tests/test_rte_assisted.py#L28) — class, line 28.
- [`AssistedTests`](../tests/test_rte_assisted.py#L53) — class, line 53.

Methods on `ConstantEstimator`:

- [`fit(self, X, y)`](../tests/test_rte_assisted.py#L19)
- [`predict(self, X)`](../tests/test_rte_assisted.py#L24)

Methods on `SyntheticData`:

- [`__init__(self, day)`](../tests/test_rte_assisted.py#L29)
- [`phase(self, start, end)`](../tests/test_rte_assisted.py#L37)
- [`columns(self, arm, rte_assisted=False)`](../tests/test_rte_assisted.py#L40)
- [`training_rows(self, origin)`](../tests/test_rte_assisted.py#L44)

Methods on `AssistedTests`:

- [`run_fixture(self, directory, day='2025-03-30')`](../tests/test_rte_assisted.py#L54)
- [`test_direct_and_residual_targets_are_distinct(self)`](../tests/test_rte_assisted.py#L61)
- [`test_residual_prediction_adds_rte_exactly_once(self)`](../tests/test_rte_assisted.py#L68)
- [`test_rte_features_ignore_unreleased_actual_demand_and_weather(self)`](../tests/test_rte_assisted.py#L74)
- [`test_both_modes_keep_dst_grid_and_equal_demand_prediction(self, factory)`](../tests/test_rte_assisted.py#L91)
- [`test_missing_rte_uses_declared_history_fallback_in_both_modes(self, factory)`](../tests/test_rte_assisted.py#L108)
- [`test_training_label_after_cutoff_fails_fit_and_records_rte_fallback(self, factory)`](../tests/test_rte_assisted.py#L120)
- [`test_sealed_cache_reused_but_modified_fit_artifact_rejected(self, factory)`](../tests/test_rte_assisted.py#L137)
- [`test_first_freeze_after_test_is_rejected(self)`](../tests/test_rte_assisted.py#L149)

### tests/test_rte_base_replay.py

[Source](../tests/test_rte_base_replay.py)

Selection timing and out-of-sample safeguards for cached base stacking.

- [`forecasts(deliveries, predictions, origins=None)`](../tests/test_rte_base_replay.py#L10) — function, line 10.
- [`CausalBaseReplayTests`](../tests/test_rte_base_replay.py#L18) — class, line 18.

Methods on `CausalBaseReplayTests`:

- [`test_quarter_boundary_excludes_late_march_labels_and_exact_cutoff(self)`](../tests/test_rte_base_replay.py#L19)
- [`test_future_outcome_changes_leave_earlier_selection_unchanged(self)`](../tests/test_rte_base_replay.py#L31)
- [`test_raw_availability_guard_uses_only_nonmissing_targets(self)`](../tests/test_rte_base_replay.py#L41)
- [`test_burnin_fixed_and_later_selection_is_causal(self)`](../tests/test_rte_base_replay.py#L54)
- [`test_rejects_outcome_columns_and_post_origin_fitting(self)`](../tests/test_rte_base_replay.py#L65)
- [`test_ties_choose_trial_zero(self)`](../tests/test_rte_base_replay.py#L76)

### tests/test_rte_feature_lab.py

[Source](../tests/test_rte_feature_lab.py)

Cutoff, data-arm and DST invariants of the expanded RTE-assisted features.

- [`FeatureLabTests`](../tests/test_rte_feature_lab.py#L13) — class, line 13.

Methods on `FeatureLabTests`:

- [`setUpClass(cls)`](../tests/test_rte_feature_lab.py#L15)
- [`fixture(self, day='2024-02-15')`](../tests/test_rte_feature_lab.py#L18)
- [`build(self, day, national, load, temp)`](../tests/test_rte_feature_lab.py#L30)
- [`view(self, national, load, temp, features)`](../tests/test_rte_feature_lab.py#L33)
- [`test_postcutoff_actuals_and_temperatures_cannot_change_features(self)`](../tests/test_rte_feature_lab.py#L36)
- [`test_future_other_delivery_day_rte_does_not_change_current_features(self)`](../tests/test_rte_feature_lab.py#L48)
- [`test_dst_days_keep_all_unique_delivery_intervals(self)`](../tests/test_rte_feature_lab.py#L60)
- [`test_regional_only_has_no_country_history_or_errors(self)`](../tests/test_rte_feature_lab.py#L70)
- [`test_publication_at_origin_is_excluded_and_lag24_masked(self)`](../tests/test_rte_feature_lab.py#L87)
- [`test_explicit_late_national_publication_is_also_excluded(self)`](../tests/test_rte_feature_lab.py#L101)
- [`test_incomplete_regional_aggregate_and_rte_day_not_silently_averaged(self)`](../tests/test_rte_feature_lab.py#L112)

### tests/test_rte_learned_features.py

[Source](../tests/test_rte_learned_features.py)

Temporal stacking must keep original-origin feature and outcome cutoffs.

- [`synthetic_data(days=240)`](../tests/test_rte_learned_features.py#L14) — function, line 14.
- [`LearnedFeatureTests`](../tests/test_rte_learned_features.py#L34) — class, line 34.

Methods on `LearnedFeatureTests`:

- [`test_future_labels_do_not_change_earlier_oos_features(self)`](../tests/test_rte_learned_features.py#L35)
- [`test_fit_ledger_proves_label_and_feature_release_cutoffs_and_arm_scope(self)`](../tests/test_rte_learned_features.py#L49)
- [`test_one_class_probability_is_explicit_training_frequency(self)`](../tests/test_rte_learned_features.py#L61)
- [`test_explicit_late_temperature_release_blocks_cold_classifier(self)`](../tests/test_rte_learned_features.py#L73)
- [`test_original_row_history_cutoff_violation_is_rejected(self)`](../tests/test_rte_learned_features.py#L80)
- [`test_label_release_before_observed_interval_end_is_rejected(self)`](../tests/test_rte_learned_features.py#L86)
- [`test_regional_only_rejects_national_history_predictor(self)`](../tests/test_rte_learned_features.py#L92)
- [`test_cache_reuse_and_tamper_detection(self)`](../tests/test_rte_learned_features.py#L98)
- [`test_cold_weights_use_forecast_probability_and_keep_missing_neutral(self)`](../tests/test_rte_learned_features.py#L109)

### tests/test_rte_risk_lab.py

[Source](../tests/test_rte_risk_lab.py)

Small numerical and causal checks for the RTE risk experiment.

- [`MeanAdapter`](../tests/test_rte_risk_lab.py#L18) — class, line 18.
- [`variant(mode='residual', tier='stacked', arm='national')`](../tests/test_rte_risk_lab.py#L33) — function, line 33.
- [`run_fixture(directory, day='2025-07-01')`](../tests/test_rte_risk_lab.py#L38) — function, line 38.
- [`RiskLabTests`](../tests/test_rte_risk_lab.py#L65) — class, line 65.

Methods on `MeanAdapter`:

- [`__init__(self, *args, **kwargs)`](../tests/test_rte_risk_lab.py#L21)
- [`fit(self, X, y, weights)`](../tests/test_rte_risk_lab.py#L24)
- [`predict(self, X)`](../tests/test_rte_risk_lab.py#L29)

Methods on `RiskLabTests`:

- [`setUp(self)`](../tests/test_rte_risk_lab.py#L66)
- [`test_weighted_mae_arithmetic_and_error_sign(self)`](../tests/test_rte_risk_lab.py#L69)
- [`test_sample_weights_change_actual_boosting_fit(self)`](../tests/test_rte_risk_lab.py#L84)
- [`test_direct_and_residual_add_rte_exactly_once(self)`](../tests/test_rte_risk_lab.py#L94)
- [`test_training_weights_ignore_future_realized_weather_and_demand(self)`](../tests/test_rte_risk_lab.py#L105)
- [`test_cache_tamper_and_source_verification(self)`](../tests/test_rte_risk_lab.py#L119)
- [`test_first_freeze_rejects_native_quantile_test_artifact(self)`](../tests/test_rte_risk_lab.py#L134)
- [`test_unfrozen_test_overlap_is_rejected_before_fit_or_artifact(self)`](../tests/test_rte_risk_lab.py#L141)
- [`test_crossing_repair_preserves_missing_quantile_rows(self)`](../tests/test_rte_risk_lab.py#L150)
- [`test_wis_interval_and_pinball_constants(self)`](../tests/test_rte_risk_lab.py#L164)
- [`test_finite_sample_tail_ranks_are_directionally_correct(self)`](../tests/test_rte_risk_lab.py#L174)
- [`test_cqr_fixed_mapping_and_strict_calibration_tail_release(self)`](../tests/test_rte_risk_lab.py#L180)

### tests/test_rte_risk_metrics.py

[Source](../tests/test_rte_risk_metrics.py)

Numerical contracts for cold-risk point losses and probabilistic scores.

- [`RiskMetricTests`](../tests/test_rte_risk_metrics.py#L12) — class, line 12.

Methods on `RiskMetricTests`:

- [`test_normalized_cold_weighted_mae_excludes_only_missing_targets(self)`](../tests/test_rte_risk_metrics.py#L13)
- [`test_upper_quantile_pinball_penalizes_shortage_more(self)`](../tests/test_rte_risk_metrics.py#L32)
- [`test_winkler_penalizes_both_interval_exceedances(self)`](../tests/test_rte_risk_metrics.py#L37)
- [`test_wis_uses_central_interval_weights_and_missing_targets_are_unscored(self)`](../tests/test_rte_risk_metrics.py#L41)
- [`test_quantile_rearrangement_preserves_keys_and_does_not_hide_missing_raw_forecasts(self)`](../tests/test_rte_risk_metrics.py#L54)
- [`test_conformal_order_statistic_is_finite_sample_rank_not_interpolated_quantile(self)`](../tests/test_rte_risk_metrics.py#L69)

## Literal file and documentation relationships

The machine graph retains every extracted observation with relation type, original source/target direction, confidence and source location. Literal rooted paths can be resolved; dynamic `run.path` values, reflection and runtime-generated strings are not exhaustively traced. Semantic document links complement the code graph without converting an inferred relation into a runtime call.

The searchable inventory records 5,414 non-environment files. The graph groups 21 generated artifact collections:

| Artifact group | Files |
|---|---:|
| `reports/figures` | 11 |
| `reports/forecasts` | 81 |
| `reports/manager_report_20261021/app` | 240 |
| `reports/manager_report_20261021/figures` | 16 |
| `reports/models` | 20 |
| `reports/notebook_previews` | 8 |
| `reports/research_v2/20261020T163949967105Z/fits` | 32 |
| `reports/research_v2/20261020T163949967105Z/forecasts` | 30 |
| `reports/research_v2/20261020T163949967105Z/ledgers` | 17 |
| `reports/research_v2/20261020T164443776583Z/figures` | 9 |
| `reports/research_v2/20261020T164443776583Z/fits` | 522 |
| `reports/research_v2/20261020T164443776583Z/forecasts` | 539 |
| `reports/research_v2/20261020T164443776583Z/ledgers` | 267 |
| `reports/rte_assisted/20261020T203527821283Z/figures` | 2 |
| `reports/rte_assisted/20261020T203527821283Z/fits` | 161 |
| `reports/rte_assisted/20261020T203527821283Z/forecasts` | 328 |
| `reports/rte_assisted/20261020T203527821283Z/ledgers` | 161 |
| `reports/rte_risk/20261020T213640021260Z/figures` | 4 |
| `reports/rte_risk/20261020T213640021260Z/fits` | 662 |
| `reports/rte_risk/20261020T213640021260Z/forecasts` | 1,330 |
| `reports/rte_risk/20261020T213640021260Z/ledgers` | 662 |

## Updating the documentation

Run the graphify skill on this project with `--update` after source changes. The custom augmentation must also be refreshed: plain Graphify cannot infer notebook-cell imports or grouped artifact lineage by itself. The rebuild helper performs the complete deterministic refresh once all changed documents have semantic cache entries:

```sh
"$(cat graphify-out/.graphify_python)" graphify-out/refresh_graph.py
```

The helper stops before replacing a graph if changed documents need host-agent semantic extraction. It never trains models, executes notebooks or modifies experiment runs. It regenerates the inventory, source hashes, full graph, extraction report, code reference and overview. Preserve each versioned experiment when updating presentation files.

## Limits and verification

Source-level imports/calls and documented dependencies are not an exhaustive runtime trace. Grouped artifact counts are metadata, not independent experiments. AST references whose implementation was not resolved remain explicitly labeled unresolved references. Cohesion and centrality describe this mixed architecture/evidence graph, not model quality or refactoring priorities.

See [verification.json](verification.json) for current endpoint, deletion, source-hash and notebook checks, and [graph_health.json](graph_health.json) for extraction diagnostics.
