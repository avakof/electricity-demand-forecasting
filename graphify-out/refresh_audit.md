# Architecture refresh audit

Audit date: 21 October 2026. Scope: read-only inspection of existing graph artifacts, Python source, notebook cells, configurations and saved research results. No notebooks, model fits, forecasts or experiment configurations were executed or changed. This document describes the **pre-refresh graph baseline**; parent-agent graph/documentation changes may occur after inspection.

## 1. Verified stale baseline

The inspected `graph.json` contained 379 nodes and 877 links; `architecture_overview.json` contained 29 nodes and 76 edges. Its old source inventory covered 199 files. The inventory omits the later research extensions and manager report. A filesystem check confirms that `src/pipeline.py` and `reports/monitoring_specification.csv` no longer exist.

| Existing representation | Required correction | Evidence |
| --- | --- | --- |
| Notebook 06 imports/calls `src.pipeline` and writes monitoring specifications | Remove these current-code edges. Notebook 06 now imports `risk_register` from `src.research_risks`. Its filename remains for compatibility. | `notebooks/06_business_and_deployment.ipynb`, code cell 3; both deleted paths absent |
| A live freshness → forecast → publication → monitoring pipeline is an implemented project component | Remove or explicitly label as historical proposal. This project has no serving, scheduling, live monitoring or procurement integration. | `README.md:3`; `config/rte_risk_experiment.json:107` |
| Six notebooks constitute the whole current architecture | Represent eight main numbered stages plus a separate legacy `02_jg_baseline_model` branch. Notebooks 07 and 08 contain three distinct versioned research sections. | `README.md:27`; notebook imports listed below |
| `pipeline.py` remains a readiness/specification module | Remove its file, function and containment nodes and related edges. Retain actual `sklearn.pipeline` preprocessing imports and serialized fitted pipelines. These are unrelated to production serving. | Old `GRAPH_REPORT.md:110`; current filesystem and `src/rte_learned_features.py:66` |
| Old GAM/bias-correction recommendation is the latest winner | Preserve it within the original `research_v2` lane; attach the current expanded selection to its own frozen selection and result files. | Latest `result_summary.json:2`; `README.md:91–130` |
| `src`/`scripts`/config coverage excludes later research | Add the current research, assisted, feature, learned-feature, base-replay, risk and uncertainty modules, their orchestration/checking scripts and three experiment configs. | `src/research_experiment.py:47`, `src/rte_assisted.py:51`, `src/rte_risk_lab.py:133` |
| Old extraction report asserts 199 current unchanged files, six notebooks, 37 cells and 18 tests | Replace with current extraction counts; distinguish inspected saved verification from tests actually run by the graph task. | Old `GRAPH_REPORT.md:104–121`; current saved artifact verification |

The original graph's obsolete node identifiers include `src_pipeline`, `src_pipeline_readiness`, `src_pipeline_monitoring_specification`, `src_pipeline_risk_register`, `reports_monitoring_specification` and `reports_final_recommendation_live_pipeline`. A historical document mentioning such concepts is not evidence that the deleted implementation still exists.

## 2. Extraction helpers requiring correction

- `build_architecture.py:11–19`: its exclusion set omits `node_modules`, `.research-deps` and bundled app/build infrastructure. A naive recursive rebuild now absorbs third-party and generated artifacts. Group forecast shards, fit/ledger files, plot exports and report-app infrastructure; retain explicit manifests and important result tables.
- `build_architecture.py:142–148`: prefix-based selection admits legacy `02_jg_baseline_model` into the ordered original workflow and excludes 07/08. It also assigns every selected notebook to the old author/executor/validator. Use explicit notebook classifications and actual script ownership.
- `render_overview.py:27–77`: hard-coded six stages, deleted pipeline/monitoring nodes and old recommendation wording require replacement. Numeric notebook IDs alone collide for the two `02` files.
- `verify_and_finish.py:31,73`: fixed assertions of six notebooks and 18 test definitions must become evidence-derived checks. A source graph build cannot certify model execution or runtime behavior.
- AST imports and simple literal paths do not resolve dynamic `run.path`, configuration reference paths or class-method calls. Add documented semantic lineage with concrete source evidence; do not label inferred runtime writes as mechanically extracted calls.
- Preserve relation direction in the data even if a force-directed network is visually undirected. Distinguish `imports`, `inherits`, `reads`, `writes`, `references`, `compares`, and `historical_version_of`.

## 3. Current research stages and versioned lanes

| Lane | Code/configuration | Notebook and artifact lineage |
| --- | --- | --- |
| Original six-stage investigation | `src/data.py`, `eda.py`, `features.py`, `backtest.py`, `metrics.py`, `uncertainty.py`, `research_risks.py`; original forecast/evaluation configs | 01 audit → 02 EDA → 03 replay/baselines → 04 comparison → 05 uncertainty/failures → 06 business interpretation. `scripts/build_notebooks.py`, `execute_notebooks.py`, `validate_artifacts.py` own this workflow. |
| Legacy baseline | `scripts/evaluation.py`; its older quarterly protocol | `02_jg_baseline_model.ipynb`, code cell 3. Historical investigation, not a mandatory predecessor or authoritative new evaluation. |
| Original broad research comparison | `ResearchRun` (`src/research_experiment.py:47`), `ResearchData`, `research_models.py`, `research_neural.py`, `config/research_experiment.json` | Notebook 07 cell 2 and notebook 08 cell 2. Run `reports/research_v2/20261020T164443776583Z`: 23 variants from 13 family labels plus four references. `build_research_notebooks.py`/`execute_research_notebooks.py`. |
| First RTE-assisted extension | `RTEAssistedRun(ResearchRun)` (`src/rte_assisted.py:51`); `config/rte_assisted_experiment.json` | Notebook 07 cell 18; notebook 08 cell 21 imports `score_extension`. Run `reports/rte_assisted/20261020T203527821283Z`: 16 variants. `extend_rte_assisted_notebooks.py` appends this study. |
| Expanded features and cold-risk comparison | `RiskLabRun` (`src/rte_risk_lab.py:133`), `rte_feature_lab.py`, `rte_learned_features.py`, `rte_base_replay.py`, `rte_risk_uncertainty.py`; `config/rte_risk_experiment.json` | Notebook 07 cell 22; notebook 08 cell 27 imports `RiskLabRun`, `uncertainty_replay`, `report`. Run `reports/rte_risk/20261020T213640021260Z`: 55 new variants. `run_rte_risk_experiment.py` orchestrates; `extend_rte_risk_notebooks.py` appends; `check_rte_risk_artifacts.py` checks saved artifacts. |
| Manager-report presentation | `reports/manager_report_20261021/diagnostics/build_diagnostics.py`, `render_research_figures.py`, `prepare_manager_snapshot.py` | Saved forecasts/truth → diagnostic CSVs → figures and reviewed `app/src/data.json` → `app/src/content/report/ReportContent.jsx`; `complete_report.md` presents the findings. Reporting-only, not a forecasting service. |

`RiskLabRun` is a separate class, **not** a subclass of `ResearchRun`. Its data class `FeatureLabData` **does** inherit `ResearchData` (`src/rte_feature_lab.py:193`). All three studies remain separate protocols and selections. Notebook placement alone does not make their calibration periods or modeling objectives identical.

## 4. Data and predictor lineage

| Source → destination | Correct relation and limitation | Source evidence |
| --- | --- | --- |
| National CSV → national targets, observed histories, supplied RTE D1 | National demand is the target in every arm. RTE D1 is assumed available at 11:00 Europe/Paris on D, not verified from issuance vintages. No D2/05:00 product enters this experiment. | `config/rte_risk_experiment.json:5–12`; `src/rte_feature_lab.py:30,115–130` |
| Regional CSV → regional histories | Twelve regional demand/temperature histories; no regional RTE forecast. `regional_only` still uses national RTE and predicts national demand. The earlier arm named `regional` instead includes both national and regional histories. | `src/rte_feature_lab.py:196–235`; `README.md:49–55,93` |
| Observed histories → engineered features | Exact origin lags, publication-gated target lags, rolling statistics/counts, EWMAs, calendar, historical RTE errors, RTE-profile transforms and interactions. Target-minus-24-hour values are masked if unavailable. | `src/rte_feature_lab.py:22–80,115–188` |
| Regional histories → aggregates | Regional sums and unweighted temperature means require complete regional composition. No claim that these temperatures equal an official weighted national temperature. | `src/rte_feature_lab.py:137–156` |
| Supplied delivery-day RTE profile → ramp/level features | All such transformations inherit the assumed RTE-availability limitation; they are not future realized demand. | `src/rte_feature_lab.py:158–170` |
| Earlier released labels + engineered inputs → learned feature bank | Monthly Ridge demand forecast, cold probability and probability of RTE underprediction exceeding 1,000 MW. Training-only imputation/scaling and strictly released labels. Future temperature is a training label, never a delivery-day predictor. | `src/rte_learned_features.py:1–7,31–44,122–165,168–180` |
| Original sealed trial forecasts + released previous-quarter labels → causal base bank | Q1 2024 fixed trial-0 burn-in; later quarters choose among original configurations using only earlier released outcomes; January 2025 final choice must match original frozen selection. Append sealed original calibration/test predictions. | `src/rte_base_replay.py:51–112,188–254` |
| Engineered + learned + optional base bank → expanded replay | Merge by unique `(origin, delivery_time)`; labels are not feature columns. Only hybrid variants receive a base-model forecast. | `src/rte_risk_lab.py:177–209` |

All three archive CSVs under `data/raw` were checked: **zero data rows** in the weather forecast, demand observation and RTE forecast archives. Hashing supplied input files or successfully checking cutoff assumptions does not recover missing historical issuance/revision provenance.

## 5. Model and selection semantics that the graph must preserve

- **55 new variants = 30 standalone tabular variants + 23 hybrids + two hybrid controls.** The 23 hybrids each fit a LightGBM residual combiner with one original candidate's causal out-of-sample forecast plus new predictors. They cover 13 original family labels; this is not native RTE-input retraining of all neural/statistical architectures (`config/rte_risk_experiment.json:65–83`).
- **98 compared policies = 27 original + 16 prior assisted + 55 new.** These are policy labels, not 98 independent architectures; some policies produce identical point forecasts. Link comparisons to common eligible origin/delivery pairs, not separate favorable samples.
- Latest selected point: `lab__lightgbm__combined__stacked__w1__residual`. Its saved variant has `base: null`, `cold_weight: 1`, `tier: stacked`: it uses three learned predictors, **not** an ensemble of all 23 base forecasts. It predicts a correction added to RTE. Source: frozen selection entry and `src/rte_risk_lab.py:200–209,254–258`.
- Selection uses normalized realized-cold-weighted development MAE plus availability and ordinary/cold-error guardrails. Training weight variants instead use previously predicted cold probability. These are different uses of cold information; no edge should route future actual cold directly into prediction features (`config/rte_risk_experiment.json:44–64`).
- Quantile selection is separate: `lab__lightgbm_quantile__combined__stacked__w5__residual`, chosen by **overall unweighted development WIS**, not a cold-interval guardrail. Do not rename it the cold-optimal model (`config/rte_risk_experiment.json:83`).
- Expanded point models refit quarterly; learned generators monthly; original/prior reference policies keep their monthly schedules. The whole policy is compared; this does not isolate architecture effects (`config/rte_risk_experiment.json:63`).
- Development: Q2–Q4 2024 plus 2 January–31 March 2025, 102 cold hours/41 days. Calibration: 2 April–29 June 2025, **zero cold evenings**. Historical evaluation: July 2025–June 2026, **already viewed**; 8,759 scorable of 8,760 expected intervals, 66 cold hours/24 days/one DJF winter. Split endpoints in JSON are exclusive (`config/rte_risk_experiment.json:13–43,106,116–117`).

## 6. Uncertainty, reporting and recommendation edges

`uncertainty_replay()` reads the separately frozen quantile selection. Native quantiles are rearranged for crossings without outcomes. A distinct engineered-only predictor remains fixed across calibration/test for split-conformal/CQR comparisons; it drops monthly learned predictors but retains training weights. CQR endpoints are calibrated coverage bounds, not nominal conditional quantiles (`src/rte_risk_uncertainty.py:35–53,63–109`).

Connect quantile grids to pinball/WIS/interval diagnostics. Connect the empirical past-calibration residual distribution to empirical CRPS. **CRPS is a metric; split conformal and CQR are calibration methods.** Sparse quantile grids are not an exact continuous-distribution CRPS. Marginal hourly intervals are not automatically daily-energy or daily-peak intervals. Time-series/conditional coverage guarantees are not established.

`rte_risk_reporting.report()` combines saved new, original and prior-assisted forecasts, trims calibration to the expanded window, scores common observations and writes tables/recommendations. The manager-report scripts consume these outputs and add descriptive diagnostics; they do not choose new winning models. Inspection of saved test errors remains diagnostic, not new development evidence.

Current result labels, if included on the graph, must identify the saved retrospective experiment:

- Point MAE: **719.63 MW** versus RTE **1,438.98 MW**, **49.99% lower**; prior selected assisted model **806.76 MW**.
- Cold MAE: **1,578.02 MW** versus RTE **2,427.77 MW**; improvement over prior selection is only **33.92 MW**, with a 14-day-block interval **−219.58 to +238.29 MW**. Incremental cold gain over the prior policy is unestablished.
- Cold coverage for nominal 90% intervals: selected-point empirical **43.94%**; separate native quantile central **53.03%**, asymmetric **59.09%**; separate fixed CQR central **63.64%**. These belong to different forecast/calibration policies.

Evidence: latest `result_summary.json`, `tables/test_scorecard.csv`, `tables/bootstrap.csv`, `tables/cold_interval_scorecard.csv`, `recommendation.md`; manager `complete_report.md` and `claim_audit.md`. The recommendation remains research-only. No authentic operational superiority, calibrated winter protection, euro savings or utility-portfolio performance is established.

## 7. Refresh acceptance questions

1. **Does every current file/function edge resolve, while deleted production nodes and invalid legacy notebook ordering are absent?** Verify endpoints and source locations, parse all nine notebooks, classify eight main notebooks plus one legacy branch, and preserve artifact-group exclusions.
2. **Can each model/result node be traced to its own versioned protocol and frozen selection without implying native RTE retraining, test-based winner choice or verified vintages?** Check the three run identities, class inheritance, feature-bank dependencies, `base: null` for the selected point, selection metrics and refit schedules.
3. **Does the uncertainty/report layer preserve the distinction between predictions, labels, calibration, scores and business decisions?** Check zero cold calibration support, previously viewed test, limited cold counts, separate interval policies and absent production/procurement implementation.

Confirmed by this audit: existing graph staleness, source/class/import mappings, empty archive schemas, latest saved selections and result boundaries. Not certified by this audit: runtime execution, freshness of external data, historical availability beyond declared assumptions, or any new model performance. Later graph counts and source hashes must be regenerated from the final refreshed corpus rather than copied from this baseline.
