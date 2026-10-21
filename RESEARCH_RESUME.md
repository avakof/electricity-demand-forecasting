# Expanded RTE feature and cold-risk study — complete and verified

Latest user instruction: ignore the D2 05:00 schedule; use supplied RTE D1 as an assumed pre-cutoff feature, and extend causal feature engineering, model comparisons and asymmetric cold-evening risk.

Run: `reports/rte_risk/20261020T213640021260Z`. Training, tuning, calibration, test and probability replays, reporting, notebook execution, artifact checks and visual review are complete. No owned training, notebook-kernel, audit or preview-server process remains. Do not restart completed training. All 55 new variants and 98 total policies are scored. Frozen point selection: `lab__lightgbm__combined__stacked__w1__residual`; probabilistic selection: `lab__lightgbm_quantile__combined__stacked__w5__residual`.

Selected test MAE 719.631404 MW versus RTE 1438.982076 and prior RTE-assisted 806.760882: 49.99% and 10.80% improvements. Cold MAE 1578.017678 MW versus RTE 2427.772728 and prior 1611.932683. Prior cold improvement 33.915 MW has 14-day 95% CI [-219.577,+238.289] MW, so extra cold improvement is unconfirmed. Native 90% cold coverage is 53.03%; fixed CQR 63.64%; selected-point empirical 43.94%. Intervals fail critical-condition diagnostics.

Notebooks 07 and 08 each contain three successfully executed extension code cells, preserving earlier studies. Final figures and exported HTML previews were visually inspected; structural checks passed. Frozen predictive sources, configuration and forecasts remain unchanged. Start with the run's `one_page_recommendation.md`, `tables/test_scorecard.csv`, and `tables/cold_interval_scorecard.csv`. No commit or push.

Development is Q2–Q4 2024 plus Jan2–Mar31 2025 (102cold hours/41days); separate calibration Apr2–Jun29 2025 has zero cold-evening observations; retrospective test remains Jul2025–Jun2026. All13 original families enter explicitly labelled RTE-feature hybrids with previously trained causal base forecasts. Enhanced tabular models compare national, regional-only and combined predictor views. No production work.

Verification: 123 tests passed and one optional test skipped. The artifact audit passed for 662 sealed forecast files, 1,927,344 forecast rows, 80,306 daily-origin ledgers and 851 fit events, with zero execution failures. It checked 938 original reference hashes and five prior-assisted reference hashes. All 8,760 delivery keys remain present; all 8,759 nonmissing targets are scored. The single missing-target/RTE DST hour is reported without interpolation. Independent review passed 505 comparisons after the final report refresh, with maximum numerical change 7.28e-12 MW. Records: `artifact_verification.json`, `independent_result_review.json`, `numeric_verification.json`, `unit_test_results.txt`, and `visual_review.json`. No further computation is pending for this bounded experiment. Native RTE retraining of every base architecture, larger neural budgets, verified vintages and a fresh unseen winter remain future experiments, not claims of this run.

# Earlier RTE-assisted extension — complete

Run `reports/rte_assisted/20261020T203527821283Z` completed both appended notebook sections and a cached reporting refresh. Artifact verification passed:503,047 forecast rows,20,961 daily-origin ledgers and817 fit events, with zero fit failures. Both notebooks executed without errors at that checkpoint. PNGs were visually inspected; HTML previews were exported/structurally checked without browser visual inspection.

Development selected `lightgbm_quantile__regional__rte_residual`. Retrospective test MAE806.76MW vs RTE1438.98MW and rolling bias correction1111.05MW:43.94% and27.39% improvements. Empirical90% cold-evening coverage51.52% (34/66), upper misses45.45% (30/66),24days/onewinter. Native three-quantile cold coverage60.61%. These intervals fail the critical-condition diagnostic; expanded risk work remains active. No owned earlier-run process remains. Details are in `result_summary.json`, `artifact_verification.json`, `visual_review.json` and `recommendation.md`.

---

# Research execution complete

Run: `reports/research_v2/20261020T164443776583Z`.

Both requested notebooks executed successfully and retain their outputs:

- `notebooks/07_statistical_and_boosting_models.ipynb`
- `notebooks/08_deep_learning_and_uncertainty.ipynb`

All 13 model families completed the bounded search: 23 national/regional variants, 184 configuration-quarter validations and four reference policies. Hyperparameters were chosen using chronological 2024 validation. Calibration is January–June 2025; the retrospective test is July 2025–June 2026. One missing evaluation target is excluded, never interpolated. Split conformal is calibration; CRPS is a score, not a model.

## Results

The complete policy selected on development data is RTE-assisted rolling bias correction. Test MAE is 1,111.0 MW versus 1,439.0 MW for supplied RTE: 22.79% improvement, with a paired 14-day-block 95% interval of 18.99%–27.80%, on 8,759 matched hours. The independently selected national GAM has MAE 1,865.7 MW.

The complete policy's nominal 90% interval covers 90.47% overall but only 69.7% of 66 realized cold-evening hours across 24 days and one represented DJF winter. Upper misses are 27.3%; conditional mean exceedance severity is 1,350 MW. The cold-evening risk diagnostic fails. No test-based interval retuning was performed.

Start with the run's `manager_recommendation.md`, `peak_report.md`, `failure_cases.md`, and `tables/test_scorecard.csv`.

## Verification

- 76 tests passed; one optional integration test skipped; none failed.
- Strict artifact check passed: 272 forecast files, 1,779,353 rows and 267 origin ledgers.
- Independent standard-library arithmetic confirmed headline MAEs, matched counts, and cold-evening interval misses.
- Both notebooks passed top-to-bottom execution. After visual inspection, notebook08 cell16 alone was genuinely re-executed against archived score tables for a legend fix; metadata records the refresh. Frozen sources, selections and forecasts remained unchanged.
- Final PNG figures were inspected directly. HTML previews were exported and structurally checked. Browser visual inspection was unavailable because CUA had no enabled browser.
- Saved verification: `artifact_verification.json`, `independent_metric_verification.json`, `unit_test_results.txt`, `visual_review.json`, and `execution_state.json` in the run directory.

## Interpretation limits

Original demand, RTE and weather forecast vintages are unavailable; release timing and hourly aggregation are assumptions. The test period had already been inspected. Results support retrospective comparison, not operational superiority or euro savings.

Neural fits used compact 20/40 optimizer-step budgets and target history only. Regional neural arms share parameters across univariate country/regional series; they are not multivariate cross-region models. Five windowed models used training anchors with an intraday phase different from the forecast origin. These limitations are documented in `execution_notes.md`; model-family rankings do not establish general architecture superiority. Statistical optimizers reported nonconvergence in 25/153 fits, separate from zero fit-execution failures.

Correcting training-anchor policy or increasing budgets requires a fresh development experiment and run identity. Do not alter frozen sources or claim an untouched confirmation test. No current training or calibration work remains. Historical pause records in `.runtime/research_pause.json` and the run's `pause_checkpoint.json` are preserved only as provenance; do not resume obsolete PIDs.

## Reproduction

Scientific interpreter: `/Users/alivakof/Documents/Codex/python-data-science/.venv/bin/python`.

Dependency files: `requirements-research.txt` and `requirements-research-neural.txt`; the compatible Intel neural stack is pinned in project-local `.research-deps`.

`python scripts/execute_research_notebooks.py` starts a fresh notebook07 run and continues it in notebook08. Rebuilding notebooks clears their outputs. `python -m unittest discover -s tests -v` runs tests. `python scripts/check_research_artifacts.py --run-dir reports/research_v2/20261020T164443776583Z` verifies this completed run.

Production-only pipeline and monitoring files were removed as requested. Raw data are unchanged. No commit or push was performed. Existing Graphify/Impeccable files are outside this task.
