> Latest expanded RTE-feature research: [one-page recommendation](rte_risk/20261020T213640021260Z/one_page_recommendation.md). This document preserves the earlier workflow results.

# Electricity demand forecasting — final research assessment

## 1. Improvement over RTE
Frozen candidate: rte_correction_proxy, an RTE-assisted additive correction, not an independent replacement. Matched test MAE: 832.05 MW; provided RTE: 1,438.98 MW. Relative improvement: 42.18%; absolute MAE reduction: 606.94 MW.
Read reports/scorecards/paired_block_bootstrap.csv for paired 7/14/28-day block sensitivity and confidence intervals. These retain exactly the same eligible paired hours as headline MAE, including partial days with their actual counts. Calendar-day blocks preserve dependence; hourly errors are not resampled independently.

## 2. Uncertainty and failures
Nominal 90% quantile interval coverage: 0.811; mean width: 2,811.5 MW. The fixed rolling past-error baseline has coverage 0.890. Quantile coverage below 90% indicates undercoverage, not successful calibration; further method development must use validation or a new prospective period, not discretionary tuning on this test. Coverage, tail exceedances and pinball losses by hour, season, cold evenings and lead time are in reports/uncertainty/. Complete residual-day scenarios evaluate daily energy and peaks separately. Unsupported DST sample sizes remain explicit.
Worst-day evidence is in reports/failure_catalogue.csv. Weather misses cannot be verified without archived forecast weather. Do not infer causes from coincidence.

## 3. Research recommendation and evidence limits
Recommendation: retain this as a retrospective research result and confirm it on fresh later data before drawing stronger conclusions. Forecast vintages, source-to-hourly conversion and original observation publication times are unverified. The chronological test was previously explored in EDA. Demand history uses a disclosed publication-delay/revision approximation.
National consumption is not a validated customer portfolio model. Energy shortage/surplus proxies are not euro savings. Prices, applicable settlement rules and a common decision policy are required for financial claims. Hourly demand scores do not establish quarter-hour settlement performance.

The research deliverables are saved predictions, matched scorecards, uncertainty calibration, a failure catalogue and reports/risk_register.csv. Serving, scheduling, live monitoring and procurement integration are outside this project.
