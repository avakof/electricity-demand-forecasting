# Manager report claim audit

Reviewed saved run `20261020T213640021260Z`; no training, model selection or changes to the frozen run. Numerical claims below refer to the frozen point policy `lab__lightgbm__combined__stacked__w1__residual`, unless stated otherwise.

## 1. Improvement over RTE: supported historical result

| Complete policy | MAE, MW | RMSE, MW | Cold-weighted MAE, MW | Cold-evening MAE, MW |
|---|---:|---:|---:|---:|
| Frozen combined LightGBM residual policy | 719.631 | 948.607 | 744.747 | 1,578.018 |
| RTE | 1,438.982 | 1,782.647 | 1,467.913 | 2,427.773 |
| Rolling RTE bias correction | 1,111.048 | 1,412.837 | 1,132.513 | 1,844.657 |
| Earlier development-selected RTE-assisted policy | 806.761 | 1,048.745 | 830.319 | 1,611.933 |

All rows use **8,759 matched nonmissing hourly outcomes**, July 2025–June 2026, from 8,760 expected intervals. One missing truth is excluded from scoring; it is not interpolated. Source: [test scorecard](../rte_risk/20261020T213640021260Z/tables/test_scorecard.csv), rows for the four policies above; [artifact verification](../rte_risk/20261020T213640021260Z/artifact_verification.json), full-grid and exclusion checks.

The frozen point policy reduces historical MAE by **49.99% versus RTE**, **35.23% versus rolling bias correction**, and **10.80% versus the earlier selected policy**. Its 14-day paired-block bootstrap MAE improvement over RTE is **719.351 MW**, 95% interval **[630.722, 815.272] MW**, or **[45.02%, 54.65%]**. Its cold-weighted improvement over RTE is **49.26%**, interval **[44.10%, 54.28%]**. These are empirical uncertainty intervals within this historical sample, not protection against test reuse or future distribution changes. Source: [bootstrap comparisons](../rte_risk/20261020T213640021260Z/tables/bootstrap.csv), primary model versus `rte_proxy`, `MAE_MW` / `weighted_MAE_MW`, `block_days=14`.

Cold MAE improves **35.00% versus RTE**, but only **2.10% versus the earlier selected policy**. The latter difference is **33.915 MW**, interval **[-219.577, 238.289] MW**. It crosses zero for 7-, 14- and 28-day blocks. **Additional cold superiority over the earlier policy is not established.** Source: the same bootstrap table, reference `lightgbm_quantile__regional__rte_residual`, `cold_MAE_MW`; [independent review](../rte_risk/20261020T213640021260Z/independent_result_review.json).

## 2. Uncertainty and failures: the critical limitation

Cold means **temperature below 5 °C at local hours 18, 19 or 20**. Development contains **102 critical hours / 41 days**; test contains **66 hours / 24 days / one DJF winter**. Calibration, **2 April–29 June 2025**, contains **2,136 hours and zero cold evenings**. This cannot establish winter-conditional calibration. Source: [frozen specification](../rte_risk/20261020T213640021260Z/manifest.json), `spec`; [calibration support](../rte_risk/20261020T213640021260Z/tables/calibration_support.csv).

| Nominal central 90% interval | Overall coverage | Cold coverage | Cold upper misses |
|---|---:|---:|---:|
| Frozen point + past-calibration empirical residuals | 82.91% | 43.94% | 30/66 |
| Separately selected native quantiles | 75.82% | 53.03% | 25/66 |
| Fixed engineered quantiles | 72.74% | 54.55% | 23/66 |
| Fixed split conformal | 83.32% | 54.55% | 26/66 |
| Fixed conformalized quantile regression (CQR) | 86.08% | 63.64% | 20/66 |

Source: [overall uncertainty scorecard](../rte_risk/20261020T213640021260Z/tables/uncertainty_scorecard.csv) and [cold interval scorecard](../rte_risk/20261020T213640021260Z/tables/cold_interval_scorecard.csv), matching method labels. These five rows compare distinct forecast constructions; fixed methods use a separate fixed engineered predictor.

The native asymmetric interval, q.075–q.975, covers **59.09%** of cold outcomes and has **20/66 upper misses (30.30%)**, versus its nominal upper-tail rate of 2.5%. Native central upper misses have **1,145.513 MW conditional mean excess**. No examined method supports a claim of adequate cold-evening interval calibration. Source: cold scorecard, `native_quantile_lightgbm`, `central90_*` and `asymmetric90_*` columns.

Proper-score improvements do not rescue this coverage failure. The frozen point's specified empirical residual distribution has overall CRPS **555.286 MW**, versus **816.947 MW** for RTE with the same calibration procedure. Native quantiles have overall WIS **472.446 MW**, versus **496.203 MW** for fixed engineered quantiles, but cold WIS reverses the ordering: **1,109.058 versus 1,052.650 MW**. CRPS is exact only for the explicitly constructed empirical distribution; no exact CRPS is claimed for the sparse native quantile grid. Source: the two uncertainty scorecards and [reporting specification](../rte_risk/20261020T213640021260Z/tables/reporting_specification.json).

Cold underprediction persists for the frozen point: **51/66 hours (77.27%)**, average shortage **1,333.833 MW** across critical hours, and **1,726.136 MW** conditional on shortage. Overall bias of **+7.005 MW** (forecast minus actual) conceals this segment failure. The worst day by absolute net daily-energy error is **1 December 2025: 63,654.855 MWh**, with all-day shortage and **3,492.441 MW** error in daily maximum demand. Its cause is explicitly unproven; do not attribute it to weather, holidays or a structural break without additional evidence. Sources: test scorecard and [worst days](../rte_risk/20261020T213640021260Z/tables/worst_days.csv).

## 3. Scope, selection and false-claim traps

| Tempting claim | Defensible replacement |
|---|---|
| “98 algorithms were independently trained with RTE.” | **98 policy/variant labels**: 27 original policies, 16 earlier assisted variants and 55 new variants. Some point variants are numerically identical. There are 23 new hybrids covering 13 original family labels, not 98 distinct architectures. |
| “Every neural/statistical model natively ingests RTE.” | Each hybrid adds a LightGBM RTE-residual model using a causal base forecast. This is not a native RTE-feature retraining of each architecture. |
| “The winner ensembles all 13 families.” | The winner has `base=null`. Its `stacked` feature tier contains causal learned predictors, not an ensemble of all base architectures. |
| “The test identifies the best model.” | The point policy was frozen using development weighted MAE and guardrails. Other variants have smaller retrospective scores; they cannot replace the selection without another unseen evaluation. |
| “714.335 MW is the selected point MAE.” | **719.631 MW** is the frozen point result. **715.202 MW** is the separate quantile point-policy row; **714.335 MW** is its rearranged native q50 diagnostic. Quantile crossing repair explains the distinction. |
| “The quantile arm was optimized for cold risk.” | Its parameters come from point tuning; its arm was selected using **unweighted overall development WIS**, without a cold-interval selection guardrail. |
| “Weight 5 means cold errors dominate.” | The 66 critical hours contribute only **3.66%** of the weighted-loss denominator. The winning point model has training weight 1; cold weighting was still used for development selection and evaluation. |
| “The expanded features alone caused the improvement.” | Prior policies retrain monthly; new policies retrain quarterly. This is a complete-policy comparison. Within-laboratory feature ablations provide the more controlled feature comparison. |
| “An operationally unseen, pre-noon forecast experiment.” | The test was previously inspected. The origin is **11:00 Europe/Paris**, not fixed CET all year. Supplied `rte_forecast_d1_mw` is assumed available; historical issue vintages are unverified. No D2/05:00 rule is used. |
| “Regional-only predicts regional demand.” | All targets are national demand. Regional-only uses regional history plus national RTE and calendar; combined adds national history. National consumption remains a customer-portfolio proxy. |
| “Conformal guarantees 90% coverage.” | Exchangeability and cold-conditional guarantees are not established for this dependent, shifting series. Fixed comparator training weights can also affect conditional quantile interpretation. Assess its bounds empirically. |
| “Tuning proves architecture superiority / eliminates overfit.” | New comparisons use two parameter configurations per variant and chronological folds. Original bounded neural budgets persist. This is a bounded experiment, not exhaustive optimization or formal bias–variance decomposition. |
| “Weighted error reduction equals financial savings.” | These are MW/MWh risk proxies. Euro savings require prices, settlement rules and a shared procurement policy. Quantile cost ratios are hypothetical decision scenarios. |

Sources: [frozen selections](../rte_risk/20261020T213640021260Z/frozen_selection.json), [variant register](../rte_risk/20261020T213640021260Z/variant_register.json), manifest `spec`, [feature ablations](../rte_risk/20261020T213640021260Z/tables/feature_ablation.csv), [hybrid controls](../rte_risk/20261020T213640021260Z/tables/hybrid_control_comparison.csv), and the independent review. The eight-quantile rearrangement is predeclared and uses no outcomes; it changes q50 on 696 rows, all flagged as crossing repairs.

## 4. Manager-facing conclusion

**Retain the development-frozen combined LightGBM RTE correction as the research candidate for a genuinely unseen evaluation. Its historical point gains are substantial. Do not approve procurement automation or cold-evening reserve sizing from these interval forecasts.** Obtain trustworthy RTE/weather vintages and winter-relevant calibration/evaluation evidence before claiming operational superiority. Maintain the distinction between the utility portfolio and the national-demand proxy. No production implementation was performed or requested in this run.

Audit basis: [artifact verification](../rte_risk/20261020T213640021260Z/artifact_verification.json) passed; [independent review](../rte_risk/20261020T213640021260Z/independent_result_review.json) records 505 final checks and a maximum numerical deviation of 7.28×10⁻¹² MW after cached artifact regeneration. These checks support artifact consistency, not unverifiable historical publication vintages or future performance guarantees.
