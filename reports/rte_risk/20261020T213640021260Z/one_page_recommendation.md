# Day-ahead demand: research recommendation

**Decision:** retain the development-selected RTE-assisted LightGBM correction for a new, unseen evaluation. The point forecast improves historical accuracy substantially. The uncertainty component has not demonstrated adequate cold-evening protection. No production implementation or procurement savings claim is supported by this study.

## 1. Does it improve on RTE?

The model adds a learned correction to `rte_forecast_d1_mw`, using country and regional histories, calendar features, eligible lags, rolling statistics and historical out-of-sample linear/classifier predictions. Selection used chronological development data and cold-weighted error; the selected model used ordinary training weights.

| Frozen policy | Overall MAE, MW | Cold-evening MAE, MW |
|---|---:|---:|
| Supplied RTE | 1,438.98 | 2,427.77 |
| Rolling RTE bias correction | 1,111.05 | 1,844.66 |
| Prior RTE-assisted selection | 806.76 | 1,611.93 |
| Expanded selected correction | **719.63** | **1,578.02** |

On 8,759 matched hours from July 2025–June 2026, overall improvement over RTE is **49.99%**, with a paired 14-day-block 95% interval of **45.02%–54.65%**. Improvement over the prior policy is **10.80%**. These compare complete policies: the new model retrains quarterly, while earlier references retrain monthly.

## 2. How uncertain is it, and when does it fail?

Cold evenings mean temperature below 5 °C at 18:00–20:00 local time. The test has only **66 such hours across 24 days and one winter**. Cold MAE improves **35.00% versus RTE**. Extra improvement over the prior model is only 33.92 MW, with interval **−219.58 to +238.29 MW**; incremental cold superiority is unproven. The selected point forecast still underpredicts 51 of these 66 hours.

Nominal 90% cold-evening coverage is **43.94%** for the selected point model's empirical interval, **53.03%** for the separately selected native quantile model, and **63.64%** for fixed-model CQR. The native asymmetric interval still misses above its upper bound in **30.30%** of cold hours, versus a nominal 2.5%. Calibration contained **zero cold evenings**. These intervals are unsuitable as evidence of reliable critical-condition coverage.

## 3. Recommendation and limits

Confirm the point policy on future unseen data with trustworthy issue vintages and winter-representative uncertainty calibration. Preserve the current results as retrospective evidence: historical RTE/observation availability is assumed, and the test period was previously inspected. All 13 original model families appear in labeled hybrids; this does not mean every architecture was natively retrained with RTE. Original compact neural budgets limit architecture rankings.

The fivefold cold weighting and asymmetric error penalties are research risk scenarios, not euro savings. National consumption remains a proxy for a utility customer portfolio.

Details: [full recommendation](recommendation.md), [98-policy scorecard](tables/test_scorecard.csv), [independent numerical review](independent_result_review.md).
