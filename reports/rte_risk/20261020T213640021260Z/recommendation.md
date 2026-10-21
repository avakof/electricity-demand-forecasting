# RTE-assisted feature and cold-risk research recommendation

## Improvement over RTE

The development-frozen point selection is **lab__lightgbm__combined__stacked__w1__residual**.
It passed the predeclared development guardrails.
On **8,759 matched nonmissing hours**, its MAE is **719.6 MW**,
versus **1,439.0 MW** for RTE and **1,111.0 MW** for rolling RTE bias correction.
The RTE-relative MAE improvement is **49.99%**.
With cold evenings weighted five times, normalized weighted MAE is **744.7 MW**;
the RTE-relative improvement is **49.26%**,
with paired 14-day-block 95% interval **[44.10, 54.28]%**. Scores use exactly the same origin-delivery pairs.
The earlier RTE-assisted study selected **lightgbm_quantile__regional__rte_residual** using development data. On these same observations it has MAE **806.8 MW** and cold-evening MAE **1,611.9 MW**. Relative to that prior selection, the new candidate's loss improvements are **+87.1 MW** overall, **+33.9 MW** on cold evenings, and **+85.6 MW** under the declared weighting (positive is improvement; negative is regression). The weighted-loss improvement's paired 14-day-block 95% interval is **[+62.4, +111.4] MW**. The cold-evening improvement's corresponding interval is **[-219.6, +238.3] MW**. Additional cold-evening point improvement over the prior policy is not established.


## Uncertainty and failures

Cold-evening MAE is **1,578.0 MW**, versus RTE's **2,427.8 MW**.
This segment contains **66 hours, 24 days, and 1 DJF winters**.
The declared chronological development selection had **102 cold-evening hours across 41 days**;
that limited support constrains claims about persistent winter superiority.
Its past-calibration empirical 90% interval covers 43.9% of cold-evening outcomes; upper-tail misses occur in 45.5%, with 1,081 MW conditional mean severity. These are dependent historical observations, not a conditional-coverage guarantee.
The separately selected native quantile model's central 90% interval covers **53.0%** of cold-evening outcomes, with **37.9%** upper misses. Its asymmetric 90% interval covers **59.1%**, with **30.3%** upper misses against a nominal 2.5% tail. The fixed engineered CQR comparator's central 90% interval covers **63.6%** of cold-evening outcomes, with **30.3%** upper misses. These results do not establish adequate cold-evening interval calibration.
The probability tables report central and asymmetric intervals, pinball loss, central-interval WIS,
upper-tail severity, and exact CRPS only for a specified empirical residual distribution.
A sparse quantile grid is not labeled exact CRPS. Genuine peaks and missing-target exclusions are retained.
The quantile arm was selected using **unweighted overall development WIS**, without a cold-interval
selection guardrail; it is not established as optimal for cold evenings. The fixed engineered-only
comparators retain forecast-cold training weights while omitting learned predictors from their inputs.
Their native endpoints and calibrated bounds require empirical coverage assessment; nominal conditional
quantiles and cold-conditional coverage are not assumed.
Raw forecast availability is **99.99%** and fallback frequency is **0.01%**.
Worst-day causes remain unproven without archived weather forecasts and input-vintage evidence.

## Recommendation and risks

Retain the frozen point model as a candidate for the next unseen research evaluation.
The interval methods do not yet support reliable cold-evening reserve-risk quantification.
This is a retrospective research comparison. Use the supplied **rte_forecast_d1_mw** under the explicitly
assumed pre-noon availability rule; no next-but-one-day product or 05:00 rule enters this experiment.
The 11:00 Europe/Paris origin leaves a pre-auction buffer. Original issue vintages remain unverified.
All 13 original families are represented through clearly labeled base-model-plus-RTE-feature hybrids;
that does not imply native exogenous support in every neural or statistical architecture.
The national, regional-history-only, and combined feature arms share national consumption as evaluation truth.
Q1 2024 supplies causal stacking burn-in; the declared subsequent chronological folds select parameters and policies.
Calibration runs from **2025-04-02** up to but excluding **2025-06-30**.
All reference methods use that same calibration period. Its seasonal and cold-support limitations are recorded in the calibration-support table.
Calibration contains **zero cold-evening outcomes**; these uncertainty methods were not calibrated on that critical segment.
The 2025–26 evaluation period was previously inspected, so prospective confirmation remains necessary.
Cold weights and shortage/surplus ratios are risk scenarios in weighted MW, not euro savings.
No production or procurement implementation is included. National demand remains a customer-portfolio proxy.
