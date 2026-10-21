# RTE-assisted research comparison

## 1. Improvement over RTE and simple bias correction

The 2024-selected assisted variant is **lightgbm_quantile__regional__rte_residual**. The complete-policy choice across all43 alternatives was **lightgbm_quantile__regional__rte_residual**. These identities were frozen before new calibration/test replay. The test is previously exposed and therefore retrospective.

| Policy | Matched test MAE MW |
|---|---:|
| Selected assisted variant | 806.8 |
| Raw RTE | 1,439.0 |
| Rolling RTE bias correction | 1,111.0 |

On 8,759 matched hours, assisted improvement over RTE is **43.94%**, or 632.2 MW; paired14-day-block95% interval [39.39, 48.31]%. Improvement over rolling bias correction is **27.39%**, or 304.3 MW; interval [22.44, 30.71]%. A negative improvement means deterioration. Tables also retain every variant and its independent comparator. No winner is selected from test scores.

## 2. Uncertainty and cold-evening failures

The fixed past-out-of-sample residual distribution has nominal90% coverage, empirical overall coverage **90.5%**, width **3,784 MW**, and CRPS **590.7 MW**. These intervals belong to the monthly-refitted policy. The separate split-conformal comparator uses one fixed fitted predictor throughout calibration/test. Native LightGBM quantiles receive quantile/interval scores, not invented CRPS. No exchangeability or conditional-coverage guarantee is established.

At realized temperature<5°C during18:00/19:00/20:00 local, there are **66 hours / 24 days / 1 represented DJF winter(s)**. Selected-assisted coverage is **51.5%**, upper exceedance **45.5%**, and conditional mean upper-miss severity **1,113 MW**. Upper-exceedance14-day-block95% interval: [32.5%, 77.4%]. Status: **observed_upper_tail_exceeds_nominal_5pct**. The5% upper target is diagnostic; symmetric conformal intervals need not allocate equal tails. Hourly intervals do not define daily energy or maximum intervals. Interval availability and sparse segment counts remain visible. Weighted-MW shortage/cold scenarios are risk proxies, not euro savings.

## 3. Recommendation and limits

Use the executed scores to assess this research candidate against both RTE and the simple correction. A gain against raw RTE alone is insufficient evidence that learned complexity adds value. Cold-tail protection requires its separate diagnostic to pass; overall MAE does not establish it. Keep RTE as the operational reference pending authentic vintage archives and fresh prospective evidence.

There are 0 failed fit events among 817 recorded fits. Original issue versions, consumption revisions, publication delay and hourly aggregation remain unverified assumptions. Missing evaluation targets are excluded without interpolation. RTE-assisted predictions use national RTE; regional arms add regional histories, not regional RTE forecasts. The16 new variants cover four tabular families with two data arms and direct/residual targets; neural and statistical architectures were not silently altered. Searches contain two configurations and four chronological quarters. Variants are separately retuned and add both RTE and recent RTE bias, so their comparison concerns complete policies, not the isolated causal effect of one predictor. Same-fit development training/next-origin error gaps and selected quarter scores are saved; such gaps can reflect season or regime changes and are not a formal bias-variance decomposition. Zero-event bootstrap intervals do not prove zero future risk. Test results must not trigger discretionary retuning. National consumption is a portfolio proxy; no production implementation or financial-savings estimate is supplied.
