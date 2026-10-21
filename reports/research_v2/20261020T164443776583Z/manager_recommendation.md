# Manager recommendation: national day-ahead demand

## 1. Can we improve on RTE?

The 2024-preselected complete policy, **rte_bias_proxy** (RTE-assisted bias adjustment), has historical-test MAE **1,111 MW**, versus **1,439 MW** for the supplied RTE proxy on 8,759 matched hourly observations. Improvement: **328 MW (22.79%)**. Paired 14-day-block 95% intervals: relative improvement [18.99, 27.80]%; absolute improvement [266, 418] MW. The block comparison supports a historical MAE improvement.

The separate independent-model choice, **gam__national**, has MAE **1,866 MW**. Its result does not summarize RTE-assisted attempts. Both choices were frozen using 2024 only; final-test results never choose the headline policy.

## 2. How uncertain, and when does it fail?

The complete policy uses the predeclared past-out-of-sample residual distribution. Independent-model/RTE intervals use their saved **pooled/pooled** choices. The independent interval decision had 33 proxy-cold observations/11 days in 2024, against the minimum 60/20. Cold evenings are 18:00, 19:00, 20:00 delivery hours with realized temperature below 5 C.

| Policy | Overall coverage / width MW | Cold hours / days / DJF winters | Cold coverage / upper misses | Mean upper-miss severity MW |
|---|---:|---:|---:|---:|
| Complete policy | 90.5% / 4,752 | 66 / 24 / 1 | 69.7% / 27.3% | 1,350 |
| Independent model | 92.0% / 9,095 | 66 / 24 / 1 | 54.5% / 45.5% | 2,540 |
| RTE | 91.2% / 4,896 | 66 / 24 / 1 | 71.2% / 27.3% | 1,271 |

Intervals target 90% coverage; widths are MW. Severity is conditional on an upper-bound miss. The complete policy's cold upper-miss 14-day-block 95% interval is **[12.3%, 55.3%]**; status: **observed upper tail exceeds nominal 5pct**. DJF counts are represented winters, not full cycles; non-DJF cold events do not increase them. Sparse segments are inconclusive; zero-event bootstrap intervals [0, 0] do not prove zero risk. The 5% upper-miss target is diagnostic; conformal bounds do not guarantee equal tails. Hourly intervals are not daily-energy/maximum intervals. Scenario triggers use past-temperature proxies; realized cold is diagnostic. Statistical nonconvergence: 25/153 recorded fits; separate fit failures across all families: 0.

## 3. What should we use, and what are the risks?

**Retain RTE as the operational reference and carry the preselected complete policy forward as a research candidate.** Vintage archives are absent, timing/aggregation are assumed, and the 2025–26 historical test was previously inspected. Fresh prospective evidence is required.

Family rankings are confounded by 20/40-step neural caps, differing histories/features, and five windowed models' training-origin mismatch; they do not establish general architecture superiority.

Development-only regional LightGBM stress: all-history loss left 0.0% availability; missing regional inputs raised MAE 3,428→11,555 MW with 0.0% fallback. These guardrail gaps concern that tested policy.

Cold weights and shortage/surplus ratios are weighted-MW risk proxies, **not euro savings or validated procurement decisions**. National consumption remains a portfolio proxy.
