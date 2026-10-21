# Independent result review

Frozen primary: `lab__lightgbm__combined__stacked__w1__residual`. Read-only scoring uses saved predictions and raw national outcomes; source hashes are recorded in the companion JSON.

Test: 1 July 2025 through 30 June 2026; 8,759 matched nonmissing hours from 8,760 expected intervals. Cold means national temperature below 5 °C at 18:00, 19:00 or 20:00 Europe/Paris: 66 hours, 24 days, one DJF winter.

| Complete policy | MAE (MW) | Weighted MAE (MW) | Cold MAE (MW) |
|---|---:|---:|---:|
| New frozen point | 719.631 | 744.747 | 1578.018 |
| RTE | 1438.982 | 1467.913 | 2427.773 |
| Rolling RTE bias correction | 1111.048 | 1132.513 | 1844.657 |
| Prior selected RTE-assisted policy | 806.761 | 830.319 | 1611.933 |

The new point policy reduces overall MAE by 49.99% versus RTE, 35.23% versus rolling bias correction and 10.80% versus the prior selected policy. Its cold MAE improves by 35.00%, 14.45% and 2.10%, respectively.

Against the prior policy, paired 14-day-block 95% intervals for absolute improvements are +87.129 MW [63.713, 112.989] overall, +85.572 MW [62.428, 111.417] weighted, and +33.915 MW [-219.577, 238.289] on cold evenings. The cold interval crosses zero under all 7/14/28-day block lengths. Incremental cold superiority over the prior policy is not established.

Cold underprediction persists: 51/66 critical hours, mean shortage 1,333.833 MW across critical hours, and 1,726.136 MW conditional on shortage. Overall bias of +7.005 MW hides this segment-specific failure.

| Nominal central 90% interval | Overall coverage | Cold coverage | Cold upper misses |
|---|---:|---:|---:|
| Selected native quantiles | 75.82% | 53.03% | 25/66 |
| Fixed engineered quantiles | 72.74% | 54.55% | 23/66 |
| Fixed split conformal | 83.32% | 54.55% | 26/66 |
| Fixed CQR | 86.08% | 63.64% | 20/66 |

Native asymmetric q.075–q.975 bounds cover 59.09% of cold hours, with 20/66 upper misses versus a nominal upper-tail rate of 2.5%. The native central interval has 1,145.513 MW mean excess conditional on an upper cold miss. These interval forecasts do not provide reliable cold-evening risk coverage.

Native overall WIS is 472.446 MW versus 496.203 for fixed engineered quantiles; cold WIS reverses that ordering (1,109.058 versus 1,052.650 MW). Do not infer cold optimality from overall proper scores. The quantile median has lower test MAE (714.335 MW), but replacing the frozen point policy after inspecting this result would be a new selection decision.

## Defensible conclusion

The expanded policy improves historical point errors over RTE, rolling bias and the earlier selected policy. Additional cold improvement over the earlier policy remains uncertain. None of the examined nominal 90% uncertainty methods achieves reliable cold coverage. Prospective confirmation and winter-relevant uncertainty calibration remain necessary; this is not a production or procurement recommendation.

## Limitations

- Retrospective test July 2025 through June 2026 was previously inspected; bootstrap intervals do not correct experiment-selection optimism.
- Supplied RTE day-ahead values are assumed available before the forecast origin; original issue vintages are not verified. No next-but-one-day or 05:00 rule is used.
- Only 66 cold-evening test hours across 24 days and one DJF winter; multi-day resampling does not establish transfer to other winters.
- April 2 through June 29 calibration contains zero cold-evening outcomes. Nominal 90% labels do not imply calibrated coverage.
- New point policy is frozen from development; do not replace it post hoc with the separately WIS-selected quantile median despite lower test MAE.
- Quantile arm was selected by overall unweighted development WIS without a cold-interval guardrail. It is not established as cold-optimal.
- Split-conformal and CQR endpoints are calibrated coverage bounds, not nominal conditional quantiles. Exchangeability and cold-conditional guarantees are not established.
- Fixed engineered-only comparator omits evolving learned predictors but retains forecast-cold-derived training weights; evaluate its conditional endpoint semantics empirically.
- The original/prior reference policies retrain monthly; new policies retrain quarterly. Comparison measures complete policies, not feature engineering alone.
- Cold-weighted error and shortage/surplus scenarios are MW risk proxies, not euro savings. National consumption remains a customer-portfolio proxy.
- CRPS is exact only for the explicitly specified empirical residual distribution; a sparse native quantile grid does not define exact CRPS.

Final generated report cross-check: passed. 359 numerical comparisons agree within 1e-7 MW, covering point metrics, all 27 bootstrap comparisons, interval coverage and scores, WIS and result-summary deltas. No arithmetic mismatch was found.

Recommendation interpretation review: resolved. The regenerated recommendation now states that the prior-policy cold improvement confidence interval crosses zero, reports the native central/asymmetric and fixed CQR cold coverage failures, and explicitly rejects adequate cold-evening interval calibration. Original findings and resolution evidence remain recorded in the companion JSON.

Final artifact provenance refresh: passed after notebook regeneration. Recomputed probability metrics from the final aggregate and rechecked all 359 report comparisons; 505 numerical checks passed in total. Maximum absolute difference was 7.28e-12 MW, below the declared 1e-7 MW tolerance. Compressed aggregate/protocol/CSV bytes changed; current file hashes and decompressed hashes are recorded, and earlier provenance/metric values are retained in JSON history. No training, replay, report regeneration or prediction modification was performed by this review.
