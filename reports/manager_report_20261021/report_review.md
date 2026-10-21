# Independent review of the manager report

Review scope: `complete_report.md`, `app/src/content/report/ReportContent.jsx`, `app/src/data.json`, cited frozen-run tables and the new descriptive diagnostic tables. No model fitting, replay, source-data changes or frozen-run edits were performed.

## Outcome

**No numerical error found in the checked Markdown claims or app snapshot.** The Markdown addresses all three manager questions and preserves the material caveats: retrospective test reuse, unverified forecast vintages, sparse cold support, failed critical-interval coverage, the distinction between point and probability selections, and lack of measured financial savings. The recommendation is proportionate to the evidence.

**Final status: passed.** All three required app wording/presentation findings below are resolved and verified in the final source. The Markdown bootstrap clarification and the new missing-target support query are also verified. No required correction remains open.

## Resolved findings

1. **Current operating policy — resolved.** The earlier app draft assumed RTE was the utility's operating policy. Both the executive summary and recommendation now state: **“Do not replace the current operating process on this evidence; retain RTE as the benchmark.”** This agrees with the Markdown and does not infer the utility's present purchasing policy.

2. **Worst-day chart clock — resolved.** The chart now uses `local_hour` and labels its axis **“Delivery hour (Europe/Paris)”**. The cited 67,480 MW observation therefore plots at **19:00 local time**, matching the narrative; its UTC timestamp remains 18:00 in source data. Source: [hourly cold-day data](diagnostics/worst_cold_day_hourly.csv).

3. **Classifier direction — resolved.** The feature narrative now specifies **RTE underprediction** and the label **actual demand minus RTE > 1,000 MW**. This agrees with [`src/rte_learned_features.py`](../../src/rte_learned_features.py), `underprediction_above_mw` and the `residual.gt(...)` construction.

## Verified clarifications and presentation distinctions

- **Bootstrap wording — resolved.** The Markdown now states **500 attempted draws**, all 500 retained for overall comparisons, and **495 valid cold-only draws** at 14-day blocks. The app table exposes the matching valid counts.
- **Missing-target support — verified.** The app's new `evaluation_support` query and narrative disclose **one missing target**, **8,759 scored hours**, no exclusion for missing policy predictions among scorable targets, and **24 observed versus 25 expected hours on 26 October 2025**. Complete-day energy/peak diagnostics omit this incomplete day; no target was interpolated.
- The interactive business chart shows **unchanged point-forecast** asymmetric losses from `asymmetric_risk_scenarios.csv`; the Markdown's figure 06 discusses **quantile-based decisions** from `quantile_decision_risk_scenarios.csv`. Both are correctly labeled. They are different experiments, not alternative displays of the same calculation.
- “Archived point forecasts” in the app denotes saved retrospective replay outputs. Its surrounding caveats explicitly state that authentic historical issue vintages are unavailable; the label must not be read as proof of a live historical forecast archive.
- The Markdown contains the numerical CRPS/WIS comparison; the app emphasizes coverage, width and upper misses and explains proper-score definitions. This presentation difference does not alter the reported critical-risk conclusion.

## Verification performed

- **6,318 numeric/Boolean values** across 13 app queries match their cited source rows within absolute `1e-7` and relative `1e-10` tolerance. Checks cover all 98 policy rows, five frozen comparison roles, all 27 selected-point bootstrap comparisons, 14 probability rows, 18 ablations, 60 point-risk scenarios, selected fit diagnostics, monthly/hourly diagnostics, five worst days, the 24-hour cold-day trajectory, both bias-diagnosis rows and the new missing-target support record. All **55 model-scope rows** also match the frozen family, arm, target, feature tier, training weight, base-model identity and selected parameters. Source `researchSummary` matches the frozen run's `result_summary.json` exactly.
- **All 34 local links/images in the Markdown resolve**, and every file cited by the app queries exists. External methodology pages were not re-fetched in this read-only review.
- Headline values agree: selected MAE **719.631 MW**, RTE **1,438.982 MW**, improvement **49.99%**; prior-policy MAE improvement **10.80%**. Overall and cold relative confidence intervals match `bootstrap.csv`, including the prior-policy cold interval **[-12.01%, 17.22%]** / **[-219.58, 238.29] MW**.
- All five uncertainty-method coverage/width/miss rows match the saved scorecards. The asymmetric native cold result is **59.09% coverage / 20 of 66 upper misses**. The report correctly refuses to infer useful cold-tail calibration from a nominal 90% label or lower aggregate proper scores.
- Additional descriptive claims match saved diagnostics: outside-cold MAE **713.114 MW**; winter/summer MAE **901.328 / 612.926 MW**; local-hour extremes **913.372 MW at 17:00 / 537.343 MW at 02:00**; lead groups **830.582 MW above 24 hours / 608.655 MW up to 24 hours**; December monthly MAE **1,016.69 MW**; the five daily failure rows and the 24 December 19:00 observation agree.
- The corrected national engineered cold MAE is **1,519.54 MW**. **11 of 23** hybrids improve weighted test loss over their matched control. Both claims are accurate and presented as descriptive comparisons rather than new selection results.
- The selected policy's four-fold training MAE range **268.52–435.16 MW** and following-fold range **527.36–1,008.58 MW** are correct after filtering `development_fit_gaps.csv` to the selected model. They must not be interpreted as a formal bias–variance decomposition.
- All 12 saved native quantile decision scenarios improve the corresponding residual-adjusted RTE proxy loss. The report correctly labels these physical-unit scenarios, not financial savings or proven optimal procurement.
- Verification totals in the report match the artifact audit: **662 forecast files, 1,927,344 rows, 80,306 daily-origin records, 851 fit events, zero failed fit executions**. The previously recorded independent review contains **505 checks**. These totals do not establish historical vintage authenticity or future operational reliability.
- **14 previously reviewed frozen-input/report hashes remain unchanged** against the independent review's final provenance: nine experiment input artifacts and five saved report artifacts. This report review changed none of those files.

## Final assessment

The Markdown's central conclusion is defensible: substantial historical point gains justify further research on the frozen candidate; cold-tail uncertainty and authentic-input evaluation remain blockers to an operational replacement. No omission was found that would reverse this conclusion. All required consistency findings are resolved; the final written and interactive reports support the same decision.
