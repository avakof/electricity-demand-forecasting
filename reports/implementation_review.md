# Methodological implementation review

The implementation was reviewed against three falsification questions:

1. **Could a future observation change an origin's features, forecast or calibrated interval?** Behavioral tests mutate post-cutoff realized data and delayed observations. Forecast issue and publication timestamps are both filtered. Training labels must be available before each fit. Historical finalized observations remain an explicitly disclosed approximation where original vintages are absent.
2. **Could inconsistent timestamps, missingness or selective availability create apparent skill?** UTC delivery grids preserve 23/24/25-hour local days, origin groups remain intact, original targets are not interpolated, duplicate source keys fail, and raw availability/common-eligible scores are separate from complete fallback-policy scores. Whole-day energy/peak metrics require complete days. Paired-loss bootstrap retains partially observed days with their actual eligible hour counts, matching the headline MAE population exactly; blocks never bridge missing dates.
3. **Could unsupported evidence become an operational or financial claim?** Operational experiments require actual archives plus verified provenance and interval lineage. The provided RTE column is only a retrospective assumed-cutoff proxy. Oracle weather cannot win selection. The previously explored test is disclosed. National-portfolio transfer and euro savings are not established.

## Deliberate boundaries

- Original source-to-hourly conversion cannot be reconstructed from hourly CSVs. It is recorded as unknown rather than invented.
- No external weather/RTE/observation archive or price history has been fabricated. Empty schema files document the integration interface.
- Deep learning is deferred as the plan requests until the simpler comparison demonstrates a need.
- Quantile and rolling empirical intervals are evaluated, not advertised as guaranteed conditional or simultaneous coverage. Whole-day residual scenarios explicitly report insufficient matching DST history.
- Source discrepancies and genuine peaks are retained. Failure catalogue classes describe observed evidence; weather misses require forecast-weather archives to verify.
- Earlier EDA exposure cannot be undone. A new prospective shadow period is necessary for a deployment claim.

Execution results, model fit cutoffs and saved forecast/source hashes provide the empirical validation trail; this review does not replace those checks.

## Corrections made during review

- Paired bootstrap eligibility was aligned with the headline MAE: partial days retain their valid paired hours and actual counts; no outcomes are interpolated.
- Every learned forecast origin is now closed to an append-only compressed run ledger before the next origin can fit on newly available outcomes. Aggregate forecasts also have content-addressed immutable copies and manifests. Predictions do not contain evaluation targets.
- Nominal quantile coverage is reported alongside achieved coverage. Undercoverage is identified explicitly and does not trigger discretionary test-period tuning.
