# Reviewed RTE D1 feature and cold-risk experiment

The supplied `rte_forecast_d1_mw` is used as an assumed pre-cutoff input. The next-but-one-day 05:00 schedule is outside this experiment. An assumed available CSV column does not verify its historical publication vintage.

## Design

1. Freeze a separate experiment and preserve the previous two studies. Keep the 11:00 Europe/Paris origin, strict interval-end-plus-two-hour assumed publication delay, and actual 23/24/25-hour delivery grids.
2. Build origin-anchored demand and temperature lags; publication-gated target-aligned lags; exact-slot rolling means, standard deviations, ranges and observation counts; EWMAs; ramps; historical released RTE errors; and same-target-day RTE profile features. Moving averages of observations are distinct from the latent innovation terms in ARIMA MA models.
3. Compare national predictors, regional predictors, and both. National demand remains the target. Do not replace it with an unreconciled regional sum. No national demand-history input is allowed in the regional-only feature view.
4. Add fixed-parameter ridge forecasts, cold probabilities and RTE-underprediction probabilities generated out of sample at historical origins. Fit their imputers, scalers and estimators only on earlier released labels.
5. Fit the four tabular families on the enhanced features. For every original statistical/neural/tabular variant, fit an explicitly named RTE-assisted hybrid using that model's causal out-of-sample forecast. Reuse actual trained base-model archives with verified hashes. Do not pretend unsupported architectures accept the new columns natively.
6. Use chronological Q2–Q4 2024 and January–March 2025 validation. Q1 is the hybrid burn-in. Base hyperparameters are chosen only from outcomes available before each new quarter; all-2024 base selection becomes available for 2025. Tune two declared configurations per new variant. Separate calibration is April 2–June 29, 2025; it has zero cold evenings, which limits cold-conditional interval validation. The development window has 102 cold hours on 41 days. Quarterly parameter refits keep this broad CPU study bounded; original references retain their documented monthly policy.
7. Penalize cold evenings in training through pre-cutoff predicted cold probability. Select point models by normalized cold-weighted MAE, with overall and cold-segment guardrails relative to simple RTE bias correction. Fivefold cold weighting is an explicit research scenario, not a price estimate. Report ordinary MAE, RMSE, bias, shortage severity, day maxima and availability too.
8. Train quantile LightGBM for an explicit quantile grid. Compare central and asymmetric intervals, pinball loss, WIS/Winkler scores, empirical CRPS and high-demand threshold-weighted CRPS. Fit fixed-predictor conformal/CQR comparators on separate calibration data. Do not equate a high quantile with an MAE-optimal demand forecast.
9. Freeze candidates before the new historical test replay. Report matched RTE and bias-correction improvements with paired multi-day block intervals; show cold counts, days and represented winters. The reused test was already seen and is a retrospective comparison, not an untouched confirmation.

## Adversarial review

- A 24-hour lag is unavailable for many late next-day targets; check source release timestamps, not just positive lag length.
- Weighting probabilistic training by future realized cold can alter the conditional distribution. Predictor-based weights avoid claiming that a tilted distribution is the original demand distribution.
- A classifier's fitted probabilities are not honest stacking inputs for its own training rows. Use recorded historical out-of-sample probabilities.
- A base model selected using all 2024 outcomes cannot be described as selected in early 2024. The causal quarterly base-selection ledger explicitly prevents this.
- No architecture, loss function or tuning procedure can promise to beat RTE. Report failures and fallback performance rather than selecting the best historical test result.
- Higher upper bounds can reduce shortage misses while increasing interval width. Evaluate proper scores and both tails, not coverage alone.

## Primary sources checked

- [Time-series cross-validation](https://otexts.com/fpp3/toolbox.html): rolling training origins and genuinely future validation.
- [LightGBM regressor API](https://lightgbm.readthedocs.io/en/stable/pythonapi/lightgbm.LGBMRegressor.html): sample weights; installed APIs are checked separately.
- [Interval-format scoring](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1008618): interval score and WIS; a sparse quantile grid is not an exact full-distribution CRPS.
- [Conformalized quantile regression](https://arxiv.org/abs/1905.03222) and [adaptive conformal inference](https://arxiv.org/abs/2106.00170): methods and limits of transferring coverage claims to this dependent time series.
- [Threshold-weighted CRPS](https://scoringrules.readthedocs.io/en/latest/generated/scoringrules.twcrps_ensemble.html): emphasize high demand using a declared transformation, not arbitrary outcome multiplication presented as a proper score.

Implementation and run results are recorded separately. This plan does not assert that the expanded models have already been trained or that a winner exists.
