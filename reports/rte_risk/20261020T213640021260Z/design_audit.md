# Read-only design audit

Scope: `src/rte_feature_lab.py`, `src/rte_learned_features.py`, `src/rte_base_replay.py`, `src/rte_risk_lab.py`, and `src/rte_risk_uncertainty.py`. Predictive modules and configuration were already frozen during this review. No predictive code, parameters, selections, or data were changed.

## Leakage and historical availability

No additional future-label leakage or invalid adaptive base-model selection was found in the reviewed paths. Historical demand and weather features are restricted to the assumed publication cutoff; target-minus-24-hour values are masked when unavailable. Learned features use earlier origins and released labels, with preprocessing fitted on training rows. The base-model selector uses released earlier-quarter outcomes, rather than using the full-year winner inside the same year's earlier quarters. Later scheduled fits can incorporate earlier evaluation-period observations only once their assumed release timestamps precede the fit.

Evidence: `src/rte_feature_lab.py:37` and `src/rte_feature_lab.py:69`; `src/rte_learned_features.py:122` and `src/rte_learned_features.py:234`; `src/rte_base_replay.py:188`; `src/rte_risk_lab.py:245`.

This finding does **not** establish operational availability. The supplied `rte_forecast_d1_mw` is assumed available at the forecast origin; original RTE issue vintages are unverified. Historical demand and observed weather also rely on documented publication/revision assumptions. Timestamp-gate correctness under those assumptions cannot demonstrate that these exact historical values were available to a real trading desk. The previously inspected test period also remains retrospective evidence, rather than an untouched confirmation set.

## Material interpretation limits

1. **Probability selection optimizes overall WIS.** The probability winner minimizes unweighted development WIS among the three quantile-LightGBM feature arms. Its selection has no cold-WIS or upper-tail guardrail. It should be described as the best overall probability score among these arms, not as a demonstrated optimum for cold-evening uncertainty. Evidence: `src/rte_risk_lab.py:375` and `config/rte_risk_experiment.json:83`.

2. **The fixed engineered comparator retains learned-probability training weights.** Its prediction columns exclude the monthly learned-feature generators, but training weights still depend on their historical cold probabilities. Those weights are therefore not necessarily solely a function of the fixed comparator's explicit predictors. Its uncalibrated estimates need not equal the unweighted conditional quantiles given those predictors. This is an estimand limitation, not identified future-label leakage. Split-conformal and CQR bounds remain separately evaluated; their advertised absence of temporal coverage guarantees must be retained. Evidence: `src/rte_risk_uncertainty.py:72`, `src/rte_risk_lab.py:257`, and `src/rte_risk_uncertainty.py:97`.

3. **Held-out calibration has no cold-evening support.** April 2–June 29, 2025 contains zero realized cold-evening observations. Aggregate calibration performance cannot substantiate winter-conditional coverage. The development segment contains 102 cold-evening hours across 41 days; that is development support, not substitute held-out calibration evidence. Evidence: `config/rte_risk_experiment.json:116` and `config/rte_risk_experiment.json:117`.

4. **The time contract uses Paris civil time.** The origin is 11:00 Europe/Paris on D. It is one hour before local noon throughout the year, but two hours before literal 12:00 fixed CET during summer. The implementation remains before either interpretation of the stated deadline; it does not use every observation that could arrive before a literal fixed-CET summer deadline. Evidence: `src/common.py:50` and the executed timezone check below.

5. **The complete family comparison uses explicitly labeled hybrids.** All 23 original variants across 13 families supply causal out-of-sample forecasts to newly trained LightGBM combiners with RTE features. This does not mean that every original neural/statistical architecture received native RTE inputs, new cold-weighted base-model training, or a fresh architecture search. Existing compact-training and base-model limitations persist. Evidence: `src/rte_risk_lab.py:77`, `src/rte_base_replay.py:188`, and `config/rte_risk_experiment.json:72`.

## Executed timezone verification

The check called the project's `forecast_origin()` and `local_day_grid()` with the frozen research contract, then converted each origin to both Europe/Paris and a fixed UTC+01:00 timezone. Results:

| Delivery date | Origin in Europe/Paris | Origin in fixed CET | Delivery intervals |
|---|---|---|---:|
| 2025-01-15 | 2025-01-14 11:00 +01:00 | 2025-01-14 11:00 +01:00 | 24 |
| 2025-07-15 | 2025-07-14 11:00 +02:00 | 2025-07-14 10:00 +01:00 | 24 |
| 2025-03-30 | 2025-03-29 11:00 +01:00 | 2025-03-29 11:00 +01:00 | 23 |
| 2025-10-26 | 2025-10-25 11:00 +02:00 | 2025-10-25 10:00 +01:00 | 25 |

The actual DST delivery grids retain unique UTC intervals, including the repeated autumn local hour. This audit reviewed design and executed the timezone check; it does not replace final artifact verification or certify the availability of historical source vintages.
