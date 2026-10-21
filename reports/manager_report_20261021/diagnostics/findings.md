# Descriptive failure diagnostics

The comparison uses **8,759 matched hours** for the development-selected point model, supplied-RTE reference policy and prior development-selected RTE-assisted model. **1 missing target** is excluded; no targets are interpolated. The critical segment contains **66 cold-evening hours across 24 days and 1 represented DJF winter**. These are retrospective diagnostics, not new model-selection results.

Signed errors are **prediction minus actual demand**: negative values indicate underforecasting. The selected model's all-hour bias is **+7.0 MW**; its cold-evening bias is **-1,089.6 MW**. Its cold-evening underforecast frequency is **77.3%**. Counts and the same statistics for both references are in `findings.json`.

The highest selected-model monthly MAE is **1,016.7 MW in 2025-12** across **744 hours**. The largest local-hour aggregate MAE is **913.4 MW at 17:00 Europe/Paris**, across **365 observations**. See `monthly_mae.csv` and `local_hour_mae.csv`; grouping uses local civil time, retaining both autumn repeated-hour observations.

The five worst selected-model days are ranked by daily MAE over matched observed hours in `worst_five_days.csv`; `worst_five_days_hourly.csv` preserves their individual observations. Complete-day energy and peak errors are withheld when a target hour is absent; partial-hour sums are labeled explicitly in `daily_errors.csv`.

| Local day | Selected MAE (MW) | RTE MAE (MW) | Prior MAE (MW) | Cold-evening hours |
| --- | ---: | ---: | ---: | ---: |
| 2025-12-01 | 2,652.3 | 3,058.4 | 1,983.4 | 0 |
| 2025-11-29 | 2,268.7 | 1,011.3 | 2,027.0 | 0 |
| 2026-01-05 | 2,094.0 | 2,502.6 | 2,084.3 | 3 |
| 2025-11-01 | 2,090.5 | 3,902.8 | 1,769.1 | 0 |
| 2026-02-01 | 1,999.7 | 784.1 | 1,656.1 | 0 |

The worst cold day by selected-model MAE over its cold-evening hours is **2025-12-24**, with **4,165.3 MW** MAE across **3 cold hours**. `worst_cold_day_hourly.csv` contains the full matched daily trajectory, actual demand, temperature, all three point forecasts, selected-model empirical intervals, and the **separate** native quantile model's central/asymmetric endpoints. These intervals must not be represented as one model's interchangeable uncertainty estimates.

Temperature and calendar columns describe measured conditions. They do **not** establish a weather-forecast miss or another causal explanation: historical weather/RTE issue vintages are unavailable. Cold support is sparse, and inspecting worst days does not create independent confirmation evidence. Forecast errors and shortage/energy quantities here are physical-unit diagnostics, not euro savings.

Reproduce with the scientific Python environment: `python reports/manager_report_20261021/diagnostics/build_diagnostics.py`. Only files in this diagnostics directory are written. `provenance.json` records source hashes; all existing experiment artifacts remain unchanged.
