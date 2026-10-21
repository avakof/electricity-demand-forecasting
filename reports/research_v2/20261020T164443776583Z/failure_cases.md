# Measured failure cases: rte_bias_proxy

Policy identity comes from development_policy_selection.json; these cases do not select or retune a policy. Bounds use the predeclared fixed_past_out_of_sample_residual_distribution.

| Local day | Diagnostic case | Scored/expected hours | MAE MW | Absolute daily energy error MWh | Largest cold upper miss MW |
|---|---|---:|---:|---:|---:|
| 2025-11-01 | worst policy MAE | 24/24 | 2,877 | 69,058 | not applicable |
| 2025-11-24 | worst absolute daily energy | 24/24 | 2,748 | 65,961 | not applicable |
| 2025-12-24 | worst cold evening upper miss | 24/24 | 2,249 | 44,271 | 2,733 |

Each criterion's global worst day and the chosen day's rank are saved. If criteria overlap, the next highest-ranked unused day supplies a distinct case; a cold-event fallback is labeled explicitly. Energy and daily peak errors require complete targets/predictions and assume hourly mean power. Missing targets remain missing. Temperature, holiday status, preceding-day changes and errors are measured observations, not verified causes. No archived forecast weather exists here to establish a weather-forecast miss. Cold shading uses realized temperature after delivery. Causes remain unknown; no financial savings are inferred.
