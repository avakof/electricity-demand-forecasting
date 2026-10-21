# Peak-risk extension

The upper tail is evaluated separately during 18:00–20:00 local delivery hours. Realized temperature below 5 C defines retrospective diagnostics and scenario loss weights only. The decision trigger uses past_temperature_mean_24h below 5 C at the forecast origin; this is a past-weather proxy, not tomorrow's forecast temperature.

Pooled versus conditional empirical intervals were selected on 2024 H2, using only released 2024 H1 out-of-sample residuals. The final residual pools use January 2–June 29, 2025 outcomes released before the first test forecast. Sparse pools back off through the frozen hierarchy; counts and reasons are saved. Main point forecasts were not retuned.

The nwinters field counts represented meteorological DJF winters, not complete seasonal cycles. Inspect upper exceedance rates, severity, coverage, width and day-block confidence intervals together. A rate above 5% signals an observed upper-tail shortfall; small segments remain inconclusive. Conformal bounds do not promise equal tails or a median center. Separate hourly intervals are not simultaneous daily-peak or daily-energy intervals.

The cost table is a scenario analysis of weighted MW errors using hypothetical shortage/surplus ratios. Both the candidate and RTE empirical reference receive the same median/q95 policy and the same origin-only trigger on exactly matched observations. It is not a euro-cost model, realized financial savings, or an optimal procurement policy. A q95 choice corresponds to a 19:1 asymmetric loss only under additional distribution and decision assumptions.
