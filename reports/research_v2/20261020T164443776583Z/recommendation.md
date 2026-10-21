# Research recommendation: national day-ahead demand

<!-- development-complete-policy-distinction -->

**Two frozen development choices.** Across all 27 predeclared policies, the development choice is `rte_bias_proxy` (2024 MAE 1,029.7 MW). Across the 23 independent model variants, the core primary is `gam__national` (2,065.9 MW). The supplied RTE baseline has development MAE 1,428.1 MW. Both choices were frozen before test replay. The complete-policy choice uses RTE assistance and is not an independent replacement. The conformal and conditional peak analyses centered on `gam__national` describe that independent candidate; they do not establish uncertainty calibration for another policy. On the same locked retrospective comparison pairs, their MAEs are 1,111.0 MW for `rte_bias_proxy`, 1,865.7 MW for `gam__national`, and 1,439.0 MW for supplied RTE. The preselected complete policy's MAE improvement over RTE is 22.79%.

## Improvement over RTE

The candidate selected using 2024 validation, **gam__national**, has historical-test MAE
**1,865.7 MW**, compared with **1,439.0 MW** for the supplied RTE proxy on identical eligible intervals.
The MAE improvement is **-29.65%**
(-426.7 MW). The paired 14-day moving-block
95% interval for percentage improvement is **[-48.39, -9.65]%**.
Its raw-forecast availability is 100.00%; fallback frequency is 0.00%.

## Uncertainty and failures

The past-out-of-sample residual distribution has empirical 90% interval coverage
**92.04%**, mean width **9,095 MW**, and empirical-distribution
CRPS **1,378.2 MW**. Consult uncertainty_segments.csv for cold-evening/hour/season counts.
This distribution is a calibrated empirical benchmark, not a conditional-coverage guarantee.
There were 0 failed fit events across development and evaluation experiments; convergence and fallback diagnostics are saved.
Worst-day causes remain unproven until inspected; weather-forecast error cannot be diagnosed without forecast vintages.

## Recommendation and risks

This is a **retrospective research comparison**, not a production implementation.
Retain RTE as the operational reference until authentic issue-time archives or prospective evidence establish availability and superiority.
The 2025–26 period was already explored; the compact two-configuration CPU search is not an exhaustive model-family comparison.
Original observation revisions, hourly aggregation and publication delays are assumptions. Future recorded weather was excluded from model inputs.
National consumption is a utility-portfolio proxy. Forecast improvements are not measured euro savings.
No serving, scheduling, live monitoring or procurement integration is included.


## Cold-evening upper-tail assessment

The validation-selected interval policy is pooled. On origin-proxy cold evenings it has 78 scored intervals, upper-bound exceedance 24.36%, and status **observed_upper_tail_exceeds_nominal_5pct**. This past-weather trigger is not a forecast of tomorrow's cold conditions. Inspect peak_risk_scorecard.csv for realized-cold diagnostic counts and day-block uncertainty. Scenario costs are weighted-MW risk proxies, not euro savings.


Realized cold-evening diagnosis for the preselected candidate: 66 intervals over 24 days; coverage 54.5%; upper-bound exceedance 45.5%; mean upper-bound miss severity when exceeded 2,540 MW. Status: observed_upper_tail_exceeds_nominal_5pct. These realized conditions are diagnostic labels, not pre-auction inputs.

<!-- recorded-research-execution-limitations -->

**Execution limitations.** Poor validation scores establish weak performance for these executed configurations. They do not establish that a neural architecture is generally inferior, prove overfitting, or prove undertraining as the sole cause. Small budgets make incomplete optimization plausible; feature differences and the five window models' origin-phase mismatch are additional limitations. Training trajectories are robust-scaled and validation trajectories are in MW; their numerical gap is not a valid overfitting diagnostic. Elapsed timers based on `perf_counter` include suspension time when a timed operation spans a pause. Host contention also affected wall times. Consequently the current timing tables describe this orchestrated execution, not a clean hardware benchmark or a fair compute-efficiency ranking of model families. See [execution notes](execution_notes.md) for the source-derived training-origin reconstruction and the recorded pause/scheduling events.
