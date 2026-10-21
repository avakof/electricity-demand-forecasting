# Execution and interpretation notes

Run: `20261020T164443776583Z`.

These notes document the executed experiment. They do not change predictor code,
the run manifest, selected parameters, cached forecasts or evaluation rules.

## Neural training-origin limitation

N-BEATS, NHiTS, TFT, PatchTST and DLinear use the adapter defaults
`step_size=24` and `start_padding_enabled=False`. The matched panel begins at
2023-01-01 23:00 UTC, corresponding to 2023-01-02 00:00 Europe/Paris. Both tested
input lengths, 168 and 336 hours, are multiples of 24. Their training forecast
blocks therefore begin at 23:00 UTC: 00:00 local in winter and 01:00 in summer.
The actual forecast bridge begins at 08:00 local, after the latest eligible
observed interval starts at 07:00. Training origins and deployment origins have
different intraday phases.

The first recorded N-BEATS national fit illustrates this precisely:

| Item | Recorded or source-derived value |
| --- | --- |
| Fit origin | 2023-12-31 10:00 UTC / 11:00 Europe/Paris |
| Panel bounds | 2023-01-01 23:00 UTC–2023-12-31 06:00 UTC |
| Panel length | 8,720 hourly rows |
| Chronological validation tail | 168 hours |
| Training padding | None on the left; 48 masked hours on the right |
| Eligible training-window starts | 350 target blocks; first 2023-01-09 00:00 local, last 2023-12-24 00:00 local |
| Local training target-start phase | 133 blocks at 00:00; 217 at 01:00 |
| Inner validation target starts | 2023-12-24–29 at 08:00 local |
| First deployment forecast step | 2023-12-31 08:00 local |

Evidence: the saved first-fit metadata supplies the bounds and validation size.
The [adapter defaults](../../../src/research_neural.py) supply the fixed stride.
The installed versioned NeuralForecast implementation
[`_base_windows.py`](https://github.com/Nixtla/neuralforecast/blob/v1.7.7/neuralforecast/common/_base_windows.py)
removes the validation tail, applies no left padding, then unfolds training
windows from the panel start with that stride. Its inner-validation window is
right-aligned to the validation tail. GRU instead uses the hourly recurrent
window logic in
[`_base_recurrent.py`](https://github.com/Nixtla/neuralforecast/blob/v1.7.7/neuralforecast/common/_base_recurrent.py).

The stride is reconstructible from the hashed source; the saved tuned-parameter
subset and fit metadata do not explicitly include every resolved default.
Future runs should record the complete resolved configuration.

No model-class substitution, double inverse scaling, or delivery-index mismatch
was identified in the read-only audit. The forecast outputs are returned in MW;
the replay selects the exact next-day UTC intervals, including 23/24/25-hour days.
The origin-phase restriction is a limitation of training policy and affects
interpretation of architecture comparisons. Inner and outer validation expose
the poor forecasts; they do not repair the training restriction.

## Optimization and architecture conclusions

Neural optimization is capped at 20 or 40 steps in two compact configurations.
The neural models use target history only; the global panel arm shares parameters
across country and regional series rather than explicitly modeling cross-region
interactions. Tabular candidates use engineered calendar/history features, so the
cross-family leaderboard compares complete feature/model/training policies.

Poor validation scores establish weak performance for these executed
configurations. They do not establish that a neural architecture is generally
inferior, prove overfitting, or prove undertraining as the sole cause. Small
budgets make incomplete optimization plausible; feature differences and the
five window models' origin-phase mismatch are additional limitations. Training
trajectories are robust-scaled and validation trajectories are in MW; their
numerical gap is not a valid overfitting diagnostic.

A corrective future development experiment should use hourly training anchors
or explicitly align anchors with the information cutoff and DST policy, allow
adequate optimization, and record full resolved configurations. It must receive
a new run identity; current predictor code and recorded results are preserved.

## Wall-time interpretation and user pause

The host is an Intel Mac with 8 GB RAM and CPU-only neural training. Parallel
validation workers caused substantial paging and compression. Scheduling first
suspended workers, then terminated owned workers so unfinished quarterly replays
could restart sequentially from completed caches. The separate
`execution_workers.json` records these scheduling events.

The user explicitly requested a pause. The original notebook executor and kernel
were suspended at **2026-10-20 17:45:16.863363 UTC** and resumed after the user's
continue instruction at **2026-10-20 18:42:02.378001 UTC**. The recorded operation
was notebook 08, cell 6, during neural validation. The exact checkpoint remains
in `pause_checkpoint.json`.

Elapsed timers based on `perf_counter` include suspension time when a timed
operation spans a pause. Host contention also affected wall times. Consequently
the current timing tables describe this orchestrated execution, not a clean
hardware benchmark or a fair compute-efficiency ranking of model families.
Future cost comparisons should use controlled, uninterrupted runs and report CPU
time, wall time, peak memory and training budgets separately.
