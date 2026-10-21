# Development fit diagnostics

Saved selected-configuration forecasts from the predeclared 2024 folds only. No model was fitted and no calibration/test target values or forecast files were read.

Each fit-gap row compares the saved in-sample MAE of one fitted predictor with its raw forecasts on subsequent observations until the next fit or fold end. Fallback predictions are reported separately because they are not outputs of the same fitted predictor. Training MAE is resubstitution error and the later observations can have different regimes; this gap does not prove overfitting or provide a formal bias/variance decomposition. Models without a saved unscaled in-sample MAE have an explicitly unavailable gap; scaled neural training trajectories must not be substituted.

Quarter ranges and standard deviations describe observed seasonal/regime dispersion; quarters share overlapping training histories. They are not independent replicates or confidence intervals. Seed summaries include every recorded predeclared seed without selecting a lucky seed.

Fit groups: 368; completed model-quarter groups: 92; missing completed caches: 0; recorded seed rows: 3.
