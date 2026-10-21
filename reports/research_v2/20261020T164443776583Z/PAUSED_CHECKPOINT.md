# Research execution paused

User requested stopping and saving all work until an explicit continue command.

## Saved execution

- Run: `reports/research_v2/20261020T164443776583Z`.
- Notebook07 completed top-to-bottom and saved executed outputs.
- Notebook08 saved through its completed setup cells; paused in code cell6, neural validation.
- Executor PID72763 and kernel PID72825 received SIGSTOP and were verified stopped. They consume no training CPU while stopped. Exec session36306.
- Last active partial ledger: NHITS regional, trial1, 2024Q3. Completed quarterly predictions, fit records, selections and checksums remain saved.
- Completed tests:74; one optional integration skipped. See `.runtime/research_tests.log`.
- No final model selection/final test comparison completed. Do not claim final model superiority or completed notebook08.
- Machine-readable checkpoint: `.runtime/research_pause.json` and this run's `pause_checkpoint.json`.

## Implemented scope

Exactly two new notebooks:07 statistical/boosting/GAM;08 neural/uncertainty/comparison. Thirteen forecasting variants plus four references. NBEATS,NHITS,TFT,GRU,PatchTST,DLinear are actual NeuralForecast implementations; GAM is explicit sklearn additive splines+Ridge. Split conformal is calibration; CRPS is a distributional score. Country and regional arms share national target; regional neural arm is global univariate shared parameters, not cross-region multivariate inputs.

Four2024 chronological validation quarters; all daily forecast origins. Matched history Jan2023.11:00 Europe/Paris cutoff, interval-end+2h assumed demand release strictly before origin;48h bridge forecast, exact23/24/25-hour next-day scoring. Calibration Jan2–Jun29,2025; Jun30 buffer. Test July2025–June2026 already previously exposed, so retrospective. Original weather/RTE/demand vintages absent. No realized future weather as predictors. Compact neural20/40-step budgets and statistical56/84-day windows are explicit limitations.

Production-only pipeline/monitoring files removed; research risk/backtesting/fallback logic retained. No git commit/push performed. Raw input files unchanged.

## Latest cold-evening requirement

`src/research_peak_risk.py` implemented and tested. Separate specification frozen2026-10-20T17:12:20.462086+00:00; SHA256 a975b051603988ec75cedcd4c844561c9071d8195758b7a6ca0da4e261c5b953. Do not edit frozen predictor/peak sources midrun.

Realized temperature<5C at local18/19/20 diagnoses cold peaks. Actionable interval trigger uses past24h temperature known at origin plus known eveninghour, explicitly a proxy. Pooled versus conditional residual intervals selected on2024 only before test. Report upper-tail exceedances/severity, q95pinball, coverage,width,WIS, counts and day-block uncertainty. Cost ratios/multipliers are weighted-MW scenarios, not euro savings. Same policies and exact matched observations for candidate/RTE. Zero observed misses and bootstrapCI[0,0] do not establish zero future risk. Recommend saved validation-selected interval policy, not whichever test sensitivity looks best.

## Outstanding neural audit finding

Read-only audit found no class/scaling/48hbridge bug. Five window architectures train using stride24 from panel start, which anchors training targetstarts near local00winter/01summer, while deployment starts08local. NF inner chronological validation right-aligns windows and matches deployment. ActualfirstNBEATSfit: origin2023-12-31 10UTC, panel2023-01-01 23UTC..2023-12-31 06UTC,8720rows,168hvalidation. Source src/research_neural.py lines183–185 step_size24/start_paddingFalse; installed NF _base_windows.py unfolds from originalstart. GRU recurrent windows step1 differ. Default stride is visible in hashedsource, not explicit fit metadata. This is a training/deployment window-phase limitation to disclose; do not attribute weak results solely to short training, or conclude architecture inherently inferior. No source/config changes made. Discuss a new development-only experiment if correcting it; do not silently invalidate existing frozen caches.

## Resume procedure (only after user says continue)

1. Read this file and `.runtime/research_pause.json`; inspect actual PID commands/state before sending signals.
2. If same processes survive and timeout permits, SIGCONT executor+kernel. If paused too long, kill these stale task processes and rerun08; caches preserve completed quarters. Never start a second executor while stopped one could resume.
3. Scientific interpreter: `/Users/alivakof/Documents/Codex/python-data-science/.venv/bin/python`.
4. Restart08 when necessary: `python scripts/execute_research_notebooks.py --from-notebook 8 --to-notebook 8 > .runtime/research_notebook08.log 2>&1` using scientific interpreter. It resumes latest run. Do not regenerate notebooks (clears outputs). Do not launch parallel workers:8GB host previously swapped heavily. All old background workers were terminated.
5. Finish remaining neural validation arms/configurations, development baselines/history/retraining/seed sensitivities, input stress tests.
6. Freeze all23variantarms using2024 only; select peak interval policy before final replay. Finish calibration/test all27point policies; strict fixed-predictor conformal replay and uncertainty/CRPS/peak reports.
7. Run full unittest suite and strict `scripts/check_research_artifacts.py`; checker already reviewed, handles nullraw+fallback and multiplemethods/DST. No final artifact check yet.
8. Inspect final plots, export/read HTML previews, update README with measured final results/caveats. Notebook07 validation plot already visually checked.
9. Final answer with two notebook links, measured point and cold-interval findings, limits; user requests four concise phases (Propose,Adversarialaudit,Groundingwith3verifiedquestions,Synthesis/confidence/Delta). Do not expose chain-of-thought.

## Files and stability

Core newmodules research_data, research_models, research_neural, research_experiment, research_extensions, research_scores and config are hash-frozen in manifest. New peakmodule separately hash-frozen. Reporting/checker/docs may be corrected without modelretuning. Notebook generator `scripts/build_research_notebooks.py`; executor saves after each completedcell. `.research-deps` contains pinnedIntel-compatible NF1.7.7/PyTorch2.2.2 helpers; scientificenv not upgraded.

Current run selected parameter snapshot and exact completed trial-row count are in pause_checkpoint.json. Earlier own aborted runs20261020T163557194394Z and20261020T163949967105Z may be removed later; preserve currentrun. Graphify/impeccable changes predate this resumed execution; do not expand scope.
