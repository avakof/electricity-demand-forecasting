"""Build exactly two research notebooks; does not execute experiments."""
from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]


def markdown(text):
    return nbf.v4.new_markdown_cell(text.strip())


def code(text):
    return nbf.v4.new_code_cell(text.strip())


SETUP = '''
import os, sys, json
from pathlib import Path
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
ROOT = Path.cwd()
if not (ROOT / "config/research_experiment.json").exists():
    ROOT = ROOT.parent
assert (ROOT / "config/research_experiment.json").exists(), "Run from project/notebooks"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, Markdown
from src.common import plotting_style
from src.research_experiment import ResearchRun, POINT_FAMILIES, NEURAL_FAMILIES
plotting_style()
'''


def save(name, cells):
    notebook = nbf.v4.new_notebook(cells=cells)
    notebook.metadata.update(kernelspec={"display_name": "Project research (Python)", "language": "python", "name": "project-research"},
                             language_info={"name": "python", "version": "3.12"})
    nbf.validate(notebook)
    nbf.write(notebook, ROOT/"notebooks"/name)


def main():
    save("07_statistical_and_boosting_models.ipynb", [
        markdown('''# Statistical and boosting research

**Question:** can an honest next-day national forecast improve on the supplied RTE proxy?

This notebook trains and tunes ARIMA, SARIMA, SARIMAX, histogram gradient boosting,
LightGBM, quantile LightGBM and a penalized additive spline GAM. Notebook 08 adds
six neural architectures and performs the frozen final comparison.

The experiment is retrospective: original demand/weather/RTE issue vintages and
hourly aggregation lineage are unverified. The 2025–26 period has already been
inspected. No operational superiority or euro savings can be claimed.

Running this notebook creates a **new versioned run**, rather than overwriting
the legacy results or changing raw CSVs. The default compact CPU budget evaluates
two configurations per family on four full chronological validation quarters.
It is a bounded research comparison, not exhaustive optimization.'''),
        code(SETUP),
        markdown('''## 1. Freeze the forecast contract before fitting

Origin is 11:00 Europe/Paris on day D. Predict every actual interval of local D+1,
including 23/24/25-hour DST days. Demand becomes available at interval end plus
two hours **by assumption**, strictly before the cutoff. Tomorrow-temperature is
excluded. Supplied RTE is a retrospective benchmark, never verified issue-time data.

`compact` runs actual training, tuning and daily replay. Change budgets only in a
new run before examining its final comparison. The shared functions are in
`src/research_data.py`, `research_models.py`, and `research_experiment.py`.'''),
        code('''PROFILE = "compact"
RESUME_LATEST = os.environ.get("RESEARCH_RESUME_RUN", "0") == "1"
run = ResearchRun.latest(ROOT) if RESUME_LATEST else ResearchRun.create(ROOT, profile=PROFILE)
data = run.data
print("Research output directory:", run.path)
display(pd.Series(run.contract, name="frozen research assumptions"))'''),
        markdown('''## 2. Audit eligibility and compare both datasets fairly

The main national and regional-assisted arms use the same 2023 onward history.
National demand remains the target; regional sums do not replace it. Missing
evaluation targets remain missing. Peaks remain in the data.

Tabular regional assistance adds pre-cutoff regional histories and temperatures.
No regional RTE forecast exists in the supplied dataset.'''),
        code('''from src.research_data import data_summary, latest_eligible_time
display(data_summary(data))
display(pd.DataFrame({"feature_group": ["national", "regional-assisted"],
                      "feature_count": [len(data.columns("national")), len(data.columns("regional"))]}))
assert data.labels.actual_mw.isna().sum() == data.national.load_mw.isna().sum()
assert not set(data.columns("regional")) & {"actual_mw", "realized_temperature_c", "oracle_temperature_c"}
data_summary(data).to_csv(run.path / "tables/data_audit.csv", index=False)
display(data.features[["origin", "delivery_time", "max_history_available_at"]].head())'''),
        markdown('''## 3. Use chronological folds, then calibration, then comparison

Four 2024 quarters are the outer validation folds. Each contains daily forecast
origins; every fit additionally filters label publication timestamps. The first
fold has approximately 328 usable supervised days after feature warm-up, so the
new contract explicitly replaces the legacy one-year guard with nine months.

Calibration deliveries: 2 January–29 June 2025. June 30 is a boundary buffer:
its evening outcomes are unavailable at the first July 1 forecast origin.
Final historical comparison: July 2025–June 2026. End dates below are exclusive.
No shuffled cross-validation is used.'''),
        code('''split_rows = [dict(stage=f["name"], start=f["start"], end=f["end"]) for f in run.contract["validation_folds"]]
split_rows += [dict(stage=k, start=run.contract[k]["start"], end=run.contract[k]["end"]) for k in ("calibration", "test")]
splits = pd.DataFrame(split_rows)
display(splits)
fig, ax = plt.subplots(figsize=(11, 3))
for i, row in splits.iterrows():
    start, end = pd.Timestamp(row.start), pd.Timestamp(row.end)
    ax.barh(i, (end-start).days, left=start, color=("#236b8e" if i<4 else "#c28635" if i==4 else "#654a91"))
ax.set_yticks(range(len(splits)), splits.stage)
ax.xaxis_date(); ax.set_title("Delivery-date blocks; publication cutoffs also apply")
fig.autofmt_xdate(); plt.tight_layout(); plt.show()'''),
        markdown('''## 4. Inspect the statistical and tabular candidates

ARIMA/SARIMA/SARIMAX use genuine statsmodels implementations. Compact statistical
fits use explicitly recent 56/84-day windows, small daily seasonal orders and
bounded optimizer iterations; convergence is recorded, not assumed.

The separate gradient-boosting model is sklearn histogram boosting. LightGBM is
another implementation. Quantile LightGBM's median is scored as its point
forecast; its 5th/95th quantiles are fitted only after parameter selection.

The GAM is independent penalized B-splines plus Ridge, explicitly **not pyGAM**.
Its smoothing is selected on the same chronological folds. Model objectives
(mean, median or quantile) are recorded rather than treated as interchangeable.'''),
        code('''from src.research_models import MODEL_REGISTRY, parameter_grid
display(pd.DataFrame(MODEL_REGISTRY).T)
candidate_parameters = {family: parameter_grid(family, "fast") for family in POINT_FAMILIES}
display(pd.DataFrame([{"family": family, "trial": i, "parameters": params}
                      for family, candidates in candidate_parameters.items() for i, params in enumerate(candidates)]))'''),
        markdown('''## 5. Train, validate and tune the statistical models

Each daily forecast starts from the latest released observation. The unobserved
remainder of day D is forecast before tomorrow's intervals are selected.
SARIMAX uses known future calendar regressors covering the entire bridge.
Between monthly parameter fits, statistical states consume only newly released
observations. A failed fit is logged and falls back until the next scheduled fit.'''),
        code('''statistical_trials = run.tune(["arima", "sarima", "sarimax"])
display(statistical_trials.tail(12))'''),
        markdown('''## 6. Train and tune boosting, quantile boosting and GAM

Both national-only and regional-assisted arms use identical delivery dates.
Imputation, spline construction and any scaling are fitted on each training slice.
Automatic random early stopping is disabled for tabular models; the external
chronological search selects capacity and iteration count.'''),
        code('''tabular_trials = run.tune(["hist_gradient_boosting", "lightgbm", "lightgbm_quantile", "gam"])
selected_point = json.loads((run.path / "selected_parameters.json").read_text())
display(pd.DataFrame(selected_point).T)'''),
        markdown('''## 7. Diagnose stability, overfit and underfit before touching final outcomes

Compare all fold scores, raw availability and fallback fractions. A large
training–validation gap can suggest overfitting; poor performance in both can
also reflect poor inputs or distribution changes. Fold variation is not a formal
bias–variance decomposition. Optimizer nonconvergence is reported separately.

The final comparison is deliberately deferred until notebook 08 has finished
neural-model validation and all policies are frozen.'''),
        code('''trials = pd.read_csv(run.path / "tables/tuning_trials.csv")
trials = trials.loc[trials.family.isin(POINT_FAMILIES)]
summary = trials.groupby(["model", "trial"]).agg(MAE_MW=("MAE_MW", "mean"),
           fold_std_MW=("MAE_MW", "std"), raw_availability=("raw_availability", "mean"),
           fallback_fraction=("fallback_fraction", "mean")).reset_index()
display(summary.sort_values("MAE_MW"))
best = summary.loc[summary.groupby("model").MAE_MW.idxmin()].sort_values("MAE_MW")
fig, ax = plt.subplots(figsize=(11, 4))
ax.barh(best.model, best.MAE_MW, xerr=best.fold_std_MW.fillna(0), color="#236b8e")
ax.invert_yaxis(); ax.set_xlabel("Mean quarterly validation MAE (MW); bars: fold standard deviation")
ax.set_title("Development results; final comparison remains unopened")
plt.tight_layout(); fig.savefig(run.path / "figures/point_validation.png", bbox_inches="tight"); plt.show()
fit_records = [r for p in (run.path / "fits").glob("*.json") for r in json.loads(p.read_text())]
fit_diagnostics = pd.DataFrame([{**{k: v for k, v in r.items() if k not in ("metadata", "params", "traceback")},
                               "converged": (r.get("metadata") or {}).get("converged")} for r in fit_records])
display(fit_diagnostics.groupby("model").agg(fits=("status", "size"), failures=("status", lambda s: s.ne("fit").sum()),
             total_fit_seconds=("elapsed_seconds", "sum")))
fit_diagnostics.to_csv(run.path / "tables/point_fit_diagnostics.csv", index=False)
print("Continue with notebook 08 using this same versioned run:", run.path)'''),
    ])

    save("08_deep_learning_and_uncertainty.ipynb", [
        markdown('''# Neural forecasting, uncertainty and the frozen comparison

This notebook trains the real N-BEATS, NHiTS, TFT, GRU, PatchTST and DLinear
implementations, then compares all statistical, tabular and neural variants.
Run notebook 07 first. This notebook resumes the same versioned research run.

Split conformal is a calibration method. CRPS is a scoring rule. Neither is
counted as a separate trained forecasting architecture.'''),
        code(SETUP),
        code('''run = ResearchRun.latest(ROOT)
data = run.data
assert (run.path / "selected_parameters.json").exists(), "Execute notebook 07 first"
print("Resuming research run:", run.path)
from src.research_neural import backend_versions, neural_configurations
versions = backend_versions()
display(pd.Series(versions, name="executed neural backend"))
from src.common import write_json
write_json(run.path / "neural_environment.json", versions)'''),
        markdown('''## 1. Define what the neural comparison actually tests

The compatible pinned implementation is NeuralForecast 1.7.7 on PyTorch 2.2.2.
The six neural variants deliberately use target history only for a comparable
architecture experiment. The regional arm trains a global shared-parameter
univariate panel (country plus 12 regions); it is not a multivariate model of
cross-region interactions. NHiTS/TFT/GRU can support covariates in separate
extensions, but this notebook does not claim to test those capabilities.

Forecast 48 consecutive UTC steps from the last eligible observation, then score
only tomorrow's actual 23/24/25 intervals. The training loss covers the bridge;
outer selection uses delivery-day MAE. Internal early stopping uses the historical
chronological tail, never outer validation or test outcomes.

Missing training labels carry mask zero. Missing predictor placeholders use
causal per-series forward fill, leading zero only. Some architectures do not
consume input masks directly; this imputation convention is an explicit limitation.
Evaluation truth remains missing and is never interpolated.

**Training-origin limitation:** N-BEATS, NHiTS, TFT, PatchTST and DLinear
use `step_size=24` and `start_padding_enabled=False`. The matched panel starts
at 23:00 UTC, so their training forecast blocks also begin at 23:00 UTC:
00:00 local in winter and 01:00 in summer. The replay bridge begins at 08:00
local. The library right-aligns its inner validation tail, so validation and
deployment expose this mismatch rather than correcting it. GRU uses hourly
recurrent windows and does not share this exact restriction. This is a limitation
of the executed training policy, not a target-timestamp or inverse-scaling error.
The fixed stride is visible in the hashed adapter source; it is not an explicit
field in the saved trial parameters. A corrected future development run should
use hourly anchors or explicitly align anchors with the cutoff and DST policy.

| Architecture | Development hypothesis |
| --- | --- |
| N-BEATS | Learned trend/seasonal basis blocks capture recurring load structure. |
| NHiTS | Multi-rate interpolation improves the direct multi-step trajectory. |
| TFT | Temporal attention helps under the explicitly history-only input restriction. |
| GRU | A compact recurrent representation is sufficient for demand history. |
| PatchTST | Attention over patches improves the long-context forecast. |
| DLinear | Simple decomposition and linear projection compete with deeper models. |

DLinear is the additional neural control, not a presumed winner; its motivation
is described in the [original paper](https://arxiv.org/abs/2205.13504).
The executed architectures use [versioned official implementations](https://github.com/Nixtla/neuralforecast/tree/v1.7.7/neuralforecast/models).
The cross-family leaderboard compares complete feature/model policies; it does
not isolate a pure architecture effect between tabular and neural families.'''),
        code('''neural_parameters = {family: neural_configurations(family) for family in NEURAL_FAMILIES}
display(pd.DataFrame([{"family": family, "trial": i, "parameters": p}
                      for family, choices in neural_parameters.items() for i, p in enumerate(choices)]))
from src.common import local_day_grid, forecast_origin
bridge_examples = []
for day in ("2025-01-02", "2025-03-30", "2025-10-26"):
    origin = forecast_origin(day, run.contract)
    latest = data.history(origin).index[-1]
    grid = local_day_grid(day)
    bridge_examples.append({"delivery_day": day, "origin": origin, "latest_observation": latest,
             "hours": len(grid), "first_step": int((grid[0]-latest)/pd.Timedelta(hours=1)),
             "last_step": int((grid[-1]-latest)/pd.Timedelta(hours=1))})
display(pd.DataFrame(bridge_examples))'''),
        markdown('''## 2. Train and tune all six neural architectures

Every architecture gets two initial configurations, all four 2024 quarters and
both national/global-panel arms. Notebook execution is sequential by default;
the recorded execution may use bounded independent CPU workers after measured memory checks.
The compact 20/40-step caps are deliberately small: they establish a reproducible
baseline experiment, not an assertion that a neural family has reached its best
performance. Nonfinite forecasts or training failures are retained in diagnostics;
there is no substitute architecture hidden behind a model name.

Scores therefore compare these bounded feature/model/training policies. The
training-origin restriction above and the history-only inputs prevent a general
claim that a poorly scoring architecture is inferior to the tabular alternatives.
Undertraining is plausible under these caps; it is not established as the sole
cause of error.'''),
        code('''neural_trials = run.tune(NEURAL_FAMILIES)
display(pd.DataFrame(json.loads((run.path / "selected_parameters.json").read_text())).T)
from src.metrics import scorecard
development_baselines = run.baselines("2024-01-01", "2025-01-01", "development_reference")
baseline_score_path = run.path / "tables/development_baselines.csv"
if baseline_score_path.exists():
    # Preserve the exact development artifact fingerprinted before final replay.
    development_baseline_scores = pd.read_csv(baseline_score_path)
else:
    development_baseline_scores = scorecard(development_baselines, data.labels)
    development_baseline_scores.to_csv(baseline_score_path, index=False)
display(development_baseline_scores)'''),
        markdown('''## 3. Check history length, refitting, seeds and input failures

Development-only sensitivities compare the best national tabular model with a
365-day window, weekly refitting and full national history. These are explicitly
separate from the matched-history comparison. Repeat the best validation neural
candidate with all three predeclared seeds; report every result rather than
selecting a lucky seed. Stress tests perturb available inputs, not outcomes.
These diagnostics do not change the predeclared main monthly-refit policy.'''),
        code('''from src.research_extensions import development_sensitivities, stress_inputs
from src.research_peak_risk import freeze_peak_spec
display(pd.Series(freeze_peak_spec(run), name="cold-evening interval specification"))
sensitivities = development_sensitivities(run)
display(sensitivities)
display(stress_inputs(run))'''),
        markdown('''## 4. Inspect neural learning curves with the correct units

The library's training trajectory is robust-scaled; its validation trajectory is
in MW. They are plotted on separate axes. Cross-model comparison and selection
use the external daily replay scores. Different seeds and quarters measure
stability, not a formal bias–variance decomposition. High validation error alone
does not identify overfit or underfit. Training progress at the step cap, an
appropriately comparable training score, input differences and the training-origin
mismatch must be considered before assigning a cause.'''),
        code('''fit_records = [r for p in sorted((run.path / "fits").glob("*.json")) for r in json.loads(p.read_text())]
neural_records = [r for r in fit_records if (r.get("metadata") or {}).get("implementation") == "neuralforecast"]
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
seen = set()
learning_curve_examples = []
for record in neural_records:
    family = record["metadata"]["family"]
    if family in seen: continue
    seen.add(family)
    learning_curve_examples.append({"family": family, "model": record.get("model"),
        "phase": record.get("phase"), "fit_origin": record.get("origin"),
        "seed": record.get("seed"), "step_cap": record.get("params", {}).get("max_steps"),
        "interpretation": "first saved example; selected-variant status was not used"})
    for ax, name in zip(axes, ("train_trajectory", "validation_trajectory")):
        values = record["metadata"].get(name, [])
        if values:
            array = np.asarray(values)
            ax.plot(array[:, 0], array[:, 1], label=family)
axes[0].set(title="Training trajectory", xlabel="Optimizer step", ylabel="Scaled MAE")
axes[1].set(title="Inner chronological validation", xlabel="Optimizer step", ylabel="MAE (MW)")
for ax in axes: ax.legend(fontsize=8)
fig.suptitle("Illustrative first saved fits; not a summary of selected variants", fontsize=11)
plt.tight_layout(rect=(0, 0, 1, .93)); fig.savefig(run.path / "figures/neural_learning_curves.png", bbox_inches="tight"); plt.show()
display(pd.DataFrame(learning_curve_examples))
from scripts.development_fit_diagnostics import build_diagnostics
development_diagnostics = build_diagnostics(run.path, ROOT)
display(development_diagnostics["fit_gaps"])
display(development_diagnostics["quarter_dispersion"])
display(development_diagnostics["seed_stability"])
from IPython.display import Image as DisplayImage
for path in development_diagnostics["figure_paths"]:
    display(DisplayImage(filename=str(path)))
display(Markdown(development_diagnostics["notes_path"].read_text()))'''),
        markdown('''## 5. Freeze every candidate before final replay

Selection requires four validation folds and adequate raw forecast availability.
Two choices have different scopes: the complete policy competes across all 23
model variants and four predeclared baselines, including RTE bias correction;
the core primary is the best independent model. Both use 2024 development MAE
and are frozen before final replay. The manager recommendation leads with the
complete policy, so a weak independent model does not hide a stronger simple
RTE-assisted policy. The historical comparison remains previously exposed;
this seal cannot make it pristine again.'''),
        code('''from scripts.freeze_research_policy import freeze_policy
complete_policy = freeze_policy(run.path)
print("Complete policy selected on 2024 only:", complete_policy["selected_policy"])
selected = run.freeze()
from src.research_peak_risk import select_peak_intervals
display(select_peak_intervals(run))
frozen = json.loads((run.path / "frozen_selection.json").read_text())
print("Primary candidate selected on 2024 only:", frozen["primary_candidate_from_validation"])
display(pd.DataFrame(selected).T)
parallel_plan = run.path / "parallel_final_plan.json"
if parallel_plan.exists():
    import time
    worker_names = json.loads(parallel_plan.read_text())["workers"]
    print("Waiting for disjoint frozen replay workers:", worker_names)
    while not all((run.path / f"final_worker_{name}.done.json").exists() for name in worker_names):
        failures = list(run.path.glob("final_worker_*.error.json"))
        if failures:
            raise RuntimeError(f"Final worker failed; inspect {failures}")
        time.sleep(5)
outputs = run.final_replay()
display(pd.DataFrame({phase: frame.groupby("model").size() for phase, frame in outputs.items()}))'''),
        markdown('''## 6. Add a genuinely fixed-predictor split-conformal comparator

A continuous replay fits once before January 2 calibration and retains the same
fitted parameters through the final comparison. It uses only calibration errors
published before the first July 1 origin. State/history updates from released
observations remain permitted. This frozen comparator is distinct from the main
monthly-refitted policies. Dependent, shifting demand does not establish classical
exchangeability or conditional coverage guarantees.

Conformal bounds target total interval coverage. Their q05/q95 schema names do
not establish equal-tailed quantiles; the center is not guaranteed to be a
conditional median. No CRPS is assigned to an interval alone.'''),
        code('''from src.research_scores import strict_conformal, build_comparison
conformal = strict_conformal(run, selected)
fixed_point_scores = scorecard(conformal, data.labels)
fixed_point_scores.to_csv(run.path / "tables/fixed_conformal_point_scorecard.csv", index=False)
display(fixed_point_scores)
display(pd.Series(json.loads((run.path / "tables/conformal_report.json").read_text())))
point_scores, uncertainty_scores = build_comparison(run, outputs, conformal)
from src.research_peak_risk import analyze_peak_risk
peak_tables = analyze_peak_risk(run, outputs)
display(peak_tables["scorecard"])
display(Markdown((run.path / "peak_report.md").read_text()))
critical = peak_tables["scorecard"].loc[peak_tables["scorecard"].segment.eq("realized_cold_evening")]
from scripts.research_comparison_plots import predeclared_plot_roles, plot_cold_evening
plot_roles = predeclared_plot_roles(run.path)
cold_plot_points = plot_cold_evening(peak_tables["scorecard"], plot_roles,
    run.path / "figures/cold_evening_upper_tail.png")
from IPython.display import Image as ComparisonImage
display(ComparisonImage(filename=str(run.path / "figures/cold_evening_upper_tail.png")))
display(cold_plot_points[["role", "model", "method", "plot_status", "interval_nobs", "ci14_status"]])
primary = frozen["primary_candidate_from_validation"]
choice = peak_tables["interval_selection"].loc[lambda f: f.model.eq(primary) & f.selected, "candidate"].iloc[0]
method = "peak_conditioned_empirical_residuals" if choice == "conditional" else "peak_pooled_empirical_residuals"
cold = critical.loc[critical.model.eq(primary) & critical.method.eq(method)].iloc[0]
diagnosis = (chr(10)*2 + f"Realized cold-evening diagnosis for the preselected candidate: {int(cold.interval_nobs)} intervals "
             f"over {int(cold.ndays)} days; coverage {cold.coverage90:.1%}; upper-bound exceedance {cold.upper_exceedance_rate:.1%}; "
             f"mean upper-bound miss severity when exceeded {cold.upper_exceedance_conditional_severity_MW:,.0f} MW. "
             f"Status: {cold.upper_guardrail_status}. These realized conditions are diagnostic labels, not pre-auction inputs." + chr(10))
recommendation = run.path / "recommendation.md"
if "Realized cold-evening diagnosis" not in recommendation.read_text():
    with recommendation.open("a") as stream: stream.write(diagnosis)
display(point_scores)
display(uncertainty_scores)'''),
        markdown('''## 7. Read point accuracy and uncertainty as different questions

Point table: common eligible observations, MAE/RMSE/bias/WAPE, RTE-relative gain,
raw forecast availability and fallback frequency. Daily energy/peak scores exclude
incomplete days explicitly. Cold-evening conditions are realized diagnostics.

Uncertainty table: native three-quantile LightGBM; empirical distributions from
past out-of-sample errors; and the fixed conformal interval. CRPS is calculated
exactly for the specified empirical residual distribution, not invented from a
three-quantile interval. Native quantiles receive pinball and interval scores.
Hourly intervals do not automatically give daily-energy or peak intervals.

Cold evenings receive their own upper-tail and interval checks. The added peak
analysis freezes definitions before final scoring, evaluates counts and multi-day
uncertainty, and compares a peak-conditioned residual interval chosen from
development data. Cold-price multipliers and asymmetric loss ratios are sensitivity
assumptions in normalized risk units, not measured euro savings. Pre-auction rules
use only a pre-cutoff past-temperature proxy; realized cold conditions are diagnostic labels.'''),
        code('''from scripts.research_comparison_plots import plot_overall_comparison
overall_plot_points = plot_overall_comparison(point_scores, peak_tables["scorecard"], plot_roles,
    run.path / "figures/final_comparison.png")
display(ComparisonImage(filename=str(run.path / "figures/final_comparison.png")))
display(overall_plot_points[["role", "model", "method", "plot_status", "interval_nobs"]])
segments = pd.read_csv(run.path / "tables/test_segments.csv")
display(segments.loc[segments.segment_type.eq("cold_evening") & segments.segment.eq("True")])
display(pd.read_csv(run.path / "tables/bootstrap.csv"))
display(pd.read_csv(run.path / "tables/worst_days.csv").head(15))
from scripts.research_failure_cases import analyze_failures
failure_cases = analyze_failures(run.path, ROOT)
display(failure_cases["catalogue"][["case_id", "local_day", "criterion_rank", "scored_hours", "expected_hours", "MAE_MW", "absolute_daily_energy_error_MWh", "cold_upper_miss_max_MW"]])
from IPython.display import Image as NotebookImage
for image_path in failure_cases["figure_paths"]:
    display(NotebookImage(filename=str(image_path)))'''),
        markdown('''## 8. Defend the result and preserve its limitations

Block bootstrap groups paired losses into contiguous multi-day blocks; hourly
errors are not treated as independent replicates. Report counts, failures,
nonconvergence, uncertainty by season/hour/cold regime and computational cost.

Interpret architecture rankings only within the executed history-only neural
inputs, 20/40-step caps and the five window models' training-origin mismatch.
These scores support screening these configurations; they do not establish a
general neural-versus-tabular result. Reported wall times also include host
contention, worker suspensions and the user-requested pause recorded in
`execution_notes.md`; they are not a clean compute-efficiency benchmark.

The recommendation addresses improvement over the supplied RTE proxy, uncertainty
and failures, and the evidence still missing. This is a research project: no
serving endpoint, production scheduler, procurement integration or live monitoring
is created. Raw data and prediction-before-outcome ledgers are retained.

The manager report distinguishes the development-selected complete policy from
the independently selected model, and uses their predeclared uncertainty methods.
It preserves the original model-focused recommendation as a separate artifact.'''),
        code('''display(pd.read_csv(run.path / "tables/fit_diagnostics.csv").groupby("model").agg(
    fit_events=("status", "size"), failed_events=("status", lambda s: s.ne("fit").sum()),
    total_fit_seconds=("elapsed_seconds", "sum")))
from scripts.write_manager_recommendation import write_manager_recommendation
manager_report = write_manager_recommendation(run.path)
display(Markdown(manager_report.read_text()))
print("Original model-focused report retained:", run.path / "recommendation.md")
print("All research artifacts:", run.path)'''),
    ])


if __name__ == "__main__":
    main()
