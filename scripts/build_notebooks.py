"""Author the notebook workflow with nbformat. Never execute the legacy notebook."""
from pathlib import Path
import shutil
import textwrap

import nbformat as nbf

ROOT=Path(__file__).resolve().parents[1]
N=ROOT/"notebooks"
legacy=N/"legacy/01_data_audit_original.ipynb"
if not legacy.exists():
    legacy.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(N/"01_data_audit.ipynb",legacy)


def md(text): return nbf.v4.new_markdown_cell(textwrap.dedent(text).strip())
def code(text): return nbf.v4.new_code_cell(textwrap.dedent(text).strip())


SETUP='''
%matplotlib inline
from pathlib import Path
import sys, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, Markdown
ROOT = next(p for p in (Path.cwd(), *Path.cwd().parents) if (p / "config/forecast_contract.json").exists())
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from src.common import load_contract, prepare_directories, plotting_style, write_json, fingerprint
contract = load_contract(ROOT)
prepare_directories(ROOT)
plotting_style()
pd.set_option("display.max_columns", 12)
pd.set_option("display.max_rows", 16)
pd.set_option("display.float_format", lambda x: f"{x:,.3f}")
'''


def save(name,cells):
    nb=nbf.v4.new_notebook(cells=cells,metadata={"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},"language_info":{"name":"python"},"project":{"regime":"retrospective; operational gate requires verified archives","contract":"config/forecast_contract.json"}})
    nbf.validate(nb)
    nbf.write(nb,N/name)


save("01_data_audit.ipynb",[
md('''# 01 · Forecast contract, availability and data audit
## Summary
This notebook fixes the research contract before model comparison, preserves raw inputs, produces the data dictionary and availability register, and reports defects without interpolating targets or deleting peaks. Read the computed audit and gate below before interpreting later scores.

The original conversational notebook is preserved in `notebooks/legacy/01_data_audit_original.ipynb`.
## Context and methods
The operational information cutoff is **11:00 Europe/Paris on D−1**; delivery is the complete next local day. Two-hour publication delay after interval end is an **assumption**, not measured metadata. The recent chronological test spans **2025-07-01 through 2026-06-30**. Earlier EDA already inspected it: it is not a pristine holdout.
### Key assumptions
Hourly interval-start labels, mean power and final observation vintages are unverified. Daily MWh and all historical replays are conditional on these approximations. Candidate weather runs are not fabricated from realized weather.'''),code(SETUP),
md("### 1. Freeze the contract and source identities"),
code('''
from src.data import run_audit, availability_gate
lock_path = ROOT / "config/evaluation_lock.json"
identity = {"contract_sha256": fingerprint(ROOT/"config/forecast_contract.json"), "country_sha256": fingerprint(ROOT/"data/france_electricity_hourly.csv"), "regional_sha256": fingerprint(ROOT/"data/france_regional_electricity_hourly.csv")}
if lock_path.exists() and json.loads(lock_path.read_text()) != identity:
    raise RuntimeError("Contract or source changed after freeze. Preserve the old run and explicitly create a new experiment/lock before rerunning.")
write_json(lock_path, identity)
display(pd.Series(contract, name="contract").to_frame())
'''),
md("### 2. Audit without modifying raw values"),
code('''
data, dictionary, audit = run_audit(ROOT, contract)
display(pd.Series(audit, name="audit_result").to_frame())
display(dictionary[["dataset","field","unit","definition","missing","operational_before_cutoff"]])
display(pd.read_csv(ROOT/"reports/audit/availability_register.csv"))
'''),
md("### 3. Missingness, frequency and population overlap"),
code('''
display(pd.read_csv(ROOT/"reports/audit/missingness.csv").rename(columns={"Unnamed: 0":"field"}).query("missing_count > 0"))
display(pd.read_csv(ROOT/"reports/audit/frequency_intervals.csv"))
display(pd.read_csv(ROOT/"reports/audit/overlap.csv"))
fig, ax = plt.subplots(figsize=(10,3))
monthly_missing = data[["load_mw","rte_forecast_d1_mw","temperature_c"]].isna().resample("MS").sum()
monthly_missing.plot(ax=ax, marker=".")
ax.set(title="Missing hourly values by month — supplied country CSV", ylabel="Missing observations", xlabel="UTC month")
plt.tight_layout(); plt.show()
'''),
md("### 4. Reconcile national and regional measurements"),
code('''
delta = data.load_country_minus_regions_mw.dropna()
display(delta.describe().to_frame("country_minus_regions_MW"))
flagged = data.loc[data.reconciliation_flag, ["load_mw","regional_load_mw","load_country_minus_regions_mw"]]
display(flagged)
temperature = data[["temperature_c","regional_temperature_mean_c","regional_temperature_load_weighted_c"]].dropna()
errors = temperature.iloc[:,1:].rsub(temperature.temperature_c,axis=0)
display(pd.DataFrame({"hours":errors.count(),"bias_C":errors.mean(),"MAE_C":errors.abs().mean(),"RMSE_C":np.sqrt(errors.pow(2).mean())}))
fig, ax = plt.subplots(figsize=(11,3))
delta.plot(ax=ax, linewidth=.6)
ax.axhline(0,color="black",linewidth=.8)
ax.set(title="Country load minus complete regional sum — flags retained",ylabel="Difference (MW)",xlabel="UTC delivery time")
plt.tight_layout(); plt.show()
'''),
md("### 5. Operational gate and source conversion limitations"),
code('''
gate = availability_gate(ROOT, contract)
display(gate)
display(Markdown((ROOT/"reports/audit/limitations.md").read_text()))
assert data.load_mw.isna().sum() == audit["country_target_missing"]
assert audit["no_targets_interpolated"] and audit["peaks_deleted"] == 0
print("Audited rows:",len(data),"; operational experiment enabled:",bool(gate.operational_experiment_enabled.all()))
'''),
md('''## Takeaways and deliverables
Outputs: audited hourly grid in `data/processed/`; dictionary, register, frequency/overlap/missingness tables and limitations in `reports/audit/`; checksummed source manifest in `data/raw/`.

No source-to-hourly transformation can be verified from these CSVs alone. Obtain the extraction script, interval convention, original observation revisions and issued forecast archives before changing the operational gate.

Sources: [ODRÉ revisions and source cadence](https://odre.opendatasoft.com/explore/dataset/eco2mix-national-tr/), [RTE demand and forecast definitions](https://www.rte-france.com/donnees-publications/eco2mix-donnees-temps-reel/consommation-france).
Next: development-only EDA.''')])

save("02_development_eda.ipynb",[
md('''# 02 · Development EDA and modeling hypotheses
## Summary
Answer modeling questions using observations strictly before the configured test start. Calendar profiles, holidays, thermal response, lagged temperature and structural changes are descriptive; they do not prove causal mechanisms or forecast availability. The legacy full-history results remain exposed and are not relabeled as development-only evidence.
## Context and methods
Target: national demand in MW. Local calendar: Europe/Paris. Development excludes delivery dates from 2025-07-01 onward. Holidays cover metropolitan national public holidays, not school calendars or regional exceptions.'''),code(SETUP),
code('''
from src.data import read_audited
from src.eda import development_frame, temperature_summary, hypothesis_register
data = read_audited(ROOT)
dev = development_frame(data, contract)
dev["rte_error_mw"] = dev.load_mw-dev.rte_forecast_d1_mw
display(pd.Series({"development_start_UTC":dev.index.min(),"development_end_UTC":dev.index.max(),"rows":len(dev),"missing_demand":int(dev.load_mw.isna().sum())}).to_frame("scope"))
'''),
md("### 1. Do seasonal and weekday profiles differ?"),
code('''
profiles = dev.groupby(["season","hour","weekend"]).load_mw.mean()
fig, axes = plt.subplots(1,4,figsize=(18,4),sharex=True,sharey=True)
for ax, season in zip(axes,["Winter","Spring","Summer","Autumn"]):
    profiles.loc[season].unstack("weekend").rename(columns={0:"Weekday",1:"Weekend"}).plot(ax=ax,title=season,style=["-","--"])
    ax.set(xlabel="Local hour",xticks=[0,6,12,18,23])
axes[0].set_ylabel("Mean demand (MW)")
fig.suptitle("Development-period mean demand profiles — national CSV")
plt.tight_layout(); plt.show()
'''),
md("### 2. Holidays and annual structural changes"),
code('''
fig, axes = plt.subplots(1,2,figsize=(13,4))
dev.loc[dev.weekend.eq(0)].groupby(["hour","holiday"]).load_mw.mean().unstack("holiday").rename(columns={0:"Other weekday",1:"National public holiday"}).plot(ax=axes[0],style=["-","--"])
axes[0].set(title="Weekday holiday comparison — seasonal mix uncontrolled",xlabel="Local hour",ylabel="Mean demand (MW)")
dev.load_mw.resample("MS").mean().plot(ax=axes[1])
axes[1].set(title="Monthly mean demand — investigate changes",xlabel="UTC month",ylabel="Mean demand (MW)")
plt.tight_layout(); plt.show()
display(dev.groupby(["year","holiday"]).load_mw.agg(["mean","count"]).tail(10))
'''),
md("### 3. Temperature response and sparse extreme bands"),
code('''
sample = dev.loc[dev.season.eq("Winter") & dev.weekend.eq(0) & dev.hour.eq(19)]
temperature_table, band_checks = temperature_summary(sample)
display(pd.Series(band_checks).to_frame("band_check")); display(temperature_table)
temperature_table.to_csv(ROOT/"reports/scorecards/development_temperature_bands.csv")
fig, axes = plt.subplots(3,1,figsize=(11,8),sharex=True)
for ax,col,label in zip(axes,["bias_MW","MAE_MW","observations"],["Mean RTE error (MW)","RTE MAE (MW)","Observations"]):
    temperature_table[col].plot.bar(ax=ax,color="steelblue")
    ax.set_ylabel(label)
axes[0].axhline(0,color="black",linewidth=.8)
axes[2].set_xlabel("Temperature band (degree C)")
axes[2].tick_params(axis="x",labelrotation=45)
fig.suptitle("Development winter weekdays, 19:00 local — actual minus provided RTE")
plt.tight_layout(); plt.show()
'''),
md("### 4. Delayed temperature and changing RTE bias"),
code('''
thermal = dev[["temperature_c","load_mw"]].copy()
thermal["temperature_previous_24h"] = thermal.temperature_c.shift(1).rolling(24,min_periods=24).mean()
thermal["temperature_lag_24h"] = thermal.temperature_c.shift(24)
display(thermal.loc[sample.index].corr()[["load_mw"]])
fig, axes = plt.subplots(1,2,figsize=(13,4))
matrix=dev.groupby(["month","hour"]).rte_error_mw.mean().unstack("hour")
bound=np.nanmax(np.abs(matrix.to_numpy()))
im=axes[0].imshow(matrix,aspect="auto",origin="lower",cmap="RdBu_r",vmin=-bound,vmax=bound)
axes[0].set(title="RTE mean error by month and local hour",xlabel="Local hour",ylabel="Month",yticks=range(12),yticklabels=range(1,13),xticks=[0,6,12,18,23])
fig.colorbar(im,ax=axes[0],label="Actual minus RTE (MW)")
dev.rte_error_mw.resample("MS").mean().plot(ax=axes[1])
axes[1].axhline(0,color="black",linewidth=.8)
axes[1].set(title="Monthly RTE bias — no cause inferred",ylabel="Mean error (MW)",xlabel="UTC month")
plt.tight_layout(); plt.show()
'''),
md('''## Takeaways: hypotheses, not automatic features
The diagnostic lagged-temperature columns above are aligned to delivery time. They are not blindly reused as next-day predictors: the replay engine reconstructs historical temperature summaries at the origin cutoff instead. Correlations can mix season, year, holidays and economic conditions. No universal differencing/stationarity prerequisite is imposed.'''),
code('''
hypotheses = hypothesis_register()
hypotheses.to_csv(ROOT/"reports/hypothesis_register.csv",index=False)
display(hypotheses)
''')])

save("03_backtest_and_baselines.ipynb",[
md('''# 03 · Origin replay, integrity checks and baselines
## Summary
All baselines share one origin–delivery feature store and fallback policy. Forecasts are saved before their outcomes are joined for scoring. RTE is a **provided-column retrospective proxy**, never an authenticated historical forecast vintage.
## Context and methods
Each origin forecasts the complete following local day. Training/evaluation splits keep origin groups intact. Demand observations are assumed available at interval end plus two hours; weekly matching uses local wall time. Both occurrences of a repeated source hour are averaged; absent source hours invoke the declared fallback. These rules are tested below.

Validation: 2023-07-01–2025-06-30. Test remains outside this notebook. Reference: [rolling-origin evaluation](https://otexts.com/fpp3/tscv.html).'''),code(SETUP),
md("### 1. Run behavioral integrity tests"),
code('''
import unittest
suite=unittest.defaultTestLoader.discover(str(ROOT/"tests"))
test_result=unittest.TextTestRunner(verbosity=1).run(suite)
assert test_result.wasSuccessful(), "Integrity tests failed: stop the backtest."
'''),
md("### 2. Reconstruct information at every historical origin"),
code('''
from src.data import read_audited, availability_gate
from src.features import build_supervised
data=read_audited(ROOT)
features,labels=build_supervised(data,contract,rte_mode="proxy")
features.to_csv(ROOT/"data/origin_snapshots/history_features.csv.gz",index=False,compression="gzip")
labels.to_csv(ROOT/"data/processed/evaluation_labels.csv.gz",index=False,compression="gzip")
write_json(ROOT/"data/origin_snapshots/manifest.json",{"contract_sha256":fingerprint(ROOT/"config/forecast_contract.json"),"rows":len(features),"origins":features.origin.nunique(),"regime":"retrospective assumptions; no authentic forecast archives"})
display(features[["origin","delivery_time","lead_hours","week_lag_1","max_history_available_at","rte_version_status"]].head())
display(features.groupby("local_day").size().value_counts().rename_axis("delivery_intervals").to_frame("days"))
assert features.max_history_available_at.dropna().lt(features.loc[features.max_history_available_at.notna(),"origin"]).all()
display(availability_gate(ROOT,contract))
'''),
md("### 3. Replay four distinct baselines and save predictions"),
code('''
from src.backtest import baseline_replay
from src.metrics import scorecard,matched_scorecard,segment_scorecard
start=contract["validation_folds"][0]["start"]
end=contract["validation_folds"][-1]["end"]
baselines=baseline_replay(features,start,end,ROOT,contract)
policy_scores=scorecard(baselines,labels)
raw_scores=scorecard(baselines,labels,"raw_prediction_mw")
common_scores,common=matched_scorecard(baselines,labels,"raw_prediction_mw")
policy_scores.to_csv(ROOT/"reports/scorecards/baselines_validation_policy.csv",index=False)
raw_scores.to_csv(ROOT/"reports/scorecards/baselines_validation_raw_availability.csv",index=False)
common_scores.to_csv(ROOT/"reports/scorecards/baselines_validation_common.csv",index=False)
display(policy_scores); display(raw_scores); display(common_scores)
'''),
md("### 4. Common observations and fallback accounting"),
code('''
display(baselines.groupby("model").agg(predictions=("prediction_mw","count"),fallbacks=("fallback_used","sum")))
fig,ax=plt.subplots(figsize=(9,3))
common_scores.set_index("model").MAE_MW.sort_values().plot.barh(ax=ax)
ax.set(title="Validation baseline MAE — matched raw forecasts",xlabel="MAE (MW)",ylabel="")
plt.tight_layout(); plt.show()
'''),
md('''## Takeaways
Weekly persistence, comparable-day smoothing, provided RTE and trailing RTE-bias adjustment answer different questions. Compare common raw forecasts and the complete fallback policy separately. Missing targets are unscored, counted and never interpolated. Missing inputs cannot improve scores simply by removing difficult days. Operational RTE and weather comparisons remain gated.''')])

save("04_model_comparison.ipynb",[
md('''# 04 · Hypothesis-driven comparison and frozen selection
## Summary
Run a bounded pooled-direct model family with chronological validation. Each row describes one origin and one next-day delivery interval; no unobserved intermediate demand is fed recursively. The primary objective is MAE, so point models use absolute-error loss (median objective).
## Context and methods
Compare calendar, demand history, training windows, retraining frequency, one capacity change, past known temperature and RTE correction. Fit every imputer/model on its training slice; use no random validation or test-driven feature choice. Selection uses validation MAE and cold-evening/availability guardrails. Within 0.5% of the best eligible MAE, choose the predefined simpler model. Oracle weather is a separately labeled sensitivity experiment and is ineligible for selection. Deep learning is deferred until simpler comparisons establish a need.'''),code(SETUP),
code('''
from dataclasses import asdict
from src.storage import read_features,read_labels,read_predictions
from src.backtest import experiments,model_replay,select_candidate,feature_columns,fit_at_origin,stress_inputs
from src.metrics import scorecard,matched_scorecard,segment_scorecard
features=read_features(ROOT/"data/origin_snapshots/history_features.csv.gz")
labels=read_labels(ROOT/"data/processed/evaluation_labels.csv.gz")
registry=experiments()
register=pd.DataFrame([asdict(e) for e in registry])
register.to_csv(ROOT/"reports/experiment_register.csv",index=False)
display(register[["name","hypothesis","window_days","retrain_days","status"]])
'''),
md("### 1. Replay non-oracle candidates on chronological validation"),
code('''
start=contract["validation_folds"][0]["start"]; end=contract["validation_folds"][-1]["end"]
predictions=[read_predictions(ROOT/"reports/forecasts/baselines_validation.csv.gz")]
for experiment in registry:
    if experiment.oracle or experiment.status.startswith("gated"): continue
    print("Validation:",experiment.name,flush=True)
    predictions.append(model_replay(features,labels,start,end,experiment,ROOT,contract))
validation=pd.concat(predictions,ignore_index=True)
validation.to_csv(ROOT/"reports/forecasts/all_validation.csv.gz",index=False,compression="gzip")
'''),
md("### 2. Score common observations, guardrails and separate validation folds"),
code('''
scores=scorecard(validation,labels)
segments=segment_scorecard(validation,labels,contract)
common_scores,_=matched_scorecard(validation,labels,"raw_prediction_mw")
winner,selection=select_candidate(scores,segments,registry,contract)
scores.to_csv(ROOT/"reports/scorecards/model_validation_policy.csv",index=False)
segments.to_csv(ROOT/"reports/scorecards/model_validation_segments.csv",index=False)
common_scores.to_csv(ROOT/"reports/scorecards/model_validation_common_raw.csv",index=False)
selection.to_csv(ROOT/"reports/scorecards/selection_guardrails.csv",index=False)
display(scores.sort_values("MAE_MW")); display(selection.sort_values("MAE_MW"))
fold_scores=[]
for fold in contract["validation_folds"]:
    subset=validation.loc[validation.local_day.ge(fold["start"]) & validation.local_day.lt(fold["end"])]
    table=scorecard(subset,labels); table["fold"]=fold["name"]; fold_scores.append(table)
pd.concat(fold_scores).to_csv(ROOT/"reports/scorecards/validation_by_fold.csv",index=False)
if winner is None: raise RuntimeError("No candidate passes frozen guardrails. Keep the baselines; investigate using validation only.")
selected=next(e for e in registry if e.name==winner)
frozen={"experiment":asdict(selected),"contract_sha256":fingerprint(ROOT/"config/forecast_contract.json"),"selection_rule":"validation MAE, guardrails, 0.5% simplicity tolerance", "oracle_ineligible":True,"test_previously_explored":True}
write_json(ROOT/"config/frozen_candidate.json",frozen)
fig,ax=plt.subplots(figsize=(10,5))
scores.set_index("model").MAE_MW.sort_values().plot.barh(ax=ax)
ax.set(title="Validation policy MAE — retrospective availability assumptions",xlabel="MAE (MW)",ylabel="")
plt.tight_layout(); plt.show()
print("Frozen candidate:",winner)
'''),
md("### 3. Input stress tests with the same fallback policy"),
code('''
from src.features import prediction_fallback
stress_origin=features.loc[features.local_day.ge("2025-06-01"),"origin"].min()
stress_day=features.loc[features.origin.eq(stress_origin)]
stress_model,_=fit_at_origin(features,labels,stress_origin,selected,contract)
stress_rows=[]
for scenario in ("missing_demand_lags","missing_rte","stale_weather","rapid_demand_change"):
    changed=stress_inputs(stress_day,scenario)
    raw=stress_model.predict(changed[feature_columns(selected)])
    if selected.correction: raw+=changed.rte_at_origin_mw.to_numpy()
    if scenario=="missing_demand_lags" and "history" in selected.groups: raw[:]=np.nan
    p,source=prediction_fallback(changed,raw,selected.correction)
    stress_rows.append({"scenario":scenario,"delivered_intervals":int(p.notna().sum()),"fallback_intervals":int(source.ne("model").sum()),"minimum_MW":p.min(),"maximum_MW":p.max()})
stress_table=pd.DataFrame(stress_rows)
stress_table.to_csv(ROOT/"reports/scorecards/input_stress_tests.csv",index=False)
display(stress_table)
'''),
md("### 4. Future recorded weather: sensitivity only, excluded from selection"),
code('''
from src.data import read_audited
data=read_audited(ROOT)
oracle=features.copy()
oracle["oracle_temperature_c"]=data.temperature_c.reindex(pd.DatetimeIndex(oracle.delivery_time)).to_numpy()
oracle["oracle_heating_c"]=(18-oracle.oracle_temperature_c).clip(lower=0)
oracle["oracle_cooling_c"]=(oracle.oracle_temperature_c-22).clip(lower=0)
oracle_experiment=next(e for e in registry if e.oracle)
oracle_predictions=model_replay(oracle,labels,start,end,oracle_experiment,ROOT,contract,phase="oracle_validation")
oracle_scores=scorecard(oracle_predictions,labels)
oracle_scores["interpretation"]="recorded future weather; not deployable evidence"
oracle_scores.to_csv(ROOT/"reports/scorecards/oracle_weather_sensitivity.csv",index=False)
display(oracle_scores)
'''),
md('''## Takeaways
The candidate and its preprocessing, window, retraining and fallback policies are frozen in `config/frozen_candidate.json`. Future test-period labels may enter later scheduled fits only after their assumed publication time; this online rule was fixed before test scoring. Changing the policy after reading test results would require a new experiment. Authentic weather and RTE-vintage experiments remain gated.''')])

save("05_uncertainty_and_failures.ipynb",[
md('''# 05 · Frozen test replay, uncertainty and failures
## Summary
Evaluate the frozen candidate and baselines on the previously explored chronological test. Archive point/quantile predictions first, then join outcomes. Compare rolling past out-of-sample error intervals with 5th/50th/95th quantile forecasts. Assess paired improvement using contiguous multi-day blocks rather than independent hourly resampling.
## Context and methods
90% is a nominal interval level, not guaranteed conditional coverage. Calibration uses only errors whose labels are available before each origin. Daily energy and daily maximum require whole residual trajectories; adding hourly quantiles is not a valid daily interval. Full daily residual scenarios preserve cross-hour dependence, while days with too little matching DST history remain unsupported.

References: [quantile regression](https://scikit-learn.org/stable/auto_examples/ensemble/plot_gradient_boosting_quantile.html), [adaptive conformal inference](https://arxiv.org/abs/2106.00170), [time-series bootstrapping](https://otexts.com/fpp3/bootstrap.html). No conformal guarantee is claimed for this empirical implementation.'''),code(SETUP),
code('''
from src.storage import read_features,read_labels,read_predictions
from src.backtest import Experiment,baseline_replay,model_replay,archive_predictions
from src.metrics import scorecard,matched_scorecard,segment_scorecard,daily_metrics,paired_block_bootstrap
from src.uncertainty import empirical_intervals,interval_scorecard,trajectory_intervals
features=read_features(ROOT/"data/origin_snapshots/history_features.csv.gz")
labels=read_labels(ROOT/"data/processed/evaluation_labels.csv.gz")
frozen=json.loads((ROOT/"config/frozen_candidate.json").read_text())
assert frozen["contract_sha256"]==fingerprint(ROOT/"config/forecast_contract.json")
selected=Experiment(**frozen["experiment"])
start=contract["test"]["start"]; end=contract["test"]["end"]
'''),
md("### 1. Save frozen point and quantile forecasts before scoring"),
code('''
baselines=baseline_replay(features,start,end,ROOT,contract,phase="test")
candidate=model_replay(features,labels,start,end,selected,ROOT,contract,phase="test",quantiles=True)
test_predictions=pd.concat([baselines,candidate],ignore_index=True)
archive_predictions(ROOT,"all_test",test_predictions,contract)
'''),
md("### 2. Matched point scores, critical segments and daily outcomes"),
code('''
scores=scorecard(test_predictions,labels)
matched_scores,matched=matched_scorecard(test_predictions,labels)
raw_scores,raw_matched=matched_scorecard(test_predictions,labels,"raw_prediction_mw")
segments=segment_scorecard(test_predictions,labels,contract)
daily=daily_metrics(test_predictions,labels)
for name,table in (("test_policy",scores),("test_common",matched_scores),("test_common_raw",raw_scores),("test_segments",segments),("test_daily",daily)):
    table.to_csv(ROOT/f"reports/scorecards/{name}.csv",index=False)
display(matched_scores); display(daily.groupby("model").mean(numeric_only=True))
ci=paired_block_bootstrap(test_predictions,labels,selected.name,block_days=contract["bootstrap_block_days"],repetitions=contract["bootstrap_repetitions"],seed=contract["random_seed"])
ci.to_csv(ROOT/"reports/scorecards/paired_block_bootstrap.csv",index=False)
display(ci)
'''),
md("### 3. Uncertainty baseline from past out-of-sample errors"),
code('''
validation=read_predictions(ROOT/"reports/forecasts/all_validation.csv.gz")
past=validation.loc[validation.model.eq(selected.name)]
point_intervals=empirical_intervals(candidate.drop(columns=["q05","q50","q95"]),past,labels,contract)
archive_predictions(ROOT,"candidate_empirical_intervals_test",point_intervals,contract)
quantile_scores=interval_scorecard(candidate,labels,contract)
empirical_scores=interval_scorecard(point_intervals,labels,contract)
quantile_scores.to_csv(ROOT/"reports/uncertainty/quantile_calibration.csv",index=False)
empirical_scores.to_csv(ROOT/"reports/uncertainty/empirical_calibration.csv",index=False)
display(quantile_scores.head()); display(empirical_scores.head())
fig,axes=plt.subplots(1,2,figsize=(12,4))
for table,label in ((quantile_scores,"Quantile model"),(empirical_scores,"Past OOS errors")):
    h=table.loc[table.segment_type.eq("hour")].copy(); h["hour"]=h.segment.astype(int); h=h.sort_values("hour")
    axes[0].plot(h.hour,h.coverage_90,marker=".",label=label)
    axes[1].plot(h.hour,h.width_MW,label=label)
axes[0].axhline(.9,color="black",linestyle="--",label="Nominal 90%")
axes[0].set(title="Test coverage by local hour",xlabel="Local hour",ylabel="Empirical coverage",ylim=(0,1)); axes[0].legend()
axes[1].set(title="Test interval width by local hour",xlabel="Local hour",ylabel="Mean width (MW)"); axes[1].legend()
plt.tight_layout(); plt.show()
'''),
md("### 4. Daily trajectory uncertainty, saved before daily outcome comparison"),
code('''
trajectory=trajectory_intervals(candidate,past,labels,contract)
trajectory.to_csv(ROOT/"reports/uncertainty/daily_trajectory_predictions.csv",index=False)
truth=labels.copy(); truth["day"]=truth.delivery_time.dt.tz_convert(contract["timezone"]).dt.strftime("%Y-%m-%d")
actual_daily=truth.groupby("day").agg(actual_energy_MWh=("actual_mw",lambda s:s.sum(min_count=len(s))),actual_peak_MW=("actual_mw","max"),valid_hours=("actual_mw","count"))
trajectory_eval=trajectory.merge(actual_daily,left_on="day",right_index=True,how="left")
ok=trajectory_eval.status.eq("ok") & trajectory_eval.valid_hours.eq(trajectory_eval.hours)
trajectory_eval["energy_covered"]=np.where(ok,trajectory_eval.actual_energy_MWh.between(trajectory_eval.energy_q05_MWh,trajectory_eval.energy_q95_MWh),np.nan)
trajectory_eval["peak_covered"]=np.where(ok,trajectory_eval.actual_peak_MW.between(trajectory_eval.peak_q05_MW,trajectory_eval.peak_q95_MW),np.nan)
trajectory_eval.to_csv(ROOT/"reports/uncertainty/daily_trajectory_evaluation.csv",index=False)
display(trajectory_eval.groupby(["status","hours"]).agg(days=("day","size"),energy_coverage=("energy_covered","mean"),peak_coverage=("peak_covered","mean")))
'''),
md("### 5. Inspect failures without inventing causes"),
code('''
errors=candidate.merge(labels,on="delivery_time",validate="many_to_one")
errors["absolute_error_MW"]=(errors.actual_mw-errors.prediction_mw).abs()
errors["day"]=errors.delivery_time.dt.tz_convert(contract["timezone"]).dt.strftime("%Y-%m-%d")
failures=errors.groupby("day").agg(MAE_MW=("absolute_error_MW","mean"),max_error_MW=("absolute_error_MW","max"),fallback_intervals=("fallback_used","sum"),minimum_temperature_C=("realized_temperature_c","min"),source_reconciliation_flags=("reconciliation_flag","sum"),scored_intervals=("actual_mw","count")).sort_values("MAE_MW",ascending=False).head(10)
failures["classification"]=np.where(failures.fallback_intervals.gt(0),"fallback/input gap observed",np.where(failures.source_reconciliation_flags.gt(0),"source inconsistency observed; cause unverified","unexplained"))
failures["weather_miss_verified"]=False
failures.to_csv(ROOT/"reports/failure_catalogue.csv")
display(failures)
worst=errors.loc[errors.day.eq(failures.index[0])].sort_values("delivery_time")
fig,ax=plt.subplots(figsize=(11,4))
ax.plot(worst.delivery_time,worst.actual_mw,label="Actual",color="black")
ax.plot(worst.delivery_time,worst.prediction_mw,label="Frozen point forecast")
ax.fill_between(worst.delivery_time,worst.q05,worst.q95,alpha=.2,label="Nominal 90% quantile interval")
ax.set(title=f"Worst test day by MAE: {failures.index[0]} — investigate, do not refit",ylabel="Demand (MW)",xlabel="UTC delivery time")
ax.legend(); plt.tight_layout(); plt.show()
'''),
md('''## Takeaways
Point metrics, interval calibration, daily trajectory risks and paired improvement uncertainty answer different questions. Cold-evening segments use recorded weather only for retrospective diagnosis; operational cold alerts cannot be evaluated without accessible forecast weather. Neither a shaded interval nor the word conformal establishes conditional or simultaneous coverage. Worst-day classes above describe evidence, not verified root causes.''')])

save("06_business_and_deployment.ipynb",[
md('''# 06 · Manager scorecard, business interpretation and research limitations
## Summary
Organize the research assessment around three questions: improvement over RTE; uncertainty and failures; recommendation and evidence limitations. Forecast skill is measured on matched observations under explicit retrospective assumptions. Procurement savings cannot be estimated from demand, temperatures and RTE alone.
## Context and methods
Use the frozen candidate and saved test forecasts. No model or feature is changed here. National consumption is a utility-demand proxy. The holdout was previously explored, weather/RTE vintages are unverified and the hourly conversion is undocumented: these constrain the recommendation even if retrospective loss improves.'''),code(SETUP),
code('''
from src.research_risks import risk_register
frozen=json.loads((ROOT/"config/frozen_candidate.json").read_text()); name=frozen["experiment"]["name"]
scores=pd.read_csv(ROOT/"reports/scorecards/test_common.csv")
ci=pd.read_csv(ROOT/"reports/scorecards/paired_block_bootstrap.csv")
'''),
md("### 1. Does it improve on the provided RTE benchmark?"),
code('''
display(scores); display(ci)
reference=scores.set_index("model").loc["rte_proxy"]
candidate_score=scores.set_index("model").loc[name]
relative=100*(1-candidate_score.MAE_MW/reference.MAE_MW)
absolute=reference.MAE_MW-candidate_score.MAE_MW
display(Markdown(f"**Frozen candidate:** `{name}`. Matched test MAE improvement over provided RTE: **{relative:.2f}%**, or **{absolute:,.1f} MW**. This is retrospective skill, not operational validation or financial savings."))
fig,ax=plt.subplots(figsize=(9,3))
scores.set_index("model").MAE_MW.sort_values().plot.barh(ax=ax)
ax.set(title="Previously explored chronological test — matched observations",xlabel="MAE (MW)",ylabel="")
plt.tight_layout(); plt.show()
'''),
md("### 2. How uncertain are forecasts, and where do they fail?"),
code('''
calibration=pd.read_csv(ROOT/"reports/uncertainty/quantile_calibration.csv")
display(calibration.loc[calibration.segment_type.isin(["overall","cold_evening"])])
display(pd.read_csv(ROOT/"reports/failure_catalogue.csv").head(5))
daily=pd.read_csv(ROOT/"reports/scorecards/test_daily.csv")
display(daily.groupby("model")[["absolute_daily_energy_error_MWh","absolute_daily_peak_error_MW","shortage_proxy_MWh","surplus_proxy_MWh"]].mean())
'''),
md('''### 3. What can the research establish, and what are its limits?
Demand forecasting and procurement decisions are separate. Euro savings require actual prices, settlement rules for the evaluation dates, market instruments, and an explicit shortage/surplus procurement policy applied identically to RTE and the candidate. Realized future prices may evaluate that policy retrospectively but cannot be auction-time input features.

EU day-ahead markets introduced **15-minute market time units for delivery from 1 October 2025**. The assignment retains hourly targets; its scores do not establish performance at quarter-hour settlement resolution. [European Commission](https://energy.ec.europa.eu/news/eu-electricity-trading-day-ahead-markets-becomes-more-dynamic-2025-10-01_en)'''),
code('''
risks=risk_register()
risks.to_csv(ROOT/"reports/risk_register.csv",index=False)
display(risks)
'''),
md("### 4. Save the manager-facing recommendation"),
code('''
overall=calibration.loc[calibration.segment_type.eq("overall")].iloc[0]
empirical_table=pd.read_csv(ROOT/"reports/uncertainty/empirical_calibration.csv")
empirical_overall=empirical_table.loc[empirical_table.segment_type.eq("overall")].iloc[0]
recommendation=f"""# Electricity demand forecasting — final research assessment

## 1. Improvement over RTE
Frozen candidate: {name}, an RTE-assisted additive correction, not an independent replacement. Matched test MAE: {candidate_score.MAE_MW:,.2f} MW; provided RTE: {reference.MAE_MW:,.2f} MW. Relative improvement: {relative:.2f}%; absolute MAE reduction: {absolute:,.2f} MW.
Read reports/scorecards/paired_block_bootstrap.csv for paired 7/14/28-day block sensitivity and confidence intervals. These retain exactly the same eligible paired hours as headline MAE, including partial days with their actual counts. Calendar-day blocks preserve dependence; hourly errors are not resampled independently.

## 2. Uncertainty and failures
Nominal 90% quantile interval coverage: {overall.coverage_90:.3f}; mean width: {overall.width_MW:,.1f} MW. The fixed rolling past-error baseline has coverage {empirical_overall.coverage_90:.3f}. Quantile coverage below 90% indicates undercoverage, not successful calibration; further method development must use validation or a new prospective period, not discretionary tuning on this test. Coverage, tail exceedances and pinball losses by hour, season, cold evenings and lead time are in reports/uncertainty/. Complete residual-day scenarios evaluate daily energy and peaks separately. Unsupported DST sample sizes remain explicit.
Worst-day evidence is in reports/failure_catalogue.csv. Weather misses cannot be verified without archived forecast weather. Do not infer causes from coincidence.

## 3. Research recommendation and evidence limits
Recommendation: retain this as a retrospective research result and confirm it on fresh later data before drawing stronger conclusions. Forecast vintages, source-to-hourly conversion and original observation publication times are unverified. The chronological test was previously explored in EDA. Demand history uses a disclosed publication-delay/revision approximation.
National consumption is not a validated customer portfolio model. Energy shortage/surplus proxies are not euro savings. Prices, applicable settlement rules and a common decision policy are required for financial claims. Hourly demand scores do not establish quarter-hour settlement performance.

The research deliverables are saved predictions, matched scorecards, uncertainty calibration, a failure catalogue and reports/risk_register.csv. Serving, scheduling, live monitoring and procurement integration are outside this project.
"""
(ROOT/"reports/final_recommendation.md").write_text(recommendation)
write_json(ROOT/"reports/run_summary.json",{"candidate":name,"test_MAE_MW":candidate_score.MAE_MW,"rte_test_MAE_MW":reference.MAE_MW,"relative_improvement_percent":relative,"absolute_improvement_MW":absolute,"coverage_90":overall.coverage_90,"operationally_validated":False,"financial_savings_estimable":False,"test_previously_explored":True,"scope":"retrospective research; no production implementation"})
display(Markdown(recommendation))
''')])

print("Authored six notebooks; original preserved at",legacy.relative_to(ROOT))
