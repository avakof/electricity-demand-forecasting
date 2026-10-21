"""Export report figures from frozen research scores; never fits or selects models."""
from pathlib import Path
import json
import hashlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
RUN = PROJECT / "reports/rte_risk/20261020T213640021260Z"
OUT = HERE / "figures"
OUT.mkdir(exist_ok=True)
summary = json.loads((RUN / "result_summary.json").read_text())
POINT = summary["selected_point"]
PRIOR = summary["previous_development_selected_policy"]
QMODEL = summary["selected_quantile"]
COLORS = ["#8b929b", "#bd8632", "#57738a", "#164f87"]
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.labelcolor": "#263340", "text.color": "#263340",
    "axes.edgecolor": "#9aa3ac", "savefig.facecolor": "white"})
inputs = set()

def read(name):
    path = RUN / "tables" / name
    inputs.add(path)
    return pd.read_csv(path)

def save(fig, stem, note):
    fig.text(.01, .015, note, fontsize=9, color="#535e69", va="bottom")
    fig.tight_layout(rect=[0, .065, 1, .94])
    fig.savefig(OUT / f"{stem}.png", dpi=190)
    fig.savefig(OUT / f"{stem}.svg")
    plt.close(fig)

scores = read("test_scorecard.csv").set_index("model")
keys = ["rte_proxy", "rte_bias_proxy", PRIOR, POINT]
names = ["RTE D1", "Rolling bias correction", "Prior selected model", "New selected model"]
fig, axes = plt.subplots(1, 2, figsize=(12, 4.9))
for ax, col, title in zip(axes, ["MAE_MW", "cold_MAE_MW"],
                         ["All matched hours (n = 8,759)", "Cold evenings (n = 66)"]):
    vals = scores.loc[keys, col].to_numpy()
    ax.barh(names, vals, color=COLORS, height=.6)
    ax.invert_yaxis()
    ax.set_xlim(0, 2900)
    ax.set_xlabel("Mean absolute error (MW); lower is better")
    ax.set_title(title, loc="left", pad=12)
    ax.grid(axis="x", alpha=.16)
    ax.set_axisbelow(True)
    for i, val in enumerate(vals): ax.text(val + 30, i, f"{val:,.1f}", va="center")
fig.suptitle("The selected correction improves on RTE, including cold evenings", x=.01, ha="left", fontsize=15)
save(fig, "01_accuracy", "July 2025–June 2026. Cold: temperature <5°C at 18:00–20:00 Europe/Paris; 24 days, one winter.\nSame national-demand targets; these are complete policy comparisons with different refit schedules.")

boot = read("bootstrap.csv")
boot = boot[(boot.model == POINT) & (boot.block_days == 14)]
fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.7), sharex=True)
for ax, metric, title in zip(axes, ["MAE_MW", "cold_MAE_MW"], ["Overall MAE improvement", "Cold-evening MAE improvement"]):
    d = boot[boot.metric == metric].set_index("reference").loc[keys[:3]]
    x = d.relative_improvement_percent.to_numpy()
    lo, hi = d.relative_low.to_numpy(), d.relative_high.to_numpy()
    ax.errorbar(x, np.arange(3), xerr=np.vstack([x-lo, hi-x]), fmt="o", capsize=5, color=COLORS[3])
    ax.axvline(0, color="#4e5860", lw=1, ls="--")
    ax.set_yticks(np.arange(3), ["vs RTE", "vs rolling bias", "vs prior selected"])
    ax.set_ylim(2.7, -.55)
    ax.set_xlim(-20, 65)
    ax.set_xlabel("Relative MAE improvement (%)")
    ax.set_title(title, loc="left", pad=10)
    for i, (v,l,h) in enumerate(zip(x,lo,hi)):
        ax.text(v, i+.23, f"{v:.1f}% [{l:.1f}, {h:.1f}]", ha="center", fontsize=9,
                bbox={"facecolor":"white", "edgecolor":"none", "pad":1})
    ax.grid(axis="x", alpha=.16)
fig.suptitle("Additional cold-evening improvement over the prior model is uncertain", x=.01, ha="left", fontsize=14)
save(fig, "02_improvement_uncertainty", "Paired 14-day block bootstrap, 500 attempted draws (495 valid cold draws), 95% intervals. Positive = lower loss.\nIntervals describe this historical sample; they do not remove vintage uncertainty or prior test exposure.")

ab = read("feature_ablation.csv")
ab = ab[ab.training_cold_weight_cap == 1]
fig, ax = plt.subplots(figsize=(10.2, 4.8))
tiers = ["basic", "engineered", "stacked"]
for arm, label, color, marker in [("national", "National history + RTE", "#bd8632", "s"),
    ("regional_only", "Regional history + RTE", "#57738a", "^"),
    ("combined", "Combined history + RTE", "#164f87", "o")]:
    vals = ab[ab.arm == arm].set_index("feature_tier").loc[tiers, "MAE_MW"].to_numpy()
    ax.plot(range(3), vals, marker=marker, color=color, lw=2, label=label)
    for i, val in enumerate(vals):
        dy = -15 if arm == "combined" else 6
        dx,ha = (-12,"right") if arm == "regional_only" and i == 0 else (5,"left")
        ax.annotate(f"{val:.1f}", (i,val), xytext=(dx,dy), textcoords="offset points", fontsize=9,ha=ha)
ax.set_xticks(range(3), ["Basic", "Engineered lags / rolling statistics", "+ Earlier-trained predictors"])
ax.set_ylim(690, 865)
ax.set_ylabel("Test MAE (MW); lower is better")
ax.set_xlim(-.15,2.35)
ax.grid(axis="y", alpha=.16)
ax.legend(loc="upper right", frameon=False)
fig.suptitle("Feature engineering improves overall error across all three input views", x=.01, ha="left", fontsize=14)
save(fig, "03_feature_ablation", "Controlled LightGBM residual variants with neutral training weights; each configuration selected on development.\nThe y-axis is focused to compare changes. Cold-evening gains are not monotonic with feature complexity.")

overall = read("uncertainty_scorecard.csv")
cold = read("cold_interval_scorecard.csv")
methods = [(POINT,"past_calibration_empirical_residual_distribution","Selected point + residuals"),
    (QMODEL,"native_quantile_lightgbm","Selected native quantiles"),
    (QMODEL+"__fixed_engineered","fixed_engineered_split_conformal","Fixed split conformal"),
    (QMODEL+"__fixed_engineered","fixed_engineered_CQR","Fixed CQR")]
fig, axes = plt.subplots(1,2, figsize=(12.5,5.1))
y=np.arange(4)
rows=[]
for model,method,label in methods:
    a=overall[(overall.model==model)&(overall.method==method)].iloc[0]
    b=cold[(cold.model==model)&(cold.method==method)].iloc[0]
    rows.append((label,a,b))
for offset, idx, label, color in [(-.16,1,"All hours", "#57738a"),(.16,2,"Cold evenings", "#bd8632")]:
    cov=[100*r[idx].central90_coverage for r in rows]
    axes[0].barh(y+offset,cov,height=.29,color=color,label=label)
    for yi,v in zip(y+offset,cov): axes[0].text(v+1,yi,f"{v:.1f}%",va="center",fontsize=9)
axes[0].axvline(90,color="#333f4c",ls="--",lw=1,label="Nominal 90%")
axes[0].set_xlim(0,105);axes[0].set_yticks(y,[r[0] for r in rows]);axes[0].invert_yaxis()
axes[0].set_xlabel("Observed interval coverage (%)")
axes[0].set_title("Central 90% coverage",loc="left")
axes[0].legend(frameon=False,loc="lower left",bbox_to_anchor=(0,-.34),ncol=3,fontsize=9)
upper=[100*r[2].central90_upper_exceedance for r in rows]
axes[1].barh(y,upper,height=.6,color="#bd8632")
axes[1].axvline(5,color="#333f4c",ls="--",lw=1)
axes[1].set_xlim(0,56);axes[1].set_yticks(y,[r[0] for r in rows]);axes[1].invert_yaxis()
axes[1].set_xlabel("Cold outcomes above the upper bound (%)")
axes[1].set_title("Cold upper-tail misses; nominal 5%",loc="left")
for yi,v in zip(y,upper):axes[1].text(v+1,yi,f"{v:.1f}%",va="center",fontsize=9)
fig.suptitle("None of the examined intervals provides adequate cold-evening coverage", x=.01, ha="left",fontsize=14)
save(fig,"04_interval_failure","8,759 overall hours; 66 cold hours across 24 days. Calibration: 2 April–29 June 2025, zero cold evenings.\nMethods use different point models / fitting policies; the fixed CQR interval is not the selected point model's interval.")

seg=read("test_segments.csv")
fig,axes=plt.subplots(1,2,figsize=(12,4.8))
for model,name,color in zip(["rte_proxy",PRIOR,POINT],["RTE D1","Prior selected","New selected"],[COLORS[0],COLORS[2],COLORS[3]]):
    d=seg[(seg.model==model)&(seg.segment_type=="hour")].copy()
    d["h"]=d.segment.astype(int);d=d.sort_values("h")
    axes[0].plot(d.h,d.MAE_MW,label=name,color=color,lw=2,marker="o",markersize=3)
axes[0].set_xticks(range(0,24,3));axes[0].set_xlim(0,23);axes[0].set_ylim(bottom=0)
axes[0].set_xlabel("Delivery hour, Europe/Paris");axes[0].set_ylabel("MAE (MW)")
axes[0].set_title("Mean error by delivery hour",loc="left");axes[0].legend(frameon=False,fontsize=9)
seasons=["Summer","Autumn","Winter","Spring"]
for off,model,name,color in [(-.22,"rte_proxy","RTE D1",COLORS[0]),(0,PRIOR,"Prior selected",COLORS[2]),(.22,POINT,"New selected",COLORS[3])]:
    d=seg[(seg.model==model)&(seg.segment_type=="season")].set_index("segment").loc[seasons]
    axes[1].bar(np.arange(4)+off,d.MAE_MW,width=.21,color=color,label=name)
axes[1].set_xticks(range(4),seasons);axes[1].set_ylabel("MAE (MW)")
axes[1].set_title("Mean error by meteorological season",loc="left")
for ax in axes:ax.grid(axis="y",alpha=.15);ax.set_axisbelow(True)
fig.suptitle("Winter and later delivery hours remain harder for the selected model",x=.01,ha="left",fontsize=14)
save(fig,"05_hour_season","Same 8,759 matched hours; descriptive test diagnostics, not selection criteria added after testing.\nLocal-hour differences combine time-of-day and forecast-lead effects; they do not identify a causal mechanism.")

risk=read("quantile_decision_risk_scenarios.csv")
risk=risk[risk.cold_weight==5]
fig,ax=plt.subplots(figsize=(9.5,4.7))
ratios=[1,3,9,19]
for model,label,color,marker in [("rte_proxy","RTE + historical residual quantile",COLORS[0],"s"),("rte_bias_proxy","Bias-corrected RTE + residual quantile",COLORS[1],"^"),(QMODEL,"Selected native quantile policy",COLORS[3],"o")]:
    d=risk[risk.model==model].set_index("shortage_surplus_ratio").loc[ratios]
    ax.plot(range(4),d.normalized_asymmetric_loss_MW,color=color,marker=marker,label=label,lw=2)
ax.set_xticks(range(4),["1:1 → q50","3:1 → q75","9:1 → q90","19:1 → q95"])
ax.set_ylim(bottom=0);ax.set_ylabel("Normalized asymmetric loss (weighted MW)")
ax.set_xlabel("Hypothetical shortage : surplus penalty → decision quantile")
ax.legend(frameon=False,fontsize=9);ax.grid(axis="y",alpha=.15)
fig.suptitle("Quantile decisions improve hypothetical risk scores; euro savings are unknown",x=.01,ha="left",fontsize=13)
save(fig,"06_risk_scenarios","Fivefold realized-cold scoring weight; identical cost rule and matched observations for all policies.\nRTE reference here includes past-error quantile adjustment. Compare within each scenario, not between cost ratios.")

# Supplemental descriptive diagnostics are produced from frozen forecasts separately.
diag=HERE/"diagnostics"
if (diag/"worst_cold_day_hourly.csv").exists():
    path=diag/"worst_cold_day_hourly.csv";inputs.add(path)
    d=pd.read_csv(path).sort_values("local_hour")
    day=d.local_day.iloc[0]
    fig,axes=plt.subplots(2,1,figsize=(11.5,7.5),sharex=True,gridspec_kw={"height_ratios":[2,1]})
    x=d.local_hour.to_numpy()
    axes[0].fill_between(x,d.selected_empirical_q05/1000,d.selected_empirical_q95/1000,
        color="#164f87",alpha=.15,label="Selected point: nominal 90% residual interval")
    for col,label,color,style in [("actual_mw","Actual demand","#242a32","-"),
        ("selected_prediction_mw","Selected point forecast",COLORS[3],"-"),
        ("rte_prediction_mw","RTE D1",COLORS[1],"--")]:
        axes[0].plot(x,d[col]/1000,color=color,ls=style,lw=2,label=label)
    axes[0].axvspan(17.5,20.5,color="#bd8632",alpha=.10)
    axes[0].set_ylabel("Hourly demand / forecast (GW)")
    axes[0].legend(frameon=False,fontsize=9,ncol=2,loc="upper left")
    axes[0].set_title(f"Worst cold day: {day}; selected cold-hour MAE 4,165 MW",loc="left",pad=12)
    axes[1].plot(x,d.selected_signed_error_mw/1000,color=COLORS[3],marker="o",lw=1.5)
    axes[1].axhline(0,color="#65717b",ls="--",lw=1)
    axes[1].axvspan(17.5,20.5,color="#bd8632",alpha=.10)
    axes[1].set_ylabel("Forecast − actual (GW)")
    axes[1].set_xlabel("Delivery hour, Europe/Paris; shaded hours satisfy the cold-evening definition")
    axes[1].annotate("19:00: −4,695 MW\nActual 67.48 GW; observed 2.21°C",xy=(19,d.loc[d.local_hour==19,"selected_signed_error_mw"].iloc[0]/1000),
                     xytext=(10,-3.2),arrowprops={"arrowstyle":"-","color":"#65717b"},fontsize=9)
    for ax in axes:ax.set_xlim(0,23);ax.grid(axis="y",alpha=.15)
    axes[1].set_xticks(range(0,24,2))
    fig.suptitle("The correction still misses a cold-evening peak",x=.01,ha="left",fontsize=15)
    save(fig,"07_worst_cold_day","Day selected descriptively by largest selected-model MAE over cold-evening hours, not for model tuning.\nAll 24 targets present. Holiday proximity and measured temperature do not establish the cause; input vintages are unavailable.")
if (diag/"monthly_mae.csv").exists():
    path=diag/"monthly_mae.csv";inputs.add(path)
    d=pd.read_csv(path)
    # Column schema is explicitly recorded by the diagnostics exporter.
    fig,ax=plt.subplots(figsize=(11,4.8))
    months=sorted(d.segment.unique())
    for role,label,color,marker in [("rte","RTE D1",COLORS[0],"s"),("prior","Prior selected",COLORS[2],"^"),("selected","New selected",COLORS[3],"o")]:
        a=d[d.role==role].set_index("segment").loc[months]
        ax.plot(range(len(months)),a.MAE_MW,color=color,marker=marker,lw=2,label=label)
    ax.set_xticks(range(len(months)),[pd.Timestamp(m).strftime("%b\n%Y") for m in months])
    ax.set_ylim(bottom=0);ax.set_ylabel("Monthly MAE (MW)");ax.legend(frameon=False,ncol=3)
    ax.grid(axis="y",alpha=.15)
    fig.suptitle("December is the selected model's highest-error month",x=.01,ha="left",fontsize=15)
    save(fig,"08_monthly_errors","Monthly aggregates of the same matched outcomes. December 2025: 1,016.7 MW selected MAE over 744 hours.\nRetrospective diagnostic; month-specific error changes do not prove a weather or structural cause.")
manifest={"run":str(RUN.relative_to(PROJECT)),"purpose":"Standalone scientific figure exports from frozen results; no training or selection", "inputs":{str(p.relative_to(PROJECT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(inputs)},"figures":[p.name for p in sorted(OUT.glob("*.png"))]}
(HERE/"figure_provenance.json").write_text(json.dumps(manifest,indent=2)+"\n")
print(f"Exported {len(manifest['figures'])} PNG figures and matching SVG files to {OUT}")
