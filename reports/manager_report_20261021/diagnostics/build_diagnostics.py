"""Descriptive failure diagnostics from frozen forecasts; no model fitting.

Run with the project's scientific Python interpreter. Outputs are restricted to
this directory; source forecasts, raw observations and experiment files are read
only. Signed errors always mean prediction minus actual demand.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
RUN = ROOT/"reports/rte_risk/20261020T213640021260Z"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.common import holiday_dates, local_day_grid


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(name, obj):
    (HERE/name).write_text(json.dumps(obj, indent=2, default=str, allow_nan=False)+"\n")


def metrics(frame, role):
    error = frame[f"{role}_signed_error_mw"]
    under = -error
    return {"n": len(frame), "days": int(frame.local_day.nunique()),
            "MAE_MW": float(error.abs().mean()), "RMSE_MW": float(np.sqrt(np.mean(error**2))),
            "bias_prediction_minus_actual_MW": float(error.mean()),
            "underforecast_rate": float(error.lt(0).mean()),
            "mean_shortage_MW": float(under.clip(lower=0).mean()),
            "mean_surplus_MW": float(error.clip(lower=0).mean()),
            "cold_hours": int(frame.cold_evening.sum())}


def grouped(frame, roles, column):
    return pd.DataFrame([{"segment": str(segment), "role": role, "model": model, **metrics(part, role)}
                         for segment, part in frame.groupby(column, sort=True)
                         for role, model in roles.items()])


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, list):
        return [clean(v) for v in value]
    if isinstance(value, (np.integer, np.bool_)):
        return value.item()
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    return value


def main():
    frozen = json.loads((RUN/"frozen_selection.json").read_text())
    summary = json.loads((RUN/"result_summary.json").read_text())
    roles = {"selected": frozen["selected_point"], "rte": "rte_proxy",
             "prior": summary["previous_development_selected_policy"]}
    spec = json.loads((ROOT/"config/rte_risk_experiment.json").read_text())
    paths = [RUN/"frozen_selection.json", RUN/"result_summary.json", RUN/"forecasts/all_test.csv.gz",
             RUN/"forecasts/selected_empirical_intervals.csv.gz", RUN/"forecasts/probabilistic_test.csv.gz",
             ROOT/"data/france_electricity_hourly.csv", ROOT/"config/rte_risk_experiment.json",
             ROOT/"src/common.py"]
    source_hashes = {str(p.relative_to(ROOT)): sha(p) for p in paths}
    raw = pd.read_csv(ROOT/"data/france_electricity_hourly.csv")
    raw["delivery_time"] = pd.to_datetime(raw.pop("datetime_utc"), utc=True)
    raw = raw.rename(columns={"load_mw": "actual_mw", "rte_forecast_d1_mw": "raw_supplied_rte_mw"})
    assert not raw.delivery_time.duplicated().any()
    parts = []
    for chunk in pd.read_csv(RUN/"forecasts/all_test.csv.gz", chunksize=100000):
        parts.append(chunk.loc[chunk.model.isin(roles.values())])
    forecasts = pd.concat(parts, ignore_index=True)
    for name in ("origin", "delivery_time"):
        forecasts[name] = pd.to_datetime(forecasts[name], utc=True)
    keys = ["origin", "delivery_time"]
    assert not forecasts.duplicated(["model", *keys]).any()
    wide = forecasts.pivot(index=keys, columns="model", values="prediction_mw").rename(
        columns={model: role+"_prediction_mw" for role, model in roles.items()}).reset_index()
    for role, model in roles.items():
        extra = forecasts.loc[forecasts.model.eq(model), keys+["fallback_used", "raw_prediction_mw"]].rename(
            columns={"fallback_used": role+"_fallback_used", "raw_prediction_mw": role+"_raw_prediction_mw"})
        wide = wide.merge(extra, on=keys, validate="one_to_one")
    wide = wide.merge(raw, on="delivery_time", validate="many_to_one")
    missing_target = wide.actual_mw.isna()
    missing_prediction = ~np.isfinite(wide[[role+"_prediction_mw" for role in roles]]).all(axis=1)
    exclusions = wide.loc[missing_target | missing_prediction, keys+["actual_mw"]].copy()
    exclusions["missing_target"] = missing_target.loc[exclusions.index]
    exclusions["missing_any_policy_prediction"] = missing_prediction.loc[exclusions.index]
    exclusions.to_csv(HERE/"excluded_hours.csv", index=False)
    frame = wide.loc[~missing_target & ~missing_prediction].sort_values(keys).reset_index(drop=True)
    assert len(frame) == summary["matched_scored_hours"] == 8759
    local = frame.delivery_time.dt.tz_convert("Europe/Paris")
    frame["delivery_local"] = local.map(lambda t: t.isoformat())
    frame["local_day"] = local.dt.strftime("%Y-%m-%d")
    frame["local_hour"] = local.dt.hour
    frame["month"] = local.dt.strftime("%Y-%m")
    frame["weekday"] = local.dt.day_name()
    holidays = set().union(*(holiday_dates(y) for y in sorted(set(local.dt.year))))
    frame["national_public_holiday"] = [day in holidays for day in local.dt.date]
    frame["cold_evening"] = frame.local_hour.isin(spec["cold_evening"]["hours_local"]) & frame.temperature_c.lt(spec["cold_evening"]["temperature_below_c"])
    for role in roles:
        frame[role+"_signed_error_mw"] = frame[role+"_prediction_mw"]-frame.actual_mw
    expected_mae = {"selected": summary["candidate_MAE_MW"], "rte": summary["RTE_MAE_MW"],
                    "prior": summary["previous_selected_MAE_MW"]}
    for role in roles:
        assert np.isclose(metrics(frame, role)["MAE_MW"], expected_mae[role], rtol=0, atol=1e-7)
    frame.to_csv(HERE/"matched_hourly.csv", index=False)
    monthly = grouped(frame, roles, "month")
    hourly = grouped(frame, roles, "local_hour")
    monthly.to_csv(HERE/"monthly_mae.csv", index=False)
    hourly.to_csv(HERE/"local_hour_mae.csv", index=False)
    daily_rows = []
    for day, part in frame.groupby("local_day", sort=True):
        expected = len(local_day_grid(day))
        complete = len(part) == expected
        conditions = {"temperature_min_c": float(part.temperature_c.min()),
                      "temperature_mean_c": float(part.temperature_c.mean()),
                      "temperature_max_c": float(part.temperature_c.max()),
                      "actual_observed_peak_mw": float(part.actual_mw.max()),
                      "weekday": part.weekday.iloc[0], "national_public_holiday": bool(part.national_public_holiday.iloc[0])}
        for role, model in roles.items():
            error = part[role+"_signed_error_mw"]
            daily_rows.append({"local_day": day, "role": role, "model": model,
                "expected_hours": expected, "complete_day": complete, **metrics(part, role), **conditions,
                "observed_hours_signed_energy_error_MWh": float(error.sum()),
                "observed_hours_shortage_proxy_MWh": float((-error).clip(lower=0).sum()),
                "observed_hours_surplus_proxy_MWh": float(error.clip(lower=0).sum()),
                "daily_signed_energy_error_MWh": float(error.sum()) if complete else np.nan,
                "daily_absolute_energy_error_MWh": float(abs(error.sum())) if complete else np.nan,
                "daily_absolute_peak_error_MW": float(abs(part[role+"_prediction_mw"].max()-part.actual_mw.max())) if complete else np.nan})
    daily = pd.DataFrame(daily_rows)
    daily.to_csv(HERE/"daily_errors.csv", index=False)
    worst = daily.loc[daily.role.eq("selected")].nlargest(5, "MAE_MW").copy()
    worst["rank"] = np.arange(1, 6)
    for role in ("rte", "prior"):
        other = daily.loc[daily.role.eq(role)].set_index("local_day")
        worst[role+"_MAE_MW"] = worst.local_day.map(other.MAE_MW)
        worst[role+"_bias_MW"] = worst.local_day.map(other.bias_prediction_minus_actual_MW)
    worst["ranking_rule"] = "Largest selected-model daily MAE over matched observed hours"
    worst["cause_status"] = "Unproven; measured calendar/temperature conditions are descriptive"
    worst.to_csv(HERE/"worst_five_days.csv", index=False)
    frame.loc[frame.local_day.isin(worst.local_day)].to_csv(HERE/"worst_five_days_hourly.csv", index=False)
    cold = frame.loc[frame.cold_evening]
    cold_daily = grouped(cold, roles, "local_day").rename(columns={"segment": "local_day"})
    cold_daily.to_csv(HERE/"cold_day_errors.csv", index=False)
    selected_cold = cold_daily.loc[cold_daily.role.eq("selected")].sort_values(["MAE_MW", "local_day"], ascending=[False, True])
    worst_cold_day = selected_cold.local_day.iloc[0]
    trajectory = frame.loc[frame.local_day.eq(worst_cold_day)].copy()
    empirical = pd.read_csv(RUN/"forecasts/selected_empirical_intervals.csv.gz")
    native = pd.read_csv(RUN/"forecasts/probabilistic_test.csv.gz")
    empirical = empirical.loc[empirical.model.eq(roles["selected"])]
    native = native.loc[native.model.eq(frozen["selected_quantile"]) & native.method.eq("native_quantile_lightgbm")]
    qcols = ["q05", "q075", "q50", "q95", "q975"]
    for interval, prefix in ((empirical, "selected_empirical_"), (native, "separate_native_quantile_model_")):
        for name in keys:
            interval[name] = pd.to_datetime(interval[name], utc=True)
        interval = interval[keys+qcols+["model", "method"]].rename(columns={c: prefix+c for c in qcols+["model", "method"]})
        trajectory = trajectory.merge(interval, on=keys, how="left", validate="one_to_one")
        assert trajectory[[prefix+"q05", prefix+"q95"]].notna().all().all()
        trajectory[prefix+"central90_covered"] = trajectory.actual_mw.ge(trajectory[prefix+"q05"]) & trajectory.actual_mw.le(trajectory[prefix+"q95"])
        trajectory[prefix+"upper_miss_mw"] = (trajectory.actual_mw-trajectory[prefix+"q95"]).clip(lower=0)
    trajectory.to_csv(HERE/"worst_cold_day_hourly.csv", index=False)
    support = {"matched_hours": len(frame), "excluded_missing_targets": int(missing_target.sum()),
               "excluded_missing_policy_predictions": int(missing_prediction.sum()), "cold_hours": len(cold),
               "cold_days": int(cold.local_day.nunique()),
               "cold_DJF_winters": int((pd.to_datetime(cold.local_day).dt.year+pd.to_datetime(cold.local_day).dt.month.eq(12).astype(int))
                                       .loc[pd.to_datetime(cold.local_day).dt.month.isin([12, 1, 2])].nunique()),
               "cold_support_by_month": cold.groupby("month").size().astype(int).to_dict(),
               "incomplete_local_days": daily.loc[daily.role.eq("selected") & ~daily.complete_day, ["local_day", "n", "expected_hours"]].to_dict("records")}
    findings = {"source_run": str(RUN.relative_to(ROOT)), "models": roles,
        "separate_native_quantile_model": frozen["selected_quantile"], "support": support,
        "overall": {role: metrics(frame, role) for role in roles}, "cold": {role: metrics(cold, role) for role in roles},
        "worst_five_selected_days": worst.to_dict("records"),
        "worst_cold_day": {"local_day": worst_cold_day, "ranking_rule": "Maximum selected-model MAE over realized cold-evening hours; not a causal diagnosis",
            "whole_day_matched_hours": len(trajectory), "cold_hours": int(trajectory.cold_evening.sum()),
            "selected_cold_metrics": metrics(cold.loc[cold.local_day.eq(worst_cold_day)], "selected"),
            "cold_policy_metrics": {role: metrics(cold.loc[cold.local_day.eq(worst_cold_day)], role) for role in roles},
            "temperature_min_c": float(trajectory.temperature_c.min()), "temperature_max_c": float(trajectory.temperature_c.max()),
            "actual_observed_peak_mw": float(trajectory.actual_mw.max())},
        "monthly_selected_maximum": monthly.loc[monthly.role.eq("selected")].sort_values("MAE_MW", ascending=False).iloc[0].to_dict(),
        "local_hour_selected_maximum": hourly.loc[hourly.role.eq("selected")].sort_values("MAE_MW", ascending=False).iloc[0].to_dict(),
        "interpretation": {"signed_error": "prediction minus actual MW; negative means underforecast",
            "daily_energy": "MW multiplied by one-hour interval duration under the project's hourly-mean convention; complete-day metrics missing for incomplete days",
            "scope": "Descriptive analysis of previously explored historical evaluation data; no model selection or retraining",
            "causes": "Recorded temperatures and calendar context are measured conditions; weather-forecast misses, market effects or behavioral explanations are unproven",
            "intervals": "Selected point model empirical intervals and separately selected native-quantile model intervals are distinct policies"}}
    findings = clean(findings)
    write_json("findings.json", findings)
    max_month = findings["monthly_selected_maximum"]
    max_hour = findings["local_hour_selected_maximum"]
    worst_table = "\n".join(f"| {r.local_day} | {r.MAE_MW:,.1f} | {r.rte_MAE_MW:,.1f} | {r.prior_MAE_MW:,.1f} | {int(r.cold_hours)} |"
                            for r in worst.itertuples())
    text = f"""# Descriptive failure diagnostics

The comparison uses **{len(frame):,} matched hours** for the development-selected point model, supplied-RTE reference policy and prior development-selected RTE-assisted model. **{int(missing_target.sum())} missing target** is excluded; no targets are interpolated. The critical segment contains **{len(cold)} cold-evening hours across {cold.local_day.nunique()} days and {support['cold_DJF_winters']} represented DJF winter**. These are retrospective diagnostics, not new model-selection results.

Signed errors are **prediction minus actual demand**: negative values indicate underforecasting. The selected model's all-hour bias is **{findings['overall']['selected']['bias_prediction_minus_actual_MW']:+,.1f} MW**; its cold-evening bias is **{findings['cold']['selected']['bias_prediction_minus_actual_MW']:+,.1f} MW**. Its cold-evening underforecast frequency is **{findings['cold']['selected']['underforecast_rate']:.1%}**. Counts and the same statistics for both references are in `findings.json`.

The highest selected-model monthly MAE is **{max_month['MAE_MW']:,.1f} MW in {max_month['segment']}** across **{max_month['n']} hours**. The largest local-hour aggregate MAE is **{max_hour['MAE_MW']:,.1f} MW at {int(max_hour['segment']):02d}:00 Europe/Paris**, across **{max_hour['n']} observations**. See `monthly_mae.csv` and `local_hour_mae.csv`; grouping uses local civil time, retaining both autumn repeated-hour observations.

The five worst selected-model days are ranked by daily MAE over matched observed hours in `worst_five_days.csv`; `worst_five_days_hourly.csv` preserves their individual observations. Complete-day energy and peak errors are withheld when a target hour is absent; partial-hour sums are labeled explicitly in `daily_errors.csv`.

| Local day | Selected MAE (MW) | RTE MAE (MW) | Prior MAE (MW) | Cold-evening hours |
| --- | ---: | ---: | ---: | ---: |
{worst_table}

The worst cold day by selected-model MAE over its cold-evening hours is **{worst_cold_day}**, with **{findings['worst_cold_day']['selected_cold_metrics']['MAE_MW']:,.1f} MW** MAE across **{findings['worst_cold_day']['cold_hours']} cold hours**. `worst_cold_day_hourly.csv` contains the full matched daily trajectory, actual demand, temperature, all three point forecasts, selected-model empirical intervals, and the **separate** native quantile model's central/asymmetric endpoints. These intervals must not be represented as one model's interchangeable uncertainty estimates.

Temperature and calendar columns describe measured conditions. They do **not** establish a weather-forecast miss or another causal explanation: historical weather/RTE issue vintages are unavailable. Cold support is sparse, and inspecting worst days does not create independent confirmation evidence. Forecast errors and shortage/energy quantities here are physical-unit diagnostics, not euro savings.

Reproduce with the scientific Python environment: `python reports/manager_report_20261021/diagnostics/build_diagnostics.py`. Only files in this diagnostics directory are written. `provenance.json` records source hashes; all existing experiment artifacts remain unchanged.
"""
    (HERE/"findings.md").write_text(text)
    assert source_hashes == {str(p.relative_to(ROOT)): sha(p) for p in paths}, "Source artifact changed during diagnosis"
    outputs = {str(p.name): sha(p) for p in HERE.iterdir() if p.is_file() and p.name != "provenance.json"}
    write_json("provenance.json", {"created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_sha256": source_hashes, "output_sha256": outputs, "no_source_edits": True,
        "no_training": True, "matched_hours": len(frame), "cold_hours": len(cold)})
    print(json.dumps({"directory": str(HERE), "matched_hours": len(frame), "cold_hours": len(cold),
                      "worst_cold_day": worst_cold_day, "outputs": sorted(outputs)}, indent=2))


if __name__ == "__main__":
    main()
