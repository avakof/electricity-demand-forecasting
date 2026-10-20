from __future__ import annotations

import numpy as np
import pandas as pd

from .common import TZ, calendar_features, local_day_grid


def point_metrics(frame, prediction="prediction_mw"):
    eligible = frame.actual_mw.notna()
    scored = frame.loc[eligible & frame[prediction].notna()]
    error = scored.actual_mw - scored[prediction]
    return {"eligible_targets": int(eligible.sum()), "scored": len(scored), "missing_targets": int((~eligible).sum()), "availability": len(scored) / eligible.sum() if eligible.sum() else np.nan, "MAE_MW": error.abs().mean(), "RMSE_MW": np.sqrt(error.pow(2).mean()), "bias_actual_minus_forecast_MW": error.mean(), "WAPE_percent": 100 * error.abs().sum() / scored.actual_mw.abs().sum() if len(scored) and scored.actual_mw.abs().sum() else np.nan}


def scorecard(predictions, labels, prediction="prediction_mw"):
    joined = predictions.merge(labels, on="delivery_time", how="left", validate="many_to_one")
    return joined.groupby("model", sort=False).apply(lambda group: pd.Series(point_metrics(group, prediction)), include_groups=False).reset_index()


def matched_scorecard(predictions, labels, prediction="prediction_mw"):
    pivot = predictions.pivot(index=["origin", "delivery_time"], columns="model", values=prediction)
    common = pivot.dropna().reset_index()[["origin", "delivery_time"]]
    matched = predictions.merge(common, on=["origin", "delivery_time"], validate="many_to_one")
    return scorecard(matched, labels, prediction), matched


def segment_scorecard(predictions, labels, contract):
    frame = predictions.merge(labels, on="delivery_time", how="left", validate="many_to_one")
    cal = calendar_features(pd.DatetimeIndex(frame.delivery_time)).reset_index(drop=True)
    for col in ("hour", "season", "holiday"):
        frame[col] = cal[col].to_numpy()
    frame["cold_evening"] = frame.hour.isin(contract["cold_evening"]["hours_local"]) & frame.realized_temperature_c.lt(contract["cold_evening"]["temperature_below_c"])
    frame["extreme_temperature"] = np.select([frame.realized_temperature_c.lt(contract["extreme_temperature_c"][0]),frame.realized_temperature_c.gt(contract["extreme_temperature_c"][1]),frame.realized_temperature_c.isna()],["cold_extreme","hot_extreme","temperature_unknown"],default="other")
    ordered_labels = labels.sort_values("delivery_time").set_index("delivery_time")
    ramps = ordered_labels.actual_mw.diff().where(ordered_labels.index.to_series().diff().eq(pd.Timedelta(hours=1)))
    frame["rapid_demand_change"] = frame.delivery_time.map(ramps.abs().gt(5000)).fillna(False)
    frame["lead_group"] = pd.cut((frame.delivery_time - frame.origin).dt.total_seconds()/3600, bins=[0, 24, 48, np.inf], labels=["<=24h", "24-48h", ">48h"])
    rows = []
    for dimension in ("hour", "season", "holiday", "cold_evening", "extreme_temperature", "rapid_demand_change", "lead_group"):
        for (model, value), group in frame.groupby(["model", dimension], observed=True):
            rows.append({"model": model, "segment_type": dimension, "segment": str(value), **point_metrics(group)})
    return pd.DataFrame(rows)


def daily_metrics(predictions, labels):
    joined = predictions.merge(labels, on="delivery_time", how="left", validate="many_to_one")
    joined["day"] = joined.delivery_time.dt.tz_convert(TZ).dt.strftime("%Y-%m-%d")
    rows = []
    for (model, day), group in joined.groupby(["model", "day"]):
        expected = len(local_day_grid(day))
        ok = len(group) == expected and group[["actual_mw", "prediction_mw"]].notna().all().all()
        if not ok:
            continue
        group = group.sort_values("delivery_time")
        diff = group.actual_mw - group.prediction_mw
        rows.append({"model": model, "day": day, "hours": expected, "absolute_daily_energy_error_MWh": abs(diff.sum()), "absolute_daily_peak_error_MW": abs(group.actual_mw.max()-group.prediction_mw.max()), "ramp_MAE_MW": (group.actual_mw.diff()-group.prediction_mw.diff()).abs().mean(), "shortage_proxy_MWh": diff.clip(lower=0).sum(), "surplus_proxy_MWh": (-diff).clip(lower=0).sum()})
    return pd.DataFrame(rows)


def paired_block_bootstrap(predictions, labels, candidate, reference="rte_proxy", block_days=(7, 14, 28), repetitions=500, seed=42):
    """Paired moving calendar-day blocks; never bridge missing-day gaps."""
    p = predictions.loc[predictions.model.isin([candidate, reference])].pivot(index=["origin", "delivery_time"], columns="model", values="prediction_mw").reset_index()
    if candidate not in p or reference not in p:
        return pd.DataFrame()
    p = p.merge(labels[["delivery_time", "actual_mw"]], on="delivery_time", validate="many_to_one").dropna(subset=[candidate, reference, "actual_mw"])
    p["day"] = p.delivery_time.dt.tz_convert(TZ).dt.tz_localize(None).dt.normalize()
    p["candidate_loss"] = (p.actual_mw-p[candidate]).abs()
    p["reference_loss"] = (p.actual_mw-p[reference]).abs()
    daily = p.groupby("day").agg(candidate_loss=("candidate_loss", "sum"), reference_loss=("reference_loss", "sum"), count=("actual_mw", "count"))
    # Loss sums/counts retain partially observed days. This preserves EXACTLY
    # the same eligible paired hours as headline MAE without imputing targets.
    # Only energy/peak trajectory metrics require a complete local day.
    incomplete_days = sum(count != len(local_day_grid(day)) for day,count in daily["count"].items())
    if daily.empty:
        return pd.DataFrame()
    array = daily.to_numpy(); days = daily.index
    rng = np.random.default_rng(seed)
    rows = []
    for width in block_days:
        windows = [np.arange(i, i+width) for i in range(len(daily)-width+1) if days[i+width-1]-days[i] == pd.Timedelta(days=width-1)]
        if not windows:
            rows.append({"block_days": width, "status": "insufficient_contiguous_days"}); continue
        improvements = []; deltas = []
        for _ in range(repetitions):
            index = np.concatenate([windows[i] for i in rng.integers(0, len(windows), size=int(np.ceil(len(daily)/width)))])[:len(daily)]
            c, r, n = array[index].sum(axis=0)
            improvements.append(100*(1-c/r) if r else np.nan)
            deltas.append((r-c)/n)
        c, r, n = array.sum(axis=0)
        rows.append({"block_days": width, "status": "ok", "paired_days": len(daily), "partially_observed_days_retained":incomplete_days, "paired_hours":int(n), "relative_improvement_percent": 100*(1-c/r), "relative_low": np.nanquantile(improvements,.025), "relative_high": np.nanquantile(improvements,.975), "absolute_MAE_improvement_MW": (r-c)/n, "absolute_low": np.quantile(deltas,.025), "absolute_high": np.quantile(deltas,.975)})
    return pd.DataFrame(rows)
