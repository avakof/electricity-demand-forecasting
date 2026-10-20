from __future__ import annotations

import numpy as np
import pandas as pd

from .common import TZ, local_day_grid


def empirical_intervals(predictions, historical_predictions, labels, contract):
    """Rolling signed out-of-sample errors available strictly before each origin.

    Earlier test observations may enter the predeclared online calibration policy.
    No in-sample training residuals or future outcomes enter a calibration pool.
    """
    all_forecasts = pd.concat([historical_predictions, predictions], ignore_index=True).drop_duplicates(["origin", "delivery_time", "model"])
    errors = all_forecasts.merge(labels, on="delivery_time", validate="many_to_one")
    errors["error"] = errors.actual_mw-errors.prediction_mw
    outputs = []
    for origin, day in predictions.groupby("origin", sort=True):
        pool = errors.loc[(errors.origin < origin) & (errors.label_available_at < origin) & errors.delivery_time.ge(origin-pd.Timedelta(days=contract["calibration_window_days"]))].error.dropna()
        out = day.copy()
        out["calibration_n"] = len(pool)
        for q in contract["quantiles"]:
            offset = pool.quantile(q) if len(pool) >= contract["calibration_minimum_observations"] else np.nan
            out[f"q{int(q*100):02d}"] = out.prediction_mw+offset
        out["interval_method"] = "rolling_past_out_of_sample_errors"
        outputs.append(out)
    return pd.concat(outputs, ignore_index=True)


def interval_metrics(frame):
    valid = frame.dropna(subset=["actual_mw", "q05", "q50", "q95"])
    if valid.empty:
        return {"eligible_targets":int(frame.actual_mw.notna().sum()),"scored": 0, "interval_availability":0., "coverage_90": np.nan, "width_MW": np.nan, "lower_exceedance": np.nan, "upper_exceedance": np.nan}
    out = {"eligible_targets":int(frame.actual_mw.notna().sum()),"scored": len(valid),"interval_availability":len(valid)/frame.actual_mw.notna().sum(), "coverage_90": ((valid.actual_mw >= valid.q05) & (valid.actual_mw <= valid.q95)).mean(), "width_MW": (valid.q95-valid.q05).mean(), "lower_exceedance": (valid.actual_mw < valid.q05).mean(), "upper_exceedance": (valid.actual_mw > valid.q95).mean()}
    for q in (.05,.5,.95):
        e = valid.actual_mw-valid[f"q{int(q*100):02d}"]
        out[f"pinball_{q:.2f}"] = np.maximum(q*e, (q-1)*e).mean()
    return out


def interval_scorecard(predictions, labels, contract):
    from .common import calendar_features
    frame = predictions.merge(labels, on="delivery_time", validate="many_to_one")
    cal = calendar_features(pd.DatetimeIndex(frame.delivery_time)).reset_index(drop=True)
    frame["hour"] = cal.hour.to_numpy()
    frame["season"] = cal.season.to_numpy()
    frame["cold_evening"] = frame.hour.isin(contract["cold_evening"]["hours_local"]) & frame.realized_temperature_c.lt(contract["cold_evening"]["temperature_below_c"])
    frame["lead_group"] = pd.cut((frame.delivery_time-frame.origin).dt.total_seconds()/3600, [0,24,48,np.inf], labels=["<=24h","24-48h",">48h"])
    rows = [{"segment_type": "overall", "segment": "all", **interval_metrics(frame)}]
    for dimension in ("hour", "season", "cold_evening", "lead_group"):
        for value, group in frame.groupby(dimension, observed=True):
            rows.append({"segment_type": dimension, "segment": str(value), **interval_metrics(group)})
    return pd.DataFrame(rows)


def trajectory_intervals(predictions, historical_predictions, labels, contract, scenarios=300):
    """Resample complete daily residual vectors; match 23/24/25-hour length.

    These are empirical scenarios, not simultaneous-coverage guarantees. DST
    lengths with insufficient complete historical days explicitly remain missing.
    """
    all_p = pd.concat([historical_predictions, predictions], ignore_index=True).drop_duplicates(["origin", "delivery_time", "model"])
    joined = all_p.merge(labels, on="delivery_time", validate="many_to_one")
    joined["day"] = joined.delivery_time.dt.tz_convert(TZ).dt.strftime("%Y-%m-%d")
    pool = []
    for day, group in joined.groupby("day"):
        group = group.sort_values("delivery_time")
        if len(group) != len(local_day_grid(day)) or group[["prediction_mw", "actual_mw"]].isna().any().any():
            continue
        pool.append({"day": day, "available_at": group.label_available_at.max(), "origin": group.origin.iloc[0], "error": (group.actual_mw-group.prediction_mw).to_numpy()})
    rng = np.random.default_rng(contract["random_seed"])
    records = []
    for origin, group in predictions.groupby("origin", sort=True):
        group = group.sort_values("delivery_time")
        histories = [item["error"] for item in pool if item["available_at"] < origin and item["origin"] < origin and len(item["error"]) == len(group) and item["available_at"] >= origin-pd.Timedelta(days=365)]
        record = {"origin": origin, "day": group.local_day.iloc[0], "hours": len(group), "historical_complete_days": len(histories), "status": "insufficient_history"}
        if len(histories) >= contract["trajectory_minimum_days"] and group.prediction_mw.notna().all():
            noise = np.array(histories)[rng.integers(0,len(histories),size=scenarios)]
            draws = group.prediction_mw.to_numpy()[None,:]+noise
            record.update({"status": "ok", "energy_q05_MWh": np.quantile(draws.sum(axis=1),.05), "energy_q95_MWh": np.quantile(draws.sum(axis=1),.95), "peak_q05_MW": np.quantile(draws.max(axis=1),.05), "peak_q95_MW": np.quantile(draws.max(axis=1),.95)})
        records.append(record)
    return pd.DataFrame(records)
