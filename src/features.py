from __future__ import annotations

import numpy as np
import pandas as pd

from .common import TZ, calendar_features, forecast_origin, local_day_grid
from .data import select_forecast_run

CALENDAR = ["hour", "weekday", "month", "weekend", "holiday", "hour_sin", "hour_cos", "annual_sin", "annual_cos", "utc_offset_hours", "lead_hours"]
HISTORY = ["week_lag_1", "week_lag_2", "week_lag_3", "week_lag_4", "comparable_mean", "last_load", "mean_24h", "mean_168h", "std_168h", "recent_ramp", "history_fraction_24h"]
PAST_WEATHER = ["last_known_temperature", "past_temperature_mean_24h"]
FORECAST_WEATHER = ["weather_forecast_c", "heating_forecast_c", "cooling_forecast_c"]
ORACLE_WEATHER = ["oracle_temperature_c", "oracle_heating_c", "oracle_cooling_c"]
RTE = ["rte_at_origin_mw", "rte_bias_recent_mw"]


def asof_history(data, origin, contract, demand_archive=None):
    """Only observed intervals ending before cutoff, published before cutoff."""
    start = origin - pd.Timedelta(days=contract["maximum_feature_lookback_days"])
    history = data.loc[start:origin].copy()
    if demand_archive is not None:
        eligible = demand_archive.loc[(demand_archive.available_at < origin) & demand_archive.valid_time.ge(start) & (demand_archive.valid_time + pd.Timedelta(hours=1) <= origin)]
        eligible = eligible.sort_values("available_at").drop_duplicates("valid_time", keep="last").set_index("valid_time")
        history["load_mw"] = eligible.value.reindex(history.index)
        history["label_available_at_assumed"] = eligible.available_at.reindex(history.index)
    else:
        history = history.loc[(history.label_available_at_assumed < origin) & (history.index + pd.Timedelta(hours=1) <= origin)].copy()
    return history


def build_origin_features(data, day, contract, rte_mode="none", rte_archive=None, weather_archive=None, demand_archive=None, oracle_weather=False):
    deliveries = local_day_grid(day)
    origin = forecast_origin(day, contract)
    f = calendar_features(deliveries)
    f["origin"] = origin
    f["delivery_time"] = deliveries
    f["lead_hours"] = (deliveries - origin).total_seconds() / 3600
    history = asof_history(data, origin, contract, demand_archive)
    valid = history.loc[history.load_mw.notna()]
    # Lookup by wall time. Repeated source hours are averaged, never silently
    # dropped. An absent spring-transition hour remains missing.
    wall_index = history.index.tz_convert(TZ).tz_localize(None)
    wall_load = pd.Series(history.load_mw.to_numpy(), index=wall_index).groupby(level=0).mean()
    local_deliveries = deliveries.tz_convert(TZ).tz_localize(None)
    for week in range(1, 5):
        lookup = local_deliveries - pd.DateOffset(weeks=week)
        f[f"week_lag_{week}"] = wall_load.reindex(lookup).to_numpy()
    f["comparable_mean"] = f[[f"week_lag_{week}" for week in range(1, 5)]].mean(axis=1)
    f["last_load"] = valid.load_mw.iloc[-1] if len(valid) else np.nan
    recent24 = history.loc[history.index >= origin - pd.Timedelta(hours=24)]
    recent168 = history.loc[history.index >= origin - pd.Timedelta(hours=168)]
    f["mean_24h"] = recent24.load_mw.mean()
    f["mean_168h"] = recent168.load_mw.mean()
    f["std_168h"] = recent168.load_mw.std()
    f["history_fraction_24h"] = recent24.load_mw.notna().sum() / 24
    f["recent_ramp"] = valid.load_mw.iloc[-1] - valid.load_mw.iloc[-2] if len(valid) >= 2 and valid.index[-1] - valid.index[-2] == pd.Timedelta(hours=1) else np.nan
    known_temperature = history.temperature_c.dropna()
    f["last_known_temperature"] = known_temperature.iloc[-1] if len(known_temperature) else np.nan
    f["past_temperature_mean_24h"] = recent24.temperature_c.mean()
    f["max_history_available_at"] = valid.label_available_at_assumed.max() if len(valid) else pd.NaT
    if rte_mode == "proxy":
        f["rte_at_origin_mw"] = data.rte_forecast_d1_mw.reindex(deliveries).to_numpy()
        past = history.loc[history.index >= origin - pd.Timedelta(days=contract["bias_window_days"])]
        f["rte_bias_recent_mw"] = (past.load_mw - past.rte_forecast_d1_mw).mean()
        f["rte_version_status"] = "provided_column_assumed_at_cutoff_RETROSPECTIVE"
    elif rte_mode == "archive":
        values, meta = select_forecast_run(rte_archive, origin, deliveries, contract["rte_archive_max_age_hours"])
        f["rte_at_origin_mw"] = values.to_numpy()
        # Historical errors must refer to the version accessible at its original
        # forecast origin, not the latest revision accessible today.
        past_forecasts = pd.Series(np.nan, index=history.index)
        for local_day in np.unique(history.index.tz_convert(TZ).strftime("%Y-%m-%d")):
            grid = local_day_grid(local_day)
            values_old, _ = select_forecast_run(rte_archive, forecast_origin(local_day, contract), grid, contract["rte_archive_max_age_hours"])
            shared = past_forecasts.index.intersection(grid)
            past_forecasts.loc[shared] = values_old.reindex(shared)
        bias = history.load_mw - past_forecasts
        f["rte_bias_recent_mw"] = bias.loc[bias.index >= origin - pd.Timedelta(days=contract["bias_window_days"])].mean()
        f["rte_version_status"] = str(meta.get("reason"))
    else:
        f["rte_at_origin_mw"] = np.nan
        f["rte_bias_recent_mw"] = np.nan
        f["rte_version_status"] = "not_used"
    weather, meta = select_forecast_run(weather_archive, origin, deliveries, contract["weather_archive_max_age_hours"])
    f["weather_forecast_c"] = weather.to_numpy()
    f["heating_forecast_c"] = (18 - f.weather_forecast_c).clip(lower=0)
    f["cooling_forecast_c"] = (f.weather_forecast_c - 22).clip(lower=0)
    f["weather_run_status"] = meta.get("reason")
    if oracle_weather:
        f["oracle_temperature_c"] = data.temperature_c.reindex(deliveries).to_numpy()
        f["oracle_heating_c"] = (18 - f.oracle_temperature_c).clip(lower=0)
        f["oracle_cooling_c"] = (f.oracle_temperature_c - 22).clip(lower=0)
    return f.reset_index(drop=True)


def build_supervised(data, contract, rte_mode="proxy", **kwargs):
    local = data.index.tz_convert(TZ)
    start = local.min().tz_localize(None).normalize() + pd.Timedelta(days=contract["maximum_feature_lookback_days"])
    end = local.max().tz_localize(None).normalize()
    feature_frames = []
    for day in pd.date_range(start, end, freq="D"):
        grid = local_day_grid(day)
        if grid.max() > data.index.max():
            continue
        feature_frames.append(build_origin_features(data, day, contract, rte_mode=rte_mode, **kwargs))
    features = pd.concat(feature_frames, ignore_index=True)
    if features.duplicated(["origin", "delivery_time"]).any():
        raise AssertionError("Duplicate origin/delivery keys.")
    labels = data[["load_mw", "temperature_c", "label_available_at_assumed", "reconciliation_flag"]].rename(columns={"load_mw": "actual_mw", "temperature_c": "realized_temperature_c", "label_available_at_assumed": "label_available_at"}).rename_axis("delivery_time").reset_index()
    if kwargs.get("demand_archive") is not None:
        truth = kwargs["demand_archive"].sort_values("available_at").drop_duplicates("valid_time", keep="last").set_index("valid_time")
        labels["actual_mw"] = truth.value.reindex(labels.delivery_time).to_numpy()
        labels["label_available_at"] = pd.to_datetime(truth.available_at.reindex(labels.delivery_time).to_numpy(), utc=True)
    return features, labels


def prediction_fallback(features, raw, rte_assisted=False):
    output = pd.Series(np.asarray(raw, dtype=float), index=features.index)
    source = pd.Series("model", index=features.index)
    if rte_assisted:
        mask = output.isna() & features.rte_at_origin_mw.notna()
        output.loc[mask] = features.loc[mask, "rte_at_origin_mw"]
        source.loc[mask] = "rte"
    for col, label in (("week_lag_1", "weekly"), ("comparable_mean", "comparable"), ("last_load", "latest_observation")):
        mask = output.isna() & features[col].notna()
        output.loc[mask] = features.loc[mask, col]
        source.loc[mask] = label
    source.loc[output.isna()] = "unavailable"
    return output, source
