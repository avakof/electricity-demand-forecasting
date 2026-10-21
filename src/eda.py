from __future__ import annotations

import numpy as np
import pandas as pd

from .common import calendar_features


def development_frame(data, contract):
    end = pd.Timestamp(contract["test"]["start"], tz=contract["timezone"]).tz_convert("UTC")
    dev = data.loc[data.index < end].copy()
    cal = calendar_features(dev.index)
    return dev.join(cal)


def temperature_summary(sample, width=2):
    valid = sample.dropna(subset=["temperature_c", "load_mw", "rte_forecast_d1_mw"]).copy()
    if valid.empty:
        return pd.DataFrame(), {"eligible": 0, "excluded_missing": len(sample)}
    t = valid.temperature_c
    lower = width*np.floor(t.min()/width)
    upper = width*np.floor(t.max()/width)+width
    edges = np.arange(lower,upper+width,width)
    valid["temperature_band"] = pd.cut(t,edges,right=False)
    valid["error"] = valid.load_mw-valid.rte_forecast_d1_mw
    valid["absolute_error"] = valid.error.abs()
    summary = valid.groupby("temperature_band",observed=False).agg(median_load_MW=("load_mw","median"), bias_MW=("error","mean"), MAE_MW=("absolute_error","mean"), observations=("error","count"))
    assert summary.observations.sum() == len(valid)
    return summary, {"eligible": len(valid), "excluded_missing": len(sample)-len(valid), "unassigned_bins": int(valid.temperature_band.isna().sum())}


def hypothesis_register():
    return pd.DataFrame([
        {"id":"H1", "question":"Do calendar cycles explain demand?", "feature":"hour/weekend/season/public holiday interactions", "experiment":"calendar", "status":"test on chronological validation"},
        {"id":"H2", "question":"Does recent demand track structural changes?", "feature":"known weekly lags and pre-cutoff rolling summaries", "experiment":"history_365 versus calendar", "status":"test on chronological validation"},
        {"id":"H3", "question":"Does prior temperature add thermal-memory information?", "feature":"last available temperature and past 24h mean", "experiment":"history_past_weather", "status":"observation vintage approximation applies"},
        {"id":"H4", "question":"Does cold-evening RTE bias recur?", "feature":"pre-cutoff calendar/history and historical errors", "experiment":"rte_correction_proxy versus rte_bias_proxy", "status":"RTE vintage unverified; retrospective only"},
        {"id":"H5", "question":"Does authentic weather improve operational forecasts?", "feature":"archived forecast run, lead time, heating/cooling transforms", "experiment":"weather_archive", "status":"gated: archive missing"},
        {"id":"H6", "question":"How sensitive is skill to future weather knowledge?", "feature":"recorded delivery-temperature heating/cooling transforms", "experiment":"oracle_weather_sensitivity", "status":"oracle; never eligible for selection/deployment"}
    ])
