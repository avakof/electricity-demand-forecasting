"""Research-v2 data views. UTC grid preserved; availability precedes every origin.

The supplied RTE column is deliberately retained as a retrospective proxy. No
future observed temperature is included in the model feature lists.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .common import TZ, calendar_features, fingerprint, forecast_origin, local_day_grid
from .data import load_inputs
from .features import CALENDAR, HISTORY, PAST_WEATHER, build_supervised


BASE_COLUMNS = CALENDAR + HISTORY + PAST_WEATHER


def load_research_contract(root):
    return json.loads((Path(root) / "config/research_experiment.json").read_text())


def latest_eligible_time(origin, contract):
    delay = contract["interval_duration_hours"] + contract["demand_publication_delay_hours"]
    return (pd.Timestamp(origin) - pd.Timedelta(hours=delay) - pd.Timedelta(nanoseconds=1)).floor("h")


def calendar_exog(index):
    """Known calendar only, including the unobserved bridge for SARIMAX."""
    frame = calendar_features(pd.DatetimeIndex(index))
    columns = ["hour_sin", "hour_cos", "annual_sin", "annual_cos", "weekend", "holiday"]
    local = pd.DatetimeIndex(index).tz_convert(TZ)
    phase = (local.dayofweek * 24 + local.hour).to_numpy()
    frame["week_sin"] = np.sin(2*np.pi*phase/168)
    frame["week_cos"] = np.cos(2*np.pi*phase/168)
    return frame[columns + ["week_sin", "week_cos"]].astype(float)


@dataclass
class ResearchData:
    root: Path
    contract: dict
    national: pd.DataFrame
    regional_load: pd.DataFrame
    regional_temperature: pd.DataFrame
    features: pd.DataFrame
    labels: pd.DataFrame

    @property
    def region_columns(self):
        return [c for c in self.features if c.startswith("region_")]

    def columns(self, arm="national", rte_assisted=False):
        columns = BASE_COLUMNS + (self.region_columns if arm == "regional" else [])
        if rte_assisted:
            columns += ["rte_at_origin_mw", "rte_bias_recent_mw"]
        forbidden = {"actual_mw", "realized_temperature_c", "oracle_temperature_c", "load_mw"}
        assert not forbidden.intersection(columns)
        return columns

    def training_rows(self, origin, window_days=None):
        # Outcomes are joined only for fitting, never put in prediction artifacts.
        train = self.features.loc[self.features.origin.lt(origin)].merge(
            self.labels[["delivery_time", "actual_mw", "label_available_at"]],
            on="delivery_time", validate="many_to_one")
        train = train.loc[train.label_available_at.lt(origin) & train.actual_mw.notna()]
        if window_days:
            train = train.loc[train.delivery_time.ge(origin-pd.Timedelta(days=window_days))]
        if len(train):
            assert train.label_available_at.max() < origin
            valid = train.max_history_available_at.notna()
            assert (train.loc[valid, "max_history_available_at"] < train.loc[valid, "origin"]).all()
        return train

    def history(self, origin, window_days=None):
        end = latest_eligible_time(origin, self.contract)
        history = self.national.loc[:end]
        if window_days:
            history = history.loc[end-pd.Timedelta(days=window_days):]
        return history

    def panel(self, origin, arm="national", window_days=None):
        """One regular UTC grid per series. Missing y remains missing here."""
        hist = self.history(origin, window_days)
        records = [pd.DataFrame({"unique_id": "country", "ds": hist.index.tz_localize(None), "y": hist.load_mw.to_numpy()})]
        if arm == "regional":
            regions = self.regional_load.reindex(hist.index)
            for name in regions:
                records.append(pd.DataFrame({"unique_id": str(name), "ds": hist.index.tz_localize(None), "y": regions[name].to_numpy()}))
        return pd.concat(records, ignore_index=True)

    def phase(self, start, end):
        return self.features.loc[self.features.local_day.ge(start) & self.features.local_day.lt(end)]


def prepare_research_data(root, history_start=None, use_cache=True):
    root = Path(root)
    contract = load_research_contract(root)
    history_start = history_start or contract["matched_history_start"]
    key = "_".join([contract["version"], history_start,
                    fingerprint(root/"data/france_electricity_hourly.csv")[:10],
                    fingerprint(root/"data/france_regional_electricity_hourly.csv")[:10],
                    fingerprint(root/"config/research_experiment.json")[:10],
                    *[fingerprint(Path(__file__).with_name(name))[:10] for name in
                      ("research_data.py", "features.py", "common.py", "data.py")]])
    cache = root/".runtime/research_data"/key
    country, regional = load_inputs(root)
    grid = pd.date_range(country.datetime_utc.min(), country.datetime_utc.max(), freq="h")
    national = country.set_index("datetime_utc").reindex(grid)
    national = national.loc[pd.Timestamp(history_start, tz=TZ).tz_convert("UTC"):].copy()
    national["label_available_at_assumed"] = national.index + pd.Timedelta(
        hours=contract["interval_duration_hours"]+contract["demand_publication_delay_hours"])
    national["reconciliation_flag"] = False  # not used as a predictor
    regional_load = regional.pivot(index="datetime_utc", columns="region", values="load_mw").reindex(national.index)
    regional_temp = regional.pivot(index="datetime_utc", columns="region", values="temperature_c").reindex(national.index)
    if use_cache and (cache/"features.parquet").exists():
        features = pd.read_parquet(cache/"features.parquet")
        labels = pd.read_parquet(cache/"labels.parquet")
    else:
        features, labels = build_supervised(national, contract, rte_mode="proxy")
        # Compute regional histories at the same national origins. Never average
        # the twelve target values into an artificial replacement national truth.
        additions = []
        wall = regional_load.index.tz_convert(TZ).tz_localize(None)
        wall_load = regional_load.set_axis(wall).groupby(level=0).mean()
        for origin, day in features.groupby("origin", sort=True):
            end = latest_eligible_time(origin, contract)
            rh = regional_load.loc[:end].tail(168)
            rt = regional_temp.loc[:end].tail(24)
            lookup = pd.DatetimeIndex(day.delivery_time).tz_convert(TZ).tz_localize(None)-pd.DateOffset(weeks=1)
            prior_week = wall_load.reindex(lookup)
            # Assert even DST-adjusted weekly source hours precede the cutoff.
            assert (lookup < origin.tz_convert(TZ).tz_localize(None)).all()
            addition = pd.DataFrame(index=day.index)
            for number, region in enumerate(regional_load.columns):
                addition[f"region_{number:02d}_last_load"] = rh[region].iloc[-1] if len(rh) else np.nan
                addition[f"region_{number:02d}_mean24"] = rh[region].tail(24).mean()
                addition[f"region_{number:02d}_week_lag"] = prior_week[region].to_numpy()
                addition[f"region_{number:02d}_past_temp24"] = rt[region].mean()
            additions.append(addition)
        features = features.join(pd.concat(additions).sort_index())
        cache.mkdir(parents=True, exist_ok=True)
        features.to_parquet(cache/"features.parquet", index=False)
        labels.to_parquet(cache/"labels.parquet", index=False)
    assert not features.duplicated(["origin", "delivery_time"]).any()
    return ResearchData(root, contract, national, regional_load, regional_temp, features, labels)


def data_summary(data):
    return pd.DataFrame([
        {"dataset": "national", "rows": len(data.national), "series": 1,
         "missing_targets": int(data.national.load_mw.isna().sum()),
         "start_utc": data.national.index.min(), "end_utc": data.national.index.max()},
        {"dataset": "regional", "rows": int(data.regional_load.size), "series": len(data.regional_load.columns),
         "missing_targets": int(data.regional_load.isna().sum().sum()),
         "start_utc": data.regional_load.index.min(), "end_utc": data.regional_load.index.max()}])
