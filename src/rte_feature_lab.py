"""Causal feature laboratory for the RTE D1-assisted retrospective experiment.

``rte_forecast_d1_mw`` is assumed available at the configured origin. Its missing
issue-time lineage is not repaired by this module. Historical actuals and weather
must pass the interval-end plus publication-delay gate. Rolling averages here are
observed-history summaries, not fitted ARIMA moving-average innovations.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .common import TZ, fingerprint, forecast_origin, holiday_dates, local_day_grid
from .features import CALENDAR, HISTORY, PAST_WEATHER, build_origin_features
from .research_data import ResearchData, latest_eligible_time, prepare_research_data


ORIGIN_LAGS = (0, 1, 2, 3, 6, 12, 24, 48, 72, 168)
TARGET_LAGS = (24, 48, 72, 168, 336)
ROLLING_WINDOWS = (6, 24, 72, 168)
ARM_LABELS = {
    "national": "National target; national historical predictors plus RTE D1",
    "regional_only": "National target; regional historical predictors plus RTE D1",
    "combined": "National target; national and regional historical predictors plus RTE D1",
}
RTE_AVAILABILITY = "Supplied D1 column assumed available at origin; original vintage unverified"


def _availability(index, contract):
    return pd.DatetimeIndex(index) + pd.Timedelta(hours=(
        contract["interval_duration_hours"] + contract["demand_publication_delay_hours"]))


def _eligible_national(national, origin, contract):
    start = origin - pd.Timedelta(days=contract["maximum_feature_lookback_days"])
    history = national.loc[start:latest_eligible_time(origin, contract)].copy()
    available = history.get("label_available_at_assumed")
    if available is None:
        available = pd.Series(_availability(history.index, contract), index=history.index)
    return history.loc[pd.to_datetime(available, utc=True).lt(origin)]


def _series_features(series, prefix, origin, deliveries, contract, *, full=True):
    """All origin lags refer to an exact UTC hour, never the last nonmissing row."""
    end = latest_eligible_time(origin, contract)
    s = series.sort_index().astype(float)
    s = s.loc[s.index <= end]
    result = {}
    lags = ORIGIN_LAGS if full else (0, 24)
    for lag in lags:
        result[f"{prefix}_origin_lag{lag}h"] = s.get(end-pd.Timedelta(hours=lag), np.nan)
    windows = ROLLING_WINDOWS if full else (24, 168)
    for window in windows:
        # Reindex to retain missing hours in counts; no interpolation.
        part = s.reindex(pd.date_range(end-pd.Timedelta(hours=window-1), end, freq="h"))
        result[f"{prefix}_mean{window}h"] = part.mean()
        if full:
            result[f"{prefix}_std{window}h"] = part.std()
            result[f"{prefix}_min{window}h"] = part.min()
            result[f"{prefix}_max{window}h"] = part.max()
            result[f"{prefix}_count{window}h"] = int(part.notna().sum())
    target_lags = TARGET_LAGS if full else (168,)
    for lag in target_lags:
        times = deliveries-pd.Timedelta(hours=lag)
        values = s.reindex(times).to_numpy().copy()
        # A target-minus-24h value can be unpublished at a day-ahead origin.
        eligible = _availability(times, contract) < origin
        values[~eligible] = np.nan
        result[f"{prefix}_target_lag{lag}h"] = values
        if full and lag == 24:
            result[f"{prefix}_target_lag24h_available"] = eligible.astype(int)
    if full:
        for span in (24, 168):
            smooth = s.ewm(span=span, adjust=False, min_periods=1, ignore_na=False).mean()
            result[f"{prefix}_ewma{span}h"] = smooth.iloc[-1] if len(smooth) else np.nan
        for lag in (1, 6, 24, 168):
            difference = s.get(end, np.nan)-s.get(end-pd.Timedelta(hours=lag), np.nan)
            result[f"{prefix}_diff{lag}h"] = difference
            if lag in (6, 24):
                result[f"{prefix}_slope{lag}h"] = difference/lag
        result[f"{prefix}_mean24_minus168"] = result[f"{prefix}_mean24h"]-result[f"{prefix}_mean168h"]
        result[f"{prefix}_ewma24_minus168"] = result[f"{prefix}_ewma24h"]-result[f"{prefix}_ewma168h"]
    return result


def _calendar_additions(deliveries):
    local = deliveries.tz_convert(TZ)
    years = range(local.year.min()-1, local.year.max()+2)
    holidays = sorted(set().union(*(holiday_dates(y) for y in years)))
    day = local[0].date()
    yesterday = (pd.Timestamp(day)-pd.Timedelta(days=1)).date()
    tomorrow = (pd.Timestamp(day)+pd.Timedelta(days=1)).date()
    workday = int(day.weekday() < 5 and day not in holidays)
    evening = np.isin(local.hour, [18, 19, 20]).astype(int)
    return {
        "cal_business_day": workday,
        "cal_previous_day_holiday": int(yesterday in holidays),
        "cal_next_day_holiday": int(tomorrow in holidays),
        "cal_days_since_holiday": min((day-d).days for d in holidays if d <= day),
        "cal_days_until_holiday": min((d-day).days for d in holidays if d >= day),
        "cal_bridge_day": int(workday and ((day.weekday() == 0 and tomorrow in holidays)
                                           or (day.weekday() == 4 and yesterday in holidays))),
        "cal_evening": evening,
        "cal_business_evening": workday*evening,
        "cal_day_hours": len(deliveries),
        "cal_week_sin": np.sin(2*np.pi*(local.dayofweek*24+local.hour)/168),
        "cal_week_cos": np.cos(2*np.pi*(local.dayofweek*24+local.hour)/168),
    }


def build_lab_origin_features(national, regional_load, regional_temperature, day, contract):
    """Build one complete 23/24/25-hour target day with a fixed pre-noon origin.

    RTE profile transforms use this delivery day's supplied D1 values only.
    Regional observations have the same explicit assumed delay as national data.
    This is not an archived-vintage operational reconstruction.
    """
    origin = forecast_origin(day, contract)
    deliveries = local_day_grid(day)
    base = build_origin_features(national, day, contract, rte_mode="proxy")
    hist = _eligible_national(national, origin, contract)
    extra = _calendar_additions(deliveries)
    for field, prefix in (("load_mw", "country_load"), ("temperature_c", "country_temperature")):
        extra.update(_series_features(hist[field], prefix, origin, deliveries, contract))
    errors = hist.load_mw-hist.rte_forecast_d1_mw
    extra.update(_series_features(errors, "country_rte_error", origin, deliveries, contract))
    hour_errors = errors.groupby(errors.index.tz_convert(TZ).hour).mean()
    extra["country_rte_error_same_local_hour_mean"] = hour_errors.reindex(deliveries.tz_convert(TZ).hour).to_numpy()
    recent_errors = errors.loc[errors.index >= origin-pd.Timedelta(days=7)]
    recent_by_hour = recent_errors.groupby(recent_errors.index.tz_convert(TZ).hour).mean()
    extra["country_rte_error_same_local_hour_mean7d"] = recent_by_hour.reindex(deliveries.tz_convert(TZ).hour).to_numpy()

    end = latest_eligible_time(origin, contract)
    start = origin-pd.Timedelta(days=contract["maximum_feature_lookback_days"])
    rload = regional_load.loc[start:end]
    rtemp = regional_temperature.reindex(columns=rload.columns).loc[start:end]
    # Requiring all regions avoids interpreting a changed regional composition
    # as a change in aggregate demand or temperature.
    count = len(rload.columns)
    aggregate_load = rload.sum(axis=1, min_count=count) if count else pd.Series(dtype=float, index=rload.index)
    aggregate_temp = rtemp.mean(axis=1).where(rtemp.notna().sum(axis=1).eq(count)) if count else pd.Series(dtype=float, index=rtemp.index)
    extra.update(_series_features(aggregate_load, "regional_load", origin, deliveries, contract))
    extra.update(_series_features(aggregate_temp, "regional_temperature", origin, deliveries, contract))
    for number, region in enumerate(rload.columns):
        prefix = f"region_{number:02d}"
        extra.update(_series_features(rload[region], prefix+"_load", origin, deliveries, contract, full=False))
        recent = rtemp[region].reindex(pd.date_range(end-pd.Timedelta(hours=23), end, freq="h"))
        extra[prefix+"_temperature_mean24h"] = recent.mean()
    for frame, prefix in ((rload, "regional_load"), (rtemp, "regional_temperature")):
        latest = frame.reindex([end]).iloc[0]
        extra[prefix+"_cross_region_std"] = latest.std() if latest.notna().all() else np.nan
        extra[prefix+"_regions_observed"] = int(latest.notna().sum())

    rte = base.rte_at_origin_mw.astype(float)
    complete_rte = rte.notna().all()
    # Day summaries remain missing for an incomplete day instead of using a
    # variable subset of delivery hours. Missingness itself is explicit.
    extra["rte_profile_available_fraction"] = float(rte.notna().mean())
    for name, value in (("mean", rte.mean()), ("max", rte.max()), ("min", rte.min()), ("std", rte.std())):
        extra[f"rte_profile_day_{name}_mw"] = value if complete_rte else np.nan
    extra["rte_profile_ramp_previous_hour_mw"] = rte.diff().to_numpy()
    extra["rte_profile_ramp_next_hour_mw"] = (rte.shift(-1)-rte).to_numpy()
    extra["rte_profile_above_day_mean_mw"] = rte.to_numpy()-extra["rte_profile_day_mean_mw"]
    extra["rte_profile_evening_level_mw"] = rte.to_numpy()*extra["cal_evening"]
    extra["country_rte_minus_weekly_load_mw"] = rte.to_numpy()-base.week_lag_1.to_numpy()
    extra["regional_rte_minus_weekly_sum_mw"] = rte.to_numpy()-extra["regional_load_target_lag168h"]
    for scope in ("country", "regional"):
        t = extra[f"{scope}_temperature_mean24h"]
        cold_proxy = float(t < contract["cold_evening"]["temperature_below_c"]) if pd.notna(t) else np.nan
        extra[f"{scope}_past_cold_evening_proxy"] = cold_proxy*extra["cal_evening"]
        extra[f"{scope}_past_heating_degree24h"] = max(18-t, 0) if pd.notna(t) else np.nan
        extra[f"{scope}_past_cooling_degree24h"] = max(t-22, 0) if pd.notna(t) else np.nan
        extra[f"{scope}_load_ramp_x_temperature_ramp"] = extra[f"{scope}_load_diff24h"]*extra[f"{scope}_temperature_diff24h"]
    result = pd.concat([base, pd.DataFrame(extra, index=base.index)], axis=1)
    available = []
    if len(hist):
        available.append(pd.to_datetime(hist.label_available_at_assumed, utc=True).max())
    if len(rload):
        available.append(_availability(rload.index, contract).max())
    if len(rtemp):
        available.append(_availability(rtemp.index, contract).max())
    result["max_history_available_at"] = max(available) if available else pd.NaT
    if available:
        assert result.max_history_available_at.lt(origin).all(), "Unpublished historical feature"
    assert result.delivery_time.is_unique
    return result


class FeatureLabData(ResearchData):
    """ResearchData-compatible view; country outcome remains identical in all arms."""

    def columns(self, arm="national", rte_assisted=True, tier="engineered"):
        # Accept columns(arm, 'basic') as well as the inherited boolean argument.
        if isinstance(rte_assisted, str):
            tier, rte_assisted = rte_assisted, True
        arm = "combined" if arm == "regional" else arm
        if arm not in ARM_LABELS:
            raise ValueError(f"Unknown feature arm: {arm}")
        if tier not in ("basic", "engineered"):
            raise ValueError(f"Unknown feature tier: {tier}")
        national = arm in ("national", "combined")
        regional = arm in ("regional_only", "combined")
        columns = list(CALENDAR)
        if national:
            columns += HISTORY+PAST_WEATHER
        if rte_assisted:
            columns += ["rte_at_origin_mw"]
            if national:
                columns += ["rte_bias_recent_mw"]
        if regional:
            columns += [c for c in self.features if c.startswith("region_")]
            columns += ["regional_load_mean24h", "regional_load_mean168h", "regional_temperature_mean24h"]
        if tier == "engineered":
            columns += [c for c in self.features if c.startswith("cal_")]
            if national:
                columns += [c for c in self.features if c.startswith("country_")
                            and (rte_assisted or "rte" not in c)]
            if regional:
                columns += [c for c in self.features if c.startswith("regional_")
                            and (rte_assisted or "rte" not in c)]
            if rte_assisted:
                columns += [c for c in self.features if c.startswith("rte_profile_")]
        columns = list(dict.fromkeys(columns))
        missing = set(columns)-set(self.features)
        if missing:
            raise ValueError(f"Missing feature columns: {sorted(missing)}")
        assert not set(columns) & {"actual_mw", "realized_temperature_c", "load_mw", "temperature_c"}
        if arm == "regional_only":
            assert not set(columns) & set(HISTORY+PAST_WEATHER+["rte_bias_recent_mw"])
            assert not any(c.startswith("country_") for c in columns)
        return columns

    def feature_register(self):
        return feature_register(self)


def feature_register(data):
    """Machine-readable availability and transformation register for every predictor."""
    all_columns = data.columns("combined", tier="engineered")
    records = []
    for column in all_columns:
        if column in CALENDAR or column.startswith("cal_"):
            source, availability = "Known metropolitan French calendar", "Known before origin"
        elif column == "rte_at_origin_mw" or column.startswith("rte_profile_"):
            source, availability = "Same target-day rte_forecast_d1_mw profile", RTE_AVAILABILITY
        elif column.startswith(("region_", "regional_")):
            source = "Regional observed demand/weather; equal-weight regional temperature mean"
            availability = "Interval end plus assumed demand publication delay, strictly before origin"
            if "rte" in column:
                source += "; target-day RTE D1"
                availability += "; "+RTE_AVAILABILITY
        else:
            source = "National observed demand/weather"
            availability = "Interval end plus assumed demand publication delay, strictly before origin"
            if "rte" in column:
                source += "; supplied historical RTE D1 and/or target-day D1"
                availability += "; "+RTE_AVAILABILITY
        if "target_lag" in column:
            transform = "Exact elapsed-UTC-hour target lag; missing unless source published before origin"
        elif "origin_lag" in column:
            transform = "Exact elapsed-hour lag from latest eligible UTC interval; no carry-forward"
        elif "ewma" in column:
            transform = "Exponentially weighted observed history, span in hours; finite declared lookback"
        elif "count" in column or "observed" in column or "fraction" in column:
            transform = "Observed-support or availability count/fraction"
        elif "mean" in column or "std" in column or "min" in column or "max" in column:
            transform = "History rolling/aggregate statistic or known target-day RTE profile statistic"
        elif "slope" in column or "ramp" in column or "diff" in column:
            transform = "Difference, hourly slope, or explicitly named interaction of eligible values"
        elif "proxy" in column:
            transform = "Historical-temperature cold proxy intersected with known target evening hour"
        else:
            transform = "Known calendar, eligible history or named deterministic interaction"
        records.append({"feature": column, "source": source, "transformation": transform,
                        "availability": availability, "missing_value_policy": "Keep missing; model preprocessing fitted on training fold only",
                        "national": column in data.columns("national"),
                        "regional_only": column in data.columns("regional_only"),
                        "combined": True,
                        "status": "Retrospective assumed-availability feature; no D2 product used"})
    return pd.DataFrame(records)


def prepare_feature_lab(root, history_start=None, use_cache=True):
    """Build and hash-cache expanded features without modifying original artifacts."""
    root = Path(root)
    base = prepare_research_data(root, history_start=history_start, use_cache=use_cache)
    inputs = [Path(__file__), root/"config/research_experiment.json",
              root/"data/france_electricity_hourly.csv", root/"data/france_regional_electricity_hourly.csv"]
    inputs += [Path(__file__).with_name(name) for name in ("research_data.py", "features.py", "common.py", "data.py")]
    hashes = {str(p.relative_to(root)): fingerprint(p) for p in inputs}
    provenance = {"source_hashes": hashes, "history_start": str(base.national.index.min()),
                  "rte_availability_assumption": RTE_AVAILABILITY, "arms": ARM_LABELS}
    key = hashlib.sha256(json.dumps(provenance, sort_keys=True).encode()).hexdigest()[:24]
    cache = root/".runtime/rte_feature_lab"/key
    if use_cache and (cache/"features.parquet").exists():
        features = pd.read_parquet(cache/"features.parquet")
    else:
        frames = [build_lab_origin_features(base.national, base.regional_load,
                    base.regional_temperature, day, base.contract)
                  for day in base.features.local_day.drop_duplicates()]
        features = pd.concat(frames, ignore_index=True)
        cache.mkdir(parents=True, exist_ok=True)
        features.to_parquet(cache/"features.parquet", index=False)
        (cache/"provenance.json").write_text(json.dumps(provenance, indent=2)+"\n")
    data = FeatureLabData(base.root, base.contract, base.national, base.regional_load,
                         base.regional_temperature, features, base.labels)
    assert not features.duplicated(["origin", "delivery_time"]).any()
    data.feature_register().to_csv(cache/"feature_register.csv", index=False)
    return data
