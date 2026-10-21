from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .common import TZ, fingerprint, load_contract, prepare_directories, write_json


def load_inputs(root):
    country = pd.read_csv(root / "data/france_electricity_hourly.csv")
    regional = pd.read_csv(root / "data/france_regional_electricity_hourly.csv")
    for frame in (country, regional):
        frame["datetime_utc"] = pd.to_datetime(frame["datetime_utc"], utc=True, errors="raise")
        if not frame.datetime_utc.eq(frame.datetime_utc.dt.floor("h")).all():
            raise ValueError("Non-hour-aligned source timestamps: audit the upstream aggregation; no silent resampling permitted.")
        for col in ("load_mw", "temperature_c", "rte_forecast_d1_mw"):
            if col in frame:
                frame[col] = pd.to_numeric(frame[col], errors="raise").replace([np.inf, -np.inf], np.nan)
    if country.duplicated("datetime_utc").any():
        raise ValueError("Duplicate country timestamps: resolve provenance; never silently aggregate.")
    if regional.duplicated(["region", "datetime_utc"]).any():
        raise ValueError("Duplicate regional keys: resolve provenance first.")
    return country.sort_values("datetime_utc"), regional.sort_values(["datetime_utc", "region"])


def availability_gate(root, contract=None):
    contract = contract or load_contract(root)
    specs = {"demand": ("demand_observation_archive.csv", ["valid_time", "available_at", "value", "revision_status"]), "rte": ("rte_forecast_archive.csv", ["valid_time", "issued_at", "available_at", "run_id", "value"]), "weather": ("weather_forecast_archive.csv", ["valid_time", "issued_at", "available_at", "run_id", "value"])}
    result = []
    evidence = contract["operational_evidence"]
    for kind, (name, required) in specs.items():
        path = root / "data/raw" / name
        frame = pd.read_csv(path) if path.exists() else pd.DataFrame()
        valid = len(frame) > 0 and set(required).issubset(frame.columns)
        if valid:
            try:
                load_archive(path, forecast=kind != "demand")
            except (ValueError, TypeError):
                valid = False
        result.append({"source": kind, "nonempty_archive_with_schema": bool(valid), "provenance_verified": evidence.get(f"{kind}_vintages_verified", False), "operationally_usable": bool(valid and evidence.get(f"{kind}_vintages_verified", False)), "rows": len(frame)})
    status = pd.DataFrame(result)
    status["interval_aggregation_verified"] = evidence["interval_aggregation_verified"]
    status["operational_experiment_enabled"] = bool(status["operationally_usable"].all() and evidence["interval_aggregation_verified"])
    return status


def field_register(country, regional):
    rows = []
    definitions = {"datetime_utc": ("timestamp", "delivery interval start ASSUMED", "alignment key"), "region": ("category", "administrative region", "geographic key"), "load_mw": ("MW", "consumption power; hourly averaging unverified", "target / past-demand predictor"), "rte_forecast_d1_mw": ("MW", "provided day-ahead forecast; cutoff version unknown", "benchmark / RTE-assisted input"), "temperature_c": ("degree C", "temperature; aggregation and observed/forecast provenance unknown", "past diagnostic; future value sensitivity only")}
    for name, frame in (("country", country), ("regional", regional)):
        for field in frame.columns:
            unit, definition, role = definitions[field]
            rows.append({"dataset": name, "field": field, "dtype": str(frame[field].dtype), "unit": unit, "definition": definition, "role": role, "geography": "country proxy: mainland excl. Corsica only if source identity confirmed" if name == "country" else "12 supplied regions", "timezone": "UTC" if field == "datetime_utc" else "not applicable", "start_utc": frame["datetime_utc"].min(), "end_utc": frame["datetime_utc"].max(), "missing": int(frame[field].isna().sum()), "source_lineage": "CSV provided; extraction/aggregation recipe missing", "revision_status": "unknown", "issued_at": "not supplied", "available_at": "not supplied", "retrospective_assumption": "interval end + 2 hours for observed history; RTE column assumed at origin ONLY in proxy experiment", "operational_before_cutoff": "unverified", "hourly_conversion": "unknown: do not infer from file name"})
    return pd.DataFrame(rows)


def run_audit(root, contract=None):
    prepare_directories(root)
    contract = contract or load_contract(root)
    country, regional = load_inputs(root)
    grid = pd.date_range(country.datetime_utc.min(), country.datetime_utc.max(), freq="h")
    data = country.set_index("datetime_utc").reindex(grid).rename_axis("datetime_utc")
    data["source_timestamp_present"] = data.index.isin(country.datetime_utc)
    data["target_missing"] = data.load_mw.isna()
    data["load_suspicious"] = data.load_mw.le(0) | data.load_mw.diff().abs().gt(15000)
    data["temperature_suspicious"] = data.temperature_c.lt(-30) | data.temperature_c.gt(50)
    data["label_available_at_assumed"] = data.index + pd.Timedelta(hours=contract["interval_duration_hours"] + contract["demand_publication_delay_hours"])
    n_regions = regional.region.nunique()
    agg = regional.assign(txw=regional.temperature_c * regional.load_mw).groupby("datetime_utc").agg(regional_load_mw=("load_mw", lambda s: s.sum(min_count=n_regions)), regional_temperature_mean_c=("temperature_c", "mean"), n_regions=("region", "nunique"), n_loads=("load_mw", "count"), n_temperatures=("temperature_c", "count"), txw=("txw", "sum"))
    agg["regional_temperature_mean_c"] = agg.regional_temperature_mean_c.where(agg.n_temperatures.eq(n_regions))
    agg["regional_temperature_load_weighted_c"] = (agg.txw / agg.regional_load_mw).where(agg.n_loads.eq(n_regions) & agg.n_temperatures.eq(n_regions) & agg.regional_load_mw.gt(0))
    data = data.join(agg.drop(columns="txw"))
    data["load_country_minus_regions_mw"] = data.load_mw - data.regional_load_mw
    data["reconciliation_flag"] = data.load_country_minus_regions_mw.abs().gt(10)
    data.to_csv(root / "data/processed/audited_country.csv.gz", compression="gzip")
    register = field_register(country, regional)
    register.to_csv(root / "reports/audit/data_dictionary.csv", index=False)
    register[["dataset", "field", "issued_at", "available_at", "revision_status", "retrospective_assumption", "operational_before_cutoff"]].to_csv(root / "reports/audit/availability_register.csv", index=False)
    missing = pd.DataFrame({"missing_count": data.isna().sum(), "missing_fraction": data.isna().mean()})
    missing.to_csv(root / "reports/audit/missingness.csv")
    missing_rows = data.loc[data[["load_mw", "rte_forecast_d1_mw", "temperature_c"]].isna().any(axis=1)]
    missing_rows.to_csv(root / "reports/audit/missing_observations.csv")
    # Differences are measured in UTC; repeated local hours retain their offsets.
    frequency = country.datetime_utc.diff().dropna().value_counts().rename_axis("interval").reset_index(name="count")
    frequency.to_csv(root / "reports/audit/frequency_intervals.csv", index=False)
    regional_rows = []
    suspicious_regional = []
    for name, group in regional.groupby("region"):
        times = group.datetime_utc.sort_values()
        expected = pd.date_range(times.min(), times.max(), freq="h")
        regional_rows.append({"region": name, "start": times.min(), "end": times.max(), "rows": len(group), "missing_timestamp_intervals": len(expected.difference(pd.DatetimeIndex(times))), "nonhourly_transitions": int(times.diff().dropna().ne(pd.Timedelta(hours=1)).sum()), "missing_load": int(group.load_mw.isna().sum()), "missing_temperature": int(group.temperature_c.isna().sum())})
        ordered = group.sort_values("datetime_utc").copy()
        change = ordered.load_mw.diff().abs()
        flag = ordered.load_mw.le(0) | ((change > 1000) & (change > .5*ordered.load_mw.shift(1).abs())) | ordered.temperature_c.lt(-30) | ordered.temperature_c.gt(50)
        suspicious_regional.append(ordered.loc[flag])
    pd.DataFrame(regional_rows).to_csv(root / "reports/audit/regional_frequency_and_missingness.csv", index=False)
    pd.concat(suspicious_regional).to_csv(root / "reports/audit/regional_suspicious_retained.csv", index=False)
    overlap = data.assign(country_present=data.source_timestamp_present, rte_present=data.rte_forecast_d1_mw.notna(), weather_present=data.temperature_c.notna(), regional_present=data.n_regions.eq(n_regions)).groupby(["country_present", "rte_present", "weather_present", "regional_present"], dropna=False).size().reset_index(name="hours")
    overlap.to_csv(root / "reports/audit/overlap.csv", index=False)
    delta = data.load_country_minus_regions_mw.dropna()
    audit = {"country_rows": len(country), "regional_rows": len(regional), "regions": n_regions, "country_duplicate_keys": 0, "regional_duplicate_keys": 0, "hourly_grid_missing_rows": int((~data.source_timestamp_present).sum()), "country_target_missing": int(data.load_mw.isna().sum()), "rte_missing": int(data.rte_forecast_d1_mw.isna().sum()), "weather_missing": int(data.temperature_c.isna().sum()), "complete_regional_overlap_hours": int(data.regional_load_mw.notna().sum()), "country_regional_comparable_hours": len(delta), "max_country_regional_difference_mw": float(delta.abs().max()), "reconciliation_flags": int(data.reconciliation_flag.sum()), "no_targets_interpolated": True, "peaks_deleted": 0}
    write_json(root / "reports/audit/audit_summary.json", audit)
    manifest = {"sources": {name: {"sha256": fingerprint(root / "data" / name), "read_only_original": True} for name in ("france_electricity_hourly.csv", "france_regional_electricity_hourly.csv")}, "contract_sha256": fingerprint(root / "config/forecast_contract.json")}
    write_json(root / "data/raw/source_manifest.json", manifest)
    availability_gate(root, contract).to_csv(root / "reports/audit/operational_gate.csv", index=False)
    limits = """# Limitations

- Supplied CSV provenance, hourly averaging, interval labels, and revision vintages are unverified. Interval-start labels and one-hour mean-power values are explicit assumptions. Daily MWh is conditional on these assumptions.
- The RTE documentation describes quarter-hourly source values and subsequent consolidation/definitive revisions. No source-to-hourly conversion recipe was provided; matching timestamps do not prove matching intervals.
- No trustworthy archived RTE forecast issues, weather runs, or original demand observation vintages were supplied. Original observation availability is approximated as interval end plus two hours in retrospective runs only.
- Provided RTE values are an explicitly assumed-cutoff retrospective benchmark. Recorded future temperature is restricted to separately named oracle sensitivity experiments. No operational superiority is established.
- Earlier conversation and legacy EDA inspected the full history. The chronological test period is previously exposed, not pristine. Freezing this implementation now cannot undo that exposure; fresh prospective evaluation is required.
- Missing targets remain missing and are counted. Peaks and the country/regional discrepancy are retained and flagged; source verification is needed before correction.
- National consumption is a utility portfolio proxy. National public holidays omit school holidays and local exceptions. No customer portfolio, price, settlement or procurement data are provided.
- Forecast skill and shortage/surplus energy proxies are not euro savings. Hourly forecasting does not implement quarter-hour market or settlement alignment.
"""
    (root / "reports/audit/limitations.md").write_text(limits)
    return data, register, audit


def read_audited(root):
    frame = pd.read_csv(root / "data/processed/audited_country.csv.gz", parse_dates=["datetime_utc", "label_available_at_assumed"])
    frame["datetime_utc"] = pd.to_datetime(frame.datetime_utc, utc=True)
    frame["label_available_at_assumed"] = pd.to_datetime(frame.label_available_at_assumed, utc=True)
    return frame.set_index("datetime_utc").sort_index()


def load_archive(path, forecast=True):
    frame = pd.read_csv(path)
    columns = ["valid_time", "available_at"] + (["issued_at", "run_id"] if forecast else [])
    if not set(columns + ["value"]).issubset(frame):
        raise ValueError(f"Archive schema missing required columns: {path}")
    for col in ("valid_time", "available_at", "issued_at"):
        if col in frame:
            frame[col] = pd.to_datetime(frame[col], utc=True, errors="raise")
    if forecast and (frame.available_at < frame.issued_at).any():
        raise ValueError("Availability cannot precede issuance.")
    if frame[columns].isna().any().any():
        raise ValueError("Archive contains missing key/publication timestamps.")
    key = ["valid_time", "issued_at", "available_at", "run_id"] if forecast else ["valid_time", "available_at"]
    if frame.duplicated(key).any():
        raise ValueError("Archive has duplicate version keys.")
    frame["value"] = pd.to_numeric(frame.value, errors="raise")
    return frame


def select_forecast_run(archive, origin, deliveries, max_age_hours):
    """Select one latest accessible, sufficiently fresh run for the entire day.

    Partial runs retain missing values; never backfill with a future issue.
    """
    out = pd.Series(np.nan, index=deliveries, dtype=float)
    if archive is None or archive.empty:
        return out, {"available": False, "reason": "missing_archive"}
    eligible = archive.loc[(archive.issued_at <= origin) & (archive.available_at <= origin) & archive.issued_at.ge(origin - pd.Timedelta(hours=max_age_hours)) & archive.valid_time.isin(deliveries)]
    if eligible.empty:
        return out, {"available": False, "reason": "stale_or_unavailable"}
    latest = eligible.sort_values(["issued_at", "available_at", "run_id"]).iloc[-1]
    run = eligible.loc[(eligible.run_id == latest.run_id) & (eligible.issued_at == latest.issued_at)].sort_values("available_at").drop_duplicates("valid_time", keep="last")
    out = run.set_index("valid_time").value.reindex(deliveries)
    return out, {"available": bool(out.notna().all()), "reason": "complete" if out.notna().all() else "incomplete_run", "run_id": latest.run_id, "issued_at": latest.issued_at, "max_available_at": run.available_at.max()}
