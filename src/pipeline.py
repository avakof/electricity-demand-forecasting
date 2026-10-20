from __future__ import annotations

import pandas as pd

from .data import availability_gate


def readiness(root, data, origin, contract):
    known = data.loc[data.label_available_at_assumed < origin].load_mw.dropna()
    age = (origin-known.index.max()).total_seconds()/3600 if len(known) else float("inf")
    gate = availability_gate(root,contract)
    return {"origin": origin, "latest_known_demand_age_hours": age, "demand_fresh_under_assumed_delay": age <= contract["demand_publication_delay_hours"]+contract["interval_duration_hours"]+2, "verified_operational_gate_passed": bool(gate.operational_experiment_enabled.all()), "publish_procurement_decision": False, "recommendation": "retrospective research / shadow design only until provenance, vintages, prospective evaluation and freshness pass"}


def monitoring_specification():
    return pd.DataFrame([
        ("ingestion", "file schema, checksums, duplicates, missing intervals", "fail run on ambiguous keys; retain last archived forecast"),
        ("freshness", "weather issue/availability, RTE publication, demand age", "reject stale runs; declared fallback or explicit no forecast"),
        ("snapshot", "origin/delivery keys; maximum input availability", "reject any input after cutoff"),
        ("forecast", "finite positive predictions; complete delivery grid", "RTE if verified/available for assisted model; weekly/comparable/latest demand"),
        ("uncertainty", "coverage, width, upper/lower exceedances, sample size", "flag uncalibrated interval; do not fabricate a band"),
        ("delivery", "runtime and completion before agreed deadline", "publish fallback before deadline or mark missed delivery"),
        ("archive", "predictions, issue versions, model/config/source hashes", "write immutable run identifier before accessing outcomes"),
        ("monitor", "MAE, cold-evening MAE, fallback frequency and drift", "shadow review; refit only under frozen online policy"),
    ],columns=["stage","check","response"])


def risk_register():
    return pd.DataFrame([
        ("unknown vintages", "revised data can leak into historical inputs", "obtain original publication/issue archives; keep operational claims gated"),
        ("unknown hourly aggregation", "target and benchmark may represent different intervals", "verify extraction recipe and interval labels before operational comparison"),
        ("previously explored holdout", "selection may be influenced by full-history EDA", "fresh prospective shadow period"),
        ("national proxy", "utility customer mix may behave differently", "portfolio-specific retraining and validation"),
        ("weather miss", "forecast weather differs from delivery conditions", "authentic weather archive; stale/missing-input policy"),
        ("rare extremes", "few cold or hot observations", "segment counts and trajectory-aware stress evaluation"),
        ("settlement alignment", "hourly target differs from 15-minute market units", "quarter-hour forecasting and settlement integration as production extension"),
        ("financial inference", "lower forecast loss need not mean lower costs", "prices, period-specific rules, instruments and procurement policy required"),
    ],columns=["risk","consequence","mitigation"])
