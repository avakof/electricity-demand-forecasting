"""Predeclared peak-risk analysis; all monetary-looking objectives are MW proxies.

Future realized temperature defines diagnostic/scoring segments only. Decisions
and conditional pools use the past-temperature value available at the origin.
This extension never alters the main point-model selection or predictor code.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .common import forecast_origin
from .research_scores import empirical_crps


SPEC = {
    "version": "peak-risk-v1",
    "evening_hours_local": [18, 19, 20], "cold_threshold_c": 5.,
    "realized_cold_role": "retrospective diagnosis and loss weighting only; never a decision trigger",
    "origin_proxy": "past_temperature_mean_24h < 5 C AND delivery local hour in [18,19,20]; past proxy, not forecast weather",
    "nominal_coverage": .9, "nominal_upper_exceedance": .05,
    "winter_count_definition": "Represented meteorological DJF winter-years; December belongs to next January/February year; not complete seasonal cycles",
    "guardrail_min_observations": 60, "guardrail_min_days": 20,
    "pool_minimums": {"coldrisk_evening": [60, 20], "evening": [120, 40], "overall": [200, 1]},
    "bootstrap_block_days": [7, 14], "bootstrap_repetitions": 500, "random_seed": 42,
    "cold_weight_scenarios": [1, 3, 5], "shortage_surplus_ratios": [1, 3, 9, 19],
    "selection_fit_dates": ["2024-01-01", "2024-07-01"],
    "selection_score_dates": ["2024-07-01", "2025-01-01"],
    "selection_rule": "Conditional only if H2 origin-proxy cold evening has >=60 observations/20 days, every conditional calibration tier is supported, proxy-cold-weighted WIS(weight3) decreases, and observed upper-tail exceedance <=5%; otherwise pooled. No realized-cold selection.",
    "final_calibration_dates": ["2025-01-02", "2025-06-30"],
    "endpoint_convention": "Delivery-date ends exclusive; calibration outcomes also strictly released before first scoring origin",
    "decision_policies": ["median_every_interval", "q95_on_origin_proxy_evening_else_median"],
    "cost_interpretation": "Dimensionless shortage/surplus ratios times MW error: risk proxies, not euros, prices, procurement savings, or optimal bids",
    "conformal_interpretation": "Symmetric conformal bounds do not establish equal-tail 5% probabilities; their center need not be a median",
}


def _write_json(path, payload):
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True, default=str))


def _test_exists(run):
    base = Path(run.path)
    return ((base/"forecasts/all_test.csv.gz").exists()
            or any((base/"forecasts").glob("*__test__*"))
            or any((base/"ledgers").glob("*__test__*")))


def freeze_peak_spec(run):
    """Freeze this extension's code and definitions without reading test outcomes."""
    source_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    payload = {**SPEC, "source_sha256": source_hash}
    path = Path(run.path)/"peak_risk_specification.json"
    if path.exists():
        existing = json.loads(path.read_text())
        if {k: v for k, v in existing.items() if k != "frozen_utc"} != payload:
            raise RuntimeError("Peak-risk specification/code changed; create a new extension run")
        return existing
    if _test_exists(run):
        raise RuntimeError("Cannot first freeze a peak-risk policy after test replay has started")
    payload["frozen_utc"] = datetime.now(timezone.utc).isoformat()
    _write_json(path, payload)
    return payload


def _read(path):
    out = pd.read_csv(path)
    for name in ("origin", "delivery_time", "fit_origin"):
        if name in out:
            out[name] = pd.to_datetime(out[name], utc=True)
    return out


def _decorate(frame, run):
    labels = run.data.labels[["delivery_time", "actual_mw", "realized_temperature_c", "label_available_at"]]
    out = frame.drop(columns=[c for c in labels if c != "delivery_time" and c in frame]).merge(labels, on="delivery_time", how="left", validate="many_to_one")
    columns = ["origin", "delivery_time", "past_temperature_mean_24h"]
    features = run.data.features[columns].drop_duplicates(["origin", "delivery_time"])
    out = out.drop(columns=["past_temperature_mean_24h"], errors="ignore").merge(features, on=["origin", "delivery_time"], how="left", validate="many_to_one")
    local = pd.to_datetime(out.delivery_time, utc=True).dt.tz_convert(run.contract["timezone"])
    out["local_day"] = local.dt.strftime("%Y-%m-%d")
    out["evening"] = local.dt.hour.isin(SPEC["evening_hours_local"])
    out["realized_cold_evening"] = out.evening & out.realized_temperature_c.lt(SPEC["cold_threshold_c"])
    out["precutoff_coldrisk_evening"] = out.evening & out.past_temperature_mean_24h.lt(SPEC["cold_threshold_c"])
    out["winter_id"] = pd.Series(np.where(local.dt.month.eq(12), local.dt.year+1, local.dt.year), index=out.index).where(local.dt.month.isin([12, 1, 2]))
    out["residual_mw"] = out.actual_mw-out.prediction_mw
    return out


def _choose_pool(pool, cold_proxy, evening, spec=SPEC):
    """Select only by origin proxy/hour. Pool requires residual_mw, local_day,
    evening, precutoff_coldrisk_evening, and optionally winter_id columns.
    """
    attempts = (["coldrisk_evening"] if cold_proxy and evening else []) + (["evening"] if evening else []) + ["overall"]
    reasons = []
    for name in attempts:
        mask = (pool.precutoff_coldrisk_evening if name == "coldrisk_evening" else pool.evening) if name != "overall" else pd.Series(True, index=pool.index)
        eligible = pool.loc[mask & np.isfinite(pool.residual_mw)]
        n, days = len(eligible), eligible.local_day.nunique()
        need_n, need_days = spec["pool_minimums"][name]
        if n >= need_n and days >= need_days:
            return eligible.residual_mw.to_numpy(dtype=float), {"pool": name, "pool_nobs": n, "pool_ndays": days,
                    "fallback_reason": "; ".join(reasons) or "requested_pool_supported"}
        reasons.append(f"{name}: {n} observations/{days} days below {need_n}/{need_days}")
    return np.array([], dtype=float), {"pool": "unavailable", "pool_nobs": 0, "pool_ndays": 0, "fallback_reason": "; ".join(reasons)}


def _intervals(target, pool, conditional):
    out = target.copy()
    details = []
    for (evening, proxy), group in out.groupby(["evening", "precutoff_coldrisk_evening"], sort=False):
        residuals, meta = _choose_pool(pool, bool(proxy) if conditional else False, bool(evening) if conditional else False)
        for q in (.05, .5, .95):
            out.loc[group.index, f"q{int(q*100):02d}"] = group.prediction_mw + (np.quantile(residuals, q) if len(residuals) else np.nan)
        out.loc[group.index, "CRPS_MW"] = empirical_crps(group.residual_mw, residuals)
        out.loc[group.index, "pool"] = meta["pool"]
        details.append({"requested_evening": bool(evening), "requested_cold_proxy": bool(proxy), "interval_policy": "conditional" if conditional else "pooled", **meta})
    return out, pd.DataFrame(details)


def _summary(frame):
    point = frame.loc[frame.actual_mw.notna() & frame.prediction_mw.notna()]
    good = frame.dropna(subset=["actual_mw", "q05", "q50", "q95"])
    error = point.actual_mw-point.prediction_mw
    result = {"nobs": len(point), "ndays": point.local_day.nunique(), "nwinters": point.winter_id.nunique(),
              "MAE_MW": error.abs().mean(), "mean_underprediction_MW": error.clip(lower=0).mean(),
              "underprediction_MAE_when_positive_MW": error.loc[error.gt(0)].mean(), "interval_nobs": len(good)}
    if not len(good):
        return {**result, "coverage90": np.nan, "width_MW": np.nan, "pinball95": np.nan,
                "upper_exceedance_rate": np.nan, "upper_exceedance_mean_severity_MW": np.nan,
                "upper_exceedance_conditional_severity_MW": np.nan, "IS90": np.nan, "WIS": np.nan,
                "upper_guardrail_status": "insufficient_interval_evidence"}
    y, lo, mid, hi = [good[c].to_numpy(dtype=float) for c in ("actual_mw", "q05", "q50", "q95")]
    score = hi-lo+20*np.where(y<lo, lo-y, 0.)+20*np.where(y>hi, y-hi, 0.)
    excess = np.maximum(y-hi, 0.)
    rate = float(np.mean(y>hi))
    supported = len(good) >= SPEC["guardrail_min_observations"] and good.local_day.nunique() >= SPEC["guardrail_min_days"]
    return {**result, "coverage90": float(np.mean((y>=lo)&(y<=hi))), "width_MW": float(np.mean(hi-lo)),
            "pinball95": float(np.maximum(.95*(y-hi), -.05*(y-hi)).mean()),
            "upper_exceedance_rate": rate, "upper_exceedance_mean_severity_MW": float(excess.mean()),
            "upper_exceedance_conditional_severity_MW": float(excess[excess>0].mean()) if (excess>0).any() else 0.,
            "IS90": float(score.mean()), "WIS": float(((.5*np.abs(y-mid)+.05*score)/1.5).mean()),
            "upper_guardrail_status": ("insufficient_segment_evidence" if not supported else "observed_rate_within_nominal_5pct" if rate <= .05 else "observed_upper_tail_exceeds_nominal_5pct"),
            "CRPS_MW": good.CRPS_MW.mean() if "CRPS_MW" in good else np.nan}


def _upper_bootstrap(full, segment, width):
    """Resample contiguous calendar-day blocks, retaining days with zero segment count."""
    first, last = full.local_day.min(), full.local_day.max()
    days = pd.date_range(first, last, freq="D")
    eligible = segment.dropna(subset=["actual_mw", "q95"]).copy()
    eligible["exceeded"] = eligible.actual_mw.gt(eligible.q95).astype(int)
    counts = eligible.groupby("local_day").agg(exceeded=("exceeded", "sum"), n=("exceeded", "size"))
    counts.index = pd.to_datetime(counts.index)
    daily = counts.reindex(days, fill_value=0).to_numpy(dtype=float)
    if len(days) < width or not daily[:, 1].sum():
        return {f"upper_rate_ci{width}_low": np.nan, f"upper_rate_ci{width}_high": np.nan}
    rng = np.random.default_rng(SPEC["random_seed"]+width)
    values = []
    for _ in range(SPEC["bootstrap_repetitions"]):
        starts = rng.integers(0, len(days)-width+1, size=int(np.ceil(len(days)/width)))
        index = np.concatenate([np.arange(i, i+width) for i in starts])[:len(days)]
        a, n = daily[index].sum(axis=0)
        if n:
            values.append(a/n)
    bounds = np.quantile(values, [.025, .975]) if values else [np.nan, np.nan]
    return {f"upper_rate_ci{width}_low": float(bounds[0]), f"upper_rate_ci{width}_high": float(bounds[1])}


def _validation_forecasts(run, key, entry):
    if key == "rte_proxy":
        paths = sorted((Path(run.path)/"forecasts").glob("rte_proxy__development_reference__*.csv.gz"))
        if paths:
            return _read(paths[-1])
        data = run.data.phase("2024-01-01", "2025-01-01").copy()
        data["prediction_mw"] = data.rte_at_origin_mw
        data["model"] = key
        return data[["origin", "delivery_time", "local_day", "prediction_mw", "model"]]
    pieces = []
    for fold in run.contract["validation_folds"]:
        prefix = f"{key}__{fold['name']}_trial{entry['trial']}__"
        paths = sorted((Path(run.path)/"forecasts").glob(prefix+"*.csv.gz"))
        if len(paths) != 1:
            raise RuntimeError(f"Expected exactly one selected-trial validation archive for {prefix}; found {len(paths)}")
        pieces.append(_read(paths[0]))
    return pd.concat(pieces, ignore_index=True)


def select_peak_intervals(run):
    """Save the validation-only interval choice BEFORE the final replay."""
    spec = freeze_peak_spec(run)
    frozen = json.loads((Path(run.path)/"frozen_selection.json").read_text())
    primary = frozen["primary_candidate_from_validation"]
    chosen = frozen["selections"]
    destination = Path(run.path)/"peak_interval_selection.json"
    if destination.exists():
        saved = json.loads(destination.read_text())
        if saved["spec_source_sha256"] != spec["source_sha256"] or saved["primary"] != primary:
            raise RuntimeError("Peak interval selection does not match frozen specification")
        return pd.DataFrame(saved["records"])
    if _test_exists(run):
        raise RuntimeError("Cannot first select peak intervals after test replay has started")
    records, pool_records = [], []
    for key in (primary, "rte_proxy"):
        frame = _decorate(_validation_forecasts(run, key, chosen.get(key)), run)
        cutoff = forecast_origin("2024-07-01", run.contract)
        fit = frame.loc[frame.local_day.ge("2024-01-01") & frame.local_day.lt("2024-07-01") & frame.label_available_at.lt(cutoff)]
        evaluation = frame.loc[frame.local_day.ge("2024-07-01") & frame.local_day.lt("2025-01-01")]
        candidates = {}
        for conditional in (False, True):
            pred, pools = _intervals(evaluation, fit, conditional)
            pools["model"], pools["phase"] = key, "2024_H1_to_H2_selection"
            pool_records.append(pools)
            good = pred.dropna(subset=["actual_mw", "q05", "q50", "q95"])
            y, lo, mid, hi = [good[c].to_numpy(dtype=float) for c in ("actual_mw", "q05", "q50", "q95")]
            wis = (.5*np.abs(y-mid)+.05*(hi-lo+20*np.maximum(lo-y, 0)+20*np.maximum(y-hi, 0)))/1.5
            weights = np.where(good.precutoff_coldrisk_evening, 3., 1.)
            critical = _summary(pred.loc[pred.precutoff_coldrisk_evening])
            candidates[conditional] = {"weighted_WIS": float(np.average(wis, weights=weights)) if len(good) else np.nan,
                                       "critical": critical, "supported_pools": bool((pools.pool != "unavailable").all()),
                                       "critical_pool_supported": bool(pools.loc[pools.requested_cold_proxy & pools.requested_evening, "pool"].eq("coldrisk_evening").all())}
        conditional = candidates[True]
        sufficient = conditional["critical"]["interval_nobs"] >= 60 and conditional["critical"]["ndays"] >= 20
        use_conditional = sufficient and conditional["supported_pools"] and conditional["critical_pool_supported"] and conditional["critical"].get("upper_exceedance_rate", np.inf) <= .05 and conditional["weighted_WIS"] < candidates[False]["weighted_WIS"]
        for flag in (False, True):
            records.append({"model": key, "candidate": "conditional" if flag else "pooled", "selected": bool(flag == use_conditional),
                            "weighted_WIS": candidates[flag]["weighted_WIS"], "critical_nobs": candidates[flag]["critical"]["interval_nobs"],
                            "critical_ndays": candidates[flag]["critical"]["ndays"], "critical_upper_exceedance_rate": candidates[flag]["critical"].get("upper_exceedance_rate", np.nan),
                            "selection_reason": "conditional passed predeclared evidence/WIS/upper-tail rules" if use_conditional else "pooled retained: conditional evidence, support, WIS or upper-tail rule not satisfied"})
    payload = {"spec_source_sha256": spec["source_sha256"], "primary": primary,
               "selected_utc": datetime.now(timezone.utc).isoformat(), "records": records}
    _write_json(destination, payload)
    result = pd.DataFrame(records)
    result.to_csv(Path(run.path)/"tables/peak_interval_selection.csv", index=False)
    pd.concat(pool_records, ignore_index=True).to_csv(Path(run.path)/"tables/peak_validation_pools.csv", index=False)
    return result


def _cost_sensitivity(distributions):
    reference = distributions.loc[distributions.model.eq("rte_proxy") & distributions.method.eq("fixed_past_out_of_sample_residual_distribution")]
    if reference.empty:
        raise RuntimeError("RTE empirical-distribution reference is required for fair decision-proxy comparison")
    rows = []
    reference = reference[["origin", "delivery_time", "q50", "q95"]].rename(columns={"q50": "ref_q50", "q95": "ref_q95"})
    for (model, method), group in distributions.groupby(["model", "method"]):
        if "conformal" in method:
            continue  # A conformal interval center is not a certified median.
        paired = group.merge(reference, on=["origin", "delivery_time"], validate="many_to_one").dropna(subset=["actual_mw", "q50", "q95", "ref_q50", "ref_q95"])
        for policy in SPEC["decision_policies"]:
            trigger = paired.precutoff_coldrisk_evening.to_numpy() if policy != "median_every_interval" else np.zeros(len(paired), dtype=bool)
            bid = np.where(trigger, paired.q95, paired.q50)
            ref_bid = np.where(trigger, paired.ref_q95, paired.ref_q50)
            y = paired.actual_mw.to_numpy()
            for weight in SPEC["cold_weight_scenarios"]:
                weights = np.where(paired.realized_cold_evening, weight, 1.)
                for ratio in SPEC["shortage_surplus_ratios"]:
                    loss = weights*(ratio*np.maximum(y-bid, 0)+np.maximum(bid-y, 0))
                    ref_loss = weights*(ratio*np.maximum(y-ref_bid, 0)+np.maximum(ref_bid-y, 0))
                    average, ref_average = (float(loss.mean()), float(ref_loss.mean())) if len(paired) else (np.nan, np.nan)
                    rows.append({"model": model, "method": method, "policy": policy, "cold_weight": weight,
                                 "shortage_surplus_ratio": ratio, "nobs": len(paired), "mean_proxy_loss": average,
                                 "RTE_same_policy_proxy_loss": ref_average,
                                 "relative_proxy_improvement_percent": 100*(1-average/ref_average) if ref_average else np.nan,
                                 "weighting": "realized cold-evening diagnostic loss weight; decision trigger uses origin proxy only",
                                 "unit": "weighted MW error proxy; not euros or realized savings"})
    return pd.DataFrame(rows)


def analyze_peak_risk(run, outputs):
    """Apply the already selected interval policy; final outcomes score only."""
    freeze_peak_spec(run)
    selection_path = Path(run.path)/"peak_interval_selection.json"
    if not selection_path.exists():
        raise RuntimeError("Call select_peak_intervals(run) before the final replay")
    selections = json.loads(selection_path.read_text())["records"]
    uncertainty = _decorate(_read(Path(run.path)/"forecasts/uncertainty_test.csv.gz"), run)
    first_test_origin = outputs["test"].origin.min()
    cal = _decorate(outputs["calibration"], run)
    cal = cal.loc[cal.local_day.ge("2025-01-02") & cal.local_day.lt("2025-06-30") & cal.label_available_at.lt(first_test_origin)]
    new_frames, pools = [], []
    for model in dict.fromkeys(row["model"] for row in selections):
        target = _decorate(outputs["test"].loc[outputs["test"].model.eq(model)], run)
        pool = cal.loc[cal.model.eq(model)]
        for conditional in (False, True):
            frame, detail = _intervals(target, pool, conditional)
            frame["method"] = "peak_conditioned_empirical_residuals" if conditional else "peak_pooled_empirical_residuals"
            chosen = next(r["candidate"] for r in selections if r["model"] == model and r["selected"])
            frame["selected_by_2024_validation"] = chosen == ("conditional" if conditional else "pooled")
            new_frames.append(frame)
            detail["model"], detail["phase"] = model, "2025_calibration_to_test"
            detail["selected_by_2024_validation"] = frame.selected_by_2024_validation.iloc[0]
            detail["latest_allowed_calibration_available_at"] = str(pool.label_available_at.max())
            pools.append(detail)
    distributions = pd.concat([uncertainty]+new_frames, ignore_index=True)
    distributions.drop(columns=["actual_mw", "realized_temperature_c", "label_available_at", "residual_mw", "realized_cold_evening", "CRPS_MW"], errors="ignore").to_csv(Path(run.path)/"forecasts/peak_uncertainty_test.csv.gz", index=False)
    rows = []
    for (model, method), full in distributions.groupby(["model", "method"]):
        for segment in ("all", "evening", "realized_cold_evening", "precutoff_coldrisk_evening"):
            part = full if segment == "all" else full.loc[full[segment]]
            record = {"model": model, "method": method, "segment": segment, **_summary(part)}
            for width in SPEC["bootstrap_block_days"]:
                record.update(_upper_bootstrap(full, part, width))
            record["upper_tail_interpretation"] = "diagnostic only: conformal does not promise equal tails" if "conformal" in method else "nominal upper 5% empirical diagnostic; no conditional guarantee"
            rows.append(record)
    scorecard = pd.DataFrame(rows)
    cost = _cost_sensitivity(distributions)
    pool_table = pd.concat(pools, ignore_index=True)
    for name, frame in {"peak_risk_scorecard": scorecard, "peak_proxy_cost_sensitivity": cost,
                        "peak_calibration_pools": pool_table}.items():
        frame.to_csv(Path(run.path)/"tables"/f"{name}.csv", index=False)
    report = """# Peak-risk extension\n\nThe upper tail is evaluated separately during 18:00–20:00 local delivery hours. Realized temperature below 5 C defines retrospective diagnostics and scenario loss weights only. The decision trigger uses past_temperature_mean_24h below 5 C at the forecast origin; this is a past-weather proxy, not tomorrow's forecast temperature.\n\nPooled versus conditional empirical intervals were selected on 2024 H2, using only released 2024 H1 out-of-sample residuals. The final residual pools use January 2–June 29, 2025 outcomes released before the first test forecast. Sparse pools back off through the frozen hierarchy; counts and reasons are saved. Main point forecasts were not retuned.\n\nThe nwinters field counts represented meteorological DJF winters, not complete seasonal cycles. Inspect upper exceedance rates, severity, coverage, width and day-block confidence intervals together. A rate above 5% signals an observed upper-tail shortfall; small segments remain inconclusive. Conformal bounds do not promise equal tails or a median center. Separate hourly intervals are not simultaneous daily-peak or daily-energy intervals.\n\nThe cost table is a scenario analysis of weighted MW errors using hypothetical shortage/surplus ratios. Both the candidate and RTE empirical reference receive the same median/q95 policy and the same origin-only trigger on exactly matched observations. It is not a euro-cost model, realized financial savings, or an optimal procurement policy. A q95 choice corresponds to a 19:1 asymmetric loss only under additional distribution and decision assumptions.\n"""
    (Path(run.path)/"peak_report.md").write_text(report)
    main_report = Path(run.path)/"recommendation.md"
    if main_report.exists():
        primary = json.loads(selection_path.read_text())["primary"]
        choice = next(r["candidate"] for r in selections if r["model"] == primary and r["selected"])
        method = "peak_conditioned_empirical_residuals" if choice == "conditional" else "peak_pooled_empirical_residuals"
        critical = scorecard.loc[scorecard.model.eq(primary) & scorecard.method.eq(method) & scorecard.segment.eq("precutoff_coldrisk_evening")].iloc[0]
        marker = "## Cold-evening upper-tail assessment"
        addition = f"\n\n{marker}\n\nThe validation-selected interval policy is {choice}. On origin-proxy cold evenings it has {int(critical.interval_nobs)} scored intervals, upper-bound exceedance {critical.upper_exceedance_rate:.2%}, and status **{critical.upper_guardrail_status}**. This past-weather trigger is not a forecast of tomorrow\'s cold conditions. Inspect peak_risk_scorecard.csv for realized-cold diagnostic counts and day-block uncertainty. Scenario costs are weighted-MW risk proxies, not euro savings.\n"
        original = main_report.read_text()
        if marker not in original:
            main_report.write_text(original+addition)
    return {"scorecard": scorecard, "proxy_cost_sensitivity": cost,
            "interval_selection": pd.DataFrame(selections), "calibration_pools": pool_table}
