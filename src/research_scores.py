"""Matched point scores, honest empirical distributions and frozen conformal.

CRPS is computed for a fully specified empirical residual distribution. Native
three-quantile forecasts receive pinball/interval scores, not invented CRPS.
"""
from __future__ import annotations

import json
import numpy as np
import pandas as pd

from .common import forecast_origin, write_json
from .metrics import (daily_metrics, matched_scorecard, paired_block_bootstrap,
                      scorecard, segment_scorecard)
from .uncertainty import interval_metrics, interval_scorecard


def empirical_crps(errors, residuals):
    """Exact CRPS of prediction + the equally weighted empirical residual CDF."""
    residuals = np.sort(np.asarray(residuals, dtype=float))
    residuals = residuals[np.isfinite(residuals)]
    if not len(residuals):
        return np.full(len(errors), np.nan)
    errors = np.asarray(errors, dtype=float)
    n = len(residuals)
    prefix = np.r_[0., np.cumsum(residuals)]
    k = np.searchsorted(residuals, errors)
    first = (errors*(2*k-n)+prefix[-1]-2*prefix[k])/n
    half_pairwise = np.sum((2*np.arange(1, n+1)-n-1)*residuals)/(n*n)
    return first-half_pairwise


def conformal_radius(scores, alpha=.1):
    if not np.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError("alpha must lie strictly between zero and one")
    scores = np.sort(np.asarray(scores, dtype=float))
    scores = scores[np.isfinite(scores)]
    if (scores < 0).any():
        raise ValueError("Absolute-error conformity scores must be nonnegative")
    k = int(np.ceil((len(scores)+1)*(1-alpha)))
    return float(scores[k-1]) if len(scores) and k <= len(scores) else float("inf")


def interval_summary(frame):
    out = interval_metrics(frame)
    good = frame.dropna(subset=["actual_mw", "q05", "q50", "q95"])
    if not len(good):
        return out
    y, lo, hi = good.actual_mw.to_numpy(), good.q05.to_numpy(), good.q95.to_numpy()
    # Multiplication by a false mask would produce NaN for infinite bounds.
    # An unbounded conformal interval has infinite width/score, not a missing score.
    interval_score = hi-lo+20*np.where(y<lo, lo-y, 0.)+20*np.where(y>hi, y-hi, 0.)
    out["interval_score_90"] = float(interval_score.mean())
    # WIS for one central interval plus median: denominator K+1/2 = 1.5.
    out["WIS_one_interval"] = float((.5*np.abs(y-good.q50.to_numpy())+.05*interval_score).mean()/1.5)
    out["CRPS_MW"] = frame.CRPS_MW.mean() if "CRPS_MW" in frame else np.nan
    return out


def strict_conformal(run, selected):
    """One continuous replay guarantees the SAME fitted predictor in both blocks."""
    frozen = json.loads((run.path/"frozen_selection.json").read_text())
    key = frozen["primary_candidate_from_validation"]
    entry = selected[key]
    tag = f"{key}__fixed_conformal"
    c, t = run.contract["calibration"], run.contract["test"]
    continuous = run.replay(entry["family"], entry["params"], c["start"], t["end"],
                            arm=entry["arm"], tag=tag, phase="fixed_calibration_and_test", fixed=True)
    assert continuous.fit_origin.dropna().nunique() == 1
    assert continuous.raw_prediction_mw.notna().mean() >= .95, "Fixed conformal predictor failed; fallback-only forecasts are not model evidence"
    first_test_origin = forecast_origin(t["start"], run.contract)
    cal = continuous.loc[continuous.local_day.ge(c["start"]) & continuous.local_day.lt(c["end"])].merge(
        run.data.labels[["delivery_time", "actual_mw", "label_available_at"]], on="delivery_time", validate="many_to_one")
    cal = cal.loc[cal.label_available_at.lt(first_test_origin)]
    scores = (cal.actual_mw-cal.prediction_mw).abs().dropna()
    radius = conformal_radius(scores)
    test = continuous.loc[continuous.local_day.ge(t["start"]) & continuous.local_day.lt(t["end"])].copy()
    test["q05"] = test.prediction_mw-radius
    test["q50"] = test.prediction_mw
    test["q95"] = test.prediction_mw+radius
    test["method"] = "fixed_predictor_split_conformal"
    test["calibration_n"] = len(scores)
    test.to_csv(run.path/"forecasts/fixed_conformal_test.csv.gz", index=False)
    write_json(run.path/"tables/conformal_report.json", {
        "point_model": key, "calibration_rows": len(scores), "radius_MW": radius,
        "single_parameter_fit_origin": str(continuous.fit_origin.dropna().iloc[0]),
        "latest_calibration_available_at": str(cal.label_available_at.max()),
        "first_test_origin": str(first_test_origin),
        "guarantee": "No exchangeability/conditional coverage guarantee established for these dependent time series",
        "calibration_seasons": "winter/spring and part of June; not a full seasonal cycle"})
    return test


def uncertainty_comparison(run, calibration, test, conformal):
    first_origin = test.origin.min()
    labels = run.data.labels
    calibrated = calibration.merge(labels[["delivery_time", "actual_mw", "label_available_at"]], on="delivery_time", validate="many_to_one")
    calibrated = calibrated.loc[calibrated.label_available_at.lt(first_origin)]
    frames = []
    rows = []
    for model, group in test.groupby("model"):
        past = calibrated.loc[calibrated.model.eq(model)]
        residuals = (past.actual_mw-past.prediction_mw).dropna().to_numpy()
        out = group.copy()
        out["method"] = "fixed_past_out_of_sample_residual_distribution"
        out["calibration_n"] = len(residuals)
        if len(residuals) >= run.contract["calibration_minimum_observations"]:
            for q in run.contract["quantiles"]:
                out[f"q{int(q*100):02d}"] = out.prediction_mw+np.quantile(residuals, q)
        else:
            out[["q05", "q50", "q95"]] = np.nan
        scored = out.merge(labels, on="delivery_time", validate="many_to_one")
        scored["CRPS_MW"] = empirical_crps(scored.actual_mw-scored.prediction_mw, residuals)
        rows.append({"model": model, "method": out.method.iloc[0], "calibration_n": len(residuals), **interval_summary(scored)})
        frames.append(out)
        if {"q05", "q50", "q95"}.issubset(group) and group.q05.notna().any():
            native = group.merge(labels, on="delivery_time", validate="many_to_one")
            rows.append({"model": model, "method": "native_lightgbm_three_quantiles", **interval_summary(native)})
            copy = group.copy()
            copy["method"] = "native_lightgbm_three_quantiles"
            frames.append(copy)
    scored = conformal.merge(labels, on="delivery_time", validate="many_to_one")
    rows.append({"model": conformal.model.iloc[0], "method": "fixed_predictor_split_conformal", **interval_summary(scored)})
    frames.append(conformal)
    distributions = pd.concat(frames, ignore_index=True)
    distributions.to_csv(run.path/"forecasts/uncertainty_test.csv.gz", index=False)
    result = pd.DataFrame(rows)
    result.to_csv(run.path/"tables/uncertainty_scorecard.csv", index=False)
    segments = []
    for (model, method), group in distributions.groupby(["model", "method"]):
        segment = interval_scorecard(group, labels, run.contract)
        segment["model"], segment["method"] = model, method
        segments.append(segment)
    pd.concat(segments, ignore_index=True).to_csv(run.path/"tables/uncertainty_segments.csv", index=False)
    return result


def build_comparison(run, outputs, conformal):
    test, cal = outputs["test"], outputs["calibration"]
    labels = run.data.labels
    matched, eligible = matched_scorecard(test, labels)
    reference = matched.loc[matched.model.eq("rte_proxy"), "MAE_MW"].iloc[0]
    matched["MAE_improvement_vs_RTE_MW"] = reference-matched.MAE_MW
    matched["MAE_improvement_vs_RTE_percent"] = 100*(1-matched.MAE_MW/reference)
    availability = test.groupby("model").agg(raw_forecast_availability=("raw_prediction_mw", lambda s: s.notna().mean()),
                                             fallback_fraction=("fallback_used", "mean"))
    matched = matched.merge(availability, on="model").sort_values("MAE_MW")
    matched.to_csv(run.path/"tables/test_scorecard.csv", index=False)
    scorecard(test, labels).to_csv(run.path/"tables/unmatched_availability_scorecard.csv", index=False)
    scorecard(test, labels, "raw_prediction_mw").to_csv(run.path/"tables/raw_forecast_scorecard.csv", index=False)
    segment_scorecard(eligible, labels, run.contract).to_csv(run.path/"tables/test_segments.csv", index=False)
    daily = daily_metrics(test, labels)
    daily.to_csv(run.path/"tables/daily_metrics.csv", index=False)
    bootstrap = []
    for model in matched.model:
        if model == "rte_proxy":
            continue
        result = paired_block_bootstrap(eligible, labels, model, block_days=run.contract["bootstrap_block_days"],
                                        repetitions=run.contract["bootstrap_repetitions"], seed=run.contract["random_seed"])
        result["model"] = model
        bootstrap.append(result)
    bootstrap = pd.concat(bootstrap, ignore_index=True)
    bootstrap.to_csv(run.path/"tables/bootstrap.csv", index=False)
    # Failure catalogue states measurements separately from unsupported causes.
    worst = daily.sort_values("absolute_daily_energy_error_MWh", ascending=False).groupby("model").head(5).copy()
    worst["cause_classification"] = "unexplained; inspect inputs/weather/calendar before attribution"
    worst["evidence"] = "Measured daily energy/peak/ramp errors; no archived forecast-weather miss available"
    worst.to_csv(run.path/"tables/worst_days.csv", index=False)
    uncertainty = uncertainty_comparison(run, cal, test, conformal)
    fit_rows = []
    for path in sorted((run.path/"fits").glob("*.json")):
        for record in json.loads(path.read_text()):
            record["artifact"] = path.name
            record["converged"] = (record.get("metadata") or {}).get("converged", np.nan)
            fit_rows.append(record)
    fits = pd.DataFrame(fit_rows)
    fits.drop(columns=["metadata", "params", "traceback"], errors="ignore").to_csv(run.path/"tables/fit_diagnostics.csv", index=False)
    write_recommendation(run, matched, uncertainty, bootstrap, fits)
    return matched, uncertainty


def write_recommendation(run, scores, uncertainty, bootstrap, fits):
    frozen = json.loads((run.path/"frozen_selection.json").read_text())
    key = frozen["primary_candidate_from_validation"]
    candidate = scores.loc[scores.model.eq(key)].iloc[0]
    rte = scores.loc[scores.model.eq("rte_proxy")].iloc[0]
    ci = bootstrap.loc[bootstrap.model.eq(key) & bootstrap.block_days.eq(14)].iloc[0]
    intervals = uncertainty.loc[uncertainty.model.eq(key) & uncertainty.method.eq("fixed_past_out_of_sample_residual_distribution")].iloc[0]
    failed = int(fits.status.ne("fit").sum()) if len(fits) else 0
    text = f"""# Research recommendation: national day-ahead demand

## Improvement over RTE

The candidate selected using 2024 validation, **{key}**, has historical-test MAE
**{candidate.MAE_MW:,.1f} MW**, compared with **{rte.MAE_MW:,.1f} MW** for the supplied RTE proxy on identical eligible intervals.
The MAE improvement is **{candidate.MAE_improvement_vs_RTE_percent:.2f}%**
({candidate.MAE_improvement_vs_RTE_MW:,.1f} MW). The paired 14-day moving-block
95% interval for percentage improvement is **[{ci.relative_low:.2f}, {ci.relative_high:.2f}]%**.
Its raw-forecast availability is {candidate.raw_forecast_availability:.2%}; fallback frequency is {candidate.fallback_fraction:.2%}.

## Uncertainty and failures

The past-out-of-sample residual distribution has empirical 90% interval coverage
**{intervals.coverage_90:.2%}**, mean width **{intervals.width_MW:,.0f} MW**, and empirical-distribution
CRPS **{intervals.CRPS_MW:,.1f} MW**. Consult uncertainty_segments.csv for cold-evening/hour/season counts.
This distribution is a calibrated empirical benchmark, not a conditional-coverage guarantee.
There were {failed} failed fit events across development and evaluation experiments; convergence and fallback diagnostics are saved.
Worst-day causes remain unproven until inspected; weather-forecast error cannot be diagnosed without forecast vintages.

## Recommendation and risks

This is a **retrospective research comparison**, not a production implementation.
Retain RTE as the operational reference until authentic issue-time archives or prospective evidence establish availability and superiority.
The 2025–26 period was already explored; the compact two-configuration CPU search is not an exhaustive model-family comparison.
Original observation revisions, hourly aggregation and publication delays are assumptions. Future recorded weather was excluded from model inputs.
National consumption is a utility-portfolio proxy. Forecast improvements are not measured euro savings.
No serving, scheduling, live monitoring or procurement integration is included.
"""
    (run.path/"recommendation.md").write_text(text)
