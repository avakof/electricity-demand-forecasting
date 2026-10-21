#!/usr/bin/env python3
"""Write a <=550-word manager summary AFTER notebook 08 has completed.

Reporting only: no fitting, selection, recalibration, predictor imports, or
modification of recommendation.md. Every model/method is taken from saved
validation selections; final-test performance never selects the report's model.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import pandas as pd


METHODS = {"pooled": "peak_pooled_empirical_residuals",
           "conditional": "peak_conditioned_empirical_residuals"}


def _json(path: Path):
    if not path.is_file():
        raise RuntimeError(f"Required completed artifact is missing: {path}")
    return json.loads(path.read_text())


def _csv(path: Path, columns: set[str]):
    if not path.is_file():
        raise RuntimeError(f"Required completed artifact is missing: {path}")
    frame = pd.read_csv(path)
    missing = columns.difference(frame.columns)
    if frame.empty or missing:
        raise RuntimeError(f"Artifact is empty or lacks required columns {sorted(missing)}: {path}")
    return frame


def _one(frame: pd.DataFrame, **keys):
    selected = frame
    for key, value in keys.items():
        selected = selected.loc[selected[key].eq(value)]
    if len(selected) != 1:
        raise RuntimeError(f"Expected one completed score row for {keys}; found {len(selected)}")
    return selected.iloc[0]


def _pct(value):
    number = float(value)
    return f"{number:.1%}" if math.isfinite(number) else "unavailable"


def _mw(value):
    number = float(value)
    if math.isnan(number):
        return "unavailable"
    return f"{number:,.0f}" if math.isfinite(number) else "unbounded"


def _rate_ci(row):
    lo, hi = float(row.upper_rate_ci14_low), float(row.upper_rate_ci14_high)
    return f"[{_pct(lo)}, {_pct(hi)}]" if math.isfinite(lo) and math.isfinite(hi) else "unavailable"


def _selected_peak_method(selection, model):
    choices = [record for record in selection["records"]
               if record["model"] == model and record["selected"] is True]
    if len(choices) != 1 or choices[0]["candidate"] not in METHODS:
        raise RuntimeError(f"No unique validation-selected peak interval method for {model}")
    choice = choices[0]["candidate"]
    return choice, METHODS[choice]


def _diagnostics(run_dir):
    path = run_dir/"tables/fit_diagnostics.csv"
    if not path.is_file():
        return ""
    frame = pd.read_csv(path)
    if frame.empty or not {"status", "model", "converged"}.issubset(frame):
        return ""
    failed = int(frame.status.ne("fit").sum())
    statistical = frame.model.str.split("__", n=1).str[0].isin(["arima", "sarima", "sarimax"]) & frame.status.eq("fit")
    convergence_known = frame.converged.astype(str).str.lower().isin(["true", "false", "1", "0", "1.0", "0.0"])
    stopped = statistical & frame.converged.astype(str).str.lower().isin(["false", "0", "0.0"])
    return (f"Statistical nonconvergence: {int(stopped.sum())}/{int((statistical & convergence_known).sum())} recorded fits; "
            f"separate fit failures across all families: {failed}.")


def _stress_note(run_dir):
    path = run_dir/"tables/development_stress_tests.csv"
    if not path.is_file():
        return ""
    frame = pd.read_csv(path)
    required = {"model", "scenario", "MAE_MW", "policy_availability", "fallback_fraction"}
    if not required.issubset(frame):
        raise RuntimeError("Development stress table lacks required reporting columns")
    lightgbm = frame.loc[frame.model.eq("lightgbm__regional")]
    clean = _one(lightgbm, scenario="clean")
    missing = _one(lightgbm, scenario="missing_regional_inputs")
    history = _one(lightgbm, scenario="missing_demand_history")
    return (f"Development-only regional LightGBM stress: all-history loss left "
            f"{_pct(history.policy_availability)} availability; missing regional inputs raised MAE "
            f"{_mw(clean.MAE_MW)}→{_mw(missing.MAE_MW)} MW with "
            f"{_pct(missing.fallback_fraction)} fallback. These guardrail gaps concern that tested policy.")


def write_manager_recommendation(run_dir: Path) -> Path:
    run_dir = run_dir.resolve()
    frozen = _json(run_dir/"frozen_selection.json")
    complete = _json(run_dir/"development_policy_selection.json")
    peak_selection = _json(run_dir/"peak_interval_selection.json")
    peak_spec = _json(run_dir/"peak_risk_specification.json")
    independent = frozen.get("primary_candidate_from_validation")
    policy = complete.get("selected_policy")
    if not independent or peak_selection.get("primary") != independent:
        raise RuntimeError("Main and peak specifications identify different independent models")
    if complete.get("selected_independent_model") != independent or not policy:
        raise RuntimeError("Complete-policy and independent-model freezes do not match")
    if len(complete.get("entries", [])) != 27:
        raise RuntimeError("Complete-policy freeze must include all 23 model variants and 4 baselines")
    if peak_selection.get("spec_source_sha256") != peak_spec.get("source_sha256"):
        raise RuntimeError("Peak selection/source freeze does not match")
    for path in (run_dir/"recommendation.md", run_dir/"peak_report.md",
                 run_dir/"forecasts/all_test.csv.gz", run_dir/"forecasts/uncertainty_test.csv.gz"):
        if not path.is_file():
            raise RuntimeError(f"Notebook 08 analysis is not ready: {path}")
    points = _csv(run_dir/"tables/test_scorecard.csv", {"model", "MAE_MW", "scored"})
    bootstrap = _csv(run_dir/"tables/bootstrap.csv", {"model", "block_days", "status"})
    peaks = _csv(run_dir/"tables/peak_risk_scorecard.csv", {
        "model", "method", "segment", "coverage90", "width_MW", "upper_exceedance_rate",
        "upper_exceedance_conditional_severity_MW", "interval_nobs", "ndays", "nwinters",
        "upper_guardrail_status", "upper_rate_ci14_low", "upper_rate_ci14_high"})
    candidate, core, rte = [_one(points, model=model) for model in (policy, independent, "rte_proxy")]
    absolute = float(rte.MAE_MW-candidate.MAE_MW)
    relative = 100*absolute/float(rte.MAE_MW) if float(rte.MAE_MW) else float("nan")
    if policy == "rte_proxy":
        interval, conclusion = "RTE itself was selected; its difference from the reference is zero.", ""
    else:
        ci = _one(bootstrap, model=policy, block_days=14)
        if ci.status == "ok":
            needed = {"relative_low", "relative_high", "absolute_low", "absolute_high"}
            if not needed.issubset(ci.index):
                raise RuntimeError("Completed 14-day bootstrap row lacks confidence bounds")
            interval = (f"Paired 14-day-block 95% intervals: relative improvement "
                        f"[{ci.relative_low:.2f}, {ci.relative_high:.2f}]%; absolute improvement "
                        f"[{ci.absolute_low:,.0f}, {ci.absolute_high:,.0f}] MW.")
            conclusion = ("The observed gain remains uncertain across day blocks."
                          if absolute>0 and float(ci.relative_low)<=0 else
                          "The preselected policy did not improve overall MAE." if absolute<=0 else
                          "The block comparison supports a historical MAE improvement.")
        else:
            interval = f"A 14-day-block confidence interval is unavailable ({ci.status})."
            conclusion = "The observed difference is descriptive."
    methods, choices = {}, {}
    for model in (independent, "rte_proxy"):
        choices[model], methods[model] = _selected_peak_method(peak_selection, model)
    support = ""
    conditional_rows = [row for row in peak_selection["records"]
                        if row["model"] == independent and row["candidate"] == "conditional"]
    if len(conditional_rows) == 1 and {"critical_nobs", "critical_ndays"}.issubset(conditional_rows[0]):
        row = conditional_rows[0]
        support = (f"The independent interval decision had {int(row['critical_nobs'])} proxy-cold observations/"
                   f"{int(row['critical_ndays'])} days in 2024, against the minimum 60/20.")
    # This is the predeclared empirical uncertainty benchmark, not the interval
    # method that happens to score best for the RTE-assisted policy in the test.
    complete_method = "fixed_past_out_of_sample_residual_distribution"
    rows, cold = [], {}
    roles = [(policy, "Complete policy", complete_method),
             (independent, "Independent model", methods[independent]),
             ("rte_proxy", "RTE", methods["rte_proxy"])]
    for model, label, method in roles:
        overall = _one(peaks, model=model, method=method, segment="all")
        row = _one(peaks, model=model, method=method, segment="realized_cold_evening")
        cold[label] = row
        rows.append(f"| {label} | {_pct(overall.coverage90)} / {_mw(overall.width_MW)} | "
                    f"{int(row.interval_nobs)} / {int(row.ndays)} / {int(row.nwinters)} | "
                    f"{_pct(row.coverage90)} / {_pct(row.upper_exceedance_rate)} | "
                    f"{_mw(row.upper_exceedance_conditional_severity_MW)} |")
    critical = cold["Complete policy"]
    diagnostics = _diagnostics(run_dir)
    stress = _stress_note(run_dir)
    kind = next(item.get("kind", "policy") for item in complete["entries"] if item["model"] == policy)
    description = "RTE-assisted bias adjustment" if policy == "rte_bias_proxy" else kind.replace("_", " ")
    text = f"""# Manager recommendation: national day-ahead demand

## 1. Can we improve on RTE?

The 2024-preselected complete policy, **{policy}** ({description}), has historical-test MAE **{candidate.MAE_MW:,.0f} MW**, versus **{rte.MAE_MW:,.0f} MW** for the supplied RTE proxy on {int(candidate.scored):,} matched hourly observations. Improvement: **{absolute:,.0f} MW ({relative:.2f}%)**. {interval} {conclusion}

The separate independent-model choice, **{independent}**, has MAE **{core.MAE_MW:,.0f} MW**. Its result does not summarize RTE-assisted attempts. Both choices were frozen using 2024 only; final-test results never choose the headline policy.

## 2. How uncertain, and when does it fail?

The complete policy uses the predeclared past-out-of-sample residual distribution. Independent-model/RTE intervals use their saved **{choices[independent]}/{choices['rte_proxy']}** choices. {support} Cold evenings are 18:00, 19:00, 20:00 delivery hours with realized temperature below 5 C.

| Policy | Overall coverage / width MW | Cold hours / days / DJF winters | Cold coverage / upper misses | Mean upper-miss severity MW |
|---|---:|---:|---:|---:|
{chr(10).join(rows)}

Intervals target 90% coverage; widths are MW. Severity is conditional on an upper-bound miss. The complete policy's cold upper-miss 14-day-block 95% interval is **{_rate_ci(critical)}**; status: **{str(critical.upper_guardrail_status).replace('_', ' ')}**. DJF counts are represented winters, not full cycles; non-DJF cold events do not increase them. Sparse segments are inconclusive; zero-event bootstrap intervals [0, 0] do not prove zero risk. The 5% upper-miss target is diagnostic; conformal bounds do not guarantee equal tails. Hourly intervals are not daily-energy/maximum intervals. Scenario triggers use past-temperature proxies; realized cold is diagnostic. {diagnostics}

## 3. What should we use, and what are the risks?

**Retain RTE as the operational reference and carry the preselected complete policy forward as a research candidate.** Vintage archives are absent, timing/aggregation are assumed, and the 2025–26 historical test was previously inspected. Fresh prospective evidence is required.

Family rankings are confounded by 20/40-step neural caps, differing histories/features, and five windowed models' training-origin mismatch; they do not establish general architecture superiority.

{stress}

Cold weights and shortage/surplus ratios are weighted-MW risk proxies, **not euro savings or validated procurement decisions**. National consumption remains a portfolio proxy.
"""
    if len(text.split())>550:
        raise RuntimeError(f"Manager recommendation exceeds 550 words ({len(text.split())}); no report written")
    destination = run_dir/"manager_recommendation.md"
    destination.write_text(text)
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True,
                        help="Completed versioned reports/research_v2 run directory")
    args = parser.parse_args()
    try:
        destination = write_manager_recommendation(args.run_dir)
    except (RuntimeError, KeyError, ValueError) as exc:
        print(f"Recommendation not written: {exc}", file=sys.stderr)
        return 1
    print(destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
