"""Matched retrospective scores for the separately frozen RTE-assisted study."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common import forecast_origin, write_json
from src.metrics import (daily_metrics, matched_scorecard, paired_block_bootstrap,
                         scorecard, segment_scorecard)
from src.research_peak_risk import _decorate, _summary, _upper_bootstrap
from src.research_scores import conformal_radius, uncertainty_comparison
from src.rte_assisted import RTEAssistedRun

EMPIRICAL = "fixed_past_out_of_sample_residual_distribution"


def read_forecasts(path):
    frame = pd.read_csv(path, low_memory=False)
    for name in ("origin", "delivery_time", "fit_origin"):
        if name in frame:
            frame[name] = pd.to_datetime(frame[name], utc=True)
    return frame


def conformal_comparator(run):
    continuous = run.fixed_conformal_replay()
    fit_origins = continuous.fit_origin.dropna().unique()
    if len(fit_origins) != 1:
        raise AssertionError("Split conformal requires one fitted predictor")
    c, t = run.contract["calibration"], run.contract["test"]
    cutoff = forecast_origin(t["start"], run.contract)
    cal = continuous.loc[continuous.local_day.ge(c["start"]) & continuous.local_day.lt(c["end"])].merge(
        run.data.labels[["delivery_time", "actual_mw", "label_available_at"]], on="delivery_time", validate="many_to_one")
    cal = cal.loc[cal.label_available_at.lt(cutoff)]
    errors = (cal.actual_mw-cal.prediction_mw).abs().dropna()
    radius = conformal_radius(errors)
    test = continuous.loc[continuous.local_day.ge(t["start"]) & continuous.local_day.lt(t["end"])].copy()
    test["q05"], test["q50"], test["q95"] = test.prediction_mw-radius, test.prediction_mw, test.prediction_mw+radius
    test["method"], test["calibration_n"] = "fixed_predictor_split_conformal", len(errors)
    test.to_csv(run.path/"forecasts/fixed_conformal_test.csv.gz", index=False)
    write_json(run.path/"tables/conformal_report.json", {
        "model": test.model.iloc[0], "single_fit_origin": str(fit_origins[0]),
        "calibration_rows": len(errors), "radius_MW": radius,
        "latest_calibration_available_at": str(cal.label_available_at.max()), "test_origin": str(cutoff),
        "interpretation": "Separate fixed-predictor comparator, not intervals around the monthly-refitted point policy; no exchangeability/conditional guarantees",
    })
    return test


def development_scores(run):
    frozen = run.freeze_assisted()
    parts = []
    gaps = []
    for key, entry in frozen["selections"].items():
        for fold in run.contract["validation_folds"]:
            p = run.replay_variant(entry["family"], entry["params"], fold["start"], fold["end"],
                arm=entry["arm"], mode=entry["mode"], phase=f"{fold['name']}_trial{entry['trial']}")
            parts.append(p)
            joined = p.merge(run.data.labels[["delivery_time", "actual_mw"]], on="delivery_time", validate="many_to_one")
            phase = f"{fold['name']}_trial{entry['trial']}"
            fit_paths = sorted((run.path/"fits").glob(f"{key}__{phase}__*.json"))
            if len(fit_paths) != 1:
                raise RuntimeError(f"Ambiguous selected development fit artifact: {key} {phase}")
            records = {pd.Timestamp(r['origin']): r for r in json.loads(fit_paths[0].read_text())}
            for fit_origin, group in joined.groupby('fit_origin'):
                record = records[fit_origin]
                valid = group.dropna(subset=['actual_mw', 'raw_prediction_mw'])
                test_mae = float((valid.actual_mw-valid.raw_prediction_mw).abs().mean())
                gaps.append({'model': key, 'fold': fold['name'], 'fit_origin': fit_origin,
                    'training_rows': record.get('training_rows'), 'scored_next_origin_hours': len(valid),
                    'train_MAE_MW': record.get('train_MAE_MW'), 'following_OOS_MAE_MW': test_mae,
                    'OOS_minus_train_MAE_MW': test_mae-record.get('train_MAE_MW', np.nan),
                    'interpretation': 'Same fitted predictor; gap may reflect regime/season shift as well as fitting. Not a formal bias-variance decomposition.'})
    forecasts = pd.concat(parts, ignore_index=True)
    segment_scorecard(forecasts, run.data.labels, run.contract).to_csv(run.path/"tables/development_segments.csv", index=False)
    pd.DataFrame(gaps).to_csv(run.path/'tables/development_fit_gaps.csv', index=False)
    trials = pd.read_csv(run.path/'tables/tuning_trials.csv')
    chosen = pd.concat([trials.loc[trials.model.eq(k) & trials.trial.eq(v['trial'])]
                        for k, v in frozen['selections'].items()], ignore_index=True)
    chosen.to_csv(run.path/'tables/selected_development_quarters.csv', index=False)
    fits = [r for path in sorted((run.path/"fits").glob("*.json")) for r in json.loads(path.read_text())]
    frame = pd.DataFrame(fits)
    frame.drop(columns=["metadata", "params", "predictor_columns", "traceback"], errors="ignore").to_csv(run.path/"tables/fit_diagnostics.csv", index=False)
    return pd.DataFrame([{"model": key, **entry} for key, entry in frozen["selections"].items()])


def score_extension(run, outputs=None):
    frozen = run.freeze_assisted()
    run.verify_manifest()
    outputs = outputs or {p: read_forecasts(run.path/"forecasts"/f"assisted_{p}.csv.gz") for p in ("calibration", "test")}
    combined = {}
    for phase, new in outputs.items():
        old = read_forecasts(run.reference/"forecasts"/f"all_{phase}.csv.gz")
        if set(old.model).intersection(new.model):
            raise AssertionError("New and original model IDs overlap")
        combined[phase] = pd.concat([old, new], ignore_index=True)
        if combined[phase].duplicated(["model", "origin", "delivery_time"]).any():
            raise AssertionError("Duplicate forecast keys")
        combined[phase].to_csv(run.path/"forecasts"/f"all_{phase}.csv.gz", index=False)
    test = combined["test"]
    matched, eligible = matched_scorecard(test, run.data.labels)
    availability = test.groupby("model").agg(raw_forecast_availability=("raw_prediction_mw", lambda x: x.notna().mean()),
                                             fallback_fraction=("fallback_used", "mean"))
    scores = matched.merge(availability, on="model")
    for reference, label in (("rte_proxy", "RTE"), ("rte_bias_proxy", "bias_correction")):
        ref = float(scores.loc[scores.model.eq(reference), "MAE_MW"].iloc[0])
        scores[f"MAE_improvement_vs_{label}_MW"] = ref-scores.MAE_MW
        scores[f"MAE_improvement_vs_{label}_percent"] = 100*(1-scores.MAE_MW/ref)
    scores["variant"] = scores.model.map(lambda m: "RTE-feature direct" if m.endswith("__rte_direct") else
        "RTE residual correction" if m.endswith("__rte_residual") else "Original policy")
    scores = scores.sort_values("MAE_MW")
    scores.to_csv(run.path/"tables/test_scorecard.csv", index=False)
    scorecard(test, run.data.labels).to_csv(run.path/"tables/unmatched_scorecard.csv", index=False)
    scorecard(test, run.data.labels, "raw_prediction_mw").to_csv(run.path/"tables/raw_scorecard.csv", index=False)
    segment_scorecard(eligible, run.data.labels, run.contract).to_csv(run.path/"tables/test_segments.csv", index=False)
    daily = daily_metrics(test, run.data.labels)
    daily.to_csv(run.path/"tables/daily_metrics.csv", index=False)
    daily.loc[daily.model.eq(frozen["selected_assisted_model"])].sort_values(
        "absolute_daily_energy_error_MWh", ascending=False).head(10).assign(
        cause="Unexplained; recorded weather/calendar are diagnostic, not verified causes").to_csv(run.path/"tables/worst_assisted_days.csv", index=False)
    bootstrap = []
    for model in frozen["selections"]:
        for ref in ("rte_proxy", "rte_bias_proxy"):
            result = paired_block_bootstrap(eligible, run.data.labels, model, reference=ref,
                block_days=run.contract["bootstrap_block_days"], repetitions=run.contract["bootstrap_repetitions"],
                seed=run.contract["random_seed"])
            result["model"], result["reference"] = model, ref
            bootstrap.append(result)
    bootstrap = pd.concat(bootstrap, ignore_index=True)
    bootstrap.to_csv(run.path/"tables/bootstrap.csv", index=False)
    conformal = conformal_comparator(run)
    uncertainty = uncertainty_comparison(run, combined["calibration"], test, conformal)
    distributions = read_forecasts(run.path/"forecasts/uncertainty_test.csv.gz")
    peak = []
    selected_roles = {frozen["selected_assisted_model"], "rte_proxy", "rte_bias_proxy"}
    for (model, method), frame in distributions.groupby(["model", "method"]):
        decorated = _decorate(frame, run)
        cold = decorated.loc[decorated.realized_cold_evening]
        row = {"model": model, "method": method, **_summary(cold)}
        if row.get('upper_exceedance_rate') == 0:
            # A conditional mean is undefined when its conditioning event never occurs.
            row['upper_exceedance_conditional_severity_MW'] = np.nan
        if model in selected_roles:
            row.update(_upper_bootstrap(decorated, cold, 14))
        peak.append(row)
    peak = pd.DataFrame(peak)
    peak.to_csv(run.path/"tables/cold_evening_scorecard.csv", index=False)
    # These ex-post loss weights are risk scenarios; they do not produce trading decisions.
    decorated = _decorate(eligible, run)
    scenarios = []
    spec = run.spec["critical_risk_scenarios"]
    for model, frame in decorated.groupby("model"):
        frame = frame.dropna(subset=["actual_mw", "prediction_mw"])
        error = frame.actual_mw-frame.prediction_mw
        for weight in spec["cold_weights"]:
            weights = np.where(frame.realized_cold_evening, weight, 1)
            for ratio in spec["shortage_surplus_ratios"]:
                scenarios.append({"model": model, "cold_weight": weight, "shortage_surplus_ratio": ratio,
                    "nobs": len(frame), "weighted_mean_loss_MW": np.mean(weights*(ratio*error.clip(lower=0)+(-error).clip(lower=0))),
                    "interpretation": "Retrospective risk proxy; not euro savings or procurement"})
    pd.DataFrame(scenarios).to_csv(run.path/"tables/cold_weighted_loss_scenarios.csv", index=False)
    ablation = []
    indexed = scores.set_index("model")
    for family in run.spec["families"]:
        for arm in run.spec["arms"]:
            independent = f"{family}__{arm}"
            for mode in run.spec["modes"]:
                model = f"{independent}__rte_{mode}"
                ablation.append({"family": family, "arm": arm, "mode": mode, "model": model,
                    "independent_MAE_MW": indexed.loc[independent, "MAE_MW"],
                    "assisted_MAE_MW": indexed.loc[model, "MAE_MW"],
                    "MAE_reduction_MW": indexed.loc[independent, "MAE_MW"]-indexed.loc[model, "MAE_MW"]})
    pd.DataFrame(ablation).to_csv(run.path/"tables/rte_feature_ablation.csv", index=False)
    development_scores(run)
    plots = draw_figures(run, scores, uncertainty, peak)
    write_report(run, scores, uncertainty, peak, bootstrap)
    return {"point_scores": scores, "uncertainty": uncertainty, "cold_scores": peak, "bootstrap": bootstrap, "figures": plots}


def draw_figures(run, scores, uncertainty, cold):
    import matplotlib.pyplot as plt
    frozen = run.freeze_assisted()
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True)
    indexed = scores.set_index("model")
    colors = ["#737d86", "#247e9b", "#317745"]
    labels = ["Independent", "RTE feature → demand", "RTE feature → correction"]
    rte = indexed.loc['rte_proxy', 'MAE_MW']
    bias = indexed.loc['rte_bias_proxy', 'MAE_MW']
    for ax, family in zip(axes.flat, run.spec['families']):
        y = np.arange(2)
        for j, suffix in enumerate(('', '__rte_direct', '__rte_residual')):
            values = [indexed.loc[f'{family}__{arm}{suffix}', 'MAE_MW'] for arm in run.spec['arms']]
            ax.barh(y+(j-1)*.23, values, height=.21, color=colors[j], label=labels[j])
        ax.axvline(rte, color='#c28635', linestyle='--', label='Raw RTE')
        ax.axvline(bias, color='#a44848', linestyle=':', label='Rolling RTE bias correction')
        ax.set(yticks=y, yticklabels=['National features', 'Regional features'],
               xlabel='Matched test MAE (MW); lower is better', title=family.replace('_', ' '), xlim=(0, None))
        ax.grid(axis='x', alpha=.2)
    handles, legend = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, legend, loc='lower center', ncol=3, frameon=False)
    fig.suptitle('Adding RTE predictors: July 2025–June 2026, 8,759 matched hours\nConfigurations selected on 2024; historical test previously inspected')
    fig.tight_layout(rect=(0, .11, 1, .91))
    p1 = run.path/'figures/rte_feature_ablation.png'
    fig.savefig(p1, dpi=150, bbox_inches='tight'); plt.close(fig)
    selected = [frozen['selected_assisted_model'], 'rte_bias_proxy', 'rte_proxy']
    titles = ['Selected RTE model', 'Rolling RTE correction', 'Raw RTE']
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    overall = uncertainty.loc[uncertainty.method.eq(EMPIRICAL)].set_index('model')
    critical = cold.loc[cold.method.eq(EMPIRICAL)].set_index('model')
    x = np.arange(3)
    axes[0].bar(x-.18, [overall.loc[k, 'coverage_90']*100 for k in selected], .36, label='All hours', color='#247e9b')
    axes[0].bar(x+.18, [critical.loc[k, 'coverage90']*100 for k in selected], .36, label='Realized cold evenings', color='#a44848')
    axes[0].axhline(90, color='#4b5055', linestyle='--', label='Nominal 90%')
    axes[0].set(ylabel='Empirical interval coverage (%)', ylim=(0, 102), title='Past-out-of-sample residual intervals')
    axes[0].legend(loc='lower left')
    rates = np.array([critical.loc[k, 'upper_exceedance_rate']*100 for k in selected])
    low = np.array([critical.loc[k, 'upper_rate_ci14_low']*100 for k in selected])
    high = np.array([critical.loc[k, 'upper_rate_ci14_high']*100 for k in selected])
    # Confidence intervals need not be centered on the observed rate.
    axes[1].vlines(x, low, high, color='#247e9b', linewidth=2)
    axes[1].scatter(x, rates, color='#247e9b', s=65, zorder=3)
    axes[1].axhline(5, color='#a44848', linestyle='--', label='Nominal upper 5% diagnostic')
    axes[1].set(ylabel='Cold-evening upper exceedances (%)', ylim=(0, max(60, float(np.nanmax(high))+5)),
                title='Upper-tail risk; paired calendar-day block interval')
    axes[1].legend(loc='upper left')
    for ax in axes:
        ax.set_xticks(x, titles)
        ax.tick_params(axis='x', labelsize=9)
        ax.grid(axis='y', alpha=.2)
    n = int(critical.loc[selected[0], 'interval_nobs'])
    days = int(critical.loc[selected[0], 'ndays'])
    fig.suptitle(f'Cold-evening diagnosis: temperature <5 °C at 18:00/19:00/20:00 local\n{n} hours / {days} days; 95% uncertainty from 14-day blocks; not operational cold alerts')
    fig.tight_layout(rect=(0, 0, 1, .9))
    p2 = run.path/'figures/rte_assisted_cold_risk.png'
    fig.savefig(p2, dpi=150, bbox_inches='tight'); plt.close(fig)
    return [p1, p2]


def write_report(run, scores, uncertainty, cold, bootstrap):
    frozen = run.freeze_assisted()
    key = frozen['selected_assisted_model']
    point = scores.set_index('model')
    selected = point.loc[key]
    own = uncertainty.loc[uncertainty.model.eq(key) & uncertainty.method.eq(EMPIRICAL)].iloc[0]
    critical = cold.loc[cold.model.eq(key) & cold.method.eq(EMPIRICAL)].iloc[0]
    ci = bootstrap.loc[bootstrap.model.eq(key) & bootstrap.block_days.eq(14)]
    ci_rte = ci.loc[ci.reference.eq('rte_proxy')].iloc[0]
    ci_bias = ci.loc[ci.reference.eq('rte_bias_proxy')].iloc[0]
    fit_records = [r for p in (run.path/'fits').glob('*.json') for r in json.loads(p.read_text())]
    failed = sum(r['status'] != 'fit' for r in fit_records)
    severity = (f"{critical.upper_exceedance_conditional_severity_MW:,.0f} MW"
                if np.isfinite(critical.upper_exceedance_conditional_severity_MW)
                else "not estimable (no observed upper misses)")
    text = f'''# RTE-assisted research comparison

## 1. Improvement over RTE and simple bias correction

The 2024-selected assisted variant is **{key}**. The complete-policy choice across all43 alternatives was **{frozen['selected_complete_policy']}**. These identities were frozen before new calibration/test replay. The test is previously exposed and therefore retrospective.

| Policy | Matched test MAE MW |
|---|---:|
| Selected assisted variant | {selected.MAE_MW:,.1f} |
| Raw RTE | {point.loc['rte_proxy', 'MAE_MW']:,.1f} |
| Rolling RTE bias correction | {point.loc['rte_bias_proxy', 'MAE_MW']:,.1f} |

On {int(selected.scored):,} matched hours, assisted improvement over RTE is **{selected.MAE_improvement_vs_RTE_percent:.2f}%**, or {selected.MAE_improvement_vs_RTE_MW:,.1f} MW; paired14-day-block95% interval [{ci_rte.relative_low:.2f}, {ci_rte.relative_high:.2f}]%. Improvement over rolling bias correction is **{selected.MAE_improvement_vs_bias_correction_percent:.2f}%**, or {selected.MAE_improvement_vs_bias_correction_MW:,.1f} MW; interval [{ci_bias.relative_low:.2f}, {ci_bias.relative_high:.2f}]%. A negative improvement means deterioration. Tables also retain every variant and its independent comparator. No winner is selected from test scores.

## 2. Uncertainty and cold-evening failures

The fixed past-out-of-sample residual distribution has nominal90% coverage, empirical overall coverage **{own.coverage_90:.1%}**, width **{own.width_MW:,.0f} MW**, and CRPS **{own.CRPS_MW:,.1f} MW**. These intervals belong to the monthly-refitted policy. The separate split-conformal comparator uses one fixed fitted predictor throughout calibration/test. Native LightGBM quantiles receive quantile/interval scores, not invented CRPS. No exchangeability or conditional-coverage guarantee is established.

At realized temperature<5°C during18:00/19:00/20:00 local, there are **{int(critical.nobs)} hours / {int(critical.ndays)} days / {int(critical.nwinters)} represented DJF winter(s)**. Selected-assisted coverage is **{critical.coverage90:.1%}**, upper exceedance **{critical.upper_exceedance_rate:.1%}**, and conditional mean upper-miss severity **{severity}**. Upper-exceedance14-day-block95% interval: [{critical.upper_rate_ci14_low:.1%}, {critical.upper_rate_ci14_high:.1%}]. Status: **{critical.upper_guardrail_status}**. The5% upper target is diagnostic; symmetric conformal intervals need not allocate equal tails. Hourly intervals do not define daily energy or maximum intervals. Interval availability and sparse segment counts remain visible. Weighted-MW shortage/cold scenarios are risk proxies, not euro savings.

## 3. Recommendation and limits

Use the executed scores to assess this research candidate against both RTE and the simple correction. A gain against raw RTE alone is insufficient evidence that learned complexity adds value. Cold-tail protection requires its separate diagnostic to pass; overall MAE does not establish it. Keep RTE as the operational reference pending authentic vintage archives and fresh prospective evidence.

There are {failed} failed fit events among {len(fit_records)} recorded fits. Original issue versions, consumption revisions, publication delay and hourly aggregation remain unverified assumptions. Missing evaluation targets are excluded without interpolation. RTE-assisted predictions use national RTE; regional arms add regional histories, not regional RTE forecasts. The16 new variants cover four tabular families with two data arms and direct/residual targets; neural and statistical architectures were not silently altered. Searches contain two configurations and four chronological quarters. Variants are separately retuned and add both RTE and recent RTE bias, so their comparison concerns complete policies, not the isolated causal effect of one predictor. Same-fit development training/next-origin error gaps and selected quarter scores are saved; such gaps can reflect season or regime changes and are not a formal bias-variance decomposition. Zero-event bootstrap intervals do not prove zero future risk. Test results must not trigger discretionary retuning. National consumption is a portfolio proxy; no production implementation or financial-savings estimate is supplied.
'''
    (run.path/'recommendation.md').write_text(text)
    write_json(run.path/'result_summary.json', {'selected_assisted_model': key,
        'selected_complete_policy': frozen['selected_complete_policy'], 'matched_test_hours': int(selected.scored),
        'selected_MAE_MW': selected.MAE_MW, 'RTE_MAE_MW': point.loc['rte_proxy','MAE_MW'],
        'bias_correction_MAE_MW': point.loc['rte_bias_proxy','MAE_MW'],
        'improvement_vs_RTE_percent': selected.MAE_improvement_vs_RTE_percent,
        'improvement_vs_bias_correction_percent': selected.MAE_improvement_vs_bias_correction_percent,
        'cold_coverage': critical.coverage90, 'cold_upper_exceedance': critical.upper_exceedance_rate,
        'cold_hours': int(critical.nobs), 'cold_days': int(critical.ndays),
        'failed_fit_events': failed, 'fit_events': len(fit_records), 'retrospective': True})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path)
    args = parser.parse_args()
    run = RTEAssistedRun(ROOT, args.run_dir) if args.run_dir else RTEAssistedRun.latest(ROOT)
    score_extension(run)
    print((run.path/'result_summary.json').read_text())
