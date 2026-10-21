"""Append independently runnable feature/risk sections to the two research notebooks."""
from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import json
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbconvert import HTMLExporter
from jupyter_client import KernelManager
from jupyter_client.kernelspec import KernelSpecManager

ROOT=Path(__file__).resolve().parents[1]
MARKER='rte_risk_extension'


def md(text):
    return nbformat.v4.new_markdown_cell(text.strip(),metadata={MARKER:True})


def code(text):
    ast.parse(text.strip())
    return nbformat.v4.new_code_cell(text.strip(),metadata={MARKER:True})


def cells07():
    return [
        md('''## RTE D1, expanded features and cold-evening risk

This separately frozen research experiment uses `rte_forecast_d1_mw` as an assumed available input at 11:00 Europe/Paris on D, to forecast every actual hourly interval of D+1. Historical issue vintages remain unverified. No next-but-one-day product or 05:00 rule is used.

The target is national consumption in all three predictor arms: national history, regional history only, and both. Regional-only includes national RTE and known calendar, but excludes national demand/weather/error history. It does not substitute a regional sum for national truth.

The feature dictionary distinguishes origin-anchored AR lags, publication-gated target lags, moving averages of observations, rolling dispersion/ranges/counts, EWMAs, RTE error history and target-day RTE profile shape. ARIMA MA terms describe latent innovations and are not the same as rolling means. Future observed weather is excluded.

New learned predictors are ridge demand forecasts and probabilities of cold/large RTE underprediction. Each value was produced out of sample using an earlier monthly fit. All imputers/scalers use training rows only.'''),
        code(r'''from src.rte_risk_lab import RiskLabRun
risk_run = RiskLabRun.latest(ROOT)
risk_run.verify_manifest()
risk_features = risk_run.frame
display(Markdown(f"**Frozen feature/risk run:** `{risk_run.path}`"))
feature_sizes = pd.DataFrame([
    {"predictor arm": arm, "basic columns": len(risk_run.data.columns(arm, tier="basic")),
     "engineered columns": len(risk_run.data.columns(arm, tier="engineered")), "learned columns added": 3}
    for arm in risk_run.spec["arms"]
])
display(feature_sizes)
display(risk_run.data.feature_register()[["feature", "source", "transformation", "availability"]].head(12))'''),
        md('''### Chronological model and feature comparison

Q1 2024 is the archived-base stacking burn-in. Q2–Q4 2024 and January 2–March 31, 2025 are development folds; they contain 102 realized cold-evening hours on 41 days. Two predeclared configurations per variant tune model capacity and regularization. New models refit at delivery-quarter boundaries. The original reference systems retain their documented monthly refit policies.

Training weights are `1 + 4 × predicted_cold_probability × evening_indicator`; the probability was available at the row's historical origin. Selection uses normalized weighted MAE with fivefold weight on realized cold evenings, while enforcing ordinary-MAE and cold-MAE guardrails against simple RTE bias correction. These weights are research scenarios, not energy prices.

The four tabular families receive the features directly. Every one of the 23 original variants also has an explicitly labelled hybrid: its causal out-of-sample forecast plus the features enters a LightGBM RTE-residual correction model. This does not turn ARIMA, N-BEATS or PatchTST into a native exogenous architecture. Their previously trained base-model budgets and limitations remain visible.

For LightGBM, basic/engineered/learned features and training weight caps of 1 or 5 are compared in all three arms. Direct-demand alternatives and hybrid controls distinguish added predictors from changed targets or shorter stacking histories.'''),
        code(r'''# Completed, hash-verified folds are reusable. A frozen run returns its
# existing selection instead of recomputing or changing it from test scores.
risk_selections = risk_run.tune()
risk_frozen = risk_run.freeze()
development_table = pd.DataFrame([
    {"model": name, "trial": entry["trial"], **entry["metrics"]}
    for name, entry in risk_frozen["selections"].items()
]).sort_values("weighted_MAE_MW")
display(development_table[["model", "trial", "MAE_MW", "weighted_MAE_MW", "cold_MAE_MW", "cold_n", "raw_availability"]].head(15).round(2))
display(Markdown(f"**Development-selected point model:** `{risk_frozen['selected_point']}`\n\n"
                 f"**Development-selected quantile model:** `{risk_frozen['selected_quantile']}`"))
display(pd.read_csv(risk_run.path / "tables/probability_development.csv").round(3))'''),
        md('''### Check selection and fitting limits

All model and feature choices above precede the new historical test replay. The test was already inspected in earlier studies, so this is a retrospective comparison. A low test error cannot establish prospective superiority.

Fit-gap diagnostics compare a sampled training MAE with the same fit's following out-of-sample error. They can reveal instability; seasonal shifts also affect the gap, so it is not a formal bias–variance decomposition. Earlier compact neural budgets of 20 or 40 training steps are not evidence that those architectures were optimally trained.

The full experiment register, parameter trials, feature availability register and source hashes are saved in this run directory. Prediction artifacts contain no evaluation outcomes.'''),
        code(r'''variant_register = json.loads((risk_run.path / "variant_register.json").read_text())
display(pd.DataFrame(variant_register.values()).groupby(["family", "arm"], dropna=False).size().rename("variants").reset_index())
display(pd.DataFrame(risk_run.spec["validation_folds"]))
display(Markdown("**Reproduce this study:** run `python scripts/run_rte_risk_experiment.py --stage all` with the project research interpreter. A fresh experiment requires `--new`; do not modify frozen predictive sources in place."))'''),
    ]


def cells08():
    return [
        md('''## Expanded RTE feature/risk evaluation

This section evaluates the new development-frozen candidates against raw RTE, rolling RTE bias correction and the previous studies on identical eligible origin–delivery pairs. Missing demand targets remain excluded and reported.

Separate calibration is April 2–June 29, 2025; it contains **zero cold-evening observations**. That is a material limitation for cold-conditional empirical and conformal coverage. The retrospective test is July 1, 2025–June 30, 2026. Earlier test outcomes may enter later scheduled fits only after release, under the frozen policy.

The native quantile model predicts q05, q075, q25, q50, q75, q90, q95 and q975. The central q05–q95 and asymmetric q075–q975 intervals both have nominal 90% mass; the latter allocates 2.5% to the upper tail. The quantile arm is selected using **unweighted overall development WIS**, without a cold-interval selection guardrail. It is not established as optimal for cold evenings.

Split conformal and CQR use a separate fixed engineered-only predictor, so an evolving learned-feature model is not incorrectly called fixed. That comparator still retains training weights based on out-of-sample cold probabilities while omitting those learned predictors from its inputs. Its native endpoints and calibrated bounds therefore require empirical coverage assessment; a nominal conditional-quantile interpretation is not assumed. No exchangeability or conditional-coverage guarantee is claimed.'''),
        code(r'''from src.rte_risk_lab import RiskLabRun
from src.rte_risk_uncertainty import uncertainty_replay
from scripts.rte_risk_reporting import report as report_rte_risk
risk_run = RiskLabRun.latest(ROOT)
risk_run.verify_manifest()
risk_frozen = risk_run.freeze()
# Sealed cached replays are verified and reused; all outputs remain auditable.
risk_results = risk_run.evaluate()
risk_probability_predictions = uncertainty_replay(risk_run)
risk_scores, risk_probability_scores, risk_bootstrap = report_rte_risk(risk_run)
selected_roles = {risk_frozen["selected_point"] or risk_frozen["diagnostic_best_point"],
                  risk_frozen["selected_quantile"], "rte_proxy", "rte_bias_proxy"}
display(risk_scores.loc[risk_scores.model.isin(selected_roles),
    ["model", "scored", "MAE_MW", "weighted_MAE_MW", "cold_MAE_MW", "cold_scored", "cold_days", "raw_availability", "fallback_fraction"]].round(3))'''),
        md('''### Accuracy, risk tradeoffs and feature ablations

Ordinary MAE measures typical point accuracy. Cold-weighted MAE applies the declared fivefold importance scenario. Cold-segment MAE and shortage severity reveal failures that an all-hour average can hide. The selected model is chosen on development, regardless of the ordering of historical test scores.

The hybrid controls use the same historical training support and feature groups without the base-model forecast. A hybrid beating RTE does not establish that its base architecture adds value. The paired control comparison tests that narrower claim. The feature-ablation table changes feature groups and training weights under the same evaluation metric.'''),
        code(r'''from IPython.display import Image
for figure in sorted((risk_run.path / "figures").glob("*.png")):
    display(Image(filename=str(figure)))
display(risk_bootstrap.loc[risk_bootstrap.block_days.eq(14),
    ["model", "reference", "metric", "relative_improvement_percent", "relative_low", "relative_high"]].round(2))
feature_ablation = pd.read_csv(risk_run.path / "tables/feature_ablation.csv")
display(feature_ablation[["arm", "feature_tier", "training_cold_weight_cap", "MAE_MW", "weighted_MAE_MW", "cold_MAE_MW"]].round(2))
hybrid_controls = pd.read_csv(risk_run.path / "tables/hybrid_control_comparison.csv")
display(hybrid_controls[["base_model", "arm", "weighted_MAE_MW_improvement_MW", "MAE_MW_improvement_MW", "cold_MAE_MW_improvement_MW", "selected_in_development"]].round(2))
learned_diagnostics = pd.read_csv(risk_run.path / "tables/learned_feature_oos_diagnostics.csv")
display(learned_diagnostics.loc[learned_diagnostics.component.ne("ridge_demand") & learned_diagnostics.segment.eq("evening"),
    ["phase", "arm", "component", "scored", "event_count", "event_base_rate", "Brier_score", "ROC_AUC", "PR_average_precision"]].round(4))'''),
        md('''### Distribution scores and expensive upper-tail errors

Pinball loss evaluates specified quantiles; WIS combines central 50% and 90% interval scores. Winkler penalizes interval width and misses. CRPS is exact here only for a fully specified past-residual empirical distribution; a few quantiles are not labelled a full distribution. Threshold-weighted CRPS emphasizes demand above a fixed pre-development threshold.

For shortage/surplus penalty ratios 1, 3, 9 and 19, the corresponding decision quantiles are q50, q75, q90 and q95. The same rule is applied to the candidate and calibrated RTE reference distributions. These are weighted-MW decision proxies, not measured euro savings. Raising the prediction is evaluated together with surplus error and interval width.

Classifier ranking metrics are undefined when a segment contains only one event class. Calibration contains one cold hour overall but no cold evening hours, so the evening cold-classifier ROC-AUC and average precision are reported as missing. This is a support limitation, not a zero performance score.'''),
        code(r'''native_roles = risk_probability_scores.loc[
    risk_probability_scores.method.ne("past_calibration_empirical_residual_distribution") &
    risk_probability_scores.segment_type.eq("all")]
display(native_roles[["model", "method", "central90_coverage", "central90_width_MW", "central90_upper_exceedance", "Winkler_central90_MW"]].round(3))
distribution_summary = risk_probability_scores.loc[risk_probability_scores.segment_type.eq("all") &
    (risk_probability_scores.model.isin(selected_roles) | risk_probability_scores.method.ne("past_calibration_empirical_residual_distribution"))]
display(distribution_summary[["model", "method", "WIS_central50_90_MW", "CRPS_empirical_MW", "upper_tail_twCRPS_empirical_MW", "pinball_q50_MW", "pinball_q95_MW"]].round(3))
cold_intervals = pd.read_csv(risk_run.path / "tables/cold_interval_scorecard.csv")
display(cold_intervals.loc[cold_intervals.model.isin(selected_roles) |
    cold_intervals.method.ne("past_calibration_empirical_residual_distribution"),
    ["model", "method", "central90_nobs", "central90_coverage", "central90_upper_exceedance", "central90_upper_miss_conditional_severity_MW"]].round(3))
display(pd.read_csv(risk_run.path / "tables/calibration_support.csv").head(8))
decision_scenarios = pd.read_csv(risk_run.path / "tables/quantile_decision_risk_scenarios.csv")
display(decision_scenarios.loc[decision_scenarios.cold_weight.eq(5),
    ["model", "method", "shortage_surplus_ratio", "decision_quantile", "scored", "normalized_asymmetric_loss_MW"]].round(3))
display(Markdown((risk_run.path / "recommendation.md").read_text()))'''),
    ]


def prepare():
    for number,factory in ((7,cells07),(8,cells08)):
        path=next((ROOT/'notebooks').glob(f'0{number}_*.ipynb'))
        nb=nbformat.read(path,as_version=4)
        previous=[c for c in nb.cells if c.metadata.get(MARKER)]
        expected=factory()
        if previous:
            if [c.source for c in previous] != [c.source for c in expected]:
                raise RuntimeError('Existing risk section sources differ; preserve its outputs and review changes')
        else:
            nb.cells.extend(expected)
            nb.metadata[MARKER]={'source':str(Path(__file__).relative_to(ROOT)),
                'execution_policy':'Section executed top-to-bottom in a fresh kernel; preceding completed studies preserved.'}
            nbformat.validate(nb)
            nbformat.write(nb,path)
        print('PREPARED',path.name,flush=True)


def execute(number):
    path=next((ROOT/'notebooks').glob(f'0{number}_*.ipynb'))
    nb=nbformat.read(path,as_version=4)
    positions=[i for i,c in enumerate(nb.cells) if c.metadata.get(MARKER)]
    if not positions:
        raise RuntimeError('Prepare section first')
    protected=json.dumps([c for c in nb.cells if not c.metadata.get(MARKER)],sort_keys=True)
    temporary=nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell(nb.cells[1].source)]+
                                     [c.copy() for c in nb.cells if c.metadata.get(MARKER)])
    manager=KernelManager(kernel_name='project-research',kernel_spec_manager=KernelSpecManager(kernel_dirs=[str(ROOT/'.runtime/kernels')]))
    def announce(cell,cell_index,**kwargs):
        if cell.cell_type=='code': print('CELL',cell_index,cell.source.splitlines()[0],flush=True)
    def checkpoint(cell,cell_index,**kwargs):
        if cell_index:
            nb.cells[positions[cell_index-1]]=cell
            nbformat.write(nb,path)
    try:
        NotebookClient(temporary,km=manager,timeout=14400,resources={'metadata':{'path':str(ROOT)}},
                       on_cell_start=announce,on_cell_executed=checkpoint).execute()
    finally:
        if manager.has_kernel: manager.shutdown_kernel(now=True)
    for i,position in enumerate(positions,1): nb.cells[position]=temporary.cells[i]
    assert protected==json.dumps([c for c in nb.cells if not c.metadata.get(MARKER)],sort_keys=True)
    nb.metadata[MARKER]['executed_utc']=datetime.now(timezone.utc).isoformat()
    nbformat.validate(nb)
    nbformat.write(nb,path)
    html,_=HTMLExporter(template_name='lab').from_notebook_node(nb)
    (ROOT/'reports/notebook_previews').mkdir(parents=True,exist_ok=True)
    (ROOT/'reports/notebook_previews'/f'{path.stem}.html').write_text(html)
    print('PASSED',path.name,'RTE risk section',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--prepare',action='store_true')
    parser.add_argument('--execute',type=int,choices=(7,8))
    args=parser.parse_args()
    if args.prepare: prepare()
    if args.execute: execute(args.execute)
