"""Append and execute the RTE extension without clearing original notebook outputs."""
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

ROOT = Path(__file__).resolve().parents[1]
MARKER = 'rte_assisted_extension'


def md(source):
    return nbformat.v4.new_markdown_cell(source.strip(), metadata={MARKER: True})


def code(source):
    ast.parse(source.strip())
    return nbformat.v4.new_code_cell(source.strip(), metadata={MARKER: True})


def cells07():
    return [
        md('''## RTE-assisted extension: direct prediction versus residual correction

This separate experiment preserves the preceding independent-model results. Four tabular families—histogram gradient boosting, LightGBM, quantile LightGBM and penalized GAM—receive `rte_at_origin_mw` and `rte_bias_recent_mw`, with national and regional feature arms. Each has a direct demand target and an RTE-error target. Both use identical predictors and fallback rules.

Two configurations per variant are evaluated across all four 2024 chronological quarters. The extension therefore has16 variants and128 configuration-quarter validations. Preprocessing is fitted inside each historical fit. The origin remains11:00 Europe/Paris; only labels released strictly before that origin can enter training. No realized future weather enters predictors. Supplied RTE is an assumed-available retrospective proxy, not a verified forecast vintage.

The best assisted variant and complete policy are frozen using development data before new calibration/test replay. The previously inspected test remains retrospective. Raw RTE and simple28-day RTE bias correction are both required references; beating raw RTE alone does not show incremental value beyond simple correction.'''),
        code(r'''from src.rte_assisted import RTEAssistedRun
assisted_run = RTEAssistedRun.latest(ROOT)
display(Markdown(f"**RTE extension run:** `{assisted_run.path}`"))
display(pd.DataFrame([
    {"mode": "direct", "predictors": "calendar/history/past weather + RTE + recent RTE bias", "target": "actual demand", "forecast": "model output"},
    {"mode": "residual", "predictors": "same predictors", "target": "actual demand minus RTE", "forecast": "RTE + predicted residual"},
]))
display(pd.DataFrame({"arm": ["national", "regional"],
    "predictor_count": [len(assisted_run.data.columns(a, rte_assisted=True)) for a in ("national", "regional")]}))'''),
        md('''### Tune chronologically and freeze the extension

The same bounded parameter grids used for independent tabular models are retuned for each RTE-assisted target/arm. Every configuration receives all daily origins within every development quarter. Forecasts are recorded before evaluation outcomes are joined. Search logs, per-fit predictor columns and chronological availability checks remain in the versioned run directory. Cached completed folds are reused only when their hashes and origin ledgers match.'''),
        code(r'''import contextlib
with (assisted_run.path / "development_execution.log").open("a") as assisted_log:
    with contextlib.redirect_stdout(assisted_log):
        assisted_trials = assisted_run.tune_assisted()
assisted_freeze = assisted_run.freeze_assisted()
assisted_development = pd.DataFrame([
    {"model": key, "family": value["family"], "arm": value["arm"], "mode": value["mode"],
     "development_MAE_MW": value["MAE_MW"], "configuration": value["trial"],
     "raw_availability": value["raw_availability"]}
    for key, value in assisted_freeze["selections"].items()
]).sort_values("development_MAE_MW")
display(assisted_development)
display(Markdown(f"**Development-selected assisted model:** `{assisted_freeze['selected_assisted_model']}`\n\n"
    f"**Development-selected complete policy:** `{assisted_freeze['selected_complete_policy']}`\n\n"
    f"**Completed configuration-quarter validations:** {len(assisted_trials)}/128. Continue in notebook08's RTE-assisted extension."))'''),
    ]


def cells08():
    return [
        md('''## RTE-assisted extension: matched historical comparison

Run notebook07's RTE-assisted development section first. This section evaluates the separately frozen16-variant tabular extension alongside the27 original policies on the same eligible origin–delivery pairs. Direct variants predict demand with RTE as a feature; residual variants learn demand minus RTE and add their output to RTE.

The 2025–26 period was previously exposed. The extension was requested after viewing earlier test results, so it is a further retrospective experiment—not a fresh confirmation set. Neither test ranking nor cold-tail outcomes select a model or trigger tuning. Original RTE issue vintages remain unavailable. National RTE is used in both data arms; no regional RTE forecast is invented.'''),
        code(r'''from src.rte_assisted import RTEAssistedRun
from scripts.rte_assisted_reporting import score_extension
import contextlib
assisted_run = RTEAssistedRun.latest(ROOT)
assert (assisted_run.path / "frozen_selection.json").exists(), "Complete notebook07's RTE extension first"
assisted_freeze = assisted_run.freeze_assisted()
display(Markdown(f"**RTE extension run:** `{assisted_run.path}`\n\n"
    f"**Frozen assisted choice:** `{assisted_freeze['selected_assisted_model']}`"))
with (assisted_run.path / "evaluation_execution.log").open("a") as assisted_log:
    with contextlib.redirect_stdout(assisted_log):
        assisted_outputs = assisted_run.replay_final_assisted()
        assisted_results = score_extension(assisted_run, assisted_outputs)
display(pd.DataFrame([json.loads((assisted_run.path / "result_summary.json").read_text())]))'''),
        md('''### Point forecasts and the value of RTE predictors

The tables retain all16 new variants, both RTE references, and each variant's original independent comparator. Scores use identical eligible observations. A negative improvement means deterioration. Separate paired7/14/28-day block resampling compares every assisted variant with raw RTE and rolling bias correction. These intervals do not eliminate selection uncertainty or turn the previously exposed test into prospective evidence.'''),
        code(r'''assisted_point = assisted_results["point_scores"]
display(assisted_point.loc[assisted_point.model.isin(assisted_freeze["selections"]) | assisted_point.model.isin(["rte_proxy", "rte_bias_proxy"]),
    ["model", "scored", "MAE_MW", "RMSE_MW", "MAE_improvement_vs_RTE_percent", "MAE_improvement_vs_bias_correction_percent", "raw_forecast_availability", "fallback_fraction"]])
display(pd.read_csv(assisted_run.path / "tables/rte_feature_ablation.csv"))
display(assisted_results["bootstrap"].loc[
    assisted_results["bootstrap"].model.eq(assisted_freeze["selected_assisted_model"]) & assisted_results["bootstrap"].block_days.eq(14)])
from IPython.display import Image as AssistedImage
display(AssistedImage(filename=str(assisted_run.path / "figures/rte_feature_ablation.png")))'''),
        md('''### Uncertainty and cold-evening upper-tail risk

Every policy receives the predeclared empirical distribution of past out-of-sample calibration errors; native LightGBM quantiles are reported separately. The development-selected assisted model also has a distinct split-conformal comparator whose fitted parameters are fixed across calibration and test. Its intervals must not be attributed to the monthly-refitted point policy. Temporal dependence means no automatic exchangeability or conditional-coverage guarantee.

Cold evenings use realized temperature<5°C at local delivery hours18/19/20 strictly for diagnosis. Counts, represented winters, upper exceedances and conditional miss severity accompany coverage and width. The upper5% diagnostic is not a guarantee, including for symmetric conformal bounds. Fixed cold weights and shortage/surplus ratios are retrospective weighted-MW risk proxies, not euro savings or actionable procurement policies. Separate hourly intervals do not define daily energy or maximum intervals.'''),
        code(r'''assisted_roles = [assisted_freeze["selected_assisted_model"], "rte_proxy", "rte_bias_proxy"]
display(assisted_results["uncertainty"].loc[assisted_results["uncertainty"].model.isin(assisted_roles)])
display(assisted_results["cold_scores"].loc[assisted_results["cold_scores"].model.isin(assisted_roles)])
display(AssistedImage(filename=str(assisted_run.path / "figures/rte_assisted_cold_risk.png")))
display(pd.read_csv(assisted_run.path / "tables/worst_assisted_days.csv").head(5))
display(Markdown((assisted_run.path / "recommendation.md").read_text()))'''),
    ]


def prepare():
    for number, factory in ((7, cells07), (8, cells08)):
        path = next((ROOT/'notebooks').glob(f'0{number}_*.ipynb'))
        notebook = nbformat.read(path, as_version=4)
        existing = [c for c in notebook.cells if c.metadata.get(MARKER)]
        expected = factory()
        if existing:
            if [c.source for c in existing] != [c.source for c in expected]:
                if any(c.cell_type == 'code' and (c.execution_count is not None or c.outputs) for c in existing):
                    raise RuntimeError('Existing executed extension sources differ; refuse to overwrite outputs')
                positions = [i for i,c in enumerate(notebook.cells) if c.metadata.get(MARKER)]
                if len(positions) != len(expected):
                    raise RuntimeError('Extension structure differs')
                for i, cell in zip(positions, expected):
                    notebook.cells[i] = cell
                nbformat.validate(notebook)
                nbformat.write(notebook, path)
        else:
            notebook.cells.extend(expected)
            notebook.metadata['rte_assisted_extension'] = {
                'source': 'scripts/extend_rte_assisted_notebooks.py',
                'execution_policy': 'Existing executed cells preserved; appended section executed in a fresh kernel with the original setup cell.',
            }
            nbformat.validate(notebook)
            nbformat.write(notebook, path)
        print(f'PREPARED {path.name}', flush=True)


def execute(number):
    path = next((ROOT/'notebooks').glob(f'0{number}_*.ipynb'))
    notebook = nbformat.read(path, as_version=4)
    positions = [i for i, c in enumerate(notebook.cells) if c.metadata.get(MARKER)]
    if not positions:
        raise RuntimeError('Prepare extension cells first')
    protected = json.dumps([c for c in notebook.cells if not c.metadata.get(MARKER)], sort_keys=True)
    temporary = nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell(notebook.cells[1].source)] +
        [c.copy() for c in notebook.cells if c.metadata.get(MARKER)])
    runtime = ROOT/'.runtime/kernels'
    manager = KernelManager(kernel_name='project-research',
        kernel_spec_manager=KernelSpecManager(kernel_dirs=[str(runtime)]))
    def announce(cell, cell_index, **kwargs):
        if cell.cell_type == 'code':
            print(f'CELL {cell_index}: {cell.source.splitlines()[0]}', flush=True)
    def checkpoint(cell, cell_index, **kwargs):
        if cell_index:
            notebook.cells[positions[cell_index-1]] = cell
            nbformat.write(notebook, path)
    try:
        NotebookClient(temporary, km=manager, timeout=14400,
            resources={'metadata': {'path': str(ROOT)}}, on_cell_start=announce,
            on_cell_executed=checkpoint).execute()
    finally:
        if manager.has_kernel:
            manager.shutdown_kernel(now=True)
    for i, position in enumerate(positions, 1):
        notebook.cells[position] = temporary.cells[i]
    if protected != json.dumps([c for c in notebook.cells if not c.metadata.get(MARKER)], sort_keys=True):
        raise AssertionError('Original notebook content changed')
    notebook.metadata['rte_assisted_extension']['executed_utc'] = datetime.now(timezone.utc).isoformat()
    nbformat.validate(notebook)
    nbformat.write(notebook, path)
    html, _ = HTMLExporter(template_name='lab').from_notebook_node(notebook)
    (ROOT/'reports/notebook_previews'/f'{path.stem}.html').write_text(html)
    print(f'PASSED {path.name} RTE extension', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--execute', type=int, choices=(7, 8))
    args = parser.parse_args()
    if args.prepare:
        prepare()
    if args.execute:
        execute(args.execute)
