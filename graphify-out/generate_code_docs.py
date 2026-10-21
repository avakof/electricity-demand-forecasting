"""Generate code and notebook reference from the inspected architecture inventory."""
import ast
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'graphify-out'

def main():
    inv=json.loads((OUT/'architecture_inventory.json').read_text())
    docs=['# Project architecture and code reference','',
      'Generated from current source definitions, notebook code cells and saved experiment evidence. This documentation does not run the forecasting code or certify historical issue-time availability.','',
      'Start with [the interactive architecture](architecture.html), [the full relationship graph](graph.html), [extraction audit](GRAPH_REPORT.md), and [verification record](verification.json).','',
      '## Research lanes and ownership','',
      '| Lane | Entry points | Implementation | Evidence |','|---|---|---|---|',
      '| Original investigation | Notebooks 01–06; `scripts/build_notebooks.py` | `data`, `eda`, `features`, `backtest`, `metrics`, `uncertainty`, `research_risks` | Original contract, scorecards and recommendation |',
      '| Legacy baseline branch | `02_jg_baseline_model.ipynb` | `scripts/evaluation.py` | Earlier protocol; not a required step in the current comparison |',
      '| Broad research comparison | Notebooks 07/08; `ResearchRun` | `research_data`, `research_experiment`, `research_models`, `research_neural`, `research_scores` | `config/research_experiment.json`; versioned `reports/research_v2/` |',
      '| RTE-assisted extension | Appended 07/08 sections; `RTEAssistedRun(ResearchRun)` | `rte_assisted`, `scripts/rte_assisted_reporting.py` | `config/rte_assisted_experiment.json`; `reports/rte_assisted/` |',
      '| Expanded features and cold risk | Appended 07/08 sections; independent `RiskLabRun` | `rte_feature_lab`, `rte_learned_features`, `rte_base_replay`, `rte_risk_lab`, `rte_risk_uncertainty` | `config/rte_risk_experiment.json`; `reports/rte_risk/` |',
      '| Manager report | Frozen results → diagnostics → figures/snapshot → report | `reports/manager_report_20261021/` scripts and authored report content | Complete written and interactive reports; no model selection or training |','',
      '## Model and information boundaries','',
      '- Every expanded predictor view forecasts national consumption. The regional-only view uses regional histories plus national RTE D1 and calendar; it has no regional RTE product.',
      '- Origin is 11:00 Europe/Paris on D for all actual hourly intervals of D+1. Observation publication/revision delays and original RTE issue vintages remain assumptions. No next-but-one-day product is used.',
      '- National/regional observed histories feed cutoff-gated lags, rolling statistics, calendar and RTE-profile features. Future actual weather is excluded. Complete regional composition is required for sums and unweighted temperature means.',
      '- Earlier monthly fits create Ridge demand predictions, cold probabilities and probabilities of RTE underprediction above 1,000 MW. Learned features and training weights use released information.',
      '- The selected point model is `lab__lightgbm__combined__stacked__w1__residual`: RTE plus a LightGBM correction with engineered and learned predictors. Its archived base-model field is null; it is not an ensemble of all original architectures.',
      '- The 23 hybrids each add one causal archived base forecast to a LightGBM residual combiner. They do not establish native RTE-input retraining of every base architecture.',
      '- The selected quantile policy is separate from the point selection. Split conformal and CQR calibrate intervals; CRPS/WIS/pinball/Winkler evaluate distributions or quantiles. They are not additional model architectures.',
      '- Expanded point fits are quarterly; learned feature generators are monthly. Older reference policies retain their own schedules. The historical evaluation was previously inspected; cold calibration has zero qualifying hours.',
      '- Deleted `src/pipeline.py` and monitoring-specification files are not current components. Scikit-learn preprocessing pipelines are unrelated to serving or procurement integration.','',
      'Evidence for these distinctions: [refresh audit](refresh_audit.md), [project README](../readme.md), [expanded specification](../config/rte_risk_experiment.json), [manager report](../reports/manager_report_20261021/complete_report.md).','',
      '## Notebook import and call map','',
      'Cell numbers below are one-based positions in the saved notebook, including Markdown cells. Calls are source observations, not proof that a cell ran.','']
    for nb in inv['notebooks']:
        docs+=['### '+Path(nb).name,'',f'[Open notebook](../{nb})','']
        imports=inv['notebook_imports'].get(nb,[])
        docs+=['Internal imports: '+(', '.join(f'[{Path(p).stem}](../{p})' for p in imports) or 'none statically resolved')+'.','']
        calls=inv['notebook_calls'].get(nb,[])
        docs+=['| Imported callable | Cell | Line within cell |','|---|---:|---:|'] if calls else []
        docs += [f'| `{x["symbol"]}` | {x["cell"]} | {x["line"]} |' for x in calls]
        docs+=['']
    docs+=['## Python API index','','Signatures and line numbers are extracted from definitions. An absent docstring remains absent; this index does not invent function behavior. Private helpers are listed because they are part of the research implementation.','']
    for module in inv['module_summary']:
        rel=module['path']; docs+=['### '+rel,'',f'[Source](../{rel})','']
        if module['docstring']:docs += [module['docstring'].split('\n\n')[0],'']
        tree=ast.parse((ROOT/rel).read_text())
        for d in module['definitions']:
            docs += [f'- [`{d["signature"]}`](../{rel}#L{d["line"]}) — {d["kind"]}, line {d["line"]}.']
            if d['docstring']: docs += ['  '+d['docstring'].split('\n\n')[0].replace('\n',' ')]
        for cls in (x for x in tree.body if isinstance(x,ast.ClassDef)):
            methods=[x for x in cls.body if isinstance(x,(ast.FunctionDef,ast.AsyncFunctionDef))]
            if methods:
                docs+=['',f'Methods on `{cls.name}`:','']
                docs += [f'- [`{m.name}({ast.unparse(m.args)})`](../{rel}#L{m.lineno})' for m in methods]
        docs+=['']
    docs+=['## Literal file and documentation relationships','','The machine graph retains every extracted observation with relation type, original source/target direction, confidence and source location. Literal rooted paths can be resolved; dynamic `run.path` values, reflection and runtime-generated strings are not exhaustively traced. Semantic document links complement the code graph without converting an inferred relation into a runtime call.','',
      f'The searchable inventory records {len(inv["files"]):,} non-environment files. The graph groups {len(inv["grouped_artifacts"])} generated artifact collections:','',
      '| Artifact group | Files |','|---|---:|']
    docs += [f'| `{x["path"]}` | {x["file_count"]:,} |' for x in inv['grouped_artifacts']]
    docs+=['','## Updating the documentation','','Run the graphify skill on this project with `--update` after source changes. The custom augmentation must also be refreshed: plain Graphify cannot infer notebook-cell imports or grouped artifact lineage by itself. The rebuild helper performs the complete deterministic refresh once all changed documents have semantic cache entries:','',
      '```sh','"$(cat graphify-out/.graphify_python)" graphify-out/refresh_graph.py','```','',
      'The helper stops before replacing a graph if changed documents need host-agent semantic extraction. It never trains models, executes notebooks or modifies experiment runs. It regenerates the inventory, source hashes, full graph, extraction report, code reference and overview. Preserve each versioned experiment when updating presentation files.','',
      '## Limits and verification','','Source-level imports/calls and documented dependencies are not an exhaustive runtime trace. Grouped artifact counts are metadata, not independent experiments. AST references whose implementation was not resolved remain explicitly labeled unresolved references. Cohesion and centrality describe this mixed architecture/evidence graph, not model quality or refactoring priorities.','',
      'See [verification.json](verification.json) for current endpoint, deletion, source-hash and notebook checks, and [graph_health.json](graph_health.json) for extraction diagnostics.']
    (OUT/'CODE_DOCUMENTATION.md').write_text('\n'.join(docs)+'\n')
    print(f'Code reference: {len(inv["module_summary"])} Python files, {len(inv["notebooks"])} notebooks')

if __name__=='__main__':main()
