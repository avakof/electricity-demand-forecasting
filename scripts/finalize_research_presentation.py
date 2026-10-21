"""Finalize reporting after notebook 08 passes; never execute model code.

Only selected notebook-08 markdown sources are mirrored from the generator. Existing
code cells, execution counts, outputs, cell identifiers and notebook metadata
are preserved. The script appends documented execution limitations once and
exports the two already executed notebooks to HTML. Predictor sources, manifests,
frozen selections and forecast artifacts remain read-only.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys

import nbformat
from nbconvert import HTMLExporter
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = ('07_statistical_and_boosting_models.ipynb',
             '08_deep_learning_and_uncertainty.ipynb')
MARKDOWN_INDICES = (3, 5, 9, 11, 17)
LIMITATION_MARKER = '<!-- recorded-research-execution-limitations -->'
POLICY_MARKER = '<!-- development-complete-policy-distinction -->'


def template_08_cells(path: Path):
    """Evaluate only the literal cell-construction list for save(notebook08).

    No generator module is imported, no main function is invoked, and notebook07
    is never constructed or written. The allowed AST consists exclusively of a
    list of markdown/code calls on literal strings or the literal SETUP string.
    """
    tree = ast.parse(path.read_text(), filename=str(path))
    constants = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == 'SETUP':
                    constants['SETUP'] = ast.literal_eval(node.value)
    main = next((node for node in tree.body if isinstance(node, ast.FunctionDef)
                 and node.name == 'main'), None)
    if main is None:
        raise ValueError('Notebook generator has no main construction function')
    matches = []
    for node in main.body:
        call = node.value if isinstance(node, ast.Expr) else None
        if not (isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
                and call.func.id == 'save' and len(call.args) == 2):
            continue
        if isinstance(call.args[0], ast.Constant) and call.args[0].value == NOTEBOOKS[1]:
            matches.append(call.args[1])
    if len(matches) != 1 or not isinstance(matches[0], ast.List):
        raise ValueError('Expected exactly one literal notebook08 cell list')
    construction = matches[0]
    for cell in construction.elts:
        if not (isinstance(cell, ast.Call) and isinstance(cell.func, ast.Name)
                and cell.func.id in {'markdown', 'code'} and len(cell.args) == 1
                and not cell.keywords):
            raise ValueError('Generator cell constructor is outside the reporting allowlist')
        argument = cell.args[0]
        allowed = (isinstance(argument, ast.Constant) and isinstance(argument.value, str)) or (
            isinstance(argument, ast.Name) and argument.id in constants)
        if not allowed:
            raise ValueError('Generator cell content must be a literal string or SETUP')
    namespace = {'__builtins__': {}, **constants,
                 'markdown': lambda text: nbformat.v4.new_markdown_cell(text.strip()),
                 'code': lambda text: nbformat.v4.new_code_cell(text.strip())}
    return eval(compile(ast.Expression(construction), str(path), 'eval'), namespace)


def assert_executed(notebook, name):
    nbformat.validate(notebook)
    cells = [cell for cell in notebook.cells if cell.cell_type == 'code']
    missing = [i for i, cell in enumerate(notebook.cells)
               if cell.cell_type == 'code' and cell.execution_count is None]
    failures = [i for i, cell in enumerate(notebook.cells)
                if any(output.get('output_type') == 'error' for output in cell.get('outputs', []))]
    if not cells or missing or failures:
        raise RuntimeError(f'{name} is not completed successfully: unexecuted={missing}, errors={failures}')
    return {'notebook': name, 'code_cells': len(cells),
            'image_outputs': sum('image/png' in output.get('data', {})
                                 for cell in cells for output in cell.get('outputs', [])),
            'executed_without_errors': True}


def _protected_snapshot(notebook):
    return json.dumps({'metadata': notebook.metadata,
                       'cells': [cell for cell in notebook.cells if cell.cell_type != 'markdown']},
                      sort_keys=True)


def mirror_markdown(notebook, expected):
    """Return changed indices, while asserting executable content is identical."""
    if len(notebook.cells) < len(expected) or any(
        not cell.metadata.get('rte_assisted_extension')
        for cell in notebook.cells[len(expected):]
    ):
        raise RuntimeError('Notebook08 structure differs from generator; refuse positional updates')
    before = _protected_snapshot(notebook)
    changed = []
    for index in MARKDOWN_INDICES:
        if notebook.cells[index].cell_type != 'markdown' or expected[index].cell_type != 'markdown':
            raise RuntimeError(f'Expected markdown at notebook08 cell {index}')
        if notebook.cells[index].source != expected[index].source:
            notebook.cells[index].source = expected[index].source
            changed.append(index)
    if before != _protected_snapshot(notebook):
        raise AssertionError('Reporting update altered protected notebook content')
    return changed


def execution_limitations(notes: str):
    """Extract the existing interpretation and timing caveats, without new claims."""
    paragraphs = [' '.join(part.split()) for part in re.split(r'\n\s*\n', notes)]
    interpretation = next((part for part in paragraphs if part.startswith('Poor validation scores establish')), None)
    timing = next((part for part in paragraphs if part.startswith('Elapsed timers based on')), None)
    if interpretation is None or timing is None:
        raise ValueError('Execution notes lack the documented interpretation/timing paragraphs')
    # Keep the timing conclusion, while leaving future-benchmark advice in its source.
    timing_sentences = re.split(r'(?<=[.!?])\s+', timing)
    timing = ' '.join(timing_sentences[:3])
    return ('**Execution limitations.** '+interpretation+' '+timing+
            ' See [execution notes](execution_notes.md) for the source-derived training-origin '
            'reconstruction and the recorded pause/scheduling events.')


def complete_policy_distinction(policy, scores=None):
    """Describe two pretest choices; never select anything using test losses."""
    entries = pd.DataFrame(policy['entries']).set_index('model')
    complete = policy['selected_policy']
    independent = policy['selected_independent_model']
    text = (f"**Two frozen development choices.** Across all {len(entries)} predeclared policies, "
            f"the development choice is `{complete}` (2024 MAE {entries.loc[complete, 'MAE_MW']:,.1f} MW). "
            f"Across the 23 independent model variants, the core primary is `{independent}` "
            f"({entries.loc[independent, 'MAE_MW']:,.1f} MW). The supplied RTE baseline has development "
            f"MAE {entries.loc['rte_proxy', 'MAE_MW']:,.1f} MW. Both choices were frozen before test replay. ")
    if complete.startswith('rte'):
        text += 'The complete-policy choice uses RTE assistance and is not an independent replacement. '
    text += (f"The conformal and conditional peak analyses centered on `{independent}` describe that "
             "independent candidate; they do not establish uncertainty calibration for another policy.")
    if scores is not None:
        indexed = scores.set_index('model')
        required = {complete, independent, 'rte_proxy'}
        if not required.issubset(indexed.index):
            raise RuntimeError('Final scorecard misses a frozen policy/model/reference')
        policy_mae = float(indexed.loc[complete, 'MAE_MW'])
        independent_mae = float(indexed.loc[independent, 'MAE_MW'])
        rte_mae = float(indexed.loc['rte_proxy', 'MAE_MW'])
        improvement = 100*(1-policy_mae/rte_mae) if rte_mae else float('nan')
        text += (f" On the same locked retrospective comparison pairs, their MAEs are "
                 f"{policy_mae:,.1f} MW for `{complete}`, {independent_mae:,.1f} MW for `{independent}`, "
                 f"and {rte_mae:,.1f} MW for supplied RTE. The preselected complete policy's MAE "
                 f"improvement over RTE is {improvement:.2f}%.")
    return text


def _atomic_text(path, text):
    temporary = path.with_name(path.name+'.presentation.tmp')
    temporary.write_text(text)
    temporary.replace(path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, help='Defaults to .runtime/latest_research_run.json')
    args = parser.parse_args(argv)
    if args.run_dir is None:
        pointer = json.loads((ROOT/'.runtime/latest_research_run.json').read_text())
        args.run_dir = Path(pointer['run_dir'])
    run = args.run_dir if args.run_dir.is_absolute() else ROOT/args.run_dir
    # Refuse to touch an unfinished active notebook or an incomplete final report.
    notebooks = {name: nbformat.read(ROOT/'notebooks'/name, as_version=4) for name in NOTEBOOKS}
    notebook_status = [assert_executed(nb, name) for name, nb in notebooks.items()]
    for name, notebook in notebooks.items():
        messages = [str(output.get('text', ''))+' '+str(output.get('data', {}).get('text/plain', ''))
                    for cell in notebook.cells for output in cell.get('outputs', [])]
        if run.name not in '\n'.join(messages):
            raise RuntimeError(f'{name} outputs do not identify the requested run {run.name}')
    for name in ('frozen_selection.json', 'recommendation.md', 'execution_notes.md',
                 'tables/test_scorecard.csv', 'tables/uncertainty_scorecard.csv'):
        if not (run/name).exists():
            raise RuntimeError(f'Final research report is incomplete: {run/name}')
    expected = template_08_cells(ROOT/'scripts/build_research_notebooks.py')
    original_markdown = {i: notebooks[NOTEBOOKS[1]].cells[i].source for i in MARKDOWN_INDICES}
    mirror_markdown(notebooks[NOTEBOOKS[1]], expected)
    policy_path = run/'development_policy_selection.json'
    policy = json.loads(policy_path.read_text()) if policy_path.exists() else None
    scores = pd.read_csv(run/'tables/test_scorecard.csv')
    distinction = complete_policy_distinction(policy, scores) if policy is not None else None
    if distinction is not None:
        protected = _protected_snapshot(notebooks[NOTEBOOKS[1]])
        cell = notebooks[NOTEBOOKS[1]].cells[17]
        if POLICY_MARKER not in cell.source:
            cell.source = cell.source.rstrip()+'\n\n'+POLICY_MARKER+'\n\n'+distinction
        if protected != _protected_snapshot(notebooks[NOTEBOOKS[1]]):
            raise AssertionError('Policy reporting altered executable notebook content')
    changed = [i for i in MARKDOWN_INDICES if notebooks[NOTEBOOKS[1]].cells[i].source != original_markdown[i]]
    nbformat.validate(notebooks[NOTEBOOKS[1]])
    # Construct every report in memory first, then perform reporting-only writes.
    recommendation_path = run/'recommendation.md'
    original_recommendation = recommendation_path.read_text()
    recommendation = original_recommendation
    if distinction is not None and POLICY_MARKER not in recommendation:
        title, _, rest = recommendation.partition('\n')
        recommendation = title+'\n\n'+POLICY_MARKER+'\n\n'+distinction+'\n\n'+rest.lstrip()
    notes = (run/'execution_notes.md').read_text()
    appended = LIMITATION_MARKER not in recommendation
    if appended:
        recommendation = recommendation.rstrip()+'\n\n'+LIMITATION_MARKER+'\n\n'+execution_limitations(notes)+'\n'
    exporter = HTMLExporter(template_name='lab')
    previews = {name: exporter.from_notebook_node(nb)[0] for name, nb in notebooks.items()}
    if changed:
        path = ROOT/'notebooks'/NOTEBOOKS[1]
        temporary = path.with_name(path.name+'.presentation.tmp')
        nbformat.write(notebooks[NOTEBOOKS[1]], temporary)
        temporary.replace(path)
    if recommendation != original_recommendation:
        _atomic_text(recommendation_path, recommendation)
    preview_dir = ROOT/'reports/notebook_previews'
    preview_dir.mkdir(parents=True, exist_ok=True)
    for name, html in previews.items():
        _atomic_text(preview_dir/(Path(name).stem+'.html'), html)
    frozen = json.loads((run/'frozen_selection.json').read_text())
    selections = frozen.get('selections', {})
    fit_path = run/'tables/fit_diagnostics.csv'
    fit_counts = Counter(pd.read_csv(fit_path, usecols=['status']).status) if fit_path.exists() else Counter()
    summary = {
        'run_directory': str(run), 'finalized_utc': datetime.now(timezone.utc).isoformat(),
        'scope': 'reporting only; no training, predictor, manifest, selection or forecast changes',
        'notebooks': notebook_status, 'updated_notebook08_markdown_indices': changed,
        'execution_limitations_appended': appended,
        'execution_notes_sha256': hashlib.sha256(notes.encode()).hexdigest(),
        'investigated_families': sorted({value['family'] for value in selections.values()}),
        'experiment_arms_investigated': len(selections),
        'selected_arms': sum(value.get('status') == 'selected' for value in selections.values()),
        'point_scorecard_rows': len(scores), 'fit_event_status_counts': dict(fit_counts),
        'primary_candidate_from_validation': frozen.get('primary_candidate_from_validation'),
        'selected_complete_policy_from_development': policy.get('selected_policy') if policy else None,
        'independent_model_from_development': policy.get('selected_independent_model') if policy else None,
        'complete_policy_reporting_added': distinction is not None,
        'html_previews': [str(preview_dir/(Path(name).stem+'.html')) for name in NOTEBOOKS],
    }
    _atomic_text(run/'presentation_summary.json', json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
