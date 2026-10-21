"""Verify architecture against the live source tree; save manifests and honest costs."""
import ast
import csv
import hashlib
import json
import re
from datetime import datetime,timezone
from pathlib import Path

import networkx as nx
from graphify.cli import _stamped_manifest_files
from graphify.detect import save_manifest

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'graphify-out'


def main():
    inventory=json.loads((OUT/'architecture_inventory.json').read_text())
    extraction=json.loads((OUT/'.graphify_extract.json').read_text())
    detection=json.loads((OUT/'.graphify_detect.json').read_text())
    hashes=json.loads((OUT/'source_hashes.json').read_text())
    changed=[p for p,h in hashes.items() if not (ROOT/p).is_file() or hashlib.sha256((ROOT/p).read_bytes()).hexdigest()!=h]
    assert not changed,changed
    from build_architecture import files_in_project,ident
    current={p.relative_to(ROOT).as_posix() for p in files_in_project()}
    assert current==set(hashes),'Project file inventory changed during graph build'
    ids={n['id'] for n in extraction['nodes']}
    assert all(ident(p) in ids for p in current)
    assert all(e['source'] in ids and e['target'] in ids for e in extraction['edges'])
    assert len(inventory['notebooks'])==6
    code_cells=0
    for nb in inventory['notebooks']:
        assert ident(nb) in ids
        assert 'src/common.py' in inventory['notebook_imports'][nb]
        content=json.loads((ROOT/nb).read_text())
        for cell in content['cells']:
            if cell['cell_type']!='code':continue
            code_cells+=1
            text=''.join(cell['source'])
            text='\n'.join('pass' if line.lstrip().startswith(('%','!')) else line for line in text.splitlines())
            ast.parse(text)
    contract=json.loads((ROOT/'config/forecast_contract.json').read_text())
    archive_rows={}
    for p in sorted((ROOT/'data/raw').glob('*archive.csv')):
        with p.open() as f:archive_rows[p.name]=sum(1 for _ in csv.DictReader(f))
    assert len(archive_rows)==3 and all(v==0 for v in archive_rows.values())
    assert not any(contract['operational_evidence'].values())
    gate=list(csv.DictReader((ROOT/'reports/audit/operational_gate.csv').open()))
    assert all(row['operational_experiment_enabled'].lower()=='false' for row in gate)
    health=json.loads((OUT/'graph_health.json').read_text())
    for key in ['dangling_endpoint_edges','missing_endpoint_edges','self_loop_edges','undirected_same_endpoint_collapsed_edges']:
        assert health[key]==0,(key,health[key])

    # Validate cycles on real import statements only, excluding local function calls.
    imports=nx.DiGraph()
    for file in current:
        if not file.endswith('.py'):continue
        imports.add_node(file)
        tree=ast.parse((ROOT/file).read_text())
        for x in ast.walk(tree):
            if not isinstance(x,ast.ImportFrom):continue
            module=x.module or ''
            if x.level:
                base=list(Path(file).parent.parts)
                if x.level>1:base=base[:-(x.level-1)]
                module='.'.join(base+([module] if module else []))
            target=module.replace('.','/')+'.py'
            if target in current:imports.add_edge(file,target)
    cycles=list(nx.simple_cycles(imports))
    tests=ast.parse((ROOT/'tests/test_integrity.py').read_text())
    test_count=sum(isinstance(x,(ast.FunctionDef,ast.AsyncFunctionDef)) and x.name.startswith('test_') for x in ast.walk(tests))
    assert test_count==18
    verification={
        'verification_questions':[
            {'question':'Does the graph include every current project file and all six notebook stages, including their code-cell imports?','confirmed':True,'project_files':len(current),'notebooks':6,'parsed_code_cells':code_cells},
            {'question':'Are all graph endpoints valid and are original project files unchanged?','confirmed':True,'nodes':len(ids),'edges':len(extraction['edges']),'changed_or_removed_project_files':changed,'source_hashes_checked':len(hashes)},
            {'question':'Does the shown operational boundary match archive contents and configuration evidence?','confirmed':True,'archive_row_counts':archive_rows,'operational_gate_open':False}
        ],
        'python_import_cycles':cycles,'integrity_test_definitions':test_count,
        'graph_health':{k:health[k] for k in ['dangling_endpoint_edges','missing_endpoint_edges','self_loop_edges','undirected_same_endpoint_collapsed_edges']},
        'not_executed':'Forecasting models, project notebooks and project integrity suite were not rerun for this architecture task.',
        'browser_preview':'Blocked local file protocol; native static plot inspected and overview behavior checked with offline JSDOM simulation.',
        'confidence_note':'Source-level dependencies and documented flow verified; dynamic runtime dependencies are not exhaustively traced.'
    }
    (OUT/'verification.json').write_text(json.dumps(verification,indent=2))
    report=(OUT/'GRAPH_REPORT.md').read_text()
    cycle_section='## Import Cycles\n'+ ('\nNo cycles found in actual internal Python import statements.\n' if not cycles else '\n'+ '\n'.join('- '+' → '.join(c) for c in cycles)+'\n')
    report=re.sub(r'## Import Cycles\n.*?(?=\n## )',cycle_section,report,flags=re.S)
    report=report.replace('- Token cost: 0 input · 0 output','- Semantic token cost: **unreported by host**. Zero values in machine metadata are placeholders; actual cost is unknown.')
    report+='\n## Final verification\n\n'
    report+=f'- SHA-256 verified unchanged for all {len(hashes)} existing project files. Outputs are in graphify-out/; isolated package setup is under ignored .runtime/. The automatic design hook also created .impeccable/hook.cache.json; it is tooling state, excluded from the research-file inventory.\n'
    report+=f'- All six notebooks and {code_cells} code cells parsed; all file-inventory nodes and graph endpoints verified.\n'
    report+='- Import-cycle claims were checked against real Python ImportFrom statements. The original automatic report incorrectly treated local calls as one-file import cycles; the corrected section reports actual module cycles.\n'
    report+='- Community cohesion describes graph connectivity, including inventory edges, not software modularity. Automatic suggested questions about splitting communities are navigation prompts, not refactoring recommendations. Low-degree external imports and generated files are expected leaves.\n'
    report+='- The overview was checked by offline DOM simulation: mode switching, selection, keyboard, reset, file search and zero script errors. Matplotlib-generated architecture.png was visually inspected. Browser preview was blocked by local-file URL policy and remains unverified.\n'
    report+='- Readability review: small overview text enlarged; no design-hook suppressions applied.\n'
    report+='- Graph HTML runtime dependency saved under vendor/ and verified against the original SHA-384 integrity digest. Both HTML views work from local assets; no third-party runtime requests are required.\n'
    (OUT/'GRAPH_REPORT.md').write_text(report)
    corpus=detection.get('all_files') or detection['files']
    stamped=_stamped_manifest_files(corpus,extraction,ROOT)
    dispatched={f for t,fl in detection['files'].items() if t in ('document','paper','image') for f in fl}
    successful={f for fl in stamped.values() for f in fl}
    save_manifest(stamped,root=ROOT,scan_corpus={f for fl in corpus.values() for f in fl},clear_semantic=(dispatched-successful) or None)
    cost={'runs':[{'date':datetime.now(timezone.utc).isoformat(),'input_tokens':0,'output_tokens':0,'files':detection['total_files'],'usage_status':'unreported by host; zeros are placeholders'}],'total_input_tokens':0,'total_output_tokens':0,'usage_status':'unknown; counters do not measure host-agent extraction usage','semantic_money_cost':None,'ast_extraction':'deterministic; no external LLM calls'}
    (OUT/'cost.json').write_text(json.dumps(cost,indent=2))
    # Retain human-curated evidence/inventory and reusable build helpers; remove run sidecars.
    cleanup=['.graphify_detect.json','.graphify_extract.json','.graphify_ast.json','.graphify_semantic.json','.graphify_analysis.json','.graphify_architecture.json','.graphify_semantic_new.json','.graphify_cached.json','.graphify_uncached.txt','semantic_result_staging.json','notebook_sources.json','.needs_update']
    for name in cleanup:(OUT/name).unlink(missing_ok=True)
    for p in OUT.glob('.graphify_chunk_*.json'):p.unlink()
    print(json.dumps(verification,indent=2))


if __name__=='__main__':main()
