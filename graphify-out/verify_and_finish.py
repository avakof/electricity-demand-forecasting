"""Validate graph/documentation against inspected sources; preserve honest run metadata."""
import ast
import csv
import hashlib
import json
import re
from datetime import datetime,timezone
from pathlib import Path
from urllib.parse import unquote
import networkx as nx
from graphify.cli import _stamped_manifest_files
from graphify.detect import save_manifest
from build_architecture import files_in_project,ident,artifact_group,clean_cell
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'graphify-out'

def main():
    inv=json.loads((OUT/'architecture_inventory.json').read_text())
    extraction=json.loads((OUT/'.graphify_extract.json').read_text())
    detection=json.loads((OUT/'.graphify_detect.json').read_text())
    hashes=json.loads((OUT/'source_hashes.json').read_text())
    changed=[p for p,h in hashes.items() if not (ROOT/p).is_file() or hashlib.sha256((ROOT/p).read_bytes()).hexdigest()!=h]
    assert not changed,changed
    current={p.relative_to(ROOT).as_posix() for p in files_in_project()}
    assert current=={x['path'] for x in inv['files']},'Inventory changed during build'
    ids={n['id'] for n in extraction['nodes']}
    assert all(ident(artifact_group(p) or p) in ids for p in current)
    assert all(e['source'] in ids and e['target'] in ids for e in extraction['edges'])
    assert not inv['issues'],inv['issues']
    removed=['src/pipeline.py','reports/monitoring_specification.csv']
    assert all(not (ROOT/p).exists() and ident(p) not in ids for p in removed)
    assert not any(str(n.get('source_file','')).endswith('/src/pipeline.py') or n.get('source_file')=='src/pipeline.py' for n in extraction['nodes'])
    expected={p.relative_to(ROOT).as_posix() for p in (ROOT/'notebooks').glob('*.ipynb')}
    assert set(inv['notebooks'])==expected and len(expected)==9
    cells=0
    for nb in expected:
        assert ident(nb) in ids
        for c in json.loads((ROOT/nb).read_text())['cells']:
            if c['cell_type']=='code': ast.parse(clean_cell(''.join(c['source'])));cells+=1
    health=json.loads((OUT/'graph_health.json').read_text())
    keys=['dangling_endpoint_edges','missing_endpoint_edges','self_loop_edges','undirected_same_endpoint_collapsed_edges']
    assert all(health[k]==0 for k in keys)
    overview=json.loads((OUT/'architecture_overview.json').read_text());byid={n['id']:n for n in overview['nodes']}
    refs=0
    for e in overview['edges']:
        assert e['source'] in byid and e['target'] in byid
        for r in e['references']:
            p=ROOT/r['path'];assert p.is_file()
            cell,line=r.get('cell'),r.get('line')
            match=re.fullmatch(r'cell (\d+), L(\d+)',r.get('location',''))
            if match:cell,line=map(int,match.groups())
            if cell:
                source=json.loads(p.read_text())['cells'][cell-1]['source'];lines=''.join(source).splitlines()
            else:lines=p.read_text().splitlines()
            assert line and 1<=line<=len(lines),(e,r)
            refs+=1
    archive_rows={}
    for p in (ROOT/'data/raw').glob('*archive.csv'):
        with p.open() as f:archive_rows[p.name]=sum(1 for _ in csv.DictReader(f))
    assert len(archive_rows)==3 and not any(archive_rows.values())
    links=0
    for filename in ['CODE_DOCUMENTATION.md','GRAPH_REPORT.md']:
        for value in re.findall(r'\]\(([^)]+)\)',(OUT/filename).read_text()):
            if '://' in value or value.startswith('#'):continue
            p=(OUT/unquote(value.split('#')[0].strip('<>'))).resolve();assert p.exists(),(filename,str(p));links+=1
    directed=nx.DiGraph()
    for m in inv['module_summary']:
        rel=m['path'];directed.add_node(rel);tree=ast.parse((ROOT/rel).read_text())
        for x in ast.walk(tree):
            if not isinstance(x,ast.ImportFrom):continue
            module=x.module or ''
            if x.level:
                base=list(Path(rel).parent.parts)
                if x.level>1:base=base[:-(x.level-1)]
                module='.'.join(base+([module] if module else []))
            target=module.replace('.','/')+'.py'
            if target in current:directed.add_edge(rel,target)
    cycles=list(nx.simple_cycles(directed))
    verification={'verified_at_utc':datetime.now(timezone.utc).isoformat(),'verification_questions':[
        {'question':'Does the graph cover current source, all nine notebooks, and grouped artifacts while removing deleted production components?','confirmed':True,'inventory_files':len(current),'notebooks':len(expected),'parsed_code_cells':cells,'removed_paths':removed},
        {'question':'Are every endpoint and overview evidence reference valid, with inspected inputs unchanged?','confirmed':True,'nodes':len(ids),'edges':len(extraction['edges']),'source_hashes_checked':len(hashes),'changed_inputs':changed,'overview_components':len(byid),'overview_relationships':len(overview['edges']),'overview_source_references_checked':refs,'documentation_links_checked':links},
        {'question':'Do availability and model distinctions match source evidence?','confirmed':True,'archive_rows':archive_rows,'note':'Three versioned later protocols, separate point/quantile/calibration methods, no verified operating deployment.'}],
        'graph_health':{k:health[k] for k in keys},'unresolved_ast_references':len(inv['unresolved_ast_references']),'unlocated_type_or_import_references':len(inv.get('unlocated_ast_references',[])),'python_import_cycles':cycles,'test_definitions_inspected':inv['test_count'],'not_executed':'No predictive models, project notebooks, or forecasting test suite were run by the graph refresh.','token_usage':'Unreported by host; counters of zero are placeholders.','browser_preview':'See interaction_validation.json and browser_validation.json for the separate UI checks.'}
    (OUT/'verification.json').write_text(json.dumps(verification,indent=2)+'\n')
    corpus=detection.get('all_files') or detection['files'];stamped=_stamped_manifest_files(corpus,extraction,ROOT)
    dispatched={f for t,fs in corpus.items() if t in ('document','paper','image') for f in fs};successful={f for fs in stamped.values() for f in fs}
    save_manifest(stamped,root=ROOT,scan_corpus={f for fs in corpus.values() for f in fs},clear_semantic=(dispatched-successful) or None)
    cost_path=OUT/'cost.json';cost=json.loads(cost_path.read_text()) if cost_path.exists() else {'runs':[],'total_input_tokens':0,'total_output_tokens':0}
    cost.setdefault('runs',[]).append({'date':datetime.now(timezone.utc).isoformat(),'files':detection['total_files'],'semantic_files':len(dispatched),'input_tokens':0,'output_tokens':0,'usage_status':'unreported by host; zeros are unmeasured placeholders'})
    cost.update(usage_status='unknown; host-agent extraction usage is not exposed',semantic_money_cost=None,ast_extraction='deterministic; no external LLM calls')
    cost_path.write_text(json.dumps(cost,indent=2)+'\n')
    (OUT/'import_cycles.json').write_text(json.dumps(cycles,indent=2)+'\n')
    print(json.dumps({'nodes':len(ids),'edges':len(extraction['edges']),'source_hashes':len(hashes),'notebooks':len(expected),'cells':cells,'overview_references':refs,'documentation_links':links,'graph_health':'passed','import_cycles':cycles},indent=2))

if __name__=='__main__':main()
