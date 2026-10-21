"""Merge verified structural/semantic evidence and rebuild the existing Graphify graph."""
import json
from pathlib import Path
from graphify.analyze import god_nodes,suggest_questions,surprising_connections
from graphify.build import build_from_json
from graphify.cluster import cluster,score_all
from graphify.diagnostics import diagnose_extraction,format_diagnostic_report
from graphify.export import to_json
from graphify.report import generate
from build_architecture import ident

ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'graphify-out'

def main():
    structural=json.loads((OUT/'.graphify_architecture.json').read_text())
    sem=json.loads((OUT/'.graphify_semantic.json').read_text())
    nodes={n['id']:n for n in structural['nodes']}; edges=list(structural['edges'])
    for n in sem['nodes']:
        n['_origin']='semantic'; nodes.setdefault(n['id'],n)
    for e in sem['edges']: e['_origin']='semantic'
    edges+=sem['edges']
    for n in sem['nodes']:
        p=Path(n['source_file']); rel=p.relative_to(ROOT).as_posix() if p.is_absolute() else p.as_posix()
        assert (ROOT/rel).exists(),rel
        anchor=ident(rel)
        if anchor in nodes and anchor!=n['id']:
            edges.append({'source':n['id'],'target':anchor,'relation':'documented_in','confidence':'EXTRACTED','confidence_score':1.0,'source_file':rel,'source_location':n.get('source_location') or 'document concept extraction','weight':1.0,'_origin':'semantic'})
    grouped={}; self_references=[]
    for e in edges:
        assert e['source'] in nodes and e['target'] in nodes,e
        if e['source']==e['target']:
            self_references.append(e); continue
        grouped.setdefault(tuple(sorted((e['source'],e['target']))),[]).append(e)
    normalized=[]
    for observations in grouped.values():
        # Exact repeated containment/import observations are redundant; preserve distinct evidence.
        unique={json.dumps(x,sort_keys=True,default=str):x for x in observations}; obs=list(unique.values())
        e=dict(obs[0]); e['relation']=' / '.join(sorted({x['relation'] for x in obs}))
        e['relationship_observations']=obs; e['observation_count']=len(obs)
        e['confidence_score']=min(x.get('confidence_score',1.0) for x in obs)
        confidences={x.get('confidence','EXTRACTED') for x in obs}
        e['confidence']='AMBIGUOUS' if 'AMBIGUOUS' in confidences else 'INFERRED' if 'INFERRED' in confidences else 'EXTRACTED'
        normalized.append(e)
    extraction={'nodes':list(nodes.values()),'edges':normalized,'hyperedges':sem.get('hyperedges',[]),'input_tokens':sem.get('input_tokens',0),'output_tokens':sem.get('output_tokens',0),'token_usage_status':'Host-agent token usage not exposed; numeric zeros are unmeasured placeholders.'}
    assert nodes,'Refusing empty graph'
    for name in ['.graphify_extract.json','architecture_evidence.json']:(OUT/name).write_text(json.dumps(extraction,indent=2)+'\n')
    (OUT/'normalization_audit.json').write_text(json.dumps({'input_relationship_observations':len(edges),'simple_graph_edges':len(normalized),'distinct_observations_retained':sum(x['observation_count'] for x in normalized),'self_reference_observations':self_references,'policy':'Simple undirected visualization; every distinct directed observation, confidence, type and source location is retained in relationship_observations. Syntactic self-reference observations are retained separately here.'},indent=2)+'\n')
    G=build_from_json(extraction,root=ROOT,directed=False)
    communities=cluster(G); cohesion=score_all(G,communities)
    gods=god_nodes(G); surprises=surprising_connections(G,communities)
    labels={k:'Community '+str(k) for k in communities}; questions=suggest_questions(G,communities,labels)
    # This requested architecture refresh intentionally removes deleted source,
    # replaces re-extracted concepts and groups generated shards. A previous
    # Graphify guard refusal is resolved only after every retired ID is audited.
    retired_path=OUT/'retired_nodes_audit.json'
    force=False
    if retired_path.exists():
        old=json.loads((OUT/'graph.json').read_text())
        retired={n['id'] for n in old['nodes']} - set(G.nodes)
        audit=json.loads(retired_path.read_text())
        force=bool(retired) and retired=={r['id'] for r in audit} and all(r['reason']!='unclassified' for r in audit)
    assert to_json(G,communities,str(OUT/'graph.json'),force=force),'Graph guard rejected an unaudited change; existing graph was preserved.'
    detection=json.loads((OUT/'.graphify_detect.json').read_text())
    report=generate(G,communities,cohesion,labels,gods,surprises,detection,{'input':extraction['input_tokens'],'output':extraction['output_tokens']},str(ROOT),suggested_questions=questions)
    (OUT/'GRAPH_REPORT.md').write_text(report)
    analysis={'communities':{str(k):v for k,v in communities.items()},'cohesion':{str(k):v for k,v in cohesion.items()},'gods':gods,'surprises':surprises,'questions':questions}
    (OUT/'.graphify_analysis.json').write_text(json.dumps(analysis,indent=2)+'\n')
    health=diagnose_extraction(extraction,directed=False,root=str(ROOT))
    (OUT/'graph_health.json').write_text(json.dumps(health,indent=2)+'\n')
    print(format_diagnostic_report(health))
    print(f'Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges, {len(communities)} communities')
    for cid,members in communities.items():
        print(cid,len(members),[nodes[k]['label'] for k in members if k in nodes][:12])

if __name__=='__main__':main()
