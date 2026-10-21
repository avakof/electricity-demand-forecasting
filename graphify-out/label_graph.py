"""Curate community names and contextualize automatic Graphify findings."""
import ast
import hashlib
import json
import re
from pathlib import Path
from collections import Counter
import networkx as nx
from graphify.analyze import suggest_questions
from graphify.build import build_from_json
from graphify.export import to_json
from graphify.report import generate
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'graphify-out'
CURATED={0:'Expanded Risk Manifest',1:'Historical Trial Manifest',2:'Interrupted Research Manifest',3:'Completed Research Manifest',4:'Notebook Execution and Presentation',5:'Learned Features and Diagnostics',6:'Quantile and Risk Tests',7:'Research Replay Integrity',8:'Learned Feature Cache',9:'Tabular and Statistical Adapters',10:'Report App Dependencies',11:'RTE Assisted Artifacts',12:'Original Peak Risk Analysis',13:'Expanded Artifact Verification',14:'Original Replay Integrity',15:'Research Commands and Serialization',16:'Original Research Score Tables',17:'Assisted Contract Settings',18:'Assisted Scores and Metrics',19:'Expanded Risk Reporting',20:'Assisted Run Manifest',21:'Research Replay Engine',22:'Frozen Selection Evidence',23:'Expanded Risk Replay',24:'RTE Target and Fallback Tests',25:'Causal Base Forecast Bank',26:'Original Notebook Authoring',27:'Neural Model Adapter',28:'Manager Report and Review',29:'Input Sources and Specifications',30:'Probabilistic Scoring Tests',31:'Original Artifact Verification',32:'Historical Failure Analysis',33:'Completed Research Artifacts',34:'Neural Panel Integrity',35:'Causal Feature Engineering',36:'Specification Source Hashes',37:'Research Comparison Figures',38:'Chronology and DST Tests',39:'Report Adoption Content',40:'Interactive Report Composition',41:'RTE Assisted Replay',42:'Research Data Preparation',43:'Feature Availability Tests',44:'Saved Forecast Replay',45:'Original Candidate Validation',46:'Project Research Documentation',47:'Expanded Risk Artifacts',48:'Original Audit and EDA',49:'Legacy Baseline Investigation',50:'Shared Report Components',51:'Manager Failure Diagnostics',52:'Expanded Recommendation and Review',53:'Model Adapter Tests',54:'Historical Research Runs',55:'Original Uncertainty and Storage',56:'Report Layout Components',57:'Failure Diagnostic Computation',58:'Forecast Contracts and Locks',59:'Development Sensitivities',60:'Report Build Integrity',61:'Report Build Commands',62:'Neural Execution Limitations',63:'Research Resume Provenance',64:'Origin Feature Store',65:'Assisted Search Budgets',66:'Original Reference Fingerprints',67:'Manager Figure Exports',68:'Evidence and Vintage Limits',69:'Scientific Environment Requirements',70:'Original Manager Recommendation',71:'Earlier Workflow Recommendation',72:'Original Peak Interval Design',73:'Original Saved Test Evidence',74:'Assisted Calibration Window',75:'Legacy Baseline Results',76:'Audited Data and Labels',77:'Development Fit Diagnostics',78:'Original Failure Catalogue',79:'Assisted Cold Definition',80:'Assisted Test Window',81:'Assisted Raw Source Identity',82:'Repository Ignore Rules'}

def main():
    extraction=json.loads((OUT/'.graphify_extract.json').read_text());detection=json.loads((OUT/'.graphify_detect.json').read_text());a=json.loads((OUT/'.graphify_analysis.json').read_text())
    G=build_from_json(extraction,root=ROOT,directed=False);communities={int(k):v for k,v in a['communities'].items()};cohesion={int(k):v for k,v in a['cohesion'].items()}
    labels={};catalog_path=OUT/'community_label_catalog.json';catalog=json.loads(catalog_path.read_text()) if catalog_path.exists() else {}
    first=not catalog
    for k,ids in communities.items():
        signature=hashlib.sha256('\n'.join(sorted(ids)).encode()).hexdigest()
        if signature in catalog:label=catalog[signature]
        elif first and k in CURATED:label=CURATED[k]
        else:
            sources=Counter(G.nodes[n].get('source_file','') for n in ids if G.nodes[n].get('source_file'))
            name=Path(sources.most_common(1)[0][0]).stem if sources else 'Source References'
            label=' '.join(name.replace('_',' ').split()[:4]).title()
        labels[k]=label;catalog[signature]=label
    catalog_path.write_text(json.dumps(catalog,indent=2)+'\n')
    questions=suggest_questions(G,communities,labels)
    assert to_json(G,communities,str(OUT/'graph.json'),community_labels=labels),'Unexpected graph-export guard refusal'
    report=generate(G,communities,cohesion,labels,a['gods'],a['surprises'],detection,{'input':0,'output':0},str(ROOT),suggested_questions=questions)
    inv=json.loads((OUT/'architecture_inventory.json').read_text())
    report=report.replace('- Token cost: 0 input · 0 output','- Semantic token cost: **unreported by host**. Numeric zeros are unmeasured placeholders; semantic monetary cost is unknown. AST extraction uses no external LLM.')
    report=re.sub(r'- Unclassified:.*\n',f'- Additional deterministic inventory: {len(inv["files"]):,} files, including unsupported CSV/compressed/model formats represented by files or artifact groups. Their contents are not semantically interpreted.\n',report)
    # Automatic cycle detection treats local calls as import loops. Use actual module imports.
    imports=nx.DiGraph();current={x['path'] for x in inv['files']}
    for m in inv['module_summary']:
        rel=m['path'];imports.add_node(rel)
        for x in ast.walk(ast.parse((ROOT/rel).read_text())):
            if not isinstance(x,ast.ImportFrom):continue
            mod=x.module or ''
            if x.level:
                base=list(Path(rel).parent.parts)
                if x.level>1:base=base[:-(x.level-1)]
                mod='.'.join(base+([mod] if mod else []))
            target=mod.replace('.','/')+'.py'
            if target in current:imports.add_edge(rel,target)
    cycles=list(nx.simple_cycles(imports));cycle_text='## Import Cycles\n\n'+ ('No cycles found in inspected internal Python `ImportFrom` statements. Local function calls are not module import cycles. JavaScript runtime cycles are not certified.\n' if not cycles else '\n'.join('- '+' → '.join(x) for x in cycles)+'\n')
    report=re.sub(r'## Import Cycles\n.*?(?=\n## )',cycle_text,report,flags=re.S)
    surprises='''## Surprising Connections (verified architecture bridges)

- Original model forecasts → causal base bank → LightGBM hybrid combiner. The extension reuses earlier out-of-sample forecasts; it does not natively retrain every base architecture with RTE. [Base-bank source](../src/rte_base_replay.py), [risk replay](../src/rte_risk_lab.py).
- Engineered history → earlier-trained Ridge/classifier features → selected point correction. The selected stacked model has `base: null`, so this path is distinct from the archived-base hybrids. [Learned features](../src/rte_learned_features.py), [frozen selection](../reports/rte_risk/20261020T213640021260Z/frozen_selection.json).
- Separately selected quantiles → fixed calibration comparisons → manager risk assessment. Point accuracy, probability selection and cold coverage remain different evidence. [Uncertainty replay](../src/rte_risk_uncertainty.py), [manager report](../reports/manager_report_20261021/complete_report.md).
'''
    report=re.sub(r'## Surprising Connections.*?(?=\n## )',surprises,report,flags=re.S)
    navigation='''## Suggested Questions

- How does an original model’s sealed forecast reach an expanded LightGBM hybrid while preserving outcome availability?
- Which inputs differ between national, regional-only and combined predictor views?
- Why does improved point accuracy not establish adequate cold-evening interval coverage?
'''
    report=re.sub(r'## Suggested Questions.*\Z',navigation,report,flags=re.S)
    report+='''
## Architecture extraction scope and verification

Start with [architecture.html](architecture.html), then inspect [CODE_DOCUMENTATION.md](CODE_DOCUMENTATION.md), [graph.json](graph.json), [verification.json](verification.json) and [refresh_audit.md](refresh_audit.md).

'''
    report+=f'- Current overview: 32 components and 72 sourced relationships. All nine notebooks are represented: eight numbered stages and a distinct legacy baseline branch. The original workflow and three later versioned protocols remain separate.\n- Source extraction: {len(inv["module_summary"])} Python files and all notebook code cells; {len(inv["files"]):,} inventoried files; {len(inv["grouped_artifacts"])} generated artifact groups. Detected JSON metadata is source structure, not execution evidence.\n'
    report+='''- Removed five obsolete nodes for the deleted production module and monitoring file; grouped 120 individual generated/tooling-era artifact nodes, replaced 44 re-extracted concepts, and excluded one local tooling node. Every retired ID is audited in [retired_nodes_audit.json](retired_nodes_audit.json); previous graph outputs are backed up under `history/`.
- Four unchanged documents had ten cached cross-references to replaced concepts. All four were re-extracted; no missing endpoint was invented to preserve a stale edge. See [semantic_cache_review.json](semantic_cache_review.json).
- The full Graphify view is undirected. Original direction, relation type, confidence and source evidence remain in each edge's `relationship_observations`. The overview renders directional relationships. Distinct repeated evidence is preserved rather than silently overwritten.
- Two syntactic self-reference observations are retained separately in [normalization_audit.json](normalization_audit.json). They are excluded from the navigation graph. Final graph diagnostics report no dangling or missing endpoints, graph self-loops or unintended endpoint collapse.
- Unresolved AST implementation references are explicitly materialized and labeled; their presence is not evidence that an external implementation was scanned. Dynamic runtime dependencies and values assembled through `run.path` are not exhaustively resolved.
- Type/decorator/import placeholders emitted without a definition location are labeled `Reference: …` with `architecture_kind: type_or_import_reference`; their incident edges retain source evidence. They are reference concepts, not inspected implementation definitions. Their IDs are listed in the inventory separately from missing-endpoint references.
- God-node degree and raw community cohesion describe this mixed source/metadata graph. Common packages and manifest fields can dominate connectivity. Neither centrality nor weak cohesion establishes an important modeling feature, software defect or refactoring need. Automatic navigation statistics remain available in [analysis.json](analysis.json).
- The selected point and quantile models, fixed conformal/CQR comparators, 23 hybrid combiners and metrics retain separate identities. Original forecast vintages remain unverified; the test was previously explored and calibration has no cold evenings. The graph does not certify operational superiority or measured financial savings.
- No predictive code, experiment configuration, notebook, raw data, fitted model or saved forecast was changed or executed for this graph refresh. The project README and architecture/documentation outputs were updated. Saved test outcomes are historical evidence, not tests rerun by this task.
- Graphify outputs were written into the working tree and may appear in `git status`; existing ignore rules were preserved. See [Graphify team workflows](https://docs.graphify.com/guides/team-workflows).
'''
    (OUT/'GRAPH_REPORT.md').write_text(report)
    (OUT/'.graphify_labels.json').write_text(json.dumps({str(k):v for k,v in labels.items()},indent=2)+'\n')
    a.update(labels={str(k):v for k,v in labels.items()},questions=questions)
    (OUT/'analysis.json').write_text(json.dumps(a,indent=2)+'\n')
    print(f'Curated {len(labels)} community labels and corrected report scope/cycle interpretation')

if __name__=='__main__':main()
