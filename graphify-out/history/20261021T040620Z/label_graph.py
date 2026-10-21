import json
from pathlib import Path
from graphify.analyze import suggest_questions
from graphify.build import build_from_json
from graphify.export import to_json
from graphify.report import generate

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'graphify-out'
labels={0:'Source Audit and Tooling',1:'Research Contract and Evidence',2:'Evaluation and Uncertainty',3:'Immutable Forecast Archives',4:'Availability and Integrity Tests',5:'Forecast Runs and Ledgers',6:'Validation and Model Selection',7:'Research Reports and Figures',8:'Fitted Model Artifacts',9:'Origin Feature Store',10:'Repository Exclusions'}
extraction=json.loads((OUT/'.graphify_extract.json').read_text())
detection=json.loads((OUT/'.graphify_detect.json').read_text())
analysis=json.loads((OUT/'.graphify_analysis.json').read_text())
G=build_from_json(extraction,root=str(ROOT),directed=False)
communities={int(k):v for k,v in analysis['communities'].items()}
cohesion={int(k):v for k,v in analysis['cohesion'].items()}
questions=suggest_questions(G,communities,labels)
assert to_json(G,communities,str(OUT/'graph.json'),community_labels=labels)
report=generate(G,communities,cohesion,labels,analysis['gods'],analysis['surprises'],detection,{'input':extraction['input_tokens'],'output':extraction['output_tokens']},str(ROOT),suggested_questions=questions)
inv=json.loads((OUT/'architecture_inventory.json').read_text())
report+='\n\n## Architecture extraction scope and verification\n\n'
report+='- Start with [architecture.html](architecture.html) for the readable stage map; use [graph.html](graph.html) for the full Graphify network.\n'
report+=f'- {len(inv["files"])} actual project files inventoried; six notebooks parsed by code cell, with original cell/line evidence. Runtime environments, Git internals and graph outputs are excluded.\n'
report+='- Graphify detection supports 58 source/configuration/document files. Notebook and CSV coverage is supplied by the additional deterministic architecture extraction. CSV contents, serialized models and generated plot pixels were not interpreted. Plot PNG and duplicate HTML preview paths remain in the inventory.\n'
report+='- Relationships are source-level imports/calls, literal path reads/writes/checks and explicitly documented notebook order. This is not a runtime trace. Dynamic paths, arbitrary reflection and external package implementations may be absent.\n'
report+='- Default Graphify graph is undirected. Original direction, relationship type and each source location remain in every edge’s `relationship_observations` array. The overview uses labeled directional links.\n'
report+='- The initial AST left 25 imported external modules as dangling identifiers. Those imports are now explicit external-reference nodes. Repeated endpoint observations were grouped, retaining every original observation in the evidence array; see normalization_audit.json. Final graph health has no missing/dangling endpoints, self-loops or unintended edge collapse.\n'
report+='- Semantic extraction: 44 nodes, 55 edges, three hyperedges; inferred connections remain labeled INFERRED. Host-agent token usage was not exposed. Any zero token counts in generated sections or cost.json are placeholders, not measured usage; semantic monetary cost is unknown.\n'
report+='- Current operational gate is closed: the archive files contain headers only and vintage/interval provenance flags are false. pipeline.py provides readiness checks/specifications; it is not an implemented ingestion/scheduling/publication/procurement service.\n'
for issue in inv['issues']: report+='- Documentation issue: '+issue+'\n'
report+='- graphify-out/ was written into the working tree. It may appear in git status unless ignored; no repository ignore rules were changed.\n'
report+='- Rebuild architecture augmentation after source changes; a plain Graphify incremental update does not reparse notebook cells or reconstruct the custom inventory edges. Existing project source, data and notebook files were not edited.\n'
(OUT/'GRAPH_REPORT.md').write_text(report)
(OUT/'.graphify_labels.json').write_text(json.dumps({str(k):v for k,v in labels.items()}))
analysis['questions']=questions
(OUT/'analysis.json').write_text(json.dumps(analysis,indent=2))
print('Curated labels and audited report saved')
