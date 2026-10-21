"""Source-only architecture refresh; no model/notebook execution or data edits."""
import ast
import hashlib
import json
import os
import re
from collections import defaultdict
from pathlib import Path
from urllib.parse import unquote

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'graphify-out'
SKIP={'.git','.runtime','.research-deps','.impeccable','.venv','venv','node_modules','__pycache__','.ipynb_checkpoints','graphify-out','.pytest_cache'}
ORDER=['01_data_audit','02_development_eda','03_backtest_and_baselines','04_model_comparison','05_uncertainty_and_failures','06_business_and_deployment']

def ident(path):
    return re.sub(r'[^a-z0-9]+','_',str(Path(path).with_suffix('')).lower()).strip('_')

def files_in_project():
    result=[]
    for directory,dirs,names in os.walk(ROOT):
        dirs[:]=sorted(d for d in dirs if d not in SKIP)
        result.extend(Path(directory)/n for n in names if n!='.DS_Store')
    return sorted(p for p in result if p.is_file())

def artifact_group(rel):
    parts=Path(rel).parts; app='reports/manager_report_20261021/app'
    if rel.startswith(app+'/') and not rel.startswith(app+'/src/content/'): return app
    for name in ('fits','forecasts','figures','notebook_previews','models','ledgers','origin_ledgers','base_replays'):
        if name in parts and parts[0] in {'reports','results'}: return Path(*parts[:parts.index(name)+1]).as_posix()
    return None

def clean_cell(text):
    return '\n'.join('pass' if line.lstrip().startswith(('%','!','?')) else line for line in text.splitlines())

def main():
    extraction=json.loads((OUT/'.graphify_ast.json').read_text())
    nodes={n['id']:n for n in extraction['nodes']}; edges=list(extraction['edges'])
    unlocated=[]
    for entity in nodes.values():
        if entity.get('_origin')=='ast' and not entity.get('source_file'):
            entity.update(file_type='concept',architecture_kind='type_or_import_reference',rationale='Syntactic type, decorator or import reference. Its implementation and definition location were not scanned; incident edges retain caller/source evidence.')
            entity['label']='Reference: '+entity['label']
            unlocated.append(entity['id'])
    detection=json.loads((OUT/'.graphify_detect.json').read_text())
    selected={str(Path(f).resolve().relative_to(ROOT)) for fs in detection['files'].values() for f in fs}
    project_files=files_in_project(); inventory=[]; hashes={}; groups=defaultdict(list)
    for p in project_files:
        rel=p.relative_to(ROOT).as_posix(); group=artifact_group(rel)
        inventory.append({'path':rel,'bytes':p.stat().st_size,'id':ident(rel),'graph_group':group})
        if group: groups[group].append(rel)
        if rel in selected or p.suffix=='.ipynb': hashes[rel]=hashlib.sha256(p.read_bytes()).hexdigest()
    (OUT/'source_hashes.json').write_text(json.dumps(hashes,indent=2)+'\n')
    def node(path,directory=False,grouped=False):
        key=ident(path)
        if key not in nodes:
            code=Path(path).suffix.lower() in {'.py','.ipynb','.json','.js','.jsx'}
            nodes[key]={'id':key,'label':str(path),'file_type':'code' if code else 'document','source_file':str(path),'source_location':'filesystem inventory','_origin':'architecture','architecture_kind':'artifact_group' if grouped else 'directory' if directory else 'file'}
        if grouped: nodes[key].update(architecture_kind='artifact_group',file_count=len(groups.get(str(path),[])),rationale='Generated members are inventoried, not individually interpreted.')
        else: nodes[key].setdefault('architecture_kind','directory' if directory else 'file')
        return key
    def edge(a,b,relation,source,location,**kw):
        if a!=b: edges.append({'source':a,'target':b,'relation':relation,'confidence':'EXTRACTED','confidence_score':1.0,'source_file':source,'source_location':location,'weight':1.0,'_origin':'architecture',**kw})
    # Graph group nodes retain individual filenames only in the searchable inventory.
    done=set()
    for item in inventory:
        rel=item['graph_group'] or item['path']
        if rel in done: continue
        done.add(rel); child=node(rel,bool(item['graph_group']),bool(item['graph_group']))
        for parent in Path(rel).parents:
            if parent.as_posix()=='.': break
            parent_id=node(parent.as_posix(),True)
            edge(parent_id,child,'contains',rel,'filesystem inventory'); child=parent_id
    def existing(path):
        try:
            p=Path(path).resolve(); rel=p.relative_to(ROOT).as_posix()
        except (ValueError,OSError): return None
        if not p.exists() or any(x in SKIP for x in Path(rel).parts): return None
        group=artifact_group(rel)
        return node(group or rel,p.is_dir() or bool(group),bool(group))
    # Some AST parsers emit disconnected nested JSON keys/docstrings. Anchor
    # their existing source locations to the file without inventing a data flow.
    for entity in list(nodes.values()):
        source=entity.get('source_file')
        if entity.get('_origin')=='ast' and source:
            p=Path(source);target=existing(p if p.is_absolute() else ROOT/p)
            if target:edge(entity['id'],target,'located_in_source',source,entity.get('source_location') or 'AST source provenance')
    imports=defaultdict(set); calls=defaultdict(list); io=[]; units=[]; issues=[]; modules=[]; test_count=0
    notebooks=sorted(p.relative_to(ROOT).as_posix() for p in project_files if p.suffix=='.ipynb')
    for p in project_files:
        rel=p.relative_to(ROOT).as_posix()
        if p.suffix=='.py' and rel in selected: units.append((rel,None,p.read_text()))
        elif p.suffix=='.ipynb':
            for i,c in enumerate(json.loads(p.read_text())['cells'],1):
                if c['cell_type']=='code': units.append((rel,i,clean_cell(''.join(c['source']))))
    def path_expr(x):
        if isinstance(x,ast.Name) and x.id in {'ROOT','root'}: return ''
        if isinstance(x,ast.Constant) and isinstance(x.value,str): return x.value
        if isinstance(x,ast.BinOp) and isinstance(x.op,ast.Div):
            a,b=path_expr(x.left),path_expr(x.right)
            if a is not None and b is not None: return str(Path(a)/b)
        return None
    for rel,cell,text in units:
        try: tree=ast.parse(text)
        except SyntaxError as exc:
            issues.append(f'Unparsed code: {rel}, cell {cell}, line {exc.lineno}: {exc.msg}'); continue
        sid=node(rel); imported={}; definitions=[]
        if not cell:
            for x in tree.body:
                if isinstance(x,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef)):
                    definitions.append({'name':x.name,'kind':'class' if isinstance(x,ast.ClassDef) else 'function','line':x.lineno,'signature':x.name if isinstance(x,ast.ClassDef) else x.name+'('+ast.unparse(x.args)+')','docstring':ast.get_docstring(x) or ''})
            modules.append({'path':rel,'docstring':ast.get_docstring(tree) or '', 'definitions':definitions})
            if rel.startswith('tests/'): test_count+=sum(isinstance(x,(ast.FunctionDef,ast.AsyncFunctionDef)) and x.name.startswith('test_') for x in ast.walk(tree))
        for x in ast.walk(tree):
            location=(f'cell {cell}, ' if cell else '')+f'L{getattr(x,"lineno",1)}'
            if isinstance(x,ast.ImportFrom):
                module=x.module or ''
                if x.level:
                    base=list(Path(rel).parent.parts)
                    if x.level>1: base=base[:-(x.level-1)]
                    module='.'.join(base+([module] if module else []))
                candidate=module.replace('.','/')+'.py'
                if (ROOT/candidate).is_file():
                    edge(sid,node(candidate),'imports',rel,location)
                    if cell: imports[rel].add(candidate)
                    for alias in x.names:
                        symbol=ident(candidate)+'_'+re.sub(r'[^a-z0-9]+','_',alias.name.lower())
                        if symbol in nodes:
                            imported[alias.asname or alias.name]=symbol
                            edge(sid,symbol,'imports_symbol',rel,location)
            elif isinstance(x,ast.Import):
                for alias in x.names:
                    candidate=alias.name.replace('.','/')+'.py'
                    if (ROOT/candidate).is_file():
                        edge(sid,node(candidate),'imports',rel,location)
                        if cell: imports[rel].add(candidate)
            if isinstance(x,ast.Call) and isinstance(x.func,ast.Name) and x.func.id in imported:
                edge(sid,imported[x.func.id],'calls',rel,location)
                if cell: calls[rel].append({'symbol':x.func.id,'cell':cell,'line':x.lineno})
            candidates=[(x,'references')] if isinstance(x,ast.BinOp) and isinstance(x.op,ast.Div) else []
            if isinstance(x,ast.Call):
                if isinstance(x.func,ast.Attribute):
                    attr=x.func.attr
                    if attr in {'read_text','read_bytes','write_text','write_bytes','to_csv','read_csv','read_json','read_parquet','to_parquet'}:
                        arg=x.args[0] if attr in {'to_csv','read_csv','read_json','read_parquet','to_parquet'} and x.args else x.func.value
                        candidates.append((arg,'writes' if attr in {'write_text','write_bytes','to_csv','to_parquet'} else 'reads'))
                elif isinstance(x.func,ast.Name) and x.args and x.func.id in {'write_json','fingerprint','read_features','read_labels','read_predictions'}:
                    candidates.append((x.args[0],{'write_json':'writes','fingerprint':'checks_hash'}.get(x.func.id,'reads')))
            for expression,relation in candidates:
                value=path_expr(expression)
                target=existing(ROOT/value) if value and not Path(value).is_absolute() else None
                if target:
                    edge(sid,target,relation,rel,location)
                    io.append({'source':rel,'target':value,'relation':relation,'location':location})
    for rel in sorted(selected):
        if not rel.endswith('.md'): continue
        for number,line in enumerate((ROOT/rel).read_text().splitlines(),1):
            for value in re.findall(r'\]\(([^)]+)\)',line):
                if '://' in value or value.startswith('#'): continue
                target=existing((ROOT/rel).parent/unquote(value.split('#')[0].strip('<>')))
                if target: edge(node(rel),target,'documents',rel,f'L{number}: explicit Markdown link')
    ordered=[f'notebooks/{stem}.ipynb' for stem in ORDER]
    for a,b in zip(ordered,ordered[1:]):
        assert (ROOT/a).exists() and (ROOT/b).exists()
        edge(node(a),node(b),'documented_investigation_order','readme.md','Notebooks table; original six-notebook investigation')
    unresolved=[]
    for e in edges:
        for endpoint in (e['source'],e['target']):
            if endpoint not in nodes:
                nodes[endpoint]={'id':endpoint,'label':endpoint.replace('_','.'),'file_type':'concept','source_file':e.get('source_file'),'source_location':e.get('source_location'),'_origin':'architecture','architecture_kind':'unresolved_reference','rationale':'Syntactic AST reference; implementation was not resolved or scanned.'}
                unresolved.append(endpoint)
    extraction.update(nodes=list(nodes.values()),edges=edges)
    (OUT/'.graphify_architecture.json').write_text(json.dumps(extraction,indent=2)+'\n')
    sizes={x['path']:x['bytes'] for x in inventory}
    inv={'files':inventory,'notebooks':notebooks,'notebook_imports':{k:sorted(v) for k,v in imports.items()},'notebook_calls':dict(calls),'io':io,'module_summary':modules,'grouped_artifacts':[{'path':k,'file_count':len(v),'bytes':sum(sizes[x] for x in v)} for k,v in sorted(groups.items())],'unresolved_ast_references':unresolved,'unlocated_ast_references':unlocated,'test_count':test_count,'scope':'Current source AST relations, notebook code cells, existing literal paths, explicit document links and non-environment file metadata. Repeated generated shards and bundled infrastructure are grouped; raw data contents, chart pixels and model binaries are not interpreted. This is not a runtime trace.','issues':issues}
    (OUT/'architecture_inventory.json').write_text(json.dumps(inv,indent=2)+'\n')
    print(f'Architecture: {len(inventory)} inventoried files; {len(notebooks)} notebooks; {len(groups)} artifact groups; {len(nodes)} nodes; {len(edges)} observations')
    print(f'{len(units)} code units; {len(issues)} parse issues; {len(unresolved)} unresolved references explicitly represented')

if __name__=='__main__': main()
