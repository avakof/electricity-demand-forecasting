"""Rebuild the source map using verified semantic cache; never run forecast models.

Changed documents require the graphify skill's host-agent extraction before this
helper proceeds. Code-only updates can then rebuild all custom outputs together.
"""
import argparse
import base64
import hashlib
from datetime import datetime, timezone
import json
import re
import runpy
import shutil
import subprocess
import sys
from pathlib import Path
from graphify.cache import check_semantic_cache
from graphify.detect import detect
from graphify.extract import extract
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'graphify-out'
SPEC=Path('/Users/alivakof/.codex/skills/graphify/references/extraction-spec.md')

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-cache',action='store_true')
    parser.add_argument('--plot-python',help='Python with matplotlib for static exports')
    args=parser.parse_args();config=json.loads((OUT/'.graphify_build.json').read_text())
    detection=detect(ROOT,extra_excludes=config['excludes'],gitignore=config.get('gitignore',True))
    if detection['total_files']>500 or detection['total_words']>2_000_000:
        raise SystemExit('Scope exceeds graphify limits; review exclusions through the graphify skill before rebuilding.')
    files=[f for kind in ('document','paper','image') for f in detection['files'].get(kind,[])]
    nodes,edges,hyperedges,uncached=check_semantic_cache(files,root=ROOT,prompt_file=SPEC)
    if uncached:
        print('Changed documents need host-agent semantic extraction. Existing graph was preserved.')
        for name in uncached:print(name)
        raise SystemExit(2)
    ids={n['id'] for n in nodes}
    if any(e['source'] not in ids or e['target'] not in ids for e in edges):
        raise SystemExit('Cached cross-document endpoints are stale. Re-extract affected documents before replacing the graph.')
    print(f'Cache verified: {len(files)} documents, {len(nodes)} semantic nodes; {detection["total_files"]} scoped files.')
    if args.check_cache:return
    plot_file=OUT/'.graphify_plot_python'
    plot_python=args.plot_python or (plot_file.read_text().strip() if plot_file.exists() else sys.executable)
    subprocess.run([plot_python,'-c','import matplotlib'],check=True)
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    backup=OUT/'history'/stamp;backup.mkdir(parents=True)
    for name in ['graph.json','graph.html','GRAPH_REPORT.md','architecture_evidence.json']:
        p=OUT/name
        if p.exists():shutil.copy2(p,backup/name)
    (OUT/'.graphify_detect.json').write_text(json.dumps(detection,ensure_ascii=False))
    ast_result=extract([Path(f) for f in detection['files'].get('code',[])],root=ROOT,cache_root=ROOT,parallel=False)
    (OUT/'.graphify_ast.json').write_text(json.dumps(ast_result,ensure_ascii=False))
    semantic={'nodes':nodes,'edges':edges,'hyperedges':hyperedges,'input_tokens':0,'output_tokens':0,'token_usage_status':'Cached host extraction; original usage unavailable.'}
    (OUT/'.graphify_semantic.json').write_text(json.dumps(semantic,ensure_ascii=False))
    from build_architecture import main as build
    from finalize_graph import main as finalize
    from label_graph import main as label
    from generate_code_docs import main as docs
    from verify_and_finish import main as verify
    build();finalize();label();docs()
    runpy.run_path(str(OUT/'render_overview.py'),run_name='__main__')
    subprocess.run([plot_python,str(OUT/'render_static.py')],cwd=ROOT,check=True)
    subprocess.run([str(Path(sys.executable).with_name('graphify')),'export','html'],cwd=ROOT,check=True)
    html=OUT/'graph.html';text=html.read_text();match=re.search(r'integrity="(sha384-[^"]+)"',text)
    asset=OUT/'vendor/vis-network.min.js'
    if match and asset.exists():
        actual='sha384-'+base64.b64encode(hashlib.sha384(asset.read_bytes()).digest()).decode()
        if actual!=match.group(1):raise SystemExit('Local graph runtime integrity mismatch; do not use the HTML until resolved.')
        html.write_text(text.replace('https://unpkg.com/vis-network@9.1.6/standalone/umd/vis-network.min.js','vendor/vis-network.min.js'))
    verify()
    print('Graph, code documentation and overview regenerated. Inspect changed visuals before relying on the presentation.')

if __name__=='__main__':main()
