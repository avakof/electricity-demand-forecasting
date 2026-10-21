"""Refresh one presentation cell from archived results; never refit models."""
from datetime import datetime, timezone
from pathlib import Path
import nbformat
from nbclient import NotebookClient
from nbconvert import HTMLExporter
from jupyter_client import KernelManager
from jupyter_client.kernelspec import KernelSpecManager

ROOT=Path(__file__).resolve().parents[1]
path=ROOT/'notebooks/08_deep_learning_and_uncertainty.ipynb'
notebook=nbformat.read(path,as_version=4)
position=[i for i,c in enumerate(notebook.cells) if c.metadata.get('rte_assisted_extension') and c.cell_type=='code'][-1]
setup='''from src.rte_assisted import RTEAssistedRun
from scripts.rte_assisted_reporting import draw_figures
from IPython.display import Image as AssistedImage
assisted_run=RTEAssistedRun.latest(ROOT)
assisted_run.verify_manifest()
assisted_freeze=assisted_run.freeze_assisted()
assisted_results={key:pd.read_csv(assisted_run.path/'tables'/name) for key,name in {
 'point_scores':'test_scorecard.csv','uncertainty':'uncertainty_scorecard.csv',
 'cold_scores':'cold_evening_scorecard.csv','bootstrap':'bootstrap.csv'}.items()}
draw_figures(assisted_run,assisted_results['point_scores'],assisted_results['uncertainty'],assisted_results['cold_scores'])
'''
temporary=nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell(notebook.cells[1].source),
    nbformat.v4.new_code_cell(setup),notebook.cells[position].copy()])
manager=KernelManager(kernel_name='project-research',kernel_spec_manager=KernelSpecManager(kernel_dirs=[str(ROOT/'.runtime/kernels')]))
try:
    NotebookClient(temporary,km=manager,timeout=600,resources={'metadata':{'path':str(ROOT)}}).execute()
finally:
    if manager.has_kernel: manager.shutdown_kernel(now=True)
notebook.cells[position]=temporary.cells[-1]
notebook.metadata['rte_assisted_extension']['presentation_refresh']={
    'executed_utc':datetime.now(timezone.utc).isoformat(),'cell_index':position,
    'scope':'Replot saved scores with shorter nonoverlapping labels; execute final presentation cell; predictive artifacts unchanged'}
nbformat.validate(notebook)
nbformat.write(notebook,path)
html,_=HTMLExporter(template_name='lab').from_notebook_node(notebook)
(ROOT/'reports/notebook_previews'/f'{path.stem}.html').write_text(html)
print('PASSED RTE assisted presentation refresh',flush=True)
