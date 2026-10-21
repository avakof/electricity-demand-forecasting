"""Run the six notebooks in order with this interpreter; no global kernel install."""
import argparse
import json
import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager
from jupyter_client.kernelspec import KernelSpecManager

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--from-notebook",type=int,default=1)
    parser.add_argument("--to-notebook",type=int,default=6)
    args=parser.parse_args()
    runtime=ROOT/".runtime/kernels"
    spec=runtime/"project-forecast"
    spec.mkdir(parents=True,exist_ok=True)
    (spec/"kernel.json").write_text(json.dumps({"argv":[sys.executable,"-m","ipykernel_launcher","-f","{connection_file}"],"display_name":"Project forecast","language":"python"}))
    validated=[]
    for path in sorted((ROOT/"notebooks").glob("0[1-6]_*.ipynb")):
        number=int(path.name[:2])
        if not args.from_notebook<=number<=args.to_notebook: continue
        print(f"EXECUTING {path.name}",flush=True)
        nb=nbformat.read(path,as_version=4)
        nbformat.validate(nb)
        km=KernelManager(kernel_name="project-forecast",kernel_spec_manager=KernelSpecManager(kernel_dirs=[str(runtime)]))
        def announce(cell,cell_index,**kwargs):
            if cell.cell_type=="code": print(f"  {path.name} cell {cell_index}",flush=True)
        client=NotebookClient(nb,km=km,timeout=2400,resources={"metadata":{"path":str(ROOT)}},on_cell_start=announce)
        try:
            client.execute()
        finally:
            # Save diagnostic outputs if execution fails, then propagate failure.
            nbformat.write(nb,path)
            if km.has_kernel:
                km.shutdown_kernel(now=True)
        validated.append(path.name)
        print(f"PASSED {path.name}",flush=True)
    print(json.dumps({"executed_notebooks":validated}),flush=True)


if __name__=="__main__": main()
