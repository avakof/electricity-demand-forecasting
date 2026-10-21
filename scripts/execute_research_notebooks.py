"""Execute the two research notebooks with the current Python, saving diagnostics."""
import argparse
import json
from pathlib import Path
import sys

import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager
from jupyter_client.kernelspec import KernelSpecManager

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--from-notebook", type=int, default=7)
    parser.add_argument("--to-notebook", type=int, default=8)
    args = parser.parse_args()
    runtime = ROOT/".runtime/kernels"
    spec = runtime/"project-research"
    spec.mkdir(parents=True, exist_ok=True)
    (spec/"kernel.json").write_text(json.dumps({"argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
                                              "display_name": "Project research", "language": "python"}))
    for path in sorted((ROOT/"notebooks").glob("0[78]_*.ipynb")):
        if not args.from_notebook <= int(path.name[:2]) <= args.to_notebook:
            continue
        print(f"EXECUTING {path.name}", flush=True)
        notebook = nbformat.read(path, as_version=4)
        manager = KernelManager(kernel_name="project-research", kernel_spec_manager=KernelSpecManager(kernel_dirs=[str(runtime)]))
        def announce(cell, cell_index, **kwargs):
            if cell.cell_type == "code":
                print(f"CELL {cell_index} {cell.source.splitlines()[0]}", flush=True)
        def checkpoint(cell, cell_index, **kwargs):
            nbformat.write(notebook, path)
        client = NotebookClient(notebook, km=manager, timeout=14400,
                                resources={"metadata": {"path": str(ROOT)}},
                                on_cell_start=announce, on_cell_executed=checkpoint)
        try:
            client.execute()
        finally:
            nbformat.write(notebook, path)
            if manager.has_kernel:
                manager.shutdown_kernel(now=True)
        print(f"PASSED {path.name}", flush=True)


if __name__ == "__main__":
    main()
