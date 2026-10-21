"""Read-only architecture extraction; all generated files stay in graphify-out."""
import ast
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "graphify-out"
SKIP = {".git", ".runtime", ".impeccable", ".venv", "venv", "__pycache__", ".ipynb_checkpoints", "graphify-out", ".pytest_cache"}


def ident(path):
    return re.sub(r"[^a-z0-9]+", "_", str(Path(path).with_suffix("")).lower()).strip("_")


def files_in_project():
    return sorted(p for p in ROOT.rglob("*") if p.is_file() and not any(x in SKIP for x in p.relative_to(ROOT).parts) and p.name != ".DS_Store")


def main():
    extraction = json.loads((OUT / ".graphify_ast.json").read_text())
    nodes = {n["id"]: n for n in extraction["nodes"]}
    edges = extraction["edges"]
    project_files = files_in_project()
    inventory = []
    hashes = {}
    for p in project_files:
        rel = p.relative_to(ROOT).as_posix()
        hashes[rel] = hashlib.sha256(p.read_bytes()).hexdigest()
        inventory.append({"path": rel, "bytes": p.stat().st_size, "id": ident(rel)})
    (OUT / "source_hashes.json").write_text(json.dumps(hashes, indent=2))

    def add_node(path, *, directory=False):
        key = ident(path)
        if key not in nodes:
            nodes[key] = {"id": key, "label": str(path), "file_type": "code" if str(path).endswith((".py", ".ipynb", ".json")) else "document", "source_file": str(path), "source_location": "directory inventory" if directory else "file inventory", "_origin": "architecture", "architecture_kind": "directory" if directory else "file"}
        nodes[key].setdefault("architecture_kind", "directory" if directory else "file")
        return key

    def edge(a, b, relation, source, location, **kw):
        if a == b:
            return
        edges.append({"source": a, "target": b, "relation": relation, "confidence": "EXTRACTED", "confidence_score": 1.0, "source_file": source, "source_location": location, "weight": 1.0, "_origin": "architecture", **kw})

    # Inventory includes generated artifact names, without interpreting model binaries or chart pixels.
    for item in inventory:
        rel = item["path"]
        child = add_node(rel)
        for parent in Path(rel).parents:
            if parent.as_posix() == ".":
                break
            parent_id = add_node(parent.as_posix(), directory=True)
            edge(parent_id, child, "contains", rel, "filesystem inventory")
            child = parent_id

    notebook_imports = defaultdict(set)
    notebook_calls = defaultdict(list)
    io = []
    source_units = []
    for p in project_files:
        rel = p.relative_to(ROOT).as_posix()
        if p.suffix == ".py":
            # The authoring script embeds entire notebooks as strings: imports remain structural,
            # but embedded notebook text is not misrepresented as executed Python.
            source_units.append((rel, None, p.read_text()))
        elif p.suffix == ".ipynb":
            nb = json.loads(p.read_text())
            for number, cell in enumerate(nb["cells"], 1):
                if cell["cell_type"] == "code":
                    source_units.append((rel, number, "".join(cell["source"])))

    def path_expr(x):
        if isinstance(x, ast.Name) and x.id in {"ROOT", "root"}:
            return ""
        if isinstance(x, ast.Constant) and isinstance(x.value, str):
            return x.value
        if isinstance(x, ast.BinOp) and isinstance(x.op, ast.Div):
            left, right = path_expr(x.left), path_expr(x.right)
            if left is not None and right is not None:
                return str(Path(left) / right)
        return None

    for rel, cell, text in source_units:
        if cell:
            # Preserve line numbers while removing IPython line magics/shell escapes.
            text = "\n".join("pass" if line.lstrip().startswith(("%", "!")) else line for line in text.splitlines())
        try:
            tree = ast.parse(text)
        except SyntaxError:
            continue
        source_id = add_node(rel)
        imported = {}
        for x in ast.walk(tree):
            location = (f"cell {cell}, " if cell else "") + f"L{x.lineno}" if hasattr(x, "lineno") else None
            if isinstance(x, ast.ImportFrom):
                module = x.module or ""
                if x.level:
                    base = list(Path(rel).parent.parts)
                    if x.level > 1:
                        base = base[: -(x.level - 1)]
                    module = ".".join(base + ([module] if module else []))
                candidate = module.replace(".", "/") + ".py"
                if (ROOT / candidate).is_file():
                    target = add_node(candidate)
                    edge(source_id, target, "imports", rel, location)
                    if cell:
                        notebook_imports[rel].add(candidate)
                    for a in x.names:
                        imported[a.asname or a.name] = ident(candidate) + "_" + re.sub(r"[^a-z0-9]+", "_", a.name.lower())
                        symbol_id = imported[a.asname or a.name]
                        if symbol_id in nodes:
                            edge(source_id, symbol_id, "imports_symbol", rel, location)
            if isinstance(x, ast.Call) and isinstance(x.func, ast.Name) and x.func.id in imported and imported[x.func.id] in nodes:
                edge(source_id, imported[x.func.id], "calls", rel, location)
                if cell:
                    notebook_calls[rel].append({"symbol": x.func.id, "cell": cell, "line": x.lineno})

            # Extract only rooted literal paths whose files/directories actually exist.
            candidates = []
            if isinstance(x, ast.BinOp) and isinstance(x.op, ast.Div):
                candidates.append((x, "references"))
            if isinstance(x, ast.Call):
                if isinstance(x.func, ast.Attribute):
                    attr = x.func.attr
                    if attr in {"read_text", "read_bytes", "write_text", "write_bytes", "to_csv"}:
                        target_expr = x.args[0] if attr == "to_csv" and x.args else x.func.value
                        candidates.append((target_expr, "writes" if attr in {"write_text", "write_bytes", "to_csv"} else "reads"))
                    elif attr in {"read_csv", "read_json"} and x.args:
                        candidates.append((x.args[0], "reads"))
                elif isinstance(x.func, ast.Name) and x.args and x.func.id in {"write_json", "fingerprint", "read_features", "read_labels", "read_predictions"}:
                    candidates.append((x.args[0], {"write_json": "writes", "fingerprint": "checks_hash"}.get(x.func.id, "reads")))
            for expr, relation in candidates:
                path = path_expr(expr)
                if path and not Path(path).is_absolute() and (ROOT / path).exists():
                    target = add_node(path, directory=(ROOT / path).is_dir())
                    edge(source_id, target, relation, rel, location)
                    io.append({"source": rel, "target": path, "relation": relation, "location": location})

    # Documented execution order is an orchestration relationship, not a hidden import.
    nbs = sorted(p.relative_to(ROOT).as_posix() for p in project_files if p.suffix == ".ipynb" and p.name[:2] in {"01", "02", "03", "04", "05", "06"})
    for a, b in zip(nbs, nbs[1:]):
        edge(ident(a), ident(b), "precedes_in_documented_workflow", "readme.md", "Read and run the six notebooks in order")
    for nb in nbs:
        edge(ident("scripts/execute_notebooks.py"), ident(nb), "executes", "scripts/execute_notebooks.py", "L28-L39")
        edge(ident("scripts/build_notebooks.py"), ident(nb), "authors", "readme.md", "scripts/build_notebooks.py is the authoring utility")
        edge(ident("scripts/validate_artifacts.py"), ident(nb), "validates", "scripts/validate_artifacts.py", "L25-L37")

    # AST leaves imported external modules as endpoints; materialize those references explicitly.
    added_external = []
    for e in list(edges):
        e.setdefault("confidence_score", 1.0 if e.get("confidence") == "EXTRACTED" else .75)
        for endpoint in (e["source"], e["target"]):
            if endpoint not in nodes:
                nodes[endpoint] = {"id": endpoint, "label": endpoint.replace("_", "."), "file_type": "concept", "source_file": e.get("source_file"), "source_location": e.get("source_location"), "_origin": "architecture", "architecture_kind": "external_import", "rationale": "Imported module reference; external package implementation was not scanned."}
                added_external.append(endpoint)

    extraction.update(nodes=list(nodes.values()), edges=edges)
    (OUT / ".graphify_architecture.json").write_text(json.dumps(extraction, indent=2))
    inventory_data = {"files": inventory, "notebooks": nbs, "notebook_imports": {k: sorted(v) for k, v in notebook_imports.items()}, "notebook_calls": notebook_calls, "io": io, "external_modules_materialized": added_external, "scope": "Source-level imports/calls, rooted literal path references, notebook code cells and full file inventory. CSV contents, serialized models and generated plot pixels are not semantically interpreted.", "issues": []}
    if not (ROOT / "notebooks/legacy/01_data_audit_original.ipynb").exists():
        inventory_data["issues"].append("readme.md references notebooks/legacy/01_data_audit_original.ipynb, but that file is absent in the current filesystem.")
    (OUT / "architecture_inventory.json").write_text(json.dumps(inventory_data, indent=2))
    print(f"Architecture: {len(inventory)} files; {len(nbs)} notebooks; {len(nodes)} nodes; {len(edges)} relationship observations; {len(added_external)} explicit external import nodes")
    print("Notebook imports:", dict(notebook_imports))
    print("Documentation issues:", inventory_data["issues"])


if __name__ == "__main__":
    main()
