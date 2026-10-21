"""Merge verified extraction, preserve relationship evidence, build Graphify outputs."""
import json
from collections import Counter
from pathlib import Path

from graphify.analyze import god_nodes, suggest_questions, surprising_connections
from graphify.build import build_from_json
from graphify.cache import save_semantic_cache
from graphify.cluster import cluster, score_all
from graphify.diagnostics import diagnose_extraction, format_diagnostic_report
from graphify.export import to_json
from graphify.report import generate

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "graphify-out"


def main():
    chunk = json.loads((OUT / ".graphify_chunk_01.json").read_text())
    assert isinstance(chunk["nodes"], list) and isinstance(chunk["edges"], list)
    allowed = (OUT / ".graphify_uncached.txt").read_text().splitlines()
    save_semantic_cache(chunk["nodes"], chunk["edges"], chunk.get("hyperedges", []), root=ROOT, allowed_source_files=allowed, prompt_file="/Users/alivakof/.codex/skills/graphify/references/extraction-spec.md")
    (OUT / ".graphify_semantic_new.json").write_text(json.dumps(chunk))
    cached = json.loads((OUT / ".graphify_cached.json").read_text()) if (OUT / ".graphify_cached.json").exists() else {"nodes": [], "edges": [], "hyperedges": []}
    sem = {"nodes": list({n["id"]: n for n in cached["nodes"] + chunk["nodes"]}.values()), "edges": cached["edges"] + chunk["edges"], "hyperedges": cached.get("hyperedges", []) + chunk.get("hyperedges", []), "input_tokens": chunk.get("input_tokens", 0), "output_tokens": chunk.get("output_tokens", 0), "token_usage_status": "Host-agent usage was not exposed by collaboration tool; zeros are placeholders, not measured free usage."}
    for n in sem["nodes"]:
        n["_origin"] = "semantic"
    for e in sem["edges"]:
        e["_origin"] = "semantic"
    (OUT / ".graphify_semantic.json").write_text(json.dumps(sem))
    structural = json.loads((OUT / ".graphify_architecture.json").read_text())
    nodes = {n["id"]: n for n in structural["nodes"]}
    for n in sem["nodes"]:
        nodes.setdefault(n["id"], n)
    edges = structural["edges"] + sem["edges"]
    # Anchor semantic concepts to the document that actually contains their evidence.
    from build_architecture import ident
    for n in sem["nodes"]:
        rel = str(Path(n["source_file"]).relative_to(ROOT))
        if ident(rel) in nodes and ident(rel) != n["id"]:
            edges.append({"source": n["id"], "target": ident(rel), "relation": "documented_in", "confidence": "EXTRACTED", "confidence_score": 1.0, "source_file": rel, "source_location": "document concept extraction", "weight": 1.0, "_origin": "semantic"})
    for i, nb in enumerate(sorted((ROOT / "notebooks").glob("0[1-6]_*.ipynb")), 1):
        edges.append({"source": f"readme_notebook_0{i}", "target": ident(nb.relative_to(ROOT)), "relation": "describes", "confidence": "EXTRACTED", "confidence_score": 1.0, "source_file": "readme.md", "source_location": "notebook main outputs table", "weight": 1.0, "_origin": "semantic"})

    # Graphify's default Graph is simple and undirected. Preserve every relationship
    # observation rather than letting NetworkX silently overwrite parallel edges.
    grouped = {}
    for e in edges:
        assert e["source"] in nodes and e["target"] in nodes, e
        if e["source"] == e["target"]:
            raise ValueError("Unexpected self-loop; review before building")
        key = tuple(sorted((e["source"], e["target"])))
        grouped.setdefault(key, []).append(e)
    normalized = []
    for key, observations in grouped.items():
        e = dict(observations[0])
        e["relation"] = " / ".join(sorted({x["relation"] for x in observations}))
        e["relationship_observations"] = observations
        e["observation_count"] = len(observations)
        e["confidence_score"] = min(x.get("confidence_score", 1.0) for x in observations)
        e["confidence"] = "EXTRACTED" if all(x.get("confidence") == "EXTRACTED" for x in observations) else "INFERRED"
        normalized.append(e)
    extraction = {"nodes": list(nodes.values()), "edges": normalized, "hyperedges": sem["hyperedges"], "input_tokens": sem["input_tokens"], "output_tokens": sem["output_tokens"], "token_usage_status": sem["token_usage_status"]}
    (OUT / ".graphify_extract.json").write_text(json.dumps(extraction, indent=2))
    # Keep the full evidence source for repeatable architecture rebuilds and edge audits.
    (OUT / "architecture_evidence.json").write_text(json.dumps(extraction, indent=2))
    normalization = {"input_relationship_observations": len(edges), "simple_graph_edges": len(normalized), "grouped_observations": len(edges)-len(normalized), "policy": "All directed source/target observations and evidence retained in relationship_observations; default Graphify visualization is undirected."}
    (OUT / "normalization_audit.json").write_text(json.dumps(normalization, indent=2))
    G = build_from_json(extraction, root=str(ROOT), directed=False)
    assert G.number_of_nodes() > 0
    communities = cluster(G)
    cohesion = score_all(G, communities)
    gods = god_nodes(G)
    surprises = surprising_connections(G, communities)
    labels = {k: "Community " + str(k) for k in communities}
    questions = suggest_questions(G, communities, labels)
    assert to_json(G, communities, str(OUT / "graph.json")), "Graph shrink guard rejected export"
    detection = json.loads((OUT / ".graphify_detect.json").read_text())
    report = generate(G, communities, cohesion, labels, gods, surprises, detection, {"input": extraction["input_tokens"], "output": extraction["output_tokens"]}, str(ROOT), suggested_questions=questions)
    (OUT / "GRAPH_REPORT.md").write_text(report)
    analysis = {"communities": {str(k): v for k, v in communities.items()}, "cohesion": {str(k): v for k, v in cohesion.items()}, "gods": gods, "surprises": surprises, "questions": questions}
    (OUT / ".graphify_analysis.json").write_text(json.dumps(analysis, indent=2))
    health = diagnose_extraction(extraction, directed=False, root=str(ROOT))
    (OUT / "graph_health.json").write_text(json.dumps(health, indent=2))
    print(format_diagnostic_report(health))
    print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges, {len(communities)} communities")
    for cid, members in communities.items():
        labels_sample = [nodes[k]["label"] for k in members if k in nodes]
        print(cid, len(members), labels_sample[:18])


if __name__ == "__main__":
    main()
