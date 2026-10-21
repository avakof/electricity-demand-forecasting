# Graph Report - Project  (2026-10-20)

## Corpus Check
- Corpus is ~10,846 words - fits in a single context window. You may not need a graph.

## Summary
- 379 nodes · 877 edges · 11 communities (6 shown, 5 thin omitted)
- Extraction: 98% EXTRACTED · 2% INFERRED · 0% AMBIGUOUS · INFERRED: 18 edges (avg confidence: 0.93)
- Semantic token cost: **unreported by host**. Zero values in machine metadata are placeholders; actual cost is unknown.

## Community Hubs (Navigation)
- Source Audit and Tooling
- Research Contract and Evidence
- Evaluation and Uncertainty
- Availability and Integrity Tests
- Validation and Model Selection
- Origin Feature Store

## God Nodes (most connected - your core abstractions)
1. `pandas` - 45 edges
2. `ForecastIntegrityTests` - 21 edges
3. `local_day_grid()` - 18 edges
4. `build_origin_features()` - 18 edges
5. `numpy` - 18 edges
6. `fingerprint()` - 15 edges
7. `model_replay()` - 14 edges
8. `load_contract()` - 14 edges
9. `write_json()` - 14 edges
10. `calendar_features()` - 12 edges

## Surprising Connections (you probably didn't know these)
- `Contract and source identity lock` --semantically_similar_to--> `Content-addressed immutable aggregate forecasts`  [INFERRED] [semantically similar]
  readme.md → reports/implementation_review.md
- `Forecast-origin available-information replay` --semantically_similar_to--> `Freshness, snapshot, forecast, fallback, archive and monitor pipeline`  [INFERRED] [semantically similar]
  readme.md → reports/final_recommendation.md
- `RTE-assisted additive correction candidate` --implements--> `Validation-selected frozen candidate`  [INFERRED]
  reports/final_recommendation.md → readme.md
- `Unverified source-to-hourly conversion` --rationale_for--> `Authentic-archive and interval-lineage gate`  [INFERRED]
  reports/audit/limitations.md → readme.md
- `main()` --calls--> `local_day_grid()`  [EXTRACTED]
  scripts/validate_artifacts.py → src/common.py

## Import Cycles

No cycles found in actual internal Python import statements.

## Hyperedges (group relationships)
- **Historical information availability controls** — readme_forecast_contract, readme_available_information_replay, readme_authentic_archives, readme_operational_gate, readme_assumed_observation_availability [EXTRACTED 1.00]
- **Six-stage research workflow** — readme_notebook_01, readme_notebook_02, readme_notebook_03, readme_notebook_04, readme_notebook_05, readme_notebook_06 [EXTRACTED 1.00]
- **Auditable prediction evidence trail** — readme_evaluation_lock, readme_forecast_archiving, reports_implementation_review_append_only_origin_ledger, reports_implementation_review_content_addressed_forecasts, reports_implementation_review_falsification_tests [INFERRED 0.95]

## Communities (11 total, 5 thin omitted)

### Community 0 - "Source Audit and Tooling"
Cohesion: 0.08
Nodes (28): argparse, base64, hashlib, json, jupyter.client, jupyter.client.kernelspec, nbclient, nbconvert (+20 more)

### Community 1 - "Research Contract and Evidence"
Cohesion: 0.08
Nodes (38): Demand, RTE and weather vintage archives, Forecast-origin available-information replay, MAE and conditional-median point objective, Research configuration, Complete residual-day trajectory scenarios, 23/24/25-hour local delivery grid, Contract and source identity lock, Predictions archived before evaluation joins (+30 more)

### Community 2 - "Evaluation and Uncertainty"
Cohesion: 0.11
Nodes (23): numpy, pandas, baseline_replay(), calendar_features(), development_frame(), hypothesis_register(), temperature_summary(), daily_metrics() (+15 more)

### Community 4 - "Availability and Integrity Tests"
Cohesion: 0.10
Nodes (16): pandas.testing, Experiment, feature_columns(), fit_at_origin(), forecast_origin(), holiday_dates(), local_day_grid(), load_archive() (+8 more)

### Community 6 - "Validation and Model Selection"
Cohesion: 0.08
Nodes (14): dataclasses, gzip, joblib, shutil, sklearn.ensemble, sklearn.impute, sklearn.pipeline, archive_predictions() (+6 more)

### Community 9 - "Origin Feature Store"
Cohesion: 0.29
Nodes (4): contract_sha256, origins, regime, rows

## Knowledge Gaps
- **11 isolated node(s):** `nbclient`, `jupyter.client`, `jupyter.client.kernelspec`, `nbconvert`, `dataclasses` (+6 more)
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 154 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **5 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `pandas` connect `Evaluation and Uncertainty` to `Source Audit and Tooling`, `Availability and Integrity Tests`, `Validation and Model Selection`?**
  _High betweenness centrality (0.070) - this node is a cross-community bridge._
- **What connects `nbclient`, `jupyter.client`, `jupyter.client.kernelspec` to the rest of the system?**
  _11 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Source Audit and Tooling` be split into smaller, more focused modules?**
  _Cohesion score 0.07638279192273925 - nodes in this community are weakly interconnected._
- **Why does `04 Model comparison` connect `Research Contract and Evidence` to `Validation and Model Selection`?**
  _High betweenness centrality (0.031) - this node is a cross-community bridge._
- **Should `Research Contract and Evidence` be split into smaller, more focused modules?**
  _Cohesion score 0.08081632653061224 - nodes in this community are weakly interconnected._
- **Why does `03 Backtest and baselines` connect `Research Contract and Evidence` to `Source Audit and Tooling`?**
  _High betweenness centrality (0.026) - this node is a cross-community bridge._
- **Should `Evaluation and Uncertainty` be split into smaller, more focused modules?**
  _Cohesion score 0.11212121212121212 - nodes in this community are weakly interconnected._

## Architecture extraction scope and verification

- Start with [architecture.html](architecture.html) for the readable stage map; use [graph.html](graph.html) for the full Graphify network.
- 199 actual project files inventoried; six notebooks parsed by code cell, with original cell/line evidence. Runtime environments, Git internals and graph outputs are excluded.
- Graphify detection supports 58 source/configuration/document files. Notebook and CSV coverage is supplied by the additional deterministic architecture extraction. CSV contents, serialized models and generated plot pixels were not interpreted. Plot PNG and duplicate HTML preview paths remain in the inventory.
- Relationships are source-level imports/calls, literal path reads/writes/checks and explicitly documented notebook order. This is not a runtime trace. Dynamic paths, arbitrary reflection and external package implementations may be absent.
- Default Graphify graph is undirected. Original direction, relationship type and each source location remain in every edge’s `relationship_observations` array. The overview uses labeled directional links.
- The initial AST left 25 imported external modules as dangling identifiers. Those imports are now explicit external-reference nodes. Repeated endpoint observations were grouped, retaining every original observation in the evidence array; see normalization_audit.json. Final graph health has no missing/dangling endpoints, self-loops or unintended edge collapse.
- Semantic extraction: 44 nodes, 55 edges, three hyperedges; inferred connections remain labeled INFERRED. Host-agent token usage was not exposed. Any zero token counts in generated sections or cost.json are placeholders, not measured usage; semantic monetary cost is unknown.
- Current operational gate is closed: the archive files contain headers only and vintage/interval provenance flags are false. pipeline.py provides readiness checks/specifications; it is not an implemented ingestion/scheduling/publication/procurement service.
- Documentation issue: readme.md references notebooks/legacy/01_data_audit_original.ipynb, but that file is absent in the current filesystem.
- graphify-out/ was written into the working tree. It may appear in git status unless ignored; no repository ignore rules were changed.
- Rebuild architecture augmentation after source changes; a plain Graphify incremental update does not reparse notebook cells or reconstruct the custom inventory edges. Existing project source, data and notebook files were not edited.

## Final verification

- SHA-256 verified unchanged for all 199 existing project files. Outputs are in graphify-out/; isolated package setup is under ignored .runtime/. The automatic design hook also created .impeccable/hook.cache.json; it is tooling state, excluded from the research-file inventory.
- All six notebooks and 37 code cells parsed; all file-inventory nodes and graph endpoints verified.
- Import-cycle claims were checked against real Python ImportFrom statements. The original automatic report incorrectly treated local calls as one-file import cycles; the corrected section reports actual module cycles.
- Community cohesion describes graph connectivity, including inventory edges, not software modularity. Automatic suggested questions about splitting communities are navigation prompts, not refactoring recommendations. Low-degree external imports and generated files are expected leaves.
- The overview was checked by offline DOM simulation: mode switching, selection, keyboard, reset, file search and zero script errors. Matplotlib-generated architecture.png was visually inspected. Browser preview was blocked by local-file URL policy and remains unverified.
- Readability review: small overview text enlarged. Later hook triage confirmed all-caps-body was a false positive on a short section heading; this one rule is suppressed only for graphify-out/overview_template.html via .impeccable/config.json. Body paragraphs remain sentence case; no UI code changed for that triage.
- Graph HTML runtime dependency saved under vendor/ and verified against the original SHA-384 integrity digest. Both HTML views work from local assets; no third-party runtime requests are required.
