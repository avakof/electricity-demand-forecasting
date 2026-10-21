# Independent architecture refresh review

Status: **passed; no open required findings**. Reviewed 2026-10-21T04:29:32.524161+00:00.

This review inspected generated graph/documentation artifacts against current Python source, notebook cells, configurations, frozen selections and saved result evidence. It performed source parsing, graph/schema checks, source-location checks, file-link checks and hashing. It did not execute notebooks, train models, replay forecasts, change configurations or modify saved research outputs. Browser testing is separately recorded by the parent task; this review does not claim an independent browser session.

## Verified structure and provenance

| Check | Independent result |
| --- | --- |
| Navigation graph | 1,696 unique nodes; 4,702 edges; 83 communities; three hyperedges |
| Relationship evidence | 6,594 distinct directed observations retained within undirected navigation edges |
| Endpoint/schema integrity | Zero missing endpoints, self-loop navigation edges, invalid node IDs, invalid node types or missing hyperedge members |
| Graph/evidence agreement | Identical node IDs, undirected edge endpoint sets and directed observation contents; graph enrichment and ordering explain serialization differences |
| Current notebook coverage | All nine notebooks represented: eight main stages and one separate legacy baseline investigation |
| Notebook parsing | 74 code cells parse after position-preserving replacement of six IPython magic lines |
| Original stage ordering | Exactly audit → EDA → baselines → comparison → uncertainty → business interpretation; legacy 02 receives no false author/executor/order relationship |
| Source references | All 127 overview references resolve; their explicit line excerpts match current source. All 5,714 mechanically parseable graph source/cell line references are within valid bounds |
| Source paths | Every nonempty recorded node/relationship source path resolves |
| Documentation links | All 659 local links in code documentation and graph report resolve |
| Inspected source preservation | All 177 hashes in the final inspected-input manifest match current files. This is the recorded corpus, not a claim that every generated forecast shard was independently hashed here |
| Retired graph elements | 170 audited retirements: 120 grouped generated artifacts, 44 re-extracted concepts, five deleted-source nodes and one excluded tooling node; prior graph backup exists under `history/20261021T040620Z` |
| Cached semantic repair | The four unchanged documents with ten obsolete cross-references were re-extracted; the saved cache review records 32 semantic documents, 89 nodes and 84 edges |

The navigation graph retains no node sourced from deleted `src/pipeline.py` or `reports/monitoring_specification.csv`, and no old live-pipeline concept. Actual scikit-learn preprocessing pipelines remain correctly represented. The original six-stage workflow, legacy branch, three later versioned protocols and reporting-only manager presentation are distinguished.

## Scientific and architectural claim checks

- **Class relationships:** `RTEAssistedRun` inherits `ResearchRun`; `FeatureLabData` inherits `ResearchData`; `RiskLabRun` is a separate class. Graph inheritance edges agree with current AST definitions.
- **Target and feature views:** national, regional-only and combined inputs all predict national consumption. Regional-only excludes national observed demand/weather/error histories while retaining national RTE D1 and calendar. The earlier arm named regional remains national-plus-regional. Unweighted regional temperature aggregates require complete composition.
- **Availability:** RTE D1 is an assumed pre-cutoff input at 11:00 Europe/Paris. Unverified original forecast vintages, demand revisions and interval conversion remain explicit limitations. No next-but-one-day/05:00 rule is introduced. Cutoff tests do not certify authentic historical availability.
- **Feature construction:** historical lags and summaries are availability-gated. Monthly learned Ridge/classifier predictors use earlier released labels. Future realized weather is excluded from the expanded predictors; realized cold scoring and predicted-cold training weights have distinct roles.
- **Hybrids:** 23 archived base variants enter newly trained LightGBM residual combiners. The graph does not claim native RTE-input retraining of all 13 base families. The selected point is `lab__lightgbm__combined__stacked__w1__residual`; its frozen variant has `base: null` and neutral training weights. Stacked here identifies learned predictors, not an ensemble of all archived architectures.
- **Selection and schedules:** point selection uses cold-weighted development MAE and guardrails; separate quantile selection uses overall unweighted development WIS. Expanded point fits are quarterly, learned generators monthly, and reference policies retain their original schedules. The comparison concerns complete policies, not isolated architecture effects.
- **Uncertainty:** native quantiles, selected-point empirical residual intervals and fixed engineered split-conformal/CQR comparators remain distinct. CRPS, WIS, pinball and interval scores are metrics, not additional architectures. Fixed comparators retain forecast-cold training weights; conditional quantile/coverage guarantees are not asserted. Zero cold calibration observations and the previously viewed test remain visible.
- **Recommendation:** the saved point-error improvement is retrospective evidence. Additional cold improvement over the prior selected policy is uncertain, and cold-evening interval coverage is inadequate. Reporting layers do not imply procurement integration, euro savings, operational validation or automatic transfer to a real customer portfolio.

These interpretations match `README.md`, `config/rte_risk_experiment.json`, the frozen selection and current source, rather than relying on node proximity or community centrality. The report correctly explains that generic graph connectivity statistics are not evidence of feature importance or a software defect.

## Resolved finding

The first pass found 21 AST type/decorator/import placeholders with empty definition locations, beyond the 67 separately materialized unresolved references. The parent corrected them. All 21 are now `file_type: concept`, `architecture_kind: type_or_import_reference`, labeled `Reference: …`, with an explicit unscanned-definition rationale. Their incoming relationship observations retain valid caller/source evidence. `architecture_inventory.json` lists their IDs separately, and `GRAPH_REPORT.md` discloses the distinction. The empty `tests/__init__.py` file legitimately lacks a source line. No fictitious implementation location was invented.

## Limits of this review

This is a source and documentation map, not a runtime trace. Dynamic paths and method dependencies are not exhaustively resolved. The source-location checks establish valid locations and exact overview excerpts; they do not convert every AST inference into observed execution. Generated artifacts are grouped, external implementations are not scanned, and saved test outcomes remain historical evidence. No forecasting suite or model was rerun for this review.

## Reviewed artifact identities

The following hashes identify the final graph/documentation snapshot reviewed here. Verification JSON, browser proof and extraction-sidecar cleanup may be finalized separately; they are not included in this table.

| Artifact | SHA-256 |
| --- | --- |
| `graph.json` | `6367b540cabc53af2c307bdc2253748cc6b3ac16ca0e7130becd812410766884` |
| `architecture_evidence.json` | `2fb1fd4927c2a12fe56dddd8aeadb702c3d478c6ff1231139c5008b59075516e` |
| `architecture_overview.json` | `4af0dede69f99b55e618394797eaa394b1c12179a0215c518100f67f11ece394` |
| `CODE_DOCUMENTATION.md` | `e1ab976a848367c147c73f942e4cc057619a4cf6f54d1942ecfbb070b8489a88` |
| `GRAPH_REPORT.md` | `e5aec77b1fb7dd6393f6759a9a669c3d784e285e290120df6b6881d3aecb0702` |
| `architecture_inventory.json` | `48e550349f9fc23e69659b0747f2c0e0a48c480a92bece29be1ba1aa17b244bd` |

## Subsequent presentation-only hook corrections

The overview heading spacing and inventory-search heading level were corrected after the original source review. The knowledge graph and relationship evidence hashes remain unchanged. Desktop/mobile rendering, heading outlines and the mechanical detector passed. Original review hashes above remain the review-time snapshot; current presentation hashes are recorded in `output_hashes.json`. Details: `hook_review.json`.
