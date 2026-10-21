# Graph Report - Project  (2026-10-21)

## Corpus Check
- 168 files · ~357,190 words
- Verdict: corpus is large enough that graph structure adds value.
- Additional deterministic inventory: 5,414 files, including unsupported CSV/compressed/model formats represented by files or artifact groups. Their contents are not semantically interpreted.

## Summary
- 1696 nodes · 4702 edges · 83 communities (62 shown, 21 thin omitted)
- Extraction: 99% EXTRACTED · 1% INFERRED · 0% AMBIGUOUS · INFERRED: 46 edges (avg confidence: 0.89)
- Semantic token cost: **unreported by host**. Numeric zeros are unmeasured placeholders; semantic monetary cost is unknown. AST extraction uses no external LLM.

## Community Hubs (Navigation)
- Expanded Risk Manifest
- Historical Trial Manifest
- Interrupted Research Manifest
- Completed Research Manifest
- Notebook Execution and Presentation
- Learned Features and Diagnostics
- Quantile and Risk Tests
- Research Replay Integrity
- Learned Feature Cache
- Tabular and Statistical Adapters
- Report App Dependencies
- RTE Assisted Artifacts
- Original Peak Risk Analysis
- Expanded Artifact Verification
- Original Replay Integrity
- Research Commands and Serialization
- Assisted Contract Settings
- Assisted Scores and Metrics
- Expanded Risk Reporting
- Assisted Run Manifest
- Research Replay Engine
- Frozen Selection Evidence
- Expanded Risk Replay
- RTE Target and Fallback Tests
- Causal Base Forecast Bank
- Original Notebook Authoring
- Neural Model Adapter
- Manager Report and Review
- Probabilistic Scoring Tests
- Original Artifact Verification
- Historical Failure Analysis
- Neural Panel Integrity
- Causal Feature Engineering
- Specification Source Hashes
- Research Comparison Figures
- Chronology and DST Tests
- Report Adoption Content
- Interactive Report Composition
- RTE Assisted Replay
- Research Data Preparation
- Feature Availability Tests
- Saved Forecast Replay
- Original Candidate Validation
- Project Research Documentation
- Original Audit and EDA
- Legacy Baseline Investigation
- Shared Report Components
- Manager Failure Diagnostics
- Expanded Recommendation and Review
- Model Adapter Tests
- Original Uncertainty and Storage
- Report Layout Components
- Failure Diagnostic Computation
- Development Sensitivities
- Report Build Integrity
- Report Build Commands
- Neural Execution Limitations
- Research Resume Provenance
- Origin Feature Store
- Assisted Search Budgets
- Original Reference Fingerprints
- Manager Figure Exports
- Evidence and Vintage Limits
- Scientific Environment Requirements
- Original Manager Recommendation
- Earlier Workflow Recommendation
- Original Peak Interval Design
- Original Saved Test Evidence
- Assisted Calibration Window
- Development Fit Diagnostics
- Original Failure Catalogue
- Assisted Cold Definition
- Assisted Test Window
- Assisted Raw Source Identity

## God Nodes (most connected - your core abstractions)
1. `pandas` - 168 edges
2. `numpy` - 114 edges
3. `forecast_origin()` - 66 edges
4. `local_day_grid()` - 55 edges
5. `fingerprint()` - 49 edges
6. `pathlib` - 48 edges
7. `write_json()` - 47 edges
8. `spec` - 35 edges
9. `contract` - 34 edges
10. `contract` - 34 edges

## Surprising Connections (verified architecture bridges)

- Original model forecasts → causal base bank → LightGBM hybrid combiner. The extension reuses earlier out-of-sample forecasts; it does not natively retrain every base architecture with RTE. [Base-bank source](../src/rte_base_replay.py), [risk replay](../src/rte_risk_lab.py).
- Engineered history → earlier-trained Ridge/classifier features → selected point correction. The selected stacked model has `base: null`, so this path is distinct from the archived-base hybrids. [Learned features](../src/rte_learned_features.py), [frozen selection](../reports/rte_risk/20261020T213640021260Z/frozen_selection.json).
- Separately selected quantiles → fixed calibration comparisons → manager risk assessment. Point accuracy, probability selection and cold coverage remain different evidence. [Uncertainty replay](../src/rte_risk_uncertainty.py), [manager report](../reports/manager_report_20261021/complete_report.md).

## Import Cycles

No cycles found in inspected internal Python `ImportFrom` statements. Local function calls are not module import cycles. JavaScript runtime cycles are not certified.

## Hyperedges (group relationships)
- **Separate original, assisted and expanded frozen research protocols** — research_resume_original_research_v2_protocol, reports_rte_assisted_20261020t203527821283z_recommendation_rte_assisted_recommendation, research_resume_expanded_rte_risk_protocol [EXTRACTED 1.00]
- **Distinct expanded point, native-quantile and fixed-conformal constructions** — reports_manager_report_20261021_claim_audit_development_frozen_point_policy, reports_manager_report_20261021_claim_audit_native_quantile_selection, reports_manager_report_20261021_claim_audit_fixed_conformal_comparators [EXTRACTED 1.00]
- **Point improvement, critical uncertainty and research recommendation** — reports_rte_risk_20261020t213640021260z_one_page_recommendation_selected_rte_correction, reports_rte_risk_20261020t213640021260z_one_page_recommendation_critical_interval_shortfall, reports_rte_risk_20261020t213640021260z_one_page_recommendation_unseen_research_recommendation [EXTRACTED 1.00]

## Communities (83 total, 21 thin omitted)

### Community 0 - "Expanded Risk Manifest"
Cohesion: 0.05
Nodes (75): end, start, hours_local, temperature_below_c, created_utc, original_manifest_sha256, raw_sha256, france_electricity_hourly.csv (+67 more)

### Community 1 - "Historical Trial Manifest"
Cohesion: 0.05
Nodes (72): boundary_buffer_day, end, start, hours_local, temperature_below_c, description, trials_per_family, contract (+64 more)

### Community 2 - "Interrupted Research Manifest"
Cohesion: 0.05
Nodes (72): boundary_buffer_day, end, start, hours_local, temperature_below_c, description, trials_per_family, contract (+64 more)

### Community 3 - "Completed Research Manifest"
Cohesion: 0.05
Nodes (72): boundary_buffer_day, end, start, hours_local, temperature_below_c, description, trials_per_family, contract (+64 more)

### Community 4 - "Notebook Execution and Presentation"
Cohesion: 0.06
Nodes (45): ast, collections, jupyter.client, jupyter.client.kernelspec, nbclient, nbconvert, nbformat, re (+37 more)

### Community 5 - "Learned Features and Diagnostics"
Cohesion: 0.11
Nodes (17): build_diagnostics(), _functional(), _labels(), main(), _metrics(), _plot_gaps(), _plot_quarters(), build_learned_feature_bank() (+9 more)

### Community 6 - "Quantile and Risk Tests"
Cohesion: 0.12
Nodes (14): assemble_quantiles(), interval_score(), pinball(), quantile_scores(), signed_quantile(), uncertainty_replay(), MeanAdapter, RiskLabTests (+6 more)

### Community 7 - "Research Replay Integrity"
Cohesion: 0.09
Nodes (13): multiprocessing, Twenty-three causal-base LightGBM hybrids across thirteen families, Zero cold-evening calibration support, Causal feature, base-selection and uncertainty arithmetic tests, Optional real DLinear integration test was skipped, Saved unit-test run: 124 tests, one optional skip, Engineered histories and chronological learned predictors, Quantiles, conformal comparators and proper-score evaluation plan (+5 more)

### Community 8 - "Learned Feature Cache"
Cohesion: 0.09
Nodes (36): artifact_sha256, features.parquet, fit_ledger.json, combined, national, regional_only, inputs, columns (+28 more)

### Community 9 - "Tabular and Statistical Adapters"
Cohesion: 0.10
Nodes (20): dataclasses, sklearn.ensemble, sklearn.exceptions, sklearn.impute, sklearn.linear.model, sklearn.pipeline, sklearn.preprocessing, _hourly_series() (+12 more)

### Community 10 - "Report App Dependencies"
Cohesion: 0.10
Nodes (38): figma-squircle, html-to-image, lexical, @lexical/history, @lexical/link, @lexical/list, @lexical/markdown, @lexical/react (+30 more)

### Community 12 - "Original Peak Risk Analysis"
Cohesion: 0.15
Nodes (15): copy, analyze_peak_risk(), _choose_pool(), _cost_sensitivity(), _decorate(), freeze_peak_spec(), _intervals(), _read() (+7 more)

### Community 13 - "Expanded Artifact Verification"
Cohesion: 0.17
Nodes (27): csv, math, _cold(), equal_number(), _interval_arithmetic(), _notebooks(), number(), _point_arithmetic() (+19 more)

### Community 14 - "Original Replay Integrity"
Cohesion: 0.14
Nodes (10): Experiment, feature_columns(), fit_at_origin(), new_model(), local_day_grid(), load_archive(), select_forecast_run(), build_origin_features() (+2 more)

### Community 15 - "Research Commands and Serialization"
Cohesion: 0.14
Nodes (18): argparse, datetime, hashlib, json, os, pathlib, query(), number() (+10 more)

### Community 17 - "Assisted Contract Settings"
Cohesion: 0.07
Nodes (30): contract, bias_window_days, bootstrap_block_days, bootstrap_repetitions, calibration_minimum_observations, calibration_window_days, date_endpoints, demand_publication_delay_hours (+22 more)

### Community 18 - "Assisted Scores and Metrics"
Cohesion: 0.22
Nodes (19): numpy, conformal_comparator(), development_scores(), draw_figures(), read_forecasts(), score_extension(), write_report(), daily_metrics() (+11 more)

### Community 19 - "Expanded Risk Reporting"
Cohesion: 0.16
Nodes (19): _ablation_and_hybrid_tables(), _bootstrap(), _decorate(), _development_fit_diagnostics(), _empirical_probabilities(), _interval_scores(), _learned_feature_diagnostics(), main() (+11 more)

### Community 20 - "Assisted Run Manifest"
Cohesion: 0.14
Nodes (25): created_utc, cold_weights, interpretation, shortage_surplus_ratios, extension_spec, arms, cold_guardrail, critical_risk_scenarios (+17 more)

### Community 21 - "Research Replay Engine"
Cohesion: 0.19
Nodes (6): fcntl, gc, platform, _json(), ResearchRun, _selection_writer()

### Community 22 - "Frozen Selection Evidence"
Cohesion: 0.11
Nodes (4): Manager report claim audit, Development-frozen combined LightGBM point policy, Fixed split-conformal and CQR comparators, Separately selected native quantile policy

### Community 23 - "Expanded Risk Replay"
Cohesion: 0.21
Nodes (9): main(), candidate_variants(), add(), cold_mask(), configurations(), point_scores(), read_frame(), RiskLabRun (+1 more)

### Community 24 - "RTE Target and Fallback Tests"
Cohesion: 0.18
Nodes (7): demand_prediction(), training_target(), AssistedTests, invalid(), ConstantEstimator, SyntheticData, unittest.mock

### Community 25 - "Causal Base Forecast Bank"
Cohesion: 0.23
Nodes (11): causal_quarter_stream(), _check_oos(), _labels(), load_causal_base_bank(), _read_sealed(), select_released_trial(), _sha256(), _utc() (+3 more)

### Community 27 - "Neural Model Adapter"
Cohesion: 0.15
Nodes (7): importlib.metadata, logging, _backend(), backend_versions(), NeuralAdapter, NeuralBackendIntegrationTests, typing

### Community 28 - "Manager Report and Review"
Cohesion: 0.13
Nodes (5): Complete manager decision report, LightGBM correction of supplied RTE D1, Manager report and interactive presentation, Independent manager report consistency review, Missing-target and complete-day support handling

### Community 30 - "Probabilistic Scoring Tests"
Cohesion: 0.21
Nodes (7): project_root(), load_research_contract(), conformal_radius(), empirical_crps(), strict_conformal(), ResearchScoreTests, replay()

### Community 32 - "Historical Failure Analysis"
Cohesion: 0.21
Nodes (14): pandas, analyze_failures(), _grid(), _load_forecasts(), main(), _measured_days(), _plot_case(), _select_cases() (+6 more)

### Community 34 - "Neural Panel Integrity"
Cohesion: 0.33
Nodes (3): pandas.testing, _panel(), NeuralPanelTests

### Community 35 - "Causal Feature Engineering"
Cohesion: 0.24
Nodes (9): latest_eligible_time(), _availability(), build_lab_origin_features(), _calendar_additions(), _eligible_national(), feature_register(), FeatureLabData, prepare_feature_lab() (+1 more)

### Community 36 - "Specification Source Hashes"
Cohesion: 0.12
Nodes (17): source_sha256, config/research_experiment.json, config/rte_assisted_experiment.json, requirements-research-neural.txt, src/common.py, src/data.py, src/features.py, src/metrics.py (+9 more)

### Community 37 - "Research Comparison Figures"
Cohesion: 0.26
Nodes (10): _annotate(), _background(), _count(), _figure(), _name(), _number(), plot_cold_evening(), plot_overall_comparison() (+2 more)

### Community 39 - "Report Adoption Content"
Cohesion: 0.16
Nodes (5): Authored content asset boundary, adoptionBrief(), number(), rate(), signed()

### Community 40 - "Interactive Report Composition"
Cohesion: 0.17
Nodes (14): bar(), fmt(), metricNames, Narrative(), palette, pc(), ReportContent(), Table() (+6 more)

### Community 41 - "RTE Assisted Replay"
Cohesion: 0.34
Nodes (4): fingerprint(), _read_predictions(), model_key(), RTEAssistedRun

### Community 42 - "Research Data Preparation"
Cohesion: 0.23
Nodes (6): load_inputs(), build_supervised(), calendar_exog(), data_summary(), prepare_research_data(), ResearchData

### Community 44 - "Saved Forecast Replay"
Cohesion: 0.25
Nodes (9): gzip, joblib, shutil, archive_predictions(), baseline_replay(), model_replay(), asof_history(), prediction_fallback() (+1 more)

### Community 45 - "Original Candidate Validation"
Cohesion: 0.19
Nodes (3): experiments(), select_candidate(), stress_inputs()

### Community 46 - "Project Research Documentation"
Cohesion: 0.19
Nodes (5): Causal national, regional-only and combined predictor views, Legacy quarterly evaluation protocol, French electricity forecasting research workflow, Earlier regional RTE-residual quantile LightGBM selection, Earlier RTE-assisted study recommendation

### Community 48 - "Original Audit and EDA"
Cohesion: 0.22
Nodes (7): load_contract(), plotting_style(), prepare_directories(), availability_gate(), field_register(), read_audited(), run_audit()

### Community 49 - "Legacy Baseline Investigation"
Cohesion: 0.26
Nodes (7): load_data(), mae(), mape(), rmse(), save_result(), score(), walk_forward_folds()

### Community 50 - "Shared Report Components"
Cohesion: 0.22
Nodes (5): react, Shared authored report components, reports.manager.report.20261021.app.src.data.app.public, reports.manager.report.20261021.app.src.data.app.public.icon, reports.manager.report.20261021.app.src.data.app.public.usedataapp

### Community 52 - "Expanded Recommendation and Review"
Cohesion: 0.27
Nodes (4): Final independent numerical and provenance cross-check, Frozen point policy improves matched historical MAE, Development-selected combined-history RTE correction, lab__lightgbm__combined__stacked__w1__residual

### Community 53 - "Model Adapter Tests"
Cohesion: 0.36
Nodes (3): make_statistical(), make_tabular(), ResearchModelTests

### Community 55 - "Original Uncertainty and Storage"
Cohesion: 0.33
Nodes (3): read_features(), read_labels(), read_predictions()

### Community 56 - "Report Layout Components"
Cohesion: 0.38
Nodes (9): AdaptiveSortableRegion(), measuredCanvasLayout(), regionLabel(), SortableDashboardLayout(), renderRegion(), visit(), sortableMetadata(), reports.manager.report.20261021.app.src.data.app.public.sortableitem (+1 more)

### Community 57 - "Failure Diagnostic Computation"
Cohesion: 0.36
Nodes (7): clean(), grouped(), main(), metrics(), sha(), write_json(), holiday_dates()

### Community 59 - "Development Sensitivities"
Cohesion: 0.43
Nodes (4): point_metrics(), development_sensitivities(), stress_inputs(), neural_configurations()

### Community 60 - "Report Build Integrity"
Cohesion: 0.25
Nodes (7): ref.node.crypto, ref.node.fs, ref.node.path, ref.node.url, configureServer(), localThreadId, snapshotIntegrityPlugin

### Community 61 - "Report Build Commands"
Cohesion: 0.25
Nodes (8): scripts, build, dev, integrity:authorize, integrity:update, prebuild, test, verify:runtime

### Community 62 - "Neural Execution Limitations"
Cohesion: 0.43
Nodes (3): Original research_v2 execution interpretation, GAM empirical residual and pooled peak intervals, Original research_v2 detailed recommendation

### Community 63 - "Research Resume Provenance"
Cohesion: 0.43
Nodes (5): Frozen original research_v2 experiment design, Historical research_v2 pause checkpoint, Expanded RTE-feature and cold-risk protocol, Original research_v2 protocol, Research completion status

### Community 64 - "Origin Feature Store"
Cohesion: 0.29
Nodes (4): contract_sha256, origins, regime, rows

### Community 65 - "Assisted Search Budgets"
Cohesion: 0.29
Nodes (7): description, trials_per_family, profiles, description, trials_per_family, compact, extended

### Community 66 - "Original Reference Fingerprints"
Cohesion: 0.29
Nodes (6): reference_sha256, development_policy_selection.json, forecasts/all_calibration.csv.gz, forecasts/all_test.csv.gz, selected_parameters.json, tables/development_baselines.csv

### Community 67 - "Manager Figure Exports"
Cohesion: 0.40
Nodes (4): matplotlib, matplotlib.pyplot, read(), save()

### Community 69 - "Scientific Environment Requirements"
Cohesion: 0.53
Nodes (3): Portable notebook execution and export lower bounds, Portable scientific and serialization package lower bounds, Recorded execution-environment pins dated 2026-10-20

### Community 70 - "Original Manager Recommendation"
Cohesion: 0.70
Nodes (3): Original separately selected national GAM, Original research_v2 manager recommendation, Original development-selected RTE rolling bias adjustment

### Community 74 - "Assisted Calibration Window"
Cohesion: 0.50
Nodes (4): boundary_buffer_day, end, start, calibration

### Community 79 - "Assisted Cold Definition"
Cohesion: 0.67
Nodes (3): hours_local, temperature_below_c, cold_evening

### Community 80 - "Assisted Test Window"
Cohesion: 0.67
Nodes (3): test, end, start

### Community 81 - "Assisted Raw Source Identity"
Cohesion: 0.67
Nodes (3): raw_sha256, france_electricity_hourly.csv, france_regional_electricity_hourly.csv

## Knowledge Gaps
- **55 isolated node(s):** `contract_sha256`, `rows`, `origins`, `regime`, `palette` (+50 more)
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 258 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **21 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions

- How does an original model’s sealed forecast reach an expanded LightGBM hybrid while preserving outcome availability?
- Which inputs differ between national, regional-only and combined predictor views?
- Why does improved point accuracy not establish adequate cold-evening interval coverage?

## Architecture extraction scope and verification

Start with [architecture.html](architecture.html), then inspect [CODE_DOCUMENTATION.md](CODE_DOCUMENTATION.md), [graph.json](graph.json), [verification.json](verification.json) and [refresh_audit.md](refresh_audit.md).

- Current overview: 32 components and 72 sourced relationships. All nine notebooks are represented: eight numbered stages and a distinct legacy baseline branch. The original workflow and three later versioned protocols remain separate.
- Source extraction: 63 Python files and all notebook code cells; 5,414 inventoried files; 21 generated artifact groups. Detected JSON metadata is source structure, not execution evidence.
- Removed five obsolete nodes for the deleted production module and monitoring file; grouped 120 individual generated/tooling-era artifact nodes, replaced 44 re-extracted concepts, and excluded one local tooling node. Every retired ID is audited in [retired_nodes_audit.json](retired_nodes_audit.json); previous graph outputs are backed up under `history/`.
- Four unchanged documents had ten cached cross-references to replaced concepts. All four were re-extracted; no missing endpoint was invented to preserve a stale edge. See [semantic_cache_review.json](semantic_cache_review.json).
- The full Graphify view is undirected. Original direction, relation type, confidence and source evidence remain in each edge's `relationship_observations`. The overview renders directional relationships. Distinct repeated evidence is preserved rather than silently overwritten.
- Two syntactic self-reference observations are retained separately in [normalization_audit.json](normalization_audit.json). They are excluded from the navigation graph. Final graph diagnostics report no dangling or missing endpoints, graph self-loops or unintended endpoint collapse.
- Unresolved AST implementation references are explicitly materialized and labeled; their presence is not evidence that an external implementation was scanned. Dynamic runtime dependencies and values assembled through `run.path` are not exhaustively resolved.
- Type/decorator/import placeholders emitted without a definition location are labeled `Reference: …` with `architecture_kind: type_or_import_reference`; their incident edges retain source evidence. They are reference concepts, not inspected implementation definitions. Their IDs are listed in the inventory separately from missing-endpoint references.
- God-node degree and raw community cohesion describe this mixed source/metadata graph. Common packages and manifest fields can dominate connectivity. Neither centrality nor weak cohesion establishes an important modeling feature, software defect or refactoring need. Automatic navigation statistics remain available in [analysis.json](analysis.json).
- The selected point and quantile models, fixed conformal/CQR comparators, 23 hybrid combiners and metrics retain separate identities. Original forecast vintages remain unverified; the test was previously explored and calibration has no cold evenings. The graph does not certify operational superiority or measured financial savings.
- No predictive code, experiment configuration, notebook, raw data, fitted model or saved forecast was changed or executed for this graph refresh. The project README and architecture/documentation outputs were updated. Saved test outcomes are historical evidence, not tests rerun by this task.
- Graphify outputs were written into the working tree and may appear in `git status`; existing ignore rules were preserved. See [Graphify team workflows](https://docs.graphify.com/guides/team-workflows).
