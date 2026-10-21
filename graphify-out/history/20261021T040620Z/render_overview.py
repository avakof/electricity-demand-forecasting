"""Render an offline architecture overview from the inspected source inventory."""
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent
ROOT = OUT.parent
inv = json.loads((OUT / "architecture_inventory.json").read_text())
N = []
E = []


def node(key, title, subtitle, kind, col, row, paths, description):
    N.append(dict(id=key, title=title, subtitle=subtitle, kind=kind, x=44+col*244, y=row, w=220, h=100, paths=paths, description=description))


def link(a, b, label, layer, evidence):
    if a != b and not any(e["source"] == a and e["target"] == b and e["layer"] == layer for e in E):
        E.append(dict(source=a, target=b, label=label, layer=layer, evidence=evidence))


stages = [
    ("Audit & contract", "Provenance, coverage, reconciliation", "01_data_audit", "Freeze the forecast contract and source hashes. Audit missingness, timestamps, country/regional reconciliation and the operational availability gate."),
    ("Development EDA", "Calendar, temperature, RTE errors", "02_development_eda", "Use development data to examine season, weekdays, holidays, temperature bands and delayed effects; record modeling hypotheses."),
    ("Replay & baselines", "Origin → next-day intervals", "03_backtest_and_baselines", "Run integrity tests, reconstruct information available at each origin, save feature/label stores and replay four baseline policies."),
    ("Model comparison", "Validate → select → freeze", "04_model_comparison", "Compare bounded feature/model ablations on chronological validation. Check guardrails and fallback stress tests. Freeze selection before test replay; keep oracle weather separate."),
    ("Uncertainty & failures", "Frozen test, intervals, bootstrap", "05_uncertainty_and_failures", "Replay the frozen candidate on the previously explored test. Score matched hours, calculate paired block uncertainty, evaluate intervals and inspect failures."),
    ("Business & deployment", "Evidence, monitoring, risks", "06_business_and_deployment", "Answer the manager's three questions: improvement over RTE, uncertainty/failures, deployment/risks. Recommend prospective shadow evidence; do not equate forecast skill with euro savings."),
]
for i, (title, subtitle, stem, desc) in enumerate(stages):
    node(f"n{i+1}", f"0{i+1}  {title}", subtitle, "notebook", i, 216, [f"notebooks/{stem}.ipynb"], desc)
    if i:
        link(f"n{i}", f"n{i+1}", "run next", "workflow", "readme.md: run the six notebooks in order")

node("contract", "Forecast contract", "11:00 Paris · D−1 → next day", "config", 0, 58, ["config/forecast_contract.json", "config/sources.json"], "Defines cutoff, 23/24/25-hour delivery days, chronological splits, MAE objective, availability assumptions, guardrails and fallbacks. Sources.json records provided source identities.")
node("raw", "Provided source data", "National + 12 regional CSVs", "data", 1, 58, ["data/france_electricity_hourly.csv", "data/france_regional_electricity_hourly.csv", "data/france_regional_electricity_hourly.csv.gz"], "Original national demand/RTE/temperature and regional demand/temperature. Source-to-hourly conversion and original vintages are unverified. Values are preserved by the audit.")
node("lock", "Evaluation lock", "Contract + source fingerprints", "config", 2, 58, ["config/evaluation_lock.json", "data/raw/source_manifest.json"], "Notebook 01 records fingerprints. A changed contract or source identity requires an explicitly versioned new research run.")
node("freeze", "Frozen candidate", "Validation-selected RTE correction", "config", 3, 58, ["config/frozen_candidate.json", "reports/experiment_register.csv"], "Notebook 04 freezes the validation-selected experiment and contract identity; notebook 05 and 06 read it. The current selected model is RTE-assisted and retrospective.")
node("archives", "Authentic vintage schemas", "Demand · RTE runs · weather runs", "data", 4, 58, ["data/raw/demand_observation_archive.csv", "data/raw/rte_forecast_archive.csv", "data/raw/weather_forecast_archive.csv"], "All three current CSVs contain headers only. Integration expects valid_time, available_at, forecast issued_at/run_id, numeric value and demand revision status. These are interfaces, not authentic historical runs.")
node("gate", "Operational gate CLOSED", "Missing archives + unverified lineage", "gate", 5, 58, ["reports/audit/operational_gate.csv", "reports/audit/limitations.md"], "No authentic operational experiment is enabled. Availability gate requires nonempty valid archives, verified vintage provenance and verified interval aggregation. Current results support retrospective research and shadow design only.")

modules = [
    ("data", "data.py", "Audit + archive selection", 0, 370),
    ("eda", "eda.py", "Development frame + hypotheses", 1, 370),
    ("features", "features.py", "Cutoff features + fallback", 2, 370),
    ("backtest", "backtest.py", "Fit, replay, freeze, archive", 3, 370),
    ("uncertainty", "uncertainty.py", "Past-error + quantile evaluation", 4, 370),
    ("pipeline", "pipeline.py", "Readiness, monitoring, risks", 5, 370),
    ("common", "common.py", "Calendar, DST, origins, hashes", 2, 494),
    ("metrics", "metrics.py", "Matched scores + block bootstrap", 3, 494),
    ("storage", "storage.py", "Restore typed saved artifacts", 4, 494),
]
for key, file, subtitle, col, row in modules:
    node(key, file, subtitle, "module", col, row, ["src/"+file], {
        "data":"Read inputs, preserve missing targets, reconcile complete regional populations, write audit artifacts, validate archive schemas and select an available forecast run.",
        "features":"Construct per-origin/per-delivery features using only cutoff-eligible history. Keep labels separate, support authentic archive snapshots and apply the declared fallback order.",
        "backtest":"One replay engine serves baselines and learned models. Training labels must be available before fitting. Per-origin prediction ledgers close before the next fit; aggregate forecasts have immutable copies.",
        "common":"Shared local calendar, French national holidays, UTC delivery grids across daylight saving, forecast origins, fingerprints and contract loading.",
        "metrics":"MAE/RMSE/bias/WAPE, availability, common eligible hours, segments, complete-day metrics and paired contiguous-day block resampling.",
        "uncertainty":"Rolling intervals from past out-of-sample errors, interval coverage/width/pinball metrics and complete residual-day trajectory scenarios. Calibration uses available outcomes only.",
        "storage":"Read saved feature, evaluation-label and forecast tables with timestamp conversion. Stored targets remain separate from forecast archives.",
        "pipeline":"Readiness checks and monitoring/risk specifications. It does not implement an external ingestion service, scheduler, publication endpoint or procurement decision.",
        "eda":"Restrict exploration to development periods; summarize temperature bands and generate a short hypothesis register. Future realized weather remains diagnostic.",
    }[key])
node("tests", "Integrity tests", "18 behavioral checks", "test", 1, 494, ["tests/"], "Tests cover unavailable future information, label delays, forecast issuance/publication, DST, missingness, fallback, reproducibility and paired bootstrap eligibility. Test count is from the current test definitions; this graph build does not rerun models.")
node("tools", "Notebook tooling", "Author → execute → validate", "tool", 0, 494, ["scripts/", "requirements.txt", "requirements-validated.txt"], "build_notebooks.py authors notebook cells; execute_notebooks.py runs them in order with a project-local kernel; validate_artifacts.py checks execution/artifact invariants and exports HTML/PNG previews. The graph build does not execute these scripts.")

outputs = [
    ("audit", "Audited data & register", "Dictionary, limitations, overlap", ["reports/audit/", "data/processed/audited_country.csv.gz"], "Audited hourly national grid includes regional reconciliation and temperature summaries. Audit reports record missingness, frequency, coverage, suspicious retained values and source limitations."),
    ("hypotheses", "Hypothesis register", "Temperature bands + EDA evidence", ["reports/hypothesis_register.csv", "reports/scorecards/development_temperature_bands.csv"], "Development findings become hypotheses for calendar, demand-history, temperature and RTE correction features. Observation counts remain visible for sparse bands."),
    ("snapshots", "Origin feature store", "Features and labels stored apart", ["data/origin_snapshots/", "data/processed/evaluation_labels.csv.gz"], "Rows represent forecast origin plus delivery interval. Information availability and label availability are explicit; complete origin groups stay together."),
    ("forecasts", "Forecasts & fitted models", "Ledgers, immutable copies, fit logs", ["reports/forecasts/", "reports/models/"], "Per-origin append-only ledgers, aggregate forecasts and manifests, serialized pipelines and fit-origin logs provide the replay evidence trail. Forecast archives exclude evaluation outcomes."),
    ("scores", "Scores & uncertainty", "Segments, intervals, failures", ["reports/scorecards/", "reports/uncertainty/", "reports/failure_catalogue.csv"], "Matched model/RTE scores, availability, temporal folds, critical segments, paired block intervals, coverage reports, daily scenarios and failure catalogue."),
    ("recommendation", "Manager recommendation", "Skill · uncertainty · deployment", ["reports/final_recommendation.md", "reports/monitoring_specification.csv", "reports/risk_register.csv", "reports/run_summary.json"], "Final report separates measured forecast losses, uncertainty and failure evidence, business proxies and deployment prerequisites. Current recommendation requires prospective shadow validation."),
]
for i,(key,title,subtitle,paths,desc) in enumerate(outputs):
    node(key,title,subtitle,"output",i,670,paths,desc)
    link(f"n{i+1}",key,"produces / reports","workflow",f"readme.md: notebook 0{i+1} main outputs")

link("contract","n1","freeze specification","workflow","notebooks/01_data_audit.ipynb: cell 4")
link("raw","n1","audit original values","workflow","src/data.py: load_inputs, run_audit")
link("n1","lock","write identity lock","data","notebooks/01_data_audit.ipynb: cell 4")
link("n4","freeze","freeze selection","workflow","notebooks/04_model_comparison.ipynb: cell 7")
link("freeze","n5","replay frozen candidate","workflow","notebooks/05_uncertainty_and_failures.ipynb: cell 3")
link("archives","gate","required evidence","workflow","src/data.py: availability_gate")
link("gate","n6","constrains recommendation","workflow","notebooks/06_business_and_deployment.ipynb: cell 9")
link("n3","tests","run checks","code","notebooks/03_backtest_and_baselines.ipynb: cell 4")
for nb,imports in inv['notebook_imports'].items():
    number=int(Path(nb).name[:2])
    for module in imports:
        key=Path(module).stem
        if any(n['id']==key for n in N):
            link(f'n{number}',key,'imports','code',f'{nb}: code-cell import')

for item in inv['io']:
    if not item['source'].startswith('notebooks/') or item['relation'] not in ('reads','writes','checks_hash'):
        continue
    number=int(Path(item['source']).name[:2])
    for target in N:
        if target['kind'] in ('notebook','module','test','tool'):
            continue
        if any(item['target']==p or (p.endswith('/') and item['target'].startswith(p)) for p in target['paths']):
            a,b=(target['id'],f'n{number}') if item['relation']=='reads' else (f'n{number}',target['id'])
            link(a,b,item['relation'],'data',f"{item['source']}: {item['location']}")

for number in range(1,7):
    link("tools",f"n{number}","authors / executes / validates","code","scripts/build_notebooks.py; scripts/execute_notebooks.py; scripts/validate_artifacts.py")

for n in N:
    n['files']=[f['path'] for f in inv['files'] if any(f['path']==p or (p.endswith('/') and f['path'].startswith(p)) for p in n['paths'])]
    n['count']=len(n['files'])
    assert n['files'],n['id']

payload={'nodes':N,'edges':E,'inventory':inv,'root':str(ROOT)}
(OUT/'architecture_overview.json').write_text(json.dumps(payload,indent=2))
template=Path(__file__).with_name('overview_template.html').read_text()
(OUT/'architecture.html').write_text(template.replace('__ARCHITECTURE_DATA__',json.dumps(payload).replace('</','<\\/')))
print(f'Overview: {len(N)} nodes, {len(E)} sourced links, {len(inv["files"])} searchable files')
