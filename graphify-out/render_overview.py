"""Evidence-backed, grouped architecture overview. Does not execute research code."""
from __future__ import annotations
import ast
from datetime import datetime, timezone
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent
ROOT = OUT.parent
inv = json.loads((OUT / 'architecture_inventory.json').read_text())
N, E = [], []
RISK = 'reports/rte_risk/20261020T213640021260Z'
RESEARCH = 'reports/research_v2/20261020T164443776583Z'
ASSISTED = 'reports/rte_assisted/20261020T203527821283Z'
MANAGER = 'reports/manager_report_20261021'
files = {f['path'] for f in inv['files']}
required = {'notebooks/07_statistical_and_boosting_models.ipynb', 'notebooks/08_deep_learning_and_uncertainty.ipynb', 'notebooks/02_jg_baseline_model.ipynb'}
assert required <= files, 'Regenerate the current source inventory before rendering.'


def evidence(path, needle):
    """Resolve a literal source anchor into a checked, exact line reference."""
    lines = (ROOT / path).read_text().splitlines()
    matches = [i for i, line in enumerate(lines, 1) if needle in line]
    assert matches, (path, needle)
    return {'path': path, 'line': matches[0], 'excerpt': lines[matches[0]-1].strip()[:240]}


def node(key, title, subtitle, kind, x, y, w, paths, description, h=100):
    existing = [p for p in paths if (ROOT/p).exists()]
    assert len(existing) == len(paths), (key, set(paths)-set(existing))
    grouped = sorted(p for p in files if any(p == q or (q.endswith('/') and p.startswith(q)) for q in paths))
    assert grouped, (key, paths)
    N.append(dict(id=key, title=title, subtitle=subtitle, kind=kind, x=x, y=y, w=w, h=h,
                  paths=paths, files=grouped, count=len(grouped), description=description))


def link(a, b, label, layer, refs, overview=False):
    assert refs and all((ROOT/r['path']).is_file() for r in refs)
    old = next((e for e in E if (e['source'], e['target'], e['layer'], e['label']) == (a,b,layer,label)), None)
    if old:
        for r in refs:
            if r not in old['references']:
                old['references'].append(r)
        return
    E.append(dict(source=a, target=b, label=label, layer=layer, references=refs, overview=overview))


panels = [
    dict(id='inputs', x=20, y=20, w=1620, h=200, title='1  Inputs, contract and evidence limits', subtitle='One national demand target; supplied RTE D1 is assumed available at the cutoff.'),
    dict(id='original', x=20, y=240, w=1620, h=300, title='2  Original notebook workflow and legacy investigation', subtitle='The original six stages remain separate from the later versioned model comparisons.'),
    dict(id='studies', x=20, y=560, w=1620, h=300, title='3  Notebooks 07–08 preserve three versioned research studies', subtitle='Each study retains its own specification, chronological selection and result directory.'),
    dict(id='risk', x=20, y=880, w=1620, h=330, title='4  Expanded RTE feature and cold-risk experiment', subtitle='National / regional-only / combined predictors → causal replay → separately selected point and probabilistic policies.'),
    dict(id='reports', x=20, y=1230, w=1620, h=230, title='5  Saved evidence and the manager’s three questions', subtitle='Improvement over RTE · uncertainty and failures · research recommendation and risks. No production system.'),
]
node('raw', 'Provided hourly data', 'National + twelve regional series', 'data', 40, 100, 300,
     ['data/france_electricity_hourly.csv','data/france_regional_electricity_hourly.csv'],
     'National consumption is the target in every comparison. Regional demand and temperature are predictors, not replacement evaluation truth. There is no regional RTE forecast. Raw observations remain unchanged.')
node('contract', 'Forecast contract', '11:00 Paris on D → all hours of D+1', 'config', 360, 100, 300,
     ['config/forecast_contract.json','config/sources.json','config/evaluation_lock.json'],
     'Cutoff-aware history, explicit publication-delay assumptions and 23/24/25-hour local delivery days. An 11:00 Europe/Paris origin is also before a literal 12:00 fixed-CET deadline in summer. Later studies use separate versioned configurations.')
node('archives', 'Vintage archive schemas', 'Demand revisions · RTE issues · weather', 'data', 680, 100, 300,
     ['data/raw/demand_observation_archive.csv','data/raw/rte_forecast_archive.csv','data/raw/weather_forecast_archive.csv'],
     'The supplied archives contain headers only. They define interfaces, not authentic historical observations or forecast runs. RTE D1 availability is assumed; no D2 timing rule is used.')
node('limits', 'Operational evidence unavailable', 'Retrospective results; exposed test period', 'gate', 1000, 100, 300,
     ['reports/audit/operational_gate.csv','reports/audit/limitations.md',f'{RISK}/design_audit.md'],
     'No identified future-label leak does not establish authentic historical input availability. Original forecast vintages, demand revisions and source-to-hourly aggregation are unverified. The test period was previously explored. No operational validation or euro savings claim follows.')
node('checks', 'Integrity and artifact checks', 'Tests + hashes + daily-origin ledgers', 'test', 1320, 100, 300,
     ['tests/','scripts/check_research_artifacts.py','scripts/check_rte_assisted_artifacts.py','scripts/check_rte_risk_artifacts.py',f'{RISK}/artifact_verification.json'],
     'Behavioral tests and saved-artifact audits cover availability, leakage, DST, complete intervals, reproducibility, frozen identities, matched arithmetic and executed notebooks. Recorded verification is evidence for this run, not a forecast-vintage guarantee.')

stages = [
 ('audit','01_data_audit','01  Data audit','Dictionary, availability, reconciliation','Audit original values, overlap, missingness and hourly conventions; record assumptions and source fingerprints.'),
 ('eda','02_development_eda','02  Development EDA','Calendar, temperature, RTE errors','Development-only patterns and observation counts support a hypothesis register.'),
 ('baseline','03_backtest_and_baselines','03  Replay and baselines','Origin → delivery interval','Cutoff-eligible features, labels stored separately and common baseline replay.'),
 ('compare','04_model_comparison','04  Model comparison','Validate → select → freeze','Earlier compact candidate/feature comparison with chronological validation and declared fallback policies.'),
 ('uncertainty','05_uncertainty_and_failures','05  Uncertainty and failures','Intervals, matched scores, bootstrap','Earlier frozen-policy evaluation and past-error uncertainty diagnostics.'),
 ('business','06_business_and_deployment','06  Research recommendation','Scorecard, limitations, business meaning','Filename retained for links. This notebook reports research evidence and limitations; it does not implement deployment or monitoring.'),
]
nb_ids = {}
for i, (key, stem, title, subtitle, desc) in enumerate(stages):
    nb_ids['notebooks/'+stem+'.ipynb'] = key
    node(key,title,subtitle,'notebook',40+i*266,320,250,['notebooks/'+stem+'.ipynb'],desc,h=100)
    if i:
        link(stages[i-1][0],key,'documented stage order','workflow',[evidence('readme.md',stem+'.ipynb')],True)
node('legacy','Legacy 02 investigation','Separate notebook; separate evaluation protocol','notebook',40,440,510,
     ['notebooks/02_jg_baseline_model.ipynb','scripts/evaluation.py'],
     'Historical baseline investigation. It is not the canonical notebook 02 and does not define the later origin-aware comparison protocol.',h=80)
nb_ids['notebooks/02_jg_baseline_model.ipynb']='legacy'
node('original_core','Shared original research modules','Audit · features · replay · metrics · uncertainty','module',575,440,510,
     ['src/common.py','src/data.py','src/eda.py','src/features.py','src/backtest.py','src/metrics.py','src/storage.py','src/uncertainty.py','src/research_risks.py'],
     'Reusable original workflow implementation. Availability, calendar/DST, fallback, matching and scoring helpers are also imported by later research modules.',h=80)
node('original_outputs','Original workflow artifacts','Audit, hypotheses, feature store and scorecards','output',1110,440,510,
     ['reports/audit/','reports/hypothesis_register.csv','data/origin_snapshots/','data/processed/','reports/scorecards/','reports/uncertainty/','reports/final_recommendation.md'],
     'Earlier workflow evidence is preserved. Do not confuse its frozen candidate or scores with any later versioned study.',h=80)
node('nb07','07  Statistical and boosting models','Original study + two RTE extensions','notebook',40,640,510,
     ['notebooks/07_statistical_and_boosting_models.ipynb'],
     'Orchestrates statistical and tabular experiments, then preserves appended RTE-assisted and expanded-feature/cold-risk sections.',h=90)
node('nb08','08  Deep learning and uncertainty','Neural comparisons + probabilistic evaluation','notebook',575,640,510,
     ['notebooks/08_deep_learning_and_uncertainty.ipynb'],
     'Resumes the original research study, compares neural candidates and evaluates separately identified uncertainty methods. Appended sections report later RTE extensions.',h=90)
nb_ids['notebooks/07_statistical_and_boosting_models.ipynb']='nb07'
nb_ids['notebooks/08_deep_learning_and_uncertainty.ipynb']='nb08'
node('tooling','Notebook and run tooling','Author / extend / execute / verify','tool',1110,640,510,
     ['scripts/build_notebooks.py','scripts/build_research_notebooks.py','scripts/execute_notebooks.py','scripts/execute_research_notebooks.py','scripts/extend_rte_assisted_notebooks.py','scripts/extend_rte_risk_notebooks.py','scripts/run_rte_risk_experiment.py'],
     'Authoring scripts serve different notebook generations; extension scripts preserve earlier study sections. The risk runner performs prepare, development, evaluation and report stages. Viewing this graph does not execute these scripts.',h=90)
node('research','Original research_v2','13 families · 23 variants + four references','module',40,760,510,
     ['config/research_experiment.json']+[str(p.relative_to(ROOT)) for p in sorted((ROOT/'src').glob('research_*.py')) if p.name != 'research_risks.py']+[RESEARCH+'/'],
     'Original bounded statistical, boosting and neural comparison; national and regional-assisted histories. Its development-selected complete policy was rolling RTE bias correction. Finite neural budgets and training-anchor limitations constrain architecture comparisons.',h=80)
node('assisted','First RTE-assisted study','16 tabular variants; direct versus residual','module',575,760,510,
     ['src/rte_assisted.py','config/rte_assisted_experiment.json','scripts/rte_assisted_reporting.py',ASSISTED+'/'],
     'Extends ResearchRun with RTE predictors. Residual target = actual − RTE; prediction = RTE + learned correction. Its old regional arm includes national plus regional history. This is not the new regional-only arm.',h=80)
node('risk_spec','Expanded risk specification','55 new variants; four development folds','config',1110,760,510,
     ['config/rte_risk_experiment.json',f'{RISK}/manifest.json',f'{RISK}/variant_register.json',f'{RISK}/frozen_selection.json'],
     'Q2–Q4 2024 and Q1 2025 development; April–June 2025 calibration; previously explored July 2025–June 2026 evaluation. Weighted-MAE selection penalizes realized cold evenings fivefold. Three predictor views and 23 labeled hybrids; all base architectures were not natively retrained with RTE.',h=80)
node('feature_views','Three causal predictor views','National / regional-only / combined','module',40,960,380,
     ['src/rte_feature_lab.py',f'{RISK}/tables/feature_register.csv'],
     'Every view predicts national demand and includes calendar plus assumed-known RTE D1. Regional-only excludes national demand, temperature and error histories. Origin/delivery lags, rolling statistics, EWMA, changes, RTE errors and delivery-day RTE shape respect declared availability. Future realized weather is excluded.',h=100)
node('learned','Chronological learned features','Ridge demand + two logistic probabilities','module',440,960,380,
     ['src/rte_learned_features.py',f'{RISK}/learned_feature_fits.json'],
     'Monthly earlier-data fits produce out-of-sample demand, cold probability and large RTE-underprediction probability. Training labels must be released before fitting; preprocessing is fitted within eligible history. Predicted cold probability can weight training.',h=100)
node('base_bank','Causal base forecasts / hybrids','23 archived base variants; new LGBM combiner','module',840,960,380,
     ['src/rte_base_replay.py',f'{RISK}/base_forecast_provenance.json'],
     'Original out-of-sample model predictions are selected using released earlier-quarter outcomes, then supplied to a new LightGBM RTE-residual combiner. A hybrid represents the original family as an input forecast; it is not native RTE-feature retraining of that architecture.',h=100)
node('risk_engine','RiskLabRun replay and selection','Train → predict → seal → score → freeze','module',1240,960,380,
     ['src/rte_risk_lab.py','scripts/run_rte_risk_experiment.py'],
     'Quarterly point-model refits; released-label training; complete daily origins. Native tabular variants and hybrids share the replay. Point selection uses development weighted MAE plus availability/ordinary-MAE/cold-MAE guardrails. Forecast files omit outcomes and carry hashes and per-origin ledgers.',h=100)
node('point','Selected point policy','Combined + learned; neutral training weights','output',40,1090,380,
     [f'{RISK}/frozen_selection.json',f'{RISK}/result_summary.json'],
     'lab__lightgbm__combined__stacked__w1__residual. RTE plus a learned correction, selected on cold-weighted development loss. Neutral training weights won this bounded comparison. Its empirical residual intervals are evaluated as a separate uncertainty construction.',h=100)
node('quantile','Separate quantile selection','Combined + learned; cold-probability weights','output',440,1090,380,
     ['src/rte_risk_uncertainty.py',f'{RISK}/tables/development_quantile_selection.csv'] if (ROOT/f'{RISK}/tables/development_quantile_selection.csv').exists() else ['src/rte_risk_uncertainty.py',f'{RISK}/frozen_selection.json'],
     'lab__lightgbm_quantile__combined__stacked__w5__residual. Chosen by overall development WIS, not best cold coverage. Quantile grid supports central/asymmetric intervals; sorted crossing repair is predeclared. Do not attach its metrics to the selected point policy.',h=100)
node('conformal','Fixed calibration comparators','Engineered-only native / split / CQR','module',840,1090,380,
     ['src/rte_risk_uncertainty.py',f'{RISK}/tables/uncertainty_protocol.json'],
     'Separately fixed predictor before calibration, excluding monthly learned-input updates. Split-conformal and CQR produce calibrated bounds, not guaranteed conditional quantiles. Calibration contains zero cold evenings; time dependence and regime shift preclude automatic coverage guarantees.',h=100)
node('scores','Matched scores and uncertainty','MAE, cold loss, pinball, WIS, CRPS, blocks','module',1240,1090,380,
     ['scripts/rte_risk_reporting.py',f'{RISK}/tables/',f'{RISK}/recommendation.md'],
     'Exactly matched observations, paired day-block intervals, feature ablations and directional tail failures. CRPS is a score; conformal is calibration. Marginal hourly intervals do not define daily energy/peak intervals. Physical-unit risk scenarios do not measure euro savings.',h=100)
node('sealed','Saved predictions and provenance','Forecasts · fit logs · ledgers · source hashes','output',40,1320,380,
     [f'{RISK}/forecasts/',f'{RISK}/fits/',f'{RISK}/ledgers/',f'{RISK}/manifest.json',f'{RISK}/artifact_verification.json'],
     'Frozen run artifacts preserve predictions before outcomes, fitting availability and source/raw identities. Reports reuse these saved outputs; graph generation never fits models.',h=110)
node('diagnostics','Descriptive failure diagnostics','Monthly/hourly errors; worst cold-day trajectory','output',440,1320,380,
     [MANAGER+'/diagnostics/'],
     'Reporting-only reconstruction from saved forecasts and raw outcomes: counts, signed errors, missing-target exclusion, worst days and interval failures. Measured cold/calendar conditions are not proof of weather forecast misses or causal explanations.',h=110)
node('written','Written manager recommendation','Three questions; eight exportable figures','output',840,1320,380,
     [MANAGER+'/complete_report.md',MANAGER+'/README.md',MANAGER+'/figures/',MANAGER+'/render_research_figures.py',MANAGER+'/claim_audit.md',MANAGER+'/report_review.md'],
     '1. Improvement over RTE. 2. Uncertainty and failures. 3. Recommendation and risks. Distinguishes measured retrospective gains from inadequate cold-tail coverage, vintage uncertainty, exposed test history and national-to-portfolio limitations.',h=110)
node('interactive','Interactive manager report','Reviewed snapshot → report app','output',1240,1320,380,
     [MANAGER+'/prepare_manager_snapshot.py',MANAGER+'/reviewed_snapshot.json',MANAGER+'/app/',MANAGER+'/verification.json'],
     'Local report app presents frozen comparisons, diagnostics and inspectable provenance. Its hypothetical point-forecast penalty chart differs from the written report’s quantile-decision scenario. Neither establishes financial savings; neither is a production forecasting service.',h=110)

# Semantic edges are supported by exact source anchors, never numeric notebook prefixes.
link('raw','original_core','loads source observations','data',[evidence('src/data.py','def load_inputs')])
link('archives','limits','archive gate checks','workflow',[evidence('src/data.py','def availability_gate')],True)
link('contract','original_core','loads contract','code',[evidence('src/common.py','def load_contract')])
link('original_core','original_outputs','writes audit artifacts','data',[evidence('src/data.py','operational_gate.csv')])
link('research','assisted','ResearchRun subclass','code',[evidence('src/rte_assisted.py','class RTEAssistedRun(ResearchRun)')],True)
link('research','base_bank','reuses sealed base predictions','data',[evidence('src/rte_base_replay.py','def load_causal_base_bank')])
link('risk_spec','risk_engine','loads versioned configuration','workflow',[evidence('src/rte_risk_lab.py',"self.spec = json.loads")],True)
link('feature_views','learned','eligible features and labels','code',[evidence('src/rte_learned_features.py','features = data.features')],True)
link('feature_views','risk_engine','prepare_feature_lab','code',[evidence('src/rte_risk_lab.py','self._data = prepare_feature_lab')])
link('learned','risk_engine','merges learned feature bank','workflow',[evidence('src/rte_risk_lab.py','learned = build_learned_feature_bank')])
link('base_bank','risk_engine','merges causal base forecasts','workflow',[evidence('src/rte_risk_lab.py','base, ledger = load_causal_base_bank')],True)
link('risk_engine','point','development point selection','workflow',[evidence('src/rte_risk_lab.py','selected_point=winner')],True)
link('risk_engine','quantile','development WIS selection','workflow',[evidence('src/rte_risk_lab.py','selected_quantile=best_quantile')],True)
link('quantile','conformal','separate fixed predictor','workflow',[evidence('src/rte_risk_uncertainty.py','fixed_entry = deepcopy(entry)')],True)
link('risk_engine','sealed','saves sealed replay artifacts','data',[evidence('src/rte_risk_lab.py',"ledgerpath = self.path/'ledgers'")])
link('conformal','sealed','writes probabilistic forecasts','data',[evidence('src/rte_risk_uncertainty.py',"forecasts/probabilistic_test.csv.gz")])
link('sealed','scores','reads saved forecasts','data',[evidence('scripts/rte_risk_reporting.py','parts = [read_forecasts(path) for path in sources]')])
link('point','scores','matched point comparison','data',[evidence('scripts/rte_risk_reporting.py','primary = frozen.get')])
link('quantile','scores','separate probability identity','data',[evidence('scripts/rte_risk_reporting.py','probability_model = frozen.get')])
link('sealed','diagnostics','reads saved outcomes-matched forecasts','workflow',[evidence(MANAGER+'/diagnostics/build_diagnostics.py','forecasts/all_test.csv.gz')],True)
link('scores','interactive','builds reviewed report snapshot','data',[evidence(MANAGER+'/prepare_manager_snapshot.py',"scores=pd.read_csv")])
link('diagnostics','written','exports failure figures','workflow',[evidence(MANAGER+'/render_research_figures.py','diag=HERE/')],True)
link('diagnostics','interactive','adds descriptive report queries','data',[evidence(MANAGER+'/prepare_manager_snapshot.py',"('monthly','monthly_mae.csv'")])
link('scores','written','source-backed figures','data',[evidence(MANAGER+'/render_research_figures.py','RUN = PROJECT')])
link('checks','sealed','audits saved forecast artifacts','code',[evidence('scripts/check_rte_risk_artifacts.py','result.update(_sealed_forecasts(run, frozen))')])

link('research','scores','compares original saved policies','data',[evidence('scripts/rte_risk_reporting.py','sources = [root/spec["reference_run"]')])
link('assisted','scores','compares prior RTE-assisted policies','data',[evidence('scripts/rte_risk_reporting.py','root/spec["rte_reference_run"]')])
link('nb07','nb08','resumes the same research run','workflow',[evidence('scripts/build_research_notebooks.py','Run notebook 07 first.')],True)
for key, stem, *_ in stages:
    link('tooling',key,'authors original notebook','code',[evidence('scripts/build_notebooks.py','save("'+stem+'.ipynb"')])
for path,key in nb_ids.items():
    if key in ('nb07','nb08'):
        link('tooling',key,'authors research notebook','code',[evidence('scripts/build_research_notebooks.py','save("'+Path(path).name+'"')])

# Every imported project module is linked using the actual notebook cell and AST line.
module_groups = {}
for n in N:
    for p in n['paths']:
        if p.startswith('src/') and p.endswith('.py'):
            module_groups.setdefault(p, []).append(n['id'])
for path, source_id in nb_ids.items():
    nb = json.loads((ROOT/path).read_text())
    for cell_no, cell in enumerate(nb['cells'],1):
        if cell['cell_type'] != 'code':
            continue
        code = ''.join(cell['source'])
        try:
            tree = ast.parse(code)
        except SyntaxError:
            # Notebook magics are preserved; remove only magic lines for import extraction.
            clean = '\n'.join('' if line.lstrip().startswith(('%','!')) else line for line in code.splitlines())
            try: tree = ast.parse(clean)
            except SyntaxError: continue
        for stmt in ast.walk(tree):
            if isinstance(stmt,ast.ImportFrom) and stmt.module and stmt.module.startswith('src.'):
                target_path=stmt.module.replace('.','/')+'.py'
                for target_id in module_groups.get(target_path,[]):
                    link(source_id,target_id,'imports project module','code',[{'path':path,'cell':cell_no,'line':stmt.lineno,'excerpt':code.splitlines()[stmt.lineno-1].strip()}])

# Add exact I/O evidence only for paths assigned to a visible group.
for item in inv['io']:
    source = nb_ids.get(item['source'])
    if not source or item['relation'] not in ('reads','writes','checks_hash'):
        continue
    for n in N:
        if n['kind'] in ('notebook','module','test','tool'): continue
        if any(item['target']==p or (p.endswith('/') and item['target'].startswith(p)) for p in n['paths']):
            a,b = (n['id'],source) if item['relation']=='reads' else (source,n['id'])
            link(a,b,item['relation'],'data',[{'path':item['source'],'location':item['location'],'excerpt':item['target']}])

ids={n['id'] for n in N}
assert all(e['source'] in ids and e['target'] in ids for e in E)
for e in E:
    def label(r):
        where=f"cell {r['cell']}, line {r['line']}" if 'cell' in r else (f"line {r['line']}" if 'line' in r else r.get('location',''))
        return f"{r['path']}: {where}"
    e['evidence']='; '.join(label(r) for r in e['references'])
payload={'nodes':N,'edges':E,'panels':panels,'inventory':inv,'root':str(ROOT),
         'canvas':{'width':1660,'height':1480},'generated_at':datetime.now(timezone.utc).isoformat(),
         'scope':'Research architecture and saved evidence. No models executed; generated artifacts are grouped by purpose.',
         'limits':'RTE D1 availability is assumed; forecast vintages are unverified. Test data were previously explored. No production system or measured financial savings.',
         'notebook_count':len(nb_ids),'study_count':4}
(OUT/'architecture_overview.json').write_text(json.dumps(payload,indent=2)+'\n')
template=(OUT/'overview_template.html').read_text()
(OUT/'architecture.html').write_text(template.replace('__ARCHITECTURE_DATA__',json.dumps(payload).replace('</','<\\/')))
print(f'Overview: {len(N)} components, {len(E)} sourced relationships, {len(nb_ids)} notebooks, {len(files)} searchable files')
