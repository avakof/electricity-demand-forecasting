"""Build the local report snapshot from frozen, reviewed research outputs only."""
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib
import pandas as pd
ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'reports/rte_risk/20261020T213640021260Z'
OUT=Path(__file__).parent
frozen=json.loads((BASE/'frozen_selection.json').read_text())
summary=json.loads((BASE/'result_summary.json').read_text())
primary=frozen['selected_point'] or frozen['diagnostic_best_point']
quantile=frozen['selected_quantile'];prior=summary['previous_development_selected_policy']
roles={'rte_proxy':'RTE','rte_bias_proxy':'Rolling correction',prior:'Prior RTE-assisted',primary:'Selected point',quantile:'Selected quantile'}
queries={}
def query(name,frame,source_files,components,definition):
    files=[str(p.relative_to(ROOT)) for p in source_files]
    rows=json.loads(frame.to_json(orient='records',date_format='iso'))
    queries[name]={'rows':rows,'source':{'label':name.replace('_',' ').title(),'provider':'Reviewed local research artifacts','files':files,'dataClassification':'observed retrospective evaluation','period':{'start':'2025-07-01','end':'2026-06-30'},'executedAt':datetime.now(timezone.utc).isoformat(),'metricDefinitions':[{'label':name.replace('_',' ').title(),'definition':definition,'componentIds':components,'sourceLineage':[{'files':files}]}],'evidenceFlow':[{'title':'Frozen source artifacts','detail':'Read saved CSV/JSON files without training, retuning, or changing source data. SHA256: '+json.dumps({p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files})},{'title':'Reviewed transformation','detail':'Select declared development-frozen roles; preserve original numeric values, units, nulls, sample counts and model identities. Any displayed percentages are original fractions multiplied by 100.'}],'caveats':['The historical test had already been explored. RTE issue vintages and original demand revisions are unverified. Results are retrospective, not verified operational performance.']},'methods':[{'language':'python','code':f'Run reports/manager_report_20261021/prepare_manager_snapshot.py; query {name}. Exact source files and hashes are recorded above.'}]}
scores=pd.read_csv(BASE/'tables/test_scorecard.csv');scores['role']=scores.model.map(roles).fillna('Other policy')
focus=scores.loc[scores.model.isin(roles)].copy();focus['role_order']=focus.model.map({m:i for i,m in enumerate(roles)});focus=focus.sort_values('role_order')
query('point_comparison',focus,[BASE/'tables/test_scorecard.csv',BASE/'frozen_selection.json',BASE/'result_summary.json'],['summary','point-comparison','cold-comparison','comparison-table','q1-findings'],'MAE is mean absolute error in MW on exactly 8,759 common eligible hours. Weighted MAE gives realized temperature below 5°C at local hours 18,19,20 weight 5 and other hours weight 1, normalized by total weight. Cold MAE uses 66 hours across 24 days.')
query('all_policies',scores,[BASE/'tables/test_scorecard.csv'],['policy-table'],'All 98 policies on the same eligible forecast origin/delivery pairs. Model names distinguish original independent, earlier RTE-assisted and 55 new feature/risk/hybrid policies. No test-based winner selection.')
b=pd.read_csv(BASE/'tables/bootstrap.csv');b=b.loc[b.model.eq(primary)&b.reference.isin(roles)].copy();b['comparison']=b.reference.map(roles);b['metric_label']=b.metric.map({'MAE_MW':'All-hour MAE','weighted_MAE_MW':'Weighted MAE','cold_MAE_MW':'Cold-evening MAE'})
query('paired_uncertainty',b,[BASE/'tables/bootstrap.csv'],['bootstrap-table','q1-inference','summary'],'Paired daily loss differences; contiguous 7,14,28-day block bootstrap, 500 resamples, nominal 95% confidence intervals. Positive absolute improvement means reference loss minus selected-point loss, MW. Cold support is limited; blocks preserve temporal dependence approximately.')
p=pd.read_csv(BASE/'tables/uncertainty_segments.csv');p=p.loc[(p.segment_type.eq('all'))|(p.segment_type.eq('cold_evening')&p.segment.astype(str).eq('True'))].copy()
emp='past_calibration_empirical_residual_distribution'
methods=[('rte_proxy',emp,'RTE residuals'),('rte_bias_proxy',emp,'Correction residuals'),(primary,emp,'Selected-point residuals'),(quantile,'native_quantile_lightgbm','Native quantiles'),(quantile+'__fixed_engineered','fixed_engineered_native_quantiles','Fixed native'),(quantile+'__fixed_engineered','fixed_engineered_split_conformal','Split conformal'),(quantile+'__fixed_engineered','fixed_engineered_CQR','CQR')]
parts=[]
for model,method,label in methods:
    x=p.loc[p.model.eq(model)&p.method.eq(method)].copy();x['display']=label;x['scope']=x.segment_type.map({'all':'All hours','cold_evening':'Cold evenings'});x['method_order']=len(parts)
    parts.append(x)
p=pd.concat(parts,ignore_index=True)
query('probability',p,[BASE/'tables/uncertainty_segments.csv',BASE/'tables/uncertainty_protocol.json'],['coverage-chart','upper-miss-chart','interval-table','proper-score-table','interval-availability','q2-findings','summary','risk-register'],'Central interval endpoints q05/q95 have nominal 90% coverage and a 5% upper-tail reference. Asymmetric endpoints q075/q975 allocate 7.5% below and 2.5% above, also nominal 90%. Fixed split/CQR intervals are calibrated bounds, not a fully calibrated conditional CDF. All references are targets, not established guarantees.')
a=pd.read_csv(BASE/'tables/feature_ablation.csv');a['arm_label']=a.arm.map({'national':'National','regional_only':'Regional only','combined':'Combined'});a['tier_label']=a.feature_tier.map({'basic':'Basic','engineered':'Engineered','stacked':'+ learned'});a['weight_label']=a.training_cold_weight_cap.map({1:'Uniform weights',5:'Forecast-cold weights'})
query('ablation',a,[BASE/'tables/feature_ablation.csv'],['ablation-chart','ablation-table','q1-features'],'Predeclared LightGBM residual ablation across three predictor arms, three feature tiers and weight caps 1/5. Every variant has the same national target and evaluation weights. Each configuration is tuned in chronological development folds; differences include that selected configuration.')
r=pd.read_csv(BASE/'tables/asymmetric_risk_scenarios.csv');r=r.loc[r.model.isin(roles)].copy();r['role']=r.model.map(roles)
query('risk_scenarios',r,[BASE/'tables/asymmetric_risk_scenarios.csv'],['risk-chart','q3-value'],'Normalized hypothetical asymmetric MW loss for unchanged point forecasts. Shortage/surplus ratios 1,3,9,19 and cold weights 1,3,5 are assumptions, not actual prices or euro savings.')
variants=json.loads((BASE/'variant_register.json').read_text());scope=pd.DataFrame([{'family':v['family'],'arm':v['arm'],'feature_tier':v['tier'],'mode':v['mode'],'selected_parameters':json.dumps(frozen['selections'][k]['params'],sort_keys=True),'cold_weight':v['cold_weight'],'base_model':v.get('base'),'model':k,'kind':'Hybrid meta-model' if v.get('base') else 'Direct tabular candidate/control'} for k,v in variants.items()])
query('model_scope',scope,[BASE/'variant_register.json',BASE/'manifest.json',BASE/'frozen_selection.json'],['scope-table','methods-models','risk-register'],'55 new policies: standalone tabular variants, feature/weight ablations, direct/residual comparisons, 23 hybrids using causally selected archived base forecasts, and two matching hybrid controls. These are not native exogenous retraining of every neural/statistical architecture.')
g=pd.read_csv(BASE/'tables/development_fit_gaps.csv');g=g.loc[g.model.eq(primary)].copy()
query('fit_diagnostics',g,[BASE/'tables/development_fit_gaps.csv'],['fit-gap-table','fit-gap-note'],'Same fitted model: last up-to-2000 training rows versus following chronological development fold. The gap mixes distribution shift and fit/generalization effects; it is not a formal bias-variance decomposition.')
for name,filename,components in [('monthly','monthly_mae.csv',['monthly-chart','seasonal-findings']),('hourly','local_hour_mae.csv',['hourly-chart']),('daily','worst_five_days.csv',['worst-days-table']),('worst_cold','worst_cold_day_hourly.csv',['worst-day-chart','q2-failure'])]:
    path=OUT/'diagnostics'/filename
    if path.exists():
        frame=pd.read_csv(path)
        query(name,frame,[path],components,'Reporting-only diagnostics calculated from saved forecast origins/delivery intervals and provided truth. Original model fits and selections are unchanged. Missing targets remain excluded; counts and complete-day flags are retained.')
findings_path=OUT/'diagnostics/findings.json'
if findings_path.exists():
    findings=json.loads(findings_path.read_text())
    bias_rows=pd.DataFrame([{'segment':'All hours',**findings['overall']['selected']},{'segment':'Cold evenings',**findings['cold']['selected']}])
    query('bias_diagnosis',bias_rows,[findings_path],['bias-findings'],'Signed bias is mean(prediction minus actual), MW. Negative values mean average underforecasting. All-hour and cold-evening populations overlap; do not add their counts. Underforecast rate is fraction of matched observations with actual above prediction.')
    support=findings['support'];incomplete=support['incomplete_local_days'][0]
    support_rows=pd.DataFrame([{'matched_scored_hours':support['matched_hours'],'excluded_missing_targets':support['excluded_missing_targets'],'excluded_missing_policy_predictions':support['excluded_missing_policy_predictions'],'incomplete_local_day':incomplete['local_day'],'observed_hours_on_incomplete_day':incomplete['n'],'expected_hours_on_incomplete_day':incomplete['expected_hours']}])
    query('evaluation_support',support_rows,[findings_path],['methods-models'],'Exact matched forecast support and missing-target exclusion from diagnostic reconstruction. The incomplete local DST day has24observedtargets outof25expectedhourly intervals. Targets were not interpolated.')
snapshot={'surface':'report','title':'RTE-assisted models halve historical error, but winter risk remains unresolved','generatedAt':datetime.now(timezone.utc).isoformat(),'status':'reviewed','buildStatus':'creating','filters':[],'report':{'asOf':'2026-06-30','audience':'Head of trading'},'queries':queries,'researchSummary':summary,'roles':roles,'frozenSelection':{'point':primary,'quantile':quantile,'prior':prior},'reportSources':{'run':str(BASE.relative_to(ROOT))}}
path=OUT/'reviewed_snapshot.json';path.write_text(json.dumps(snapshot,indent=2,allow_nan=False)+'\n')
appdata=OUT/'app/src/data.json'
if appdata.exists():
    existing=json.loads(appdata.read_text());snapshot['id']=existing['id'];snapshot['buildStatus']=existing.get('buildStatus','creating');appdata.write_text(json.dumps(snapshot,indent=2,allow_nan=False)+'\n')
print(path)
