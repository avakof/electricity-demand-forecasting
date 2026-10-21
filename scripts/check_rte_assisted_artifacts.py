"""Verify extension provenance and recompute headline losses independently."""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from datetime import datetime, timezone
import gzip
import json
import math
from pathlib import Path
import sys
from zoneinfo import ZoneInfo

import nbformat
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.common import fingerprint, forecast_origin, local_day_grid
from src.rte_assisted import RTEAssistedRun, model_key


def number(value):
    try:
        return float(value)
    except (ValueError, TypeError):
        return float('nan')


def utc(value):
    return datetime.fromisoformat(value).astimezone(timezone.utc)


def verify(run):
    result = {'status': 'passed', 'checked_at_utc': datetime.now(timezone.utc).isoformat(),
              'run_dir': str(run.path), 'errors': []}
    run.verify_manifest()
    frozen = json.loads((run.path/'frozen_selection.json').read_text())
    for name, expected in frozen['selection_inputs_sha256'].items():
        assert fingerprint(run.path/name) == expected, name
    expected = {model_key(f,a,m) for f in run.spec['families'] for a in run.spec['arms'] for m in run.spec['modes']}
    assert set(frozen['selections']) == expected
    trials = pd.read_csv(run.path/'tables/tuning_trials.csv')
    assert len(trials) == 128 and not trials.duplicated(['model','trial','fold']).any()
    for key, entry in frozen['selections'].items():
        t = trials.loc[trials.model.eq(key)]
        assert t.groupby('trial').fold.nunique().eq(4).all()
        pooled = t.assign(total_loss=t.MAE_MW*t.scored).groupby('trial').agg(loss=('total_loss','sum'), n=('scored','sum'))
        computed = pooled.loss/pooled.n
        assert int(computed.idxmin()) == entry['trial']
        assert math.isclose(computed.loc[entry['trial']], entry['MAE_MW'], abs_tol=1e-7)
    assert frozen['selected_assisted_model'] == min(frozen['selections'], key=lambda k:frozen['selections'][k]['MAE_MW'])
    assert frozen['selected_complete_policy'] == min(frozen['alternatives'], key=lambda x:x['MAE_MW'])['model']
    freeze_time = datetime.fromisoformat(frozen['frozen_utc']).timestamp()
    rows = ledgers = fits = failures = 0
    for path in sorted((run.path/'forecasts').glob('*__*.json')):
        meta = json.loads(path.read_text())
        stem = path.stem
        forecast = path.with_suffix('.csv.gz')
        ledger = run.path/'ledgers'/f'{stem}.jsonl.gz'
        fit = run.path/'fits'/f'{stem}.json'
        assert fingerprint(forecast) == meta['sha256']
        assert fingerprint(ledger) == meta['ledger_sha256']
        assert fingerprint(fit) == meta['fits_sha256']
        assert {'rte_at_origin_mw','rte_bias_recent_mw'}.issubset(meta['predictor_columns'])
        frame = pd.read_csv(forecast, low_memory=False)
        assert 'actual_mw' not in frame and len(frame) == meta['rows']
        assert not frame.duplicated(['origin','delivery_time']).any()
        frame['origin'] = pd.to_datetime(frame.origin, utc=True)
        frame['delivery_time'] = pd.to_datetime(frame.delivery_time, utc=True)
        for day, group in frame.groupby('local_day'):
            assert len(group) == len(local_day_grid(day))
            assert pd.DatetimeIndex(group.delivery_time).equals(local_day_grid(day))
            assert group.origin.eq(forecast_origin(day,run.contract)).all()
        if meta['phase'] == 'test' or meta['phase'] == 'fixed_calibration_and_test':
            assert getattr(ledger.stat(), 'st_birthtime', ledger.stat().st_mtime) >= freeze_time
        records = json.loads(fit.read_text())
        if meta['fixed_parameters']:
            assert len(records) == 1 and frame.fit_origin.nunique() == 1
        for record in records:
            fits += 1
            if record['status'] != 'fit':
                failures += 1
                continue
            assert pd.Timestamp(record['latest_label_available_at']) < pd.Timestamp(record['origin'])
            assert record['predictor_columns'] == meta['predictor_columns']
            assert record['mode'] == meta['mode']
        # Compare the pre-outcome daily ledger with the archived forecast values.
        indexed = frame.set_index(['origin','delivery_time'])
        count = 0
        with gzip.open(ledger, 'rt') as stream:
            for line in stream:
                event = json.loads(line)
                ledgers += 1
                for prediction in event['predictions']:
                    assert 'actual_mw' not in prediction
                    key = (pd.Timestamp(prediction['origin']), pd.Timestamp(prediction['delivery_time']))
                    saved = indexed.loc[key]
                    for name in ('prediction_mw','raw_prediction_mw'):
                        a, b = number(prediction.get(name)), number(saved[name])
                        assert (math.isnan(a) and math.isnan(b)) or math.isclose(a,b,abs_tol=1e-6), (stem,key,name)
                    count += 1
        assert count == len(frame)
        rows += len(frame)
    result.update(forecast_rows_checked=rows, daily_origin_ledgers_checked=ledgers,
                  fit_events_checked=fits, failed_fit_events=failures,
                  assisted_variants=len(expected), development_fold_replays=len(trials))
    # Independent arithmetic below does not call the project's metric functions.
    actual = {}
    with (ROOT/'data/france_electricity_hourly.csv').open() as stream:
        for row in csv.DictReader(stream):
            actual[utc(row['datetime_utc'])] = (number(row['load_mw']),number(row['temperature_c']))
    losses = defaultdict(list)
    counts = defaultdict(int)
    missing = defaultdict(int)
    with gzip.open(run.path/'forecasts/all_test.csv.gz','rt') as stream:
        for row in csv.DictReader(stream):
            y, _ = actual[utc(row['delivery_time'])]
            model, p = row['model'], number(row['prediction_mw'])
            counts[model] += 1
            if math.isfinite(y) and math.isfinite(p):
                losses[model].append(abs(y-p))
            elif not math.isfinite(y):
                missing[model] += 1
    mae = {k: math.fsum(v)/len(v) for k,v in losses.items()}
    saved = pd.read_csv(run.path/'tables/test_scorecard.csv').set_index('model')
    assert len(saved) == 43 and set(saved.index) == set(mae)
    for model in saved.index:
        assert math.isclose(mae[model],saved.loc[model,'MAE_MW'],abs_tol=1e-7)
        assert len(losses[model]) == 8759 and missing[model] == 1 and counts[model] == 8760
    selected = frozen['selected_assisted_model']
    cold = defaultdict(lambda: {'n':0,'covered':0,'upper':0,'severity_sum':0.,'days':set()})
    with gzip.open(run.path/'forecasts/uncertainty_test.csv.gz','rt') as stream:
        for row in csv.DictReader(stream):
            if row['model'] not in (selected,'rte_proxy','rte_bias_proxy'):
                continue
            delivery = utc(row['delivery_time'])
            local = delivery.astimezone(ZoneInfo('Europe/Paris'))
            y, temperature = actual[delivery]
            lo,hi = number(row['q05']),number(row['q95'])
            if local.hour not in (18,19,20) or not temperature<5 or not all(math.isfinite(x) for x in (y,lo,hi)):
                continue
            c = cold[(row['model'],row['method'])]
            c['n'] += 1; c['covered'] += lo<=y<=hi; c['upper'] += y>hi
            c['severity_sum'] += max(y-hi,0); c['days'].add(local.date())
    critical = pd.read_csv(run.path/'tables/cold_evening_scorecard.csv').set_index(['model','method'])
    computed_cold = []
    for key, c in cold.items():
        assert c['n'] == 66 and len(c['days']) == 24
        row = critical.loc[key]
        coverage, upper = c['covered']/c['n'], c['upper']/c['n']
        severity = c['severity_sum']/c['upper'] if c['upper'] else None
        assert math.isclose(coverage,row.coverage90,abs_tol=1e-9)
        assert math.isclose(upper,row.upper_exceedance_rate,abs_tol=1e-9)
        assert (math.isclose(severity,row.upper_exceedance_conditional_severity_MW,abs_tol=1e-6)
                if severity is not None else pd.isna(row.upper_exceedance_conditional_severity_MW))
        computed_cold.append({'model':key[0],'method':key[1],'n':c['n'],'days':len(c['days']),
                             'covered':c['covered'],'upper_misses':c['upper'],'coverage':coverage,
                             'upper_miss_rate':upper,'conditional_severity_MW':severity})
    notebooks = []
    for p in sorted((ROOT/'notebooks').glob('0[78]_*.ipynb')):
        n=nbformat.read(p,as_version=4); nbformat.validate(n)
        cells=[c for c in n.cells if c.cell_type=='code']
        assert all(c.execution_count is not None for c in cells)
        assert not any(o.output_type=='error' for c in cells for o in c.get('outputs',[]))
        assert any(c.metadata.get('rte_assisted_extension') for c in cells)
        notebooks.append({'name':p.name,'all_code_executed_without_errors':True})
    result.update(independent_test_MAE_MW={k:mae[k] for k in (selected,'rte_proxy','rte_bias_proxy')},
        independent_cold_checks=computed_cold, matched_test_hours=8759,missing_targets_per_policy=1,
        notebooks=notebooks, selected_assisted_model=selected,
        selected_complete_policy=frozen['selected_complete_policy'])
    return result


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path)
    args=parser.parse_args()
    run=RTEAssistedRun(ROOT,args.run_dir) if args.run_dir else RTEAssistedRun.latest(ROOT)
    try:
        result=verify(run)
    except Exception as exc:
        import traceback
        result={'status':'failed','errors':[f'{type(exc).__name__}: {exc}'], 'traceback':traceback.format_exc()}
    (run.path/'artifact_verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
    sys.exit(0 if result['status']=='passed' else 1)
