"""Quantile grid and fixed-predictor calibration for the RTE risk study."""
from __future__ import annotations

from copy import deepcopy
import numpy as np
import pandas as pd

from .common import forecast_origin, write_json
from .research_scores import conformal_radius

Q_NAMES = {.05:'q05', .075:'q075', .25:'q25', .5:'q50', .75:'q75', .9:'q90', .95:'q95', .975:'q975'}


def pinball(y, prediction, q):
    error = np.asarray(y)-np.asarray(prediction)
    return np.maximum(q*error, (q-1)*error)


def interval_score(y, lo, hi, alpha):
    y, lo, hi = map(np.asarray, (y,lo,hi))
    return hi-lo+(2/alpha)*np.maximum(lo-y,0)+(2/alpha)*np.maximum(y-hi,0)


def quantile_scores(predictions, labels):
    f = predictions.merge(labels[['delivery_time','actual_mw']], on='delivery_time', validate='many_to_one')
    f = f.dropna(subset=['actual_mw','q05','q25','q50','q75','q95'])
    y = f.actual_mw.to_numpy()
    wis = (.5*np.abs(y-f.q50)+.05*interval_score(y,f.q05,f.q95,.1)+.25*interval_score(y,f.q25,f.q75,.5))/2.5
    return {'WIS_MW':float(wis.mean()), 'n':len(f),
            'coverage_90':float(((y>=f.q05)&(y<=f.q95)).mean()),
            'width_90_MW':float((f.q95-f.q05).mean()),
            'upper_exceedance_rate':float((y>f.q95).mean())}


def assemble_quantiles(run, entry, start, end, phase, fixed=False):
    out = None
    validity = []
    for q in run.spec['quantiles']:
        p = run.replay(entry['variant'], entry['params'], start, end, phase, q=q, fixed=fixed)
        if out is None:
            out = p[['model','origin','delivery_time','local_day','fit_origin']].copy()
        else:
            assert p[['origin','delivery_time']].equals(out[['origin','delivery_time']])
        out[Q_NAMES[q]] = p.raw_prediction_mw.to_numpy()
        validity.append(p.raw_prediction_mw.notna().to_numpy())
    names = [Q_NAMES[q] for q in run.spec['quantiles']]
    values = out[names].to_numpy()
    out['quantile_crossing_repaired'] = (np.diff(values,axis=1)<0).any(axis=1)
    # Increasing rearrangement is predeclared and independent of outcomes.
    out[names] = np.sort(values, axis=1)
    out.loc[~np.column_stack(validity).all(axis=1), names] = np.nan
    out['method'] = 'native_quantile_lightgbm'
    return out


def signed_quantile(scores, alpha):
    values = np.sort(np.asarray(scores,dtype=float))
    values = values[np.isfinite(values)]
    rank = int(np.ceil((len(values)+1)*(1-alpha)))
    return float(values[rank-1]) if 0<rank<=len(values) else np.inf


def uncertainty_replay(run):
    frozen = run.freeze()
    entry = frozen['selections'][frozen['selected_quantile']]
    c,t = run.spec['calibration'],run.spec['test']
    native = assemble_quantiles(run,entry,t['start'],t['end'],'native_quantile_test')
    # Fix the ENTIRE fitted forecast mapping: remove learned, monthly-updated
    # feature generators from this separately labelled comparator. Original
    # engineered histories are deterministic functions of each eligible snapshot.
    fixed_entry = deepcopy(entry)
    fixed_entry['variant'].update(model=entry['variant']['model']+'__fixed_engineered',tier='engineered')
    continuous = assemble_quantiles(run,fixed_entry,c['start'],t['end'],'fixed_probability_calibration_test',fixed=True)
    assert continuous.fit_origin.nunique() == 1
    cutoff = forecast_origin(t['start'],run.contract)
    cal = continuous.loc[continuous.local_day.ge(c['start']) & continuous.local_day.lt(c['end'])].merge(
        run.data.labels[['delivery_time','actual_mw','label_available_at']],on='delivery_time',validate='many_to_one')
    cal = cal.loc[cal.label_available_at.lt(cutoff)].dropna(subset=['actual_mw','q05','q50','q95','q075','q975'])
    test = continuous.loc[continuous.local_day.ge(t['start']) & continuous.local_day.lt(t['end'])].copy()
    abs_radius = conformal_radius(np.abs(cal.actual_mw-cal.q50),alpha=.1)
    cqr_delta = max(0.,signed_quantile(np.maximum(cal.q05-cal.actual_mw,cal.actual_mw-cal.q95),.1))
    lower_delta = max(0.,signed_quantile(cal.q075-cal.actual_mw,.075))
    upper_delta = max(0.,signed_quantile(cal.actual_mw-cal.q975,.025))
    raw_fixed = test.copy()
    raw_fixed['method'] = 'fixed_engineered_native_quantiles'
    split = test.copy()
    split['method'] = 'fixed_engineered_split_conformal'
    split['q05'],split['q95'] = split.q50-abs_radius,split.q50+abs_radius
    # This comparator has only a median and one calibrated90% interval. Do not
    # borrow unrelated native endpoints and call the result one coherent CDF.
    split[['q075','q25','q75','q90','q975']] = np.nan
    cqr = test.copy()
    cqr['method'] = 'fixed_engineered_CQR'
    cqr['q05'],cqr['q95'] = cqr.q05-cqr_delta,cqr.q95+cqr_delta
    cqr['q075'],cqr['q975'] = cqr.q075-lower_delta,cqr.q975+upper_delta
    cqr[['q25','q75','q90']] = np.nan
    # Calibrated endpoints represent coverage bounds, not nominal conditional
    # quantiles; report interval scores/coverage, not native quantile optimality.
    result = pd.concat([native,raw_fixed,split,cqr],ignore_index=True)
    result.to_csv(run.path/'forecasts/probabilistic_test.csv.gz',index=False)
    write_json(run.path/'tables/uncertainty_protocol.json',{
        'selected_quantile_model':frozen['selected_quantile'],'calibration_n':len(cal),
        'calibration_latest_available_at':str(cal.label_available_at.max()),'first_test_origin':str(cutoff),
        'fixed_fit_origin':str(continuous.fit_origin.iloc[0]),'split_radius_MW':abs_radius,
        'CQR_central_expansion_MW':cqr_delta,'CQR_lower_expansion_MW':lower_delta,'CQR_upper_expansion_MW':upper_delta,
        'fixed_comparator':'Separate engineered-only predictor, all fitted parameters frozen before calibration; no learned-feature monthly updates',
        'native_asymmetric_interval':'q.075 to q.975 has nominal90% mass:7.5% below and2.5% above',
        'coverage_guarantee':'No exchangeability or conditional coverage guarantee established for this dependent series',
        'selection':'All methods predeclared; none selected or retuned using test outcomes'})
    return result
