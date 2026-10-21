"""Reporting-only failure cases for the development-frozen complete policy.

Call after final point/peak scorecards exist. Outcomes are joined only in memory
for diagnosis, then written to reports/tables and reports/figures. No fitting,
selection changes, forecast rewriting or weather-forecast attribution occurs.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
METHOD = 'fixed_past_out_of_sample_residual_distribution'
CAUSE = 'Unknown; observed load, temperature and calendar alone do not identify a cause'


def _grid(day, timezone):
    start = pd.Timestamp(day).normalize().tz_localize(timezone)
    end = (pd.Timestamp(day).normalize()+pd.DateOffset(days=1)).tz_localize(timezone)
    return pd.date_range(start.tz_convert('UTC'), end.tz_convert('UTC'), freq='h', inclusive='left')


def _load_forecasts(path):
    frame = pd.read_csv(path, low_memory=False)
    for name in ('origin', 'delivery_time'):
        frame[name] = pd.to_datetime(frame[name], utc=True)
    return frame


def _measured_days(frame, country, timezone, evening_hours, cold_threshold, holidays):
    """Measure errors and contemporaneous observations; never infer causes."""
    source = country.copy()
    source['local_day'] = source.delivery_time.dt.tz_convert(timezone).dt.strftime('%Y-%m-%d')
    means = source.groupby('local_day').actual_mw.mean()
    observed = source.groupby('local_day').actual_mw.count()
    complete = {day: int(observed.loc[day]) == len(_grid(day, timezone)) for day in means.index}
    rows = []
    for day, group in frame.groupby('local_day', sort=True):
        group = group.sort_values('delivery_time')
        expected = _grid(day, timezone)
        regular = pd.DatetimeIndex(group.delivery_time).equals(expected)
        valid = np.isfinite(group.actual_mw) & np.isfinite(group.prediction_mw)
        point = group.loc[valid]
        error = point.actual_mw-point.prediction_mw
        full = regular and int(valid.sum()) == len(expected)
        interval = np.isfinite(group.actual_mw) & np.isfinite(group.q05) & np.isfinite(group.q95)
        evening = group.local_hour.isin(evening_hours)
        cold = evening & group.temperature_c.lt(cold_threshold)
        upper_excess = (group.actual_mw-group.q95).clip(lower=0)
        cold_excess = upper_excess.loc[cold & interval]
        evening_excess = upper_excess.loc[evening & interval]
        evening_point = (group.actual_mw-group.prediction_mw).loc[evening & valid]
        previous = (pd.Timestamp(day)-pd.Timedelta(days=1)).strftime('%Y-%m-%d')
        change = means.get(day, np.nan)-means.get(previous, np.nan) if complete.get(day) and complete.get(previous) else np.nan
        peak_error = float(group.actual_mw.max()-group.prediction_mw.max()) if full else np.nan
        date = pd.Timestamp(day)
        rows.append({
            'local_day': day, 'origin': str(group.origin.iloc[0]),
            'expected_hours': len(expected), 'scored_hours': int(valid.sum()),
            'interval_scored_hours': int(interval.sum()), 'complete_delivery_grid': regular,
            'full_point_day': full, 'full_interval_day': regular and int(interval.sum()) == len(expected),
            'MAE_MW': float(error.abs().mean()),
            'signed_daily_energy_error_actual_minus_policy_MWh': float(error.sum()) if full else np.nan,
            'absolute_daily_energy_error_MWh': float(abs(error.sum())) if full else np.nan,
            'observed_actual_peak_MW': group.actual_mw.max(),
            'policy_peak_MW': group.prediction_mw.max(),
            'daily_peak_error_actual_minus_policy_MW': peak_error,
            'absolute_daily_peak_error_MW': abs(peak_error),
            'recorded_temperature_observations': int(group.temperature_c.notna().sum()),
            'recorded_temperature_mean_c': group.temperature_c.mean(),
            'recorded_temperature_min_c': group.temperature_c.min(),
            'cold_evening_observed_hours': int((cold & group.actual_mw.notna()).sum()),
            'cold_interval_scored_hours': int((cold & interval).sum()),
            'cold_upper_miss_hours': int(cold_excess.gt(0).sum()),
            'cold_upper_miss_max_MW': float(cold_excess.max()) if len(cold_excess) else np.nan,
            'cold_upper_miss_mean_when_positive_MW': float(cold_excess.loc[cold_excess.gt(0)].mean()),
            'evening_upper_miss_max_MW': float(evening_excess.max()) if len(evening_excess) else np.nan,
            'evening_underprediction_max_MW': float(evening_point.clip(lower=0).max()) if len(evening_point) else np.nan,
            'evening_absolute_error_max_MW': float(evening_point.abs().max()) if len(evening_point) else np.nan,
            'mean_actual_load_change_vs_previous_day_MW': float(change),
            'weekday': date.day_name(), 'weekend': date.dayofweek >= 5,
            'national_public_holiday': date.date() in holidays,
            'raw_RTE_available_hours': int(group.raw_rte_mw.notna().sum()),
            'policy_fallback_hours': int(group.fallback_used.fillna(False).astype(bool).sum()) if 'fallback_used' in group else np.nan,
            'cause_classification': CAUSE,
            'weather_evidence': 'Recorded temperature only; archived forecast-weather error is unavailable',
        })
    return pd.DataFrame(rows)


def _select_cases(daily):
    """Select distinct diagnostic days; explicitly retain rank when criteria overlap."""
    selected, used = [], set()

    def choose(case_id, criterion, description, positive=False):
        ranked = daily.loc[np.isfinite(daily[criterion])].copy()
        if positive:
            ranked = ranked.loc[ranked[criterion].gt(0)]
        ranked = ranked.sort_values([criterion, 'local_day'], ascending=[False, True]).reset_index(drop=True)
        if ranked.empty:
            return False
        candidates = ranked.loc[~ranked.local_day.isin(used)]
        if candidates.empty:
            return False
        choice = candidates.iloc[0].to_dict()
        rank = int(candidates.index[0])+1
        choice.update(case_id=case_id, criterion=criterion, criterion_rank=rank,
                      criterion_global_worst_day=ranked.local_day.iloc[0], diagnostic=description,
                      selection_note='Overall worst for this criterion' if rank == 1 else
                      f'Highest-ranked unused day (rank {rank}); the overall worst day is already another case')
        selected.append(choice); used.add(choice['local_day'])
        return True

    if not choose('worst_policy_MAE', 'MAE_MW', 'Daily policy MAE, retaining actual observation counts'):
        raise RuntimeError('No observed point errors are available for a failure catalogue')
    if not choose('worst_absolute_daily_energy', 'absolute_daily_energy_error_MWh',
                  'Absolute daily energy error on complete days; hourly-mean interpretation assumed'):
        choose('daily_energy_unavailable', 'MAE_MW', 'Fallback: another MAE day because complete-day energy is unavailable')
    if not choose('worst_cold_evening_upper_miss', 'cold_upper_miss_max_MW',
                  'Largest realized-cold evening upper-bound miss; temperature is retrospective diagnosis', positive=True):
        if not choose('fallback_evening_upper_miss', 'evening_upper_miss_max_MW',
                      'Fallback: evening upper-bound miss; no unused realized-cold upper-miss day exists', positive=True):
            if not choose('fallback_evening_underprediction', 'evening_underprediction_max_MW',
                          'Fallback: evening underprediction; no unused evening upper-bound miss exists', positive=True):
                choose('fallback_evening_absolute_error', 'evening_absolute_error_max_MW',
                       'Fallback: evening absolute error; no unused upper miss or underprediction exists')
    if len(selected) != 3:
        raise RuntimeError('Fewer than three distinct eligible diagnostic days; report this limitation explicitly')
    return pd.DataFrame(selected)


def _plot_case(frame, case, destination, policy, timezone, evening_hours, cold_threshold):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.dates import DateFormatter
    from matplotlib.figure import Figure

    group = frame.loc[frame.local_day.eq(case.local_day)].sort_values('delivery_time').copy()
    times = group.delivery_time.dt.tz_convert(timezone)
    x = pd.DatetimeIndex(times).to_pydatetime()
    figure = Figure(figsize=(12, 7), layout='constrained')
    FigureCanvasAgg(figure)
    load_ax, temp_ax = figure.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios': [3, 1]})
    load_ax.plot(x, group.actual_mw, color='black', lw=1.8, label='Observed national demand')
    load_ax.plot(x, group.raw_rte_mw, color='#ad4b42', ls='--', label='Supplied RTE, raw values (gaps retained)')
    load_ax.plot(x, group.prediction_mw, color='#176d83', label=f'{policy}, complete policy')
    load_ax.fill_between(x, group.q05.to_numpy(dtype=float), group.q95.to_numpy(dtype=float),
                         color='#176d83', alpha=.18, label='Nominal 90% past-error interval')
    load_ax.plot(x, group.q95, color='#176d83', lw=.8, ls=':')
    wall_start = pd.Timestamp(case.local_day)
    first_evening = True; first_cold = True
    for hour in evening_hours:
        left = (wall_start+pd.Timedelta(hours=hour)).tz_localize(timezone)
        right = left+pd.Timedelta(hours=1)
        for axis in (load_ax, temp_ax):
            axis.axvspan(left, right, color='#cb9c30', alpha=.13,
                         label='18:00–20:00 delivery intervals' if first_evening and axis is load_ax else None)
        first_evening = False
    for timestamp, actual_temp, hour in zip(times, group.temperature_c, group.local_hour):
        if hour in evening_hours and pd.notna(actual_temp) and actual_temp < cold_threshold:
            for axis in (load_ax, temp_ax):
                axis.axvspan(timestamp, timestamp+pd.Timedelta(hours=1), color='#416cac', alpha=.12,
                             label='Recorded cold evening (diagnostic)' if first_cold and axis is load_ax else None)
            first_cold = False
    cold = group.local_hour.isin(evening_hours) & group.temperature_c.lt(cold_threshold)
    misses = cold & group.actual_mw.gt(group.q95)
    if misses.any():
        load_ax.scatter(np.asarray(x)[misses.to_numpy()], group.loc[misses, 'actual_mw'],
                        marker='^', color='#a82d36', zorder=5, label='Cold-evening actual above upper bound')
    load_ax.set(title=f"{case.local_day}: {case.case_id.replace('_', ' ')} (criterion rank {int(case.criterion_rank)})",
                ylabel='Demand (MW)')
    load_ax.legend(loc='best', fontsize=8, ncol=2)
    temp_ax.plot(x, group.temperature_c, color='#8e4c32', label='Recorded temperature')
    temp_ax.axhline(cold_threshold, color='#416cac', ls='--', lw=.8, label=f'Cold threshold {cold_threshold:g} °C')
    temp_ax.set(ylabel='Temperature (°C)', xlabel=f'{timezone} local time; recorded weather is retrospective')
    temp_ax.legend(loc='best', fontsize=8)
    template = '%H:%M\n%z' if int(case.expected_hours) != 24 else '%H:%M'
    temp_ax.xaxis.set_major_formatter(DateFormatter(template, tz=ZoneInfo(timezone)))
    for axis in (load_ax, temp_ax):
        axis.grid(alpha=.2)
        axis.spines[['top', 'right']].set_visible(False)
    figure.savefig(destination, dpi=150)
    figure.clear()


def analyze_failures(run_dir, root=ROOT):
    """Create measured tables/figures after final scores, without changing policy."""
    run_dir, root = Path(run_dir).resolve(), Path(root).resolve()
    required = ('development_policy_selection.json', 'manifest.json', 'tables/test_scorecard.csv',
                'tables/peak_risk_scorecard.csv', 'forecasts/all_test.csv.gz',
                'forecasts/uncertainty_test.csv.gz')
    for name in required:
        if not (run_dir/name).is_file():
            raise RuntimeError(f'Final analysis is not ready: {run_dir/name}')
    policy_spec = json.loads((run_dir/'development_policy_selection.json').read_text())
    policy = policy_spec['selected_policy']
    contract = json.loads((run_dir/'manifest.json').read_text())['contract']
    timezone = contract['timezone']
    peak_path = run_dir/'peak_risk_specification.json'
    peak = json.loads(peak_path.read_text()) if peak_path.exists() else {}
    evening_hours = peak.get('evening_hours_local', contract['cold_evening']['hours_local'])
    cold_threshold = peak.get('cold_threshold_c', contract['cold_evening']['temperature_below_c'])
    predictions = _load_forecasts(run_dir/'forecasts/all_test.csv.gz')
    model = predictions.loc[predictions.model.eq(policy)].copy()
    rte = predictions.loc[predictions.model.eq('rte_proxy'), ['origin', 'delivery_time', 'raw_prediction_mw']].rename(columns={'raw_prediction_mw': 'raw_rte_mw'})
    if model.empty or rte.empty or model.duplicated(['origin', 'delivery_time']).any():
        raise RuntimeError('Frozen complete policy or unique RTE reference is missing')
    intervals = _load_forecasts(run_dir/'forecasts/uncertainty_test.csv.gz')
    intervals = intervals.loc[intervals.model.eq(policy) & intervals.method.eq(METHOD),
                              ['origin', 'delivery_time', 'q05', 'q50', 'q95']]
    if intervals.empty:
        raise RuntimeError('Predeclared complete-policy residual intervals are missing')
    model = model.drop(columns=['q05', 'q50', 'q95'], errors='ignore').merge(
        intervals, on=['origin', 'delivery_time'], how='left', validate='one_to_one')
    country = pd.read_csv(root/'data/france_electricity_hourly.csv',
                          usecols=['datetime_utc', 'load_mw', 'temperature_c'])
    country = country.rename(columns={'datetime_utc': 'delivery_time', 'load_mw': 'actual_mw'})
    country['delivery_time'] = pd.to_datetime(country.delivery_time, utc=True)
    if country.delivery_time.duplicated().any():
        raise RuntimeError('Duplicate country targets cannot be silently aggregated')
    frame = model.merge(rte, on=['origin', 'delivery_time'], how='left', validate='one_to_one').merge(
        country, on='delivery_time', how='left', validate='many_to_one')
    local = frame.delivery_time.dt.tz_convert(timezone)
    frame['local_day'], frame['local_hour'] = local.dt.strftime('%Y-%m-%d'), local.dt.hour
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from src.common import holiday_dates
    holidays = set().union(*(holiday_dates(year) for year in local.dt.year.unique()))
    daily = _measured_days(frame, country, timezone, evening_hours, cold_threshold, holidays)
    cases = _select_cases(daily)
    cases['model'], cases['interval_method'] = policy, METHOD
    tables, figures = run_dir/'tables', run_dir/'figures/failure_cases'
    tables.mkdir(exist_ok=True); figures.mkdir(parents=True, exist_ok=True)
    figure_paths = []
    for _, case in cases.iterrows():
        destination = figures/f"{case.case_id}__{case.local_day}.png"
        _plot_case(frame, case, destination, policy, timezone, evening_hours, cold_threshold)
        figure_paths.append(destination)
    cases['figure'] = [str(path.relative_to(run_dir)) for path in figure_paths]
    daily.to_csv(tables/'complete_policy_daily_failure_metrics.csv', index=False)
    cases.to_csv(tables/'complete_policy_failure_cases.csv', index=False)
    report_rows = []
    for row in cases.itertuples():
        energy = f'{row.absolute_daily_energy_error_MWh:,.0f}' if pd.notna(row.absolute_daily_energy_error_MWh) else 'not available'
        cold_miss = (f'{row.cold_upper_miss_max_MW:,.0f}' if pd.notna(row.cold_upper_miss_max_MW) else
                     'not applicable' if row.cold_evening_observed_hours == 0 else 'not available')
        report_rows.append(f"| {row.local_day} | {row.case_id.replace('_', ' ')} | {row.scored_hours}/{row.expected_hours} | "
                           f"{row.MAE_MW:,.0f} | {energy} | {cold_miss} |")
    rows = '\n'.join(report_rows)
    report = f"""# Measured failure cases: {policy}

Policy identity comes from development_policy_selection.json; these cases do not select or retune a policy. Bounds use the predeclared {METHOD}.

| Local day | Diagnostic case | Scored/expected hours | MAE MW | Absolute daily energy error MWh | Largest cold upper miss MW |
|---|---|---:|---:|---:|---:|
{rows}

Each criterion's global worst day and the chosen day's rank are saved. If criteria overlap, the next highest-ranked unused day supplies a distinct case; a cold-event fallback is labeled explicitly. Energy and daily peak errors require complete targets/predictions and assume hourly mean power. Missing targets remain missing. Temperature, holiday status, preceding-day changes and errors are measured observations, not verified causes. No archived forecast weather exists here to establish a weather-forecast miss. Cold shading uses realized temperature after delivery. Causes remain unknown; no financial savings are inferred.
"""
    report_path = run_dir/'failure_cases.md'; report_path.write_text(report)
    return {'catalogue': cases, 'daily_metrics': daily, 'figure_paths': figure_paths,
            'report_path': report_path}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--root', type=Path, default=ROOT)
    args = parser.parse_args(argv)
    run = args.run_dir if args.run_dir.is_absolute() else args.root/args.run_dir
    result = analyze_failures(run, args.root)
    print(result['catalogue'][['case_id', 'local_day', 'criterion_rank', 'scored_hours', 'expected_hours', 'figure']].to_string(index=False))
    print(result['report_path'])
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
