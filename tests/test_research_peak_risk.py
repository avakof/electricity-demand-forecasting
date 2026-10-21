"""Behavioral checks for the frozen cold-evening uncertainty extension."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import numpy as np
import pandas as pd

from src.common import forecast_origin, local_day_grid, project_root
from src.research_data import load_research_contract
from src.research_peak_risk import (
    SPEC, _choose_pool, _cost_sensitivity, _decorate, _intervals, _summary,
    analyze_peak_risk, freeze_peak_spec, select_peak_intervals,
)


class PeakRiskTests(unittest.TestCase):
    def setUp(self):
        self.contract = deepcopy(load_research_contract(project_root()))

    def pool(self, days=50, cold_days=0):
        rows = []
        for day in pd.date_range('2024-01-01', periods=days):
            for hour in range(24):
                evening = hour in (18, 19, 20)
                rows.append({'local_day': day.strftime('%Y-%m-%d'), 'residual_mw': float(hour),
                             'evening': evening,
                             'precutoff_coldrisk_evening': evening and (day.dayofyear <= cold_days)})
        return pd.DataFrame(rows)

    def test_sparse_cold_pool_explicitly_falls_back_to_evening(self):
        pool = self.pool(days=50, cold_days=5)
        values, meta = _choose_pool(pool, cold_proxy=True, evening=True)
        self.assertEqual(meta['pool'], 'evening')
        self.assertEqual(meta['pool_nobs'], 150)
        self.assertEqual(meta['pool_ndays'], 50)
        self.assertIn('coldrisk_evening', meta['fallback_reason'])
        self.assertEqual(len(values), 150)

    def test_cold_pool_requires_both_observations_and_distinct_days(self):
        pool = self.pool(days=50, cold_days=20)
        _, meta = _choose_pool(pool, cold_proxy=True, evening=True)
        self.assertEqual(meta['pool'], 'coldrisk_evening')
        self.assertEqual(meta['pool_nobs'], 60)
        self.assertEqual(meta['pool_ndays'], 20)
        # Many repeated observations on a single day do not create independent days.
        pool.loc[pool.precutoff_coldrisk_evening, 'local_day'] = '2024-01-01'
        _, fallback = _choose_pool(pool, cold_proxy=True, evening=True)
        self.assertNotEqual(fallback['pool'], 'coldrisk_evening')
        self.assertIn('60 observations/1 days', fallback['fallback_reason'])

    def test_evening_pool_falls_back_overall_and_insufficient_is_explicit(self):
        _, meta = _choose_pool(self.pool(days=10), cold_proxy=True, evening=True)
        self.assertEqual(meta['pool'], 'overall')
        self.assertEqual(meta['pool_nobs'], 240)
        empty, meta = _choose_pool(self.pool(days=2), cold_proxy=True, evening=True)
        self.assertEqual(meta['pool'], 'unavailable')
        self.assertEqual(len(empty), 0)
        self.assertIn('overall', meta['fallback_reason'])

    def test_peak_summary_counts_and_upper_exceedance_severity(self):
        frame = pd.DataFrame({'actual_mw': [100., 120., 90., np.nan],
                              'prediction_mw': [100., 100., 100., 100.],
                              'q05': [90.]*4, 'q50': [100.]*4, 'q95': [110.]*4,
                              'local_day': ['2025-01-01', '2025-01-01', '2025-01-02', '2025-01-02'],
                              'winter_id': [2025]*4})
        summary = _summary(frame)
        self.assertEqual(summary['nobs'], 3)
        self.assertEqual(summary['ndays'], 2)
        self.assertEqual(summary['nwinters'], 1)
        self.assertEqual(summary['interval_nobs'], 3)
        self.assertAlmostEqual(summary['upper_exceedance_rate'], 1/3)
        self.assertAlmostEqual(summary['upper_exceedance_mean_severity_MW'], 10/3)
        self.assertEqual(summary['upper_exceedance_conditional_severity_MW'], 10.)
        self.assertEqual(summary['upper_guardrail_status'], 'insufficient_segment_evidence')
        self.assertAlmostEqual(summary['MAE_MW'], 10.)

    def test_realized_weather_changes_diagnostics_but_not_origin_proxy(self):
        delivery = local_day_grid('2025-01-02')
        features = pd.DataFrame({'origin': forecast_origin('2025-01-02', self.contract),
                                 'delivery_time': delivery, 'past_temperature_mean_24h': 10.})
        labels = pd.DataFrame({'delivery_time': delivery, 'actual_mw': 100.,
                               'realized_temperature_c': -10.,
                               'label_available_at': delivery+pd.Timedelta(hours=3)})
        frame = features[['origin', 'delivery_time']].assign(prediction_mw=99., model='m')
        run = SimpleNamespace(contract=self.contract, data=SimpleNamespace(features=features, labels=labels))
        cold = _decorate(frame, run)
        self.assertEqual(cold.realized_cold_evening.sum(), 3)
        self.assertEqual(cold.precutoff_coldrisk_evening.sum(), 0)
        labels['realized_temperature_c'] = 30.
        warm = _decorate(frame, run)
        self.assertEqual(warm.realized_cold_evening.sum(), 0)
        pd.testing.assert_series_equal(cold.precutoff_coldrisk_evening, warm.precutoff_coldrisk_evening)
        # Historical cold may flag tomorrow's evening, while future warm weather
        # remains merely an outcome diagnostic rather than suppressing that flag.
        features['past_temperature_mean_24h'] = 0.
        proxy = _decorate(frame, run)
        self.assertEqual(proxy.precutoff_coldrisk_evening.sum(), 3)
        self.assertEqual(proxy.realized_cold_evening.sum(), 0)

    def test_cost_ratios_are_labeled_proxies_and_policy_is_shared(self):
        delivery = local_day_grid('2025-01-02')[:2]
        frames = []
        for model, median, upper in [('candidate', 90., 110.), ('rte_proxy', 80., 120.)]:
            frames.append(pd.DataFrame({'model': model, 'method': 'fixed_past_out_of_sample_residual_distribution',
                                        'origin': forecast_origin('2025-01-02', self.contract),
                                        'delivery_time': delivery, 'actual_mw': [100., 100.],
                                        'q50': median, 'q95': upper,
                                        'precutoff_coldrisk_evening': [True, False],
                                        'realized_cold_evening': [True, False]}))
        scores = _cost_sensitivity(pd.concat(frames, ignore_index=True))
        self.assertTrue(scores.unit.str.contains('not euros').all())
        self.assertEqual(set(scores.shortage_surplus_ratio), set(SPEC['shortage_surplus_ratios']))
        row = scores.loc[scores.model.eq('candidate') & scores.policy.eq('q95_on_origin_proxy_evening_else_median')
                         & scores.cold_weight.eq(1) & scores.shortage_surplus_ratio.eq(3)].iloc[0]
        # Candidate: (10 surplus + 3*10 shortage)/2=20.
        # RTE with the SAME trigger: (20 surplus + 3*20 shortage)/2=40.
        self.assertEqual(row.mean_proxy_loss, 20.)
        self.assertEqual(row.RTE_same_policy_proxy_loss, 40.)
        self.assertEqual(row.nobs, 2)
        self.assertEqual(row.relative_proxy_improvement_percent, 50.)
        reference = scores.loc[scores.model.eq('rte_proxy')]
        np.testing.assert_allclose(reference.mean_proxy_loss, reference.RTE_same_policy_proxy_loss)

    def final_fixture(self, path):
        first_origin = forecast_origin(self.contract['test']['start'], self.contract)
        fit_days = [d.strftime('%Y-%m-%d') for d in pd.date_range('2025-01-02', periods=10)]
        calibration_days = fit_days+['2025-06-29', '2025-06-30']
        test_days = ['2025-07-01', '2025-07-02']
        features, labels, output = [], [], {'calibration': [], 'test': []}
        for phase, days in [('calibration', calibration_days), ('test', test_days)]:
            for day in days:
                grid = local_day_grid(day)
                origin = forecast_origin(day, self.contract)
                local = grid.tz_convert(self.contract['timezone'])
                proxy_temp = np.where(np.isin(local.hour, [18, 19, 20]), 0., 10.)
                features.append(pd.DataFrame({'origin': origin, 'delivery_time': grid,
                                              'past_temperature_mean_24h': proxy_temp}))
                labels.append(pd.DataFrame({'delivery_time': grid, 'actual_mw': 1e12 if day == '2025-06-30' else (102. if phase == 'calibration' else 104.),
                                            'realized_temperature_c': 0.,
                                            'label_available_at': first_origin-pd.Timedelta(hours=1) if phase == 'calibration' else grid+pd.Timedelta(hours=3)}))
                for model, point in [('candidate', 100.), ('rte_proxy', 90.)]:
                    output[phase].append(pd.DataFrame({'origin': origin, 'delivery_time': grid,
                                                       'local_day': day, 'prediction_mw': point, 'model': model}))
        output = {key: pd.concat(value, ignore_index=True) for key, value in output.items()}
        labels = pd.concat(labels, ignore_index=True)
        labels.loc[0, ['actual_mw', 'label_available_at']] = [1e12, first_origin]
        labels.loc[1, ['actual_mw', 'label_available_at']] = [1e12, first_origin+pd.Timedelta(seconds=1)]
        data = SimpleNamespace(features=pd.concat(features, ignore_index=True), labels=labels)
        run = SimpleNamespace(path=path, contract=self.contract, data=data)
        freeze_peak_spec(run)
        (path/'peak_interval_selection.json').write_text(json.dumps({'records': [
            {'model': model, 'candidate': choice, 'selected': choice == 'pooled'}
            for model in ('candidate', 'rte_proxy') for choice in ('pooled', 'conditional')]}))
        empirical = output['test'].copy()
        empirical['method'] = 'fixed_past_out_of_sample_residual_distribution'
        empirical['q05'] = empirical.prediction_mw-1
        empirical['q50'] = empirical.prediction_mw
        empirical['q95'] = empirical.prediction_mw+1
        empirical.to_csv(path/'forecasts/uncertainty_test.csv.gz', index=False)
        return run, output

    def test_final_calibration_is_strict_and_ignores_june30_and_test_outcomes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory); (path/'forecasts').mkdir(); (path/'tables').mkdir()
            run, outputs = self.final_fixture(path)
            first = analyze_peak_risk(run, outputs)
            pools = first['calibration_pools']
            # 10 days + June29 =264; the two boundary/future publication values excluded.
            pooled = pools.loc[pools.interval_policy.eq('pooled')]
            self.assertTrue(pooled.pool_nobs.eq(262).all())
            before = pd.read_csv(path/'forecasts/peak_uncertainty_test.csv.gz')
            intervals = before.loc[before.method.eq('peak_pooled_empirical_residuals')]
            self.assertTrue(intervals.q95.eq(102.).all())
            self.assertNotIn('actual_mw', before)
            self.assertNotIn('realized_temperature_c', before)
            self.assertNotIn('CRPS_MW', before)
            cutoff = outputs['test'].origin.min()
            run.data.labels.loc[run.data.labels.label_available_at.ge(cutoff), 'actual_mw'] = -1e15
            run.data.labels.loc[run.data.labels.delivery_time.isin(local_day_grid('2025-06-30')), 'actual_mw'] = -1e15
            run.data.labels['realized_temperature_c'] = 40.
            second = analyze_peak_risk(run, outputs)
            after = pd.read_csv(path/'forecasts/peak_uncertainty_test.csv.gz')
            # Outcomes can alter measured errors/segments, never the chosen bounds.
            columns = ['model', 'method', 'origin', 'delivery_time', 'q05', 'q50', 'q95',
                       'precutoff_coldrisk_evening', 'selected_by_2024_validation']
            pd.testing.assert_frame_equal(before[columns], after[columns])
            pd.testing.assert_frame_equal(first['interval_selection'], second['interval_selection'])

    def test_cannot_first_freeze_or_select_after_test_replay_starts(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory); (path/'forecasts').mkdir()
            run = SimpleNamespace(path=path)
            (path/'forecasts/all_test.csv.gz').touch()
            with self.assertRaisesRegex(RuntimeError, 'first freeze'):
                freeze_peak_spec(run)
            (path/'forecasts/all_test.csv.gz').unlink()
            freeze_peak_spec(run)
            (path/'frozen_selection.json').write_text(json.dumps({
                'primary_candidate_from_validation': 'candidate', 'selections': {}}))
            (path/'forecasts/candidate__test__abc.json').touch()
            with self.assertRaisesRegex(RuntimeError, 'first select'):
                select_peak_intervals(run)

    def test_freeze_is_reusable_without_touching_outcomes_and_rejects_drift(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            run = SimpleNamespace(path=path)
            first = freeze_peak_spec(run)
            self.assertEqual(freeze_peak_spec(run), first)
            spec = json.loads((path/'peak_risk_specification.json').read_text())
            spec['cold_threshold_c'] = -99.
            (path/'peak_risk_specification.json').write_text(json.dumps(spec))
            with self.assertRaises(RuntimeError):
                freeze_peak_spec(run)


if __name__ == '__main__':
    unittest.main()
