"""RTE-feature/residual distinction, cutoff rules, fallbacks and replay caches."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.common import forecast_origin, local_day_grid, project_root
from src.features import build_origin_features
from src.research_data import load_research_contract
from src.rte_assisted import RTEAssistedRun, demand_prediction, model_key, training_target


class ConstantEstimator:
    def fit(self, X, y):
        self.value = float(y.mean())
        self.metadata_ = {"feature_columns": list(X.columns)}
        return self

    def predict(self, X):
        return np.full(len(X), self.value)


class SyntheticData:
    def __init__(self, day):
        self.origin = forecast_origin(day, load_research_contract(project_root()))
        grid = local_day_grid(day)
        self.features = pd.DataFrame({"origin": self.origin, "delivery_time": grid,
            "local_day": day, "rte_at_origin_mw": 100., "rte_bias_recent_mw": 10.,
            "week_lag_1": 95., "comparable_mean": 96., "last_load": 97.})
        self.calls = 0

    def phase(self, start, end):
        return self.features.copy()

    def columns(self, arm, rte_assisted=False):
        assert rte_assisted
        return ["rte_at_origin_mw", "rte_bias_recent_mw"]

    def training_rows(self, origin):
        self.calls += 1
        times = pd.date_range(origin-pd.Timedelta(days=400), periods=280, freq='D')
        return pd.DataFrame({"delivery_time": times, "origin": times-pd.Timedelta(days=1),
            "actual_mw": 110., "rte_at_origin_mw": 100., "rte_bias_recent_mw": 10.,
            "label_available_at": times+pd.Timedelta(hours=3),
            "max_history_available_at": times-pd.Timedelta(days=2)})


class AssistedTests(unittest.TestCase):
    def run_fixture(self, directory, day='2025-03-30'):
        root = project_root()
        run = RTEAssistedRun(root, Path(directory)/'run')
        run.verify_manifest = lambda: 'synthetic-manifest'
        run._data = SyntheticData(day)
        return run

    def test_direct_and_residual_targets_are_distinct(self):
        train = pd.DataFrame({'actual_mw': [105., 115.], 'rte_at_origin_mw': [100., 110.]})
        np.testing.assert_array_equal(training_target(train, 'direct'), [105., 115.])
        np.testing.assert_array_equal(training_target(train, 'residual'), [5., 5.])
        with self.assertRaises(ValueError):
            training_target(train, 'unknown')

    def test_residual_prediction_adds_rte_exactly_once(self):
        day = pd.DataFrame({'rte_at_origin_mw': [100., 110.]})
        np.testing.assert_array_equal(demand_prediction([5., -5.], day, 'residual'), [105., 105.])
        np.testing.assert_array_equal(demand_prediction([105., 105.], day, 'direct'), [105., 105.])
        self.assertNotEqual(model_key('gam', 'national', 'direct'), model_key('gam', 'national', 'residual'))

    def test_rte_features_ignore_unreleased_actual_demand_and_weather(self):
        contract = load_research_contract(project_root())
        day = '2024-02-15'
        origin = forecast_origin(day, contract)
        grid = pd.date_range('2024-01-01', '2024-02-17', freq='h', tz='UTC')
        data = pd.DataFrame({'load_mw': 110.+np.sin(np.arange(len(grid))),
                             'rte_forecast_d1_mw': 100., 'temperature_c': 4.,
                             'label_available_at_assumed': grid+pd.Timedelta(hours=3)}, index=grid)
        before = build_origin_features(data, day, contract, rte_mode='proxy')
        changed = data.copy()
        changed.loc[changed.label_available_at_assumed.ge(origin), ['load_mw', 'temperature_c']] = 1e12
        after = build_origin_features(changed, day, contract, rte_mode='proxy')
        pd.testing.assert_frame_equal(before, after)
        self.assertTrue(before.rte_at_origin_mw.eq(100.).all())
        self.assertTrue(before.rte_bias_recent_mw.notna().all())

    @patch('src.rte_assisted.make_tabular', side_effect=lambda *a, **k: ConstantEstimator())
    def test_both_modes_keep_dst_grid_and_equal_demand_prediction(self, factory):
        for day, expected in [('2025-03-30', 23), ('2025-10-26', 25)]:
            with self.subTest(day=day), TemporaryDirectory() as directory:
                run = self.run_fixture(directory, day)
                for mode in ('direct', 'residual'):
                    p = run.replay_variant('gam', {}, day, '2025-10-27', arm='national', mode=mode)
                    self.assertEqual(len(p), expected)
                    self.assertFalse(p.delivery_time.duplicated().any())
                    np.testing.assert_allclose(p.prediction_mw, 110.)
                    self.assertFalse(p.fallback_used.any())
                    self.assertNotIn('actual_mw', p)
                fits = [r for path in (run.path/'fits').glob('*.json') for r in json.loads(path.read_text())]
                self.assertEqual({tuple(r['predictor_columns']) for r in fits},
                                 {('rte_at_origin_mw', 'rte_bias_recent_mw')})
                self.assertTrue(all(pd.Timestamp(r['latest_label_available_at']) < pd.Timestamp(r['origin']) for r in fits))

    @patch('src.rte_assisted.make_tabular', side_effect=lambda *a, **k: ConstantEstimator())
    def test_missing_rte_uses_declared_history_fallback_in_both_modes(self, factory):
        with TemporaryDirectory() as directory:
            run = self.run_fixture(directory)
            run.data.features.loc[0, 'rte_at_origin_mw'] = np.nan
            for mode in ('direct', 'residual'):
                p = run.replay_variant('gam', {}, '2025-03-30', '2025-03-31', arm='national', mode=mode)
                self.assertTrue(p.loc[0, 'fallback_used'])
                self.assertEqual(p.loc[0, 'fallback_source'], 'weekly')
                self.assertEqual(p.loc[0, 'prediction_mw'], 95.)
                self.assertTrue(pd.isna(p.loc[0, 'raw_prediction_mw']))

    @patch('src.rte_assisted.make_tabular', side_effect=lambda *a, **k: ConstantEstimator())
    def test_training_label_after_cutoff_fails_fit_and_records_rte_fallback(self, factory):
        with TemporaryDirectory() as directory:
            run = self.run_fixture(directory)
            original = run.data.training_rows
            def invalid(origin):
                train = original(origin)
                train.loc[0, 'label_available_at'] = origin
                return train
            run.data.training_rows = invalid
            p = run.replay_variant('gam', {}, '2025-03-30', '2025-03-31', arm='national', mode='residual')
            self.assertTrue(p.fallback_used.all())
            self.assertTrue(p.fallback_source.eq('rte').all())
            record = json.loads(next((run.path/'fits').glob('*.json')).read_text())[0]
            self.assertEqual(record['status'], 'failed')
            self.assertIn('released after fit cutoff', record['error'])

    @patch('src.rte_assisted.make_tabular', side_effect=lambda *a, **k: ConstantEstimator())
    def test_sealed_cache_reused_but_modified_fit_artifact_rejected(self, factory):
        with TemporaryDirectory() as directory:
            run = self.run_fixture(directory)
            kwargs = dict(arm='national', mode='residual')
            run.replay_variant('gam', {}, '2025-03-30', '2025-03-31', **kwargs)
            self.assertEqual(run.data.calls, 1)
            run.replay_variant('gam', {}, '2025-03-30', '2025-03-31', **kwargs)
            self.assertEqual(run.data.calls, 1)
            next((run.path/'fits').glob('*.json')).write_text('[]')
            with self.assertRaisesRegex(RuntimeError, 'Sealed forecast cache'):
                run.replay_variant('gam', {}, '2025-03-30', '2025-03-31', **kwargs)

    def test_first_freeze_after_test_is_rejected(self):
        with TemporaryDirectory() as directory:
            run = self.run_fixture(directory)
            (run.path/'forecasts/gam__national__rte_direct__test__dummy.csv.gz').touch()
            with self.assertRaisesRegex(RuntimeError, 'after test replay'):
                run.freeze_assisted()


if __name__ == '__main__':
    unittest.main()
