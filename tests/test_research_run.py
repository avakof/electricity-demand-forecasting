"""Cross-process selection persistence and immutable-input replay-cache checks."""
import gzip
import json
import multiprocessing
from pathlib import Path
import tempfile
import time
import unittest
from types import SimpleNamespace

import pandas as pd

from src.common import forecast_origin, local_day_grid, project_root
from src.research_data import load_research_contract
from src.research_experiment import ResearchRun


def _selection_writer(root, run_path, family, start):
    """Top-level target supports macOS spawn rather than relying on fork state."""
    run = ResearchRun(root, run_path)
    if not start.wait(20):
        raise RuntimeError('Concurrent test start signal timed out')
    for trial in range(2):
        for fold in range(4):
            run._record_trial({'model': f'{family}__national', 'family': family,
                               'trial': trial, 'fold': f'2024_Q{fold+1}',
                               'MAE_MW': 100.+trial+fold})
    for arm in ('national', 'regional'):
        key = f'{family}__{arm}'
        run._record_selection(key, {'family': family, 'arm': arm,
                                    'status': 'selected', 'params': {'depth': 2},
                                    'MAE_MW': 100.})


class _PhaseData:
    def __init__(self, features):
        self.features, self.calls = features, 0

    def phase(self, start, end):
        self.calls += 1
        return self.features.copy()

    def columns(self, *args):
        return []


class ResearchRunTests(unittest.TestCase):
    def fixture(self, directory):
        root = Path(directory)
        (root/'config').mkdir()
        (root/'data').mkdir()
        (root/'src').mkdir()
        contract = load_research_contract(project_root())
        (root/'config/research_experiment.json').write_text(json.dumps(contract))
        # Dummy dependency contents are sufficient: these tests verify provenance
        # checks and synthetic replay, not source-data preparation or model fitting.
        for name in ('features.py', 'common.py', 'data.py', 'metrics.py', 'uncertainty.py', 'research_stub.py'):
            (root/'src'/name).write_text('# immutable test dependency\n')
        (root/'requirements-research-neural.txt').write_text('# immutable test requirements\n')
        for name in ('france_electricity_hourly.csv', 'france_regional_electricity_hourly.csv'):
            (root/'data'/name).write_text('value\n1\n')
        run = ResearchRun.create(root)
        return root, run

    def cache_fixture(self, directory):
        root, run = self.fixture(directory)
        day = '2025-03-30'
        grid = local_day_grid(day)
        features = pd.DataFrame({'origin': forecast_origin(day, run.contract),
                                 'delivery_time': grid, 'local_day': day,
                                 'week_lag_1': 100., 'comparable_mean': 101.,
                                 'last_load': 102., 'rte_at_origin_mw': 99.,
                                 'rte_bias_recent_mw': 1.})
        run._data = _PhaseData(features)
        args = ('seasonal_naive', {}, day, '2025-03-31')
        return root, run, args

    def test_disjoint_process_writes_preserve_all_trials_and_selections(self):
        with tempfile.TemporaryDirectory() as directory:
            root, run = self.fixture(directory)
            context = multiprocessing.get_context('spawn')
            start = context.Event()
            families = ('lightgbm', 'gru')
            processes = [context.Process(target=_selection_writer,
                                         args=(str(root), str(run.path), family, start))
                         for family in families]
            try:
                for process in processes:
                    process.start()
                start.set()
                deadline = time.monotonic()+45
                # Readers should see a complete old/new file, never an incomplete
                # write, while writers hold a separate POSIX lock and rename.
                while any(process.is_alive() for process in processes):
                    if time.monotonic() > deadline:
                        self.fail('Concurrent persistence check timed out')
                    path = run.path/'selected_parameters.json'
                    if path.exists():
                        self.assertIsInstance(json.loads(path.read_text()), dict)
                    path = run.path/'tables/tuning_trials.csv'
                    if path.exists():
                        self.assertFalse(pd.read_csv(path).empty)
                    time.sleep(.01)
                for process in processes:
                    process.join(timeout=2)
                    self.assertEqual(process.exitcode, 0)
            finally:
                for process in processes:
                    if process.is_alive():
                        process.terminate()
                    process.join(timeout=2)
            selections = json.loads((run.path/'selected_parameters.json').read_text())
            expected = {f'{family}__{arm}' for family in families for arm in ('national', 'regional')}
            self.assertEqual(set(selections), expected)
            trials = pd.read_csv(run.path/'tables/tuning_trials.csv')
            self.assertEqual(len(trials), 16)
            self.assertFalse(trials.duplicated(['model', 'trial', 'fold']).any())
            self.assertTrue(trials.groupby(['model', 'trial']).fold.nunique().eq(4).all())

    def test_record_trial_replaces_one_key_without_losing_other_trials(self):
        with tempfile.TemporaryDirectory() as directory:
            _, run = self.fixture(directory)
            first = {'model': 'gam__national', 'trial': 0, 'fold': '2024_Q1', 'MAE_MW': 10.}
            second = {**first, 'fold': '2024_Q2', 'MAE_MW': 20.}
            run._record_trial(first); run._record_trial(second)
            run._record_trial({**first, 'MAE_MW': 5.})
            trials = pd.read_csv(run.path/'tables/tuning_trials.csv')
            self.assertEqual(len(trials), 2)
            self.assertEqual(trials.loc[trials.fold.eq('2024_Q1'), 'MAE_MW'].iloc[0], 5.)
            self.assertEqual(trials.loc[trials.fold.eq('2024_Q2'), 'MAE_MW'].iloc[0], 20.)

    def test_latest_pointer_and_manifest_are_consistent(self):
        with tempfile.TemporaryDirectory() as directory:
            root, run = self.fixture(directory)
            latest = ResearchRun.latest(root)
            self.assertEqual(latest.path, run.path)
            self.assertEqual(latest.profile, run.profile)
            self.assertEqual(latest.verify_manifest(), run.verify_manifest())
            manifest = json.loads((run.path/'manifest.json').read_text())
            for name in ('src/features.py', 'src/common.py', 'src/data.py', 'src/metrics.py',
                         'src/uncertainty.py', 'requirements-research-neural.txt'):
                self.assertIn(name, manifest['source_sha256'])

    def test_manifest_rejects_changed_raw_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root, run = self.fixture(directory)
            (root/'data/france_electricity_hourly.csv').write_text('value\n999\n')
            with self.assertRaisesRegex(RuntimeError, 'inputs/code changed'):
                run.verify_manifest()

    def test_manifest_rejects_changed_feature_code_and_contract(self):
        for filename in ('src/features.py', 'src/research_stub.py', 'config/research_experiment.json'):
            with self.subTest(filename=filename), tempfile.TemporaryDirectory() as directory:
                root, run = self.fixture(directory)
                path = root/filename
                path.write_text(path.read_text()+'\n# changed\n')
                with self.assertRaisesRegex(RuntimeError, 'inputs/code changed'):
                    run.verify_manifest()

    def test_valid_cache_skips_replay_but_corrupt_ledger_forces_replay(self):
        with tempfile.TemporaryDirectory() as directory:
            _, run, args = self.cache_fixture(directory)
            first = run.replay(*args)
            self.assertEqual(run._data.calls, 1)
            second = run.replay(*args)
            self.assertEqual(run._data.calls, 1)
            self.assertEqual(len(first), 23)
            pd.testing.assert_series_equal(first.prediction_mw, second.prediction_mw)
            ledger = next((run.path/'ledgers').glob('*.jsonl.gz'))
            with gzip.open(ledger, 'at') as stream:
                stream.write('{}\n')
            third = run.replay(*args)
            self.assertEqual(run._data.calls, 2)
            pd.testing.assert_series_equal(first.prediction_mw, third.prediction_mw)
            with gzip.open(ledger, 'rt') as stream:
                self.assertEqual(len(stream.readlines()), 1)

    def test_corrupt_cached_forecast_forces_replay(self):
        with tempfile.TemporaryDirectory() as directory:
            _, run, args = self.cache_fixture(directory)
            run.replay(*args)
            path = next((run.path/'forecasts').glob('*.csv.gz'))
            frame = pd.read_csv(path)
            frame['prediction_mw'] = 1e12
            frame.to_csv(path, index=False)
            result = run.replay(*args)
            self.assertEqual(run._data.calls, 2)
            self.assertTrue(result.prediction_mw.eq(100.).all())

    def test_cache_cannot_bypass_manifest_change(self):
        with tempfile.TemporaryDirectory() as directory:
            root, run, args = self.cache_fixture(directory)
            run.replay(*args)
            (root/'src/common.py').write_text('# changed calendar semantics\n')
            with self.assertRaisesRegex(RuntimeError, 'inputs/code changed'):
                run.replay(*args)
            self.assertEqual(run._data.calls, 1)


if __name__ == '__main__':
    unittest.main()
