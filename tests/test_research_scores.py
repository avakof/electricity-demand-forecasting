"""Independent numeric and cutoff checks for research uncertainty methods."""
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
from src.research_scores import (
    conformal_radius, empirical_crps, interval_summary, strict_conformal,
)


class ResearchScoreTests(unittest.TestCase):
    def test_empirical_crps_matches_brute_pairwise_definition(self):
        errors = np.array([-5., -1., 0., .5, 2., 10., np.nan])
        residuals = np.array([-2., -2., 0., 3., 8., np.nan, np.inf])
        valid = residuals[np.isfinite(residuals)]
        expected = np.array([np.mean(np.abs(valid-e))
                             -.5*np.mean(np.abs(valid[:, None]-valid[None, :])) for e in errors])
        np.testing.assert_allclose(empirical_crps(errors, residuals), expected, equal_nan=True)

    def test_empirical_crps_degenerate_and_empty_distributions(self):
        np.testing.assert_allclose(empirical_crps(np.array([-1., 2., 5.]), [2., 2.]), [3., 0., 3.])
        self.assertTrue(np.isnan(empirical_crps(np.array([0., np.nan]), [np.nan])).all())
        self.assertEqual(len(empirical_crps(np.array([]), [1., 2.])), 0)

    def test_conformal_finite_sample_ceil_not_interpolated_quantile(self):
        # n=10: ceil((10+1)*.9)=10, so the radius is the maximum, not q=.9.
        self.assertEqual(conformal_radius(np.arange(10.)), 9.)
        self.assertEqual(conformal_radius(np.arange(19.)), 17.)
        self.assertEqual(conformal_radius([1., 2., 3., np.nan], alpha=.5), 2.)

    def test_conformal_reports_unbounded_for_insufficient_scores(self):
        self.assertTrue(np.isinf(conformal_radius([])))
        self.assertTrue(np.isinf(conformal_radius(np.arange(8.))))
        self.assertEqual(conformal_radius(np.arange(9.)), 8.)

    def test_conformal_rejects_invalid_probability_or_negative_abs_scores(self):
        for alpha in (-.1, 0., 1., 1.1, np.nan):
            with self.assertRaises(ValueError):
                conformal_radius([1., 2.], alpha)
        with self.assertRaises(ValueError):
            conformal_radius([-1., 2.])

    def test_interval_score_and_wis_known_values(self):
        frame = pd.DataFrame({'actual_mw': [0., 2., 5., np.nan],
                              'q05': [1., 1., 1., 1.], 'q50': [2., 2., 2., 2.],
                              'q95': [3., 3., 3., 3.]})
        scored = interval_summary(frame)
        # Scores: 2+20*(1-0)=22; 2; 2+20*(5-3)=42.
        self.assertAlmostEqual(scored['interval_score_90'], 22.)
        expected_wis = np.mean([(.5*2+.05*22)/1.5,
                                (.5*0+.05*2)/1.5,
                                (.5*3+.05*42)/1.5])
        self.assertAlmostEqual(scored['WIS_one_interval'], expected_wis)
        self.assertEqual(scored['scored'], 3)
        self.assertAlmostEqual(scored['coverage_90'], 1/3)

    def test_unbounded_interval_score_is_infinite_not_nan(self):
        frame = pd.DataFrame({'actual_mw': [10.], 'q05': [-np.inf],
                              'q50': [10.], 'q95': [np.inf]})
        score = interval_summary(frame)
        self.assertEqual(score['coverage_90'], 1.)
        self.assertTrue(np.isinf(score['width_MW']))
        self.assertTrue(np.isinf(score['interval_score_90']))
        self.assertTrue(np.isinf(score['WIS_one_interval']))

    def test_strict_conformal_single_fit_and_strict_label_availability(self):
        contract = deepcopy(load_research_contract(project_root()))
        first_test_origin = forecast_origin(contract['test']['start'], contract)
        fit_origin = forecast_origin(contract['calibration']['start'], contract)
        rows, labels = [], []
        # Include the excluded June30 buffer. Its enormous errors must not enter
        # calibration even if fake metadata makes those outcomes already available.
        for day, value in [('2025-06-28', 101.), ('2025-06-29', 102.),
                           ('2025-06-30', 1e12), ('2025-07-01', 1e12)]:
            for delivery in local_day_grid(day):
                rows.append({'origin': forecast_origin(day, contract), 'delivery_time': delivery,
                             'local_day': day, 'fit_origin': fit_origin,
                             'model': 'gam__national__fixed_conformal', 'prediction_mw': 100., 'raw_prediction_mw': 100.})
                labels.append({'delivery_time': delivery, 'actual_mw': value,
                               'label_available_at': first_test_origin-pd.Timedelta(hours=1)})
        continuous = pd.DataFrame(rows)
        labels = pd.DataFrame(labels)
        # Two otherwise-eligible calibration values are published exactly at and
        # after the first test origin. Their values must not affect the interval.
        labels.loc[0, ['actual_mw', 'label_available_at']] = [1e12, first_test_origin]
        labels.loc[1, ['actual_mw', 'label_available_at']] = [1e12, first_test_origin+pd.Timedelta(seconds=1)]
        selected = {'gam__national': {'family': 'gam', 'params': {'alpha': 20.}, 'arm': 'national'}}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path/'forecasts').mkdir(); (path/'tables').mkdir()
            (path/'frozen_selection.json').write_text(json.dumps({'primary_candidate_from_validation': 'gam__national'}))
            calls = []
            def replay(*args, **kwargs):
                calls.append((args, kwargs))
                return continuous.copy()
            run = SimpleNamespace(path=path, contract=contract,
                                  data=SimpleNamespace(labels=labels), replay=replay)
            out = strict_conformal(run, selected)
            self.assertEqual(len(calls), 1)
            self.assertIs(calls[0][1]['fixed'], True)
            self.assertEqual(calls[0][0][2], contract['calibration']['start'])
            self.assertEqual(calls[0][0][3], contract['test']['end'])
            self.assertTrue(out.local_day.eq('2025-07-01').all())
            self.assertTrue(out.calibration_n.eq(46).all())
            self.assertTrue(out.q05.eq(98.).all())
            self.assertTrue(out.q95.eq(102.).all())
            report = json.loads((path/'tables/conformal_report.json').read_text())
            self.assertEqual(report['calibration_rows'], 46)
            self.assertLess(pd.Timestamp(report['latest_calibration_available_at']), first_test_origin)
            self.assertNotIn('actual_mw', pd.read_csv(path/'forecasts/fixed_conformal_test.csv.gz'))
            # Mutating all labels that should be excluded leaves predictions equal.
            labels.loc[labels.label_available_at.ge(first_test_origin), 'actual_mw'] = -1e15
            labels.loc[labels.delivery_time.isin(local_day_grid('2025-06-30')), 'actual_mw'] = -1e15
            second = strict_conformal(run, selected)
            pd.testing.assert_frame_equal(out, second)

    def test_strict_conformal_rejects_parameter_refitting(self):
        contract = deepcopy(load_research_contract(project_root()))
        rows = pd.DataFrame({'fit_origin': pd.to_datetime(['2025-01-01', '2025-02-01'], utc=True)})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path/'frozen_selection.json').write_text(json.dumps({'primary_candidate_from_validation': 'gam__national'}))
            run = SimpleNamespace(path=path, contract=contract, replay=lambda *args, **kwargs: rows)
            selected = {'gam__national': {'family': 'gam', 'params': {}, 'arm': 'national'}}
            with self.assertRaises(AssertionError):
                strict_conformal(run, selected)

    def test_strict_conformal_rejects_fallback_only_predictor(self):
        contract = deepcopy(load_research_contract(project_root()))
        rows = pd.DataFrame({'fit_origin': pd.to_datetime(['2025-01-01'], utc=True),
                             'raw_prediction_mw': [np.nan]})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path/'frozen_selection.json').write_text(json.dumps({'primary_candidate_from_validation': 'gam__national'}))
            run = SimpleNamespace(path=path, contract=contract, replay=lambda *args, **kwargs: rows)
            selected = {'gam__national': {'family': 'gam', 'params': {}, 'arm': 'national'}}
            with self.assertRaisesRegex(AssertionError, 'fallback-only'):
                strict_conformal(run, selected)


if __name__ == '__main__':
    unittest.main()
