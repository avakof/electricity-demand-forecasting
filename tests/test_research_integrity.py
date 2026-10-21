"""Behavioral checks for research-v2 chronology, panel views and feature causality."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from src.common import forecast_origin, local_day_grid, project_root
from src.research_data import (
    ResearchData, calendar_exog, latest_eligible_time,
    load_research_contract, prepare_research_data,
)


class ResearchIntegrityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.contract = load_research_contract(project_root())

    def synthetic_data(self):
        origin = forecast_origin('2025-01-02', self.contract)
        index = pd.date_range('2024-12-30', '2025-01-02', freq='h', tz='UTC')
        national = pd.DataFrame({'load_mw': np.arange(len(index), dtype=float),
                                 'temperature_c': 10.,
                                 'label_available_at_assumed': index + pd.Timedelta(hours=3)}, index=index)
        national.loc[index[5], 'load_mw'] = np.nan
        regional = pd.DataFrame({'r1': national.load_mw / 2, 'r2': national.load_mw / 3}, index=index)
        regional.loc[index[7], 'r2'] = np.nan
        features = pd.DataFrame({
            'origin': [origin-pd.Timedelta(days=1)]*4+[origin],
            'delivery_time': index[[10, 11, 12, 13, 14]],
            'max_history_available_at': [origin-pd.Timedelta(days=2)]*4+[origin-pd.Timedelta(hours=1)],
            'region_00_last_load': [1., 2., 3., 4., 5.],
            'actual_mw': [9e9]*5, 'realized_temperature_c': [9e9]*5,
        })
        labels = pd.DataFrame({
            'delivery_time': features.delivery_time,
            'actual_mw': [10., 20., 30., np.nan, 50.],
            'label_available_at': [origin-pd.Timedelta(seconds=1), origin,
                                   origin+pd.Timedelta(seconds=1),
                                   origin-pd.Timedelta(hours=1), origin-pd.Timedelta(hours=1)],
        })
        # Avoid duplicate label columns on merge; forbidden feature checks below
        # use their existence as a separate adversarial condition.
        features = features.drop(columns=['actual_mw'])
        return ResearchData(Path('.'), deepcopy(self.contract), national, regional,
                            regional*0+10., features, labels), origin

    def test_latest_eligible_time_uses_strict_publication_boundary(self):
        for day in ('2025-01-02', '2025-03-30', '2025-10-26'):
            origin = forecast_origin(day, self.contract)
            end = latest_eligible_time(origin, self.contract)
            delay = pd.Timedelta(hours=3)
            self.assertLess(end+delay, origin)
            self.assertGreaterEqual(end+pd.Timedelta(hours=1)+delay, origin)
            self.assertEqual(end.tz_convert('Europe/Paris').hour, 7)

    def test_sequence_bridge_covers_complete_dst_delivery_days(self):
        for day, count, last_step in (('2025-01-02', 24, 40),
                                      ('2025-03-30', 23, 39),
                                      ('2025-10-26', 25, 41)):
            origin = forecast_origin(day, self.contract)
            end = latest_eligible_time(origin, self.contract)
            bridge = pd.date_range(end+pd.Timedelta(hours=1),
                                   periods=self.contract['sequence_horizon_hours'], freq='h')
            delivery = local_day_grid(day)
            steps = (delivery-end).total_seconds()/3600
            self.assertEqual(len(delivery), count)
            self.assertEqual(steps.min(), 17)
            self.assertEqual(steps.max(), last_step)
            self.assertTrue(delivery.isin(bridge).all())
            self.assertTrue(delivery.is_unique)

    def test_known_calendar_covers_bridge_without_outcomes(self):
        origin = forecast_origin('2025-10-26', self.contract)
        end = latest_eligible_time(origin, self.contract)
        bridge = pd.date_range(end+pd.Timedelta(hours=1), periods=48, freq='h')
        calendar = calendar_exog(bridge)
        self.assertTrue(calendar.index.equals(bridge))
        self.assertTrue(np.isfinite(calendar.to_numpy()).all())
        self.assertFalse(any('load' in c or 'temperature' in c for c in calendar))

    def test_calibration_starts_after_last_validation_label_release(self):
        last_validation = local_day_grid('2024-12-31')[-1]
        released = last_validation+pd.Timedelta(hours=3)
        first_calibration_origin = forecast_origin(self.contract['calibration']['start'], self.contract)
        self.assertLess(released, first_calibration_origin)
        self.assertEqual(self.contract['calibration']['start'], '2025-01-02')

    def test_calibration_end_buffer_precedes_first_test_origin(self):
        calibration = self.contract['calibration']
        days = pd.date_range(calibration['start'], calibration['end'], freq='D', inclusive='left')
        first_test_origin = forecast_origin(self.contract['test']['start'], self.contract)
        releases = local_day_grid(days[-1])+pd.Timedelta(hours=3)
        self.assertEqual(len(days), 179)
        self.assertTrue((releases < first_test_origin).all())
        unavailable_buffer = local_day_grid(calibration['boundary_buffer_day'])+pd.Timedelta(hours=3)
        self.assertTrue((unavailable_buffer >= first_test_origin).any())

    def test_training_rows_exclude_equal_future_and_missing_labels(self):
        data, origin = self.synthetic_data()
        training = data.training_rows(origin)
        self.assertEqual(training.actual_mw.tolist(), [10.])
        self.assertTrue(training.label_available_at.lt(origin).all())
        self.assertTrue(training.origin.lt(origin).all())

    def test_future_label_values_cannot_change_training_rows(self):
        data, origin = self.synthetic_data()
        before = data.training_rows(origin)
        data.labels.loc[data.labels.label_available_at.ge(origin), 'actual_mw'] = 1e12
        after = data.training_rows(origin)
        assert_frame_equal(before, after)

    def test_selected_features_exclude_targets_and_future_weather(self):
        data, _ = self.synthetic_data()
        forbidden = {'actual_mw', 'realized_temperature_c', 'oracle_temperature_c',
                     'load_mw', 'weather_forecast_c', 'rte_at_origin_mw'}
        for arm in ('national', 'regional'):
            self.assertFalse(forbidden.intersection(data.columns(arm)))
        self.assertIn('region_00_last_load', data.columns('regional'))
        self.assertNotIn('region_00_last_load', data.columns('national'))
        self.assertIn('rte_at_origin_mw', data.columns('national', rte_assisted=True))

    def test_panel_retains_missing_values_and_common_utc_cutoff(self):
        data, origin = self.synthetic_data()
        panel = data.panel(origin, arm='regional')
        end = latest_eligible_time(origin, self.contract).tz_localize(None)
        counts = panel.groupby('unique_id').size()
        self.assertEqual(counts.nunique(), 1)
        self.assertEqual(panel.groupby('unique_id').ds.max().nunique(), 1)
        self.assertTrue(panel.groupby('unique_id').ds.max().eq(end).all())
        self.assertGreater(panel.y.isna().sum(), 0)
        for _, group in panel.groupby('unique_id'):
            self.assertTrue(group.ds.diff().iloc[1:].eq(pd.Timedelta(hours=1)).all())
        country = panel.loc[panel.unique_id.eq('country')]
        self.assertEqual(len(country), len(data.history(origin)))
        self.assertEqual(country.y.isna().sum(), data.history(origin).load_mw.isna().sum())

    def test_prepare_features_ignore_unreleased_national_and_regional_values(self):
        """Exercise the real preparation path on a small source, without cache reuse."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'data').mkdir()
            (root/'config').mkdir()
            contract = deepcopy(self.contract)
            contract['matched_history_start'] = '2024-01-01'
            (root/'config/research_experiment.json').write_text(json.dumps(contract))
            index = pd.date_range('2024-01-01', '2024-02-25', freq='h', tz='UTC')
            country = pd.DataFrame({'datetime_utc': index,
                                    'load_mw': 1000.+np.arange(len(index)),
                                    'temperature_c': 10., 'rte_forecast_d1_mw': 990.})
            regional = pd.concat([country[['datetime_utc', 'load_mw', 'temperature_c']].assign(region=r)
                                  for r in ('A', 'B')], ignore_index=True)
            country_path = root/'data/france_electricity_hourly.csv'
            region_path = root/'data/france_regional_electricity_hourly.csv'
            country.to_csv(country_path, index=False)
            regional.to_csv(region_path, index=False)
            first = prepare_research_data(root, use_cache=False)
            origin = forecast_origin('2024-02-20', contract)
            expected = first.features.loc[first.features.origin.eq(origin)].copy()
            for frame in (country, regional):
                unreleased = frame.datetime_utc+pd.Timedelta(hours=3) >= origin
                frame.loc[unreleased, ['load_mw', 'temperature_c']] = 1e12
            country.to_csv(country_path, index=False)
            regional.to_csv(region_path, index=False)
            second = prepare_research_data(root, use_cache=False)
            actual = second.features.loc[second.features.origin.eq(origin)]
            self.assertEqual(len(actual), 24)
            assert_frame_equal(expected, actual)


if __name__ == '__main__':
    unittest.main()
