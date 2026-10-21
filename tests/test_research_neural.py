"""Neural-panel causality checks; optional real-backend reproducibility test.

Pure checks do not import PyTorch/NeuralForecast. Enable the bounded real DLinear
fit with RUN_NEURAL_INTEGRATION_TESTS=1; it is a functional test, not a score.
"""
from copy import deepcopy
import os
from pathlib import Path
import unittest

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from src.common import forecast_origin, local_day_grid, project_root
from src.research_data import ResearchData, latest_eligible_time, load_research_contract
from src.research_neural import NeuralAdapter, _panel


class NeuralPanelTests(unittest.TestCase):
    def example(self):
        times = pd.date_range("2025-01-01", periods=5, freq="h", tz="UTC")
        return pd.DataFrame({
            "unique_id": ["a"]*5+["b"]*5, "ds": list(times)*2,
            "y": [np.nan, 10., np.nan, 30., 40., np.nan, 100., np.nan, 300., 400.],
        })

    def test_missing_targets_keep_grid_and_zero_loss_mask(self):
        source = self.example()
        original = source.copy(deep=True)
        result = _panel(source)
        self.assertEqual(len(result), len(source))
        self.assertEqual(result.available_mask.tolist(), [0, 1, 0, 1, 1]*2)
        assert_frame_equal(source, original)  # Evaluation truth is never mutated.
        self.assertEqual(source.y.isna().sum(), 4)

    def test_predictor_fill_is_causal_with_leading_zero_only(self):
        result = _panel(self.example())
        self.assertEqual(result.y.tolist(), [0, 10, 10, 30, 40, 0, 100, 100, 300, 400])
        changed = self.example()
        changed.loc[3:4, "y"] = 1e12
        changed.loc[8:9, "y"] = 1e12
        after = _panel(changed)
        self.assertEqual(after.y.iloc[0], 0)
        self.assertEqual(after.y.iloc[2], 10)
        self.assertEqual(after.y.iloc[7], 100)

    def test_fill_never_crosses_series_boundary(self):
        result = _panel(self.example())
        self.assertEqual(result.loc[result.unique_id.eq("b"), "y"].iloc[0], 0)

    def test_explicitly_masked_values_do_not_enter_predictor_fill(self):
        source = self.example()
        source["available_mask"] = source.y.notna().astype(int)
        source.loc[2, "y"] = 1e12
        result = _panel(source)
        self.assertEqual(result.y.iloc[2], 10)
        self.assertEqual(result.available_mask.iloc[2], 0)

    def test_unsorted_panel_mask_stays_attached_to_its_timestamp(self):
        source = self.example()
        source["available_mask"] = source.y.notna().astype(int)
        expected = _panel(source)
        actual = _panel(source.sample(frac=1, random_state=7))
        assert_frame_equal(expected, actual)

    def test_timestamps_preserve_utc_in_naive_nanosecond_format(self):
        source = self.example()
        source["ds"] = source.ds.dt.tz_convert("Europe/Paris")
        result = _panel(source)
        self.assertEqual(result.ds.dtype, np.dtype("datetime64[ns]"))
        self.assertEqual(result.ds.iloc[0], pd.Timestamp("2025-01-01 00:00"))

    def test_missing_timestamp_row_is_rejected_instead_of_compressed(self):
        with self.assertRaisesRegex(ValueError, "regular hourly grid"):
            _panel(self.example().drop(index=2))

    def test_duplicate_timestamp_key_is_rejected(self):
        source = self.example()
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            _panel(pd.concat([source, source.iloc[[2]]], ignore_index=True))

    def test_all_missing_series_is_rejected(self):
        source = self.example()
        source.loc[source.unique_id.eq("b"), "y"] = np.nan
        with self.assertRaisesRegex(ValueError, "at least one observed"):
            _panel(source)

    def test_invalid_loss_mask_is_rejected(self):
        source = self.example()
        source["available_mask"] = 1.
        source.loc[2, "available_mask"] = .5
        with self.assertRaisesRegex(ValueError, "only 0 or 1"):
            _panel(source)

    def test_unreleased_future_cannot_change_cutoff_panel_or_input_fill(self):
        contract = load_research_contract(project_root())
        origin = forecast_origin("2025-03-30", contract)
        index = pd.date_range("2025-03-01", "2025-04-01", freq="h", tz="UTC")
        national = pd.DataFrame({"load_mw": 1000.+np.arange(len(index))}, index=index)
        national.loc[index[10], "load_mw"] = np.nan
        regional = pd.DataFrame({"r1": national.load_mw/2, "r2": national.load_mw/3})
        data = ResearchData(Path("."), deepcopy(contract), national, regional,
                            regional*0, pd.DataFrame(), pd.DataFrame())
        before = _panel(data.panel(origin, arm="regional", window_days=10))
        cutoff = latest_eligible_time(origin, contract)
        data.national.loc[data.national.index > cutoff, "load_mw"] = 1e12
        data.regional_load.loc[data.regional_load.index > cutoff, :] = 1e12
        after = _panel(data.panel(origin, arm="regional", window_days=10))
        assert_frame_equal(before, after)

    def test_48_hour_output_grid_covers_both_dst_delivery_days(self):
        contract = load_research_contract(project_root())
        for day, expected in (("2025-03-30", 23), ("2025-10-26", 25)):
            end = latest_eligible_time(forecast_origin(day, contract), contract)
            future = pd.date_range(end+pd.Timedelta(hours=1), periods=48, freq="h")
            delivery = local_day_grid(day)
            self.assertEqual(len(delivery), expected)
            self.assertTrue(delivery.isin(future).all())
            self.assertTrue(delivery.is_unique)


@unittest.skipUnless(os.getenv("RUN_NEURAL_INTEGRATION_TESTS") == "1",
                     "Enable RUN_NEURAL_INTEGRATION_TESTS=1 for a real bounded DLinear fit")
class NeuralBackendIntegrationTests(unittest.TestCase):
    def test_real_dlinear_reproducibility_masked_values_and_cpu_policy(self):
        times = pd.date_range("2025-01-01", periods=600, freq="h")
        source = pd.DataFrame({"unique_id": "country", "ds": times,
                               "y": 1000.+100*np.sin(np.arange(600)*2*np.pi/24),
                               "available_mask": 1})
        source.loc[50, "available_mask"] = 0
        forecasts = []
        for masked_value in (np.nan, 1e12):
            panel = source.copy()
            panel.loc[50, "y"] = masked_value
            adapter = NeuralAdapter("dlinear", {
                "max_steps": 2, "val_check_steps": 1,
                "accelerator": "gpu",  # Adapter must enforce the declared CPU policy.
            }, seed=42).fit(panel, validation_size=168)
            self.assertEqual(next(adapter.forecaster.models[0].parameters()).device.type, "cpu")
            predicted = adapter.predict(panel.tail(336))
            self.assertEqual(len(predicted), 48)
            self.assertTrue(np.isfinite(predicted.prediction_mw).all())
            self.assertEqual(predicted.ds.min(), times[-1]+pd.Timedelta(hours=1))
            self.assertEqual(predicted.ds.max(), times[-1]+pd.Timedelta(hours=48))
            forecasts.append(predicted)
            adapter.close()
        assert_frame_equal(forecasts[0], forecasts[1], check_exact=True)


if __name__ == "__main__":
    unittest.main()
