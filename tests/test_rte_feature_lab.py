"""Cutoff, data-arm and DST invariants of the expanded RTE-assisted features."""
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

from src.common import forecast_origin, local_day_grid, project_root
from src.research_data import latest_eligible_time, load_research_contract
from src.rte_feature_lab import FeatureLabData, build_lab_origin_features


class FeatureLabTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.contract = load_research_contract(project_root())

    def fixture(self, day="2024-02-15"):
        origin = forecast_origin(day, self.contract)
        grid = pd.date_range(origin.floor("D")-pd.Timedelta(days=40),
                             origin.floor("D")+pd.Timedelta(days=5), freq="h")
        t = np.arange(len(grid), dtype=float)
        national = pd.DataFrame({"load_mw": 1000+t,
            "temperature_c": 7+np.sin(t/24), "rte_forecast_d1_mw": 990+t*.95,
            "label_available_at_assumed": grid+pd.Timedelta(hours=3)}, index=grid)
        load = pd.DataFrame({"north": 200+t*.4, "south": 300+t*.6}, index=grid)
        temp = pd.DataFrame({"north": 3+np.cos(t/24), "south": 9+np.sin(t/24)}, index=grid)
        return origin, national, load, temp

    def build(self, day, national, load, temp):
        return build_lab_origin_features(national, load, temp, day, self.contract)

    def view(self, national, load, temp, features):
        return FeatureLabData(Path("."), self.contract, national, load, temp, features, pd.DataFrame())

    def test_postcutoff_actuals_and_temperatures_cannot_change_features(self):
        day = "2024-02-15"
        origin, national, load, temp = self.fixture(day)
        before = self.build(day, national, load, temp)
        unpublished = national.label_available_at_assumed.ge(origin)
        national.loc[unpublished, ["load_mw", "temperature_c"]] = 1e12
        load.loc[unpublished] = 2e12
        temp.loc[unpublished] = -1e12
        after = self.build(day, national, load, temp)
        pd.testing.assert_frame_equal(before, after)
        self.assertTrue(before.max_history_available_at.lt(origin).all())

    def test_future_other_delivery_day_rte_does_not_change_current_features(self):
        day = "2024-02-15"
        origin, national, load, temp = self.fixture(day)
        before = self.build(day, national, load, temp)
        other_future = (national.index >= origin) & ~national.index.isin(local_day_grid(day))
        national.loc[other_future, "rte_forecast_d1_mw"] = 1e12
        after = self.build(day, national, load, temp)
        pd.testing.assert_frame_equal(before, after)
        national.loc[local_day_grid(day)[-1], "rte_forecast_d1_mw"] += 123
        changed_target_day = self.build(day, national, load, temp)
        self.assertFalse(before.rte_profile_day_mean_mw.equals(changed_target_day.rte_profile_day_mean_mw))

    def test_dst_days_keep_all_unique_delivery_intervals(self):
        for day, expected in (("2025-03-30", 23), ("2025-10-26", 25)):
            with self.subTest(day=day):
                origin, national, load, temp = self.fixture(day)
                f = self.build(day, national, load, temp)
                self.assertEqual(len(f), expected)
                self.assertTrue(f.delivery_time.is_unique)
                self.assertTrue(f.cal_day_hours.eq(expected).all())
                self.assertTrue(f.max_history_available_at.lt(origin).all())

    def test_regional_only_has_no_country_history_or_errors(self):
        day = "2024-02-15"
        _, national, load, temp = self.fixture(day)
        before = self.build(day, national, load, temp)
        view = self.view(national, load, temp, before)
        columns = view.columns("regional_only")
        self.assertIn("rte_at_origin_mw", columns)
        self.assertIn("regional_load_mean24h", columns)
        self.assertNotIn("rte_bias_recent_mw", columns)
        self.assertFalse(any(c.startswith("country_") for c in columns))
        national.loc[:, "load_mw"] = -9999
        national.loc[:, "temperature_c"] = 999
        after = self.build(day, national, load, temp)
        pd.testing.assert_frame_equal(before[columns], after[columns])
        self.assertTrue(set(view.columns("national", tier="basic")) < set(view.columns("national")))
        self.assertTrue(set(view.columns("regional_only")) < set(view.columns("combined")))

    def test_publication_at_origin_is_excluded_and_lag24_masked(self):
        day = "2024-02-15"
        origin, national, load, temp = self.fixture(day)
        f = self.build(day, national, load, temp)
        end = latest_eligible_time(origin, self.contract)
        self.assertEqual(f.country_load_origin_lag0h.iloc[0], national.loc[end, "load_mw"])
        excluded = end+pd.Timedelta(hours=1)
        self.assertEqual(national.loc[excluded, "label_available_at_assumed"], origin)
        times = pd.DatetimeIndex(f.delivery_time)-pd.Timedelta(hours=24)
        allowed = times+pd.Timedelta(hours=3) < origin
        self.assertTrue(f.loc[~allowed, "country_load_target_lag24h"].isna().all())
        np.testing.assert_array_equal(f.country_load_target_lag24h_available, allowed.astype(int))
        np.testing.assert_allclose(f.loc[allowed, "country_load_target_lag24h"], national.load_mw.reindex(times[allowed]))

    def test_explicit_late_national_publication_is_also_excluded(self):
        day = "2024-02-15"
        origin, national, load, temp = self.fixture(day)
        end = latest_eligible_time(origin, self.contract)
        national.loc[end, "label_available_at_assumed"] = origin+pd.Timedelta(minutes=1)
        before = self.build(day, national, load, temp)
        national.loc[end, ["load_mw", "temperature_c"]] = 1e12
        after = self.build(day, national, load, temp)
        pd.testing.assert_frame_equal(before, after)
        self.assertTrue(before.country_load_origin_lag0h.isna().all())

    def test_incomplete_regional_aggregate_and_rte_day_not_silently_averaged(self):
        day = "2024-02-15"
        origin, national, load, temp = self.fixture(day)
        end = latest_eligible_time(origin, self.contract)
        load.loc[end, "north"] = np.nan
        temp.loc[end, "south"] = np.nan
        national.loc[local_day_grid(day)[3], "rte_forecast_d1_mw"] = np.nan
        f = self.build(day, national, load, temp)
        self.assertTrue(f.regional_load_origin_lag0h.isna().all())
        self.assertTrue(f.regional_temperature_origin_lag0h.isna().all())
        self.assertTrue(f.rte_profile_day_mean_mw.isna().all())
        self.assertTrue(f.rte_profile_available_fraction.eq(23/24).all())
        register = self.view(national, load, temp, f).feature_register()
        self.assertFalse(register.feature.duplicated().any())
        self.assertEqual(set(register.feature), set(self.view(national, load, temp, f).columns("combined")))


if __name__ == "__main__":
    unittest.main()
