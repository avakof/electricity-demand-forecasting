"""Temporal stacking must keep original-origin feature and outcome cutoffs."""
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import numpy as np
import pandas as pd

from src.rte_learned_features import build_learned_feature_bank, learned_columns, predicted_cold_evening_weights


def synthetic_data(days=240):
    origins = pd.date_range("2023-01-01 10:00", periods=days, freq="D", tz="UTC")
    delivery = origins+pd.Timedelta(hours=32)
    x = np.sin(np.arange(days)/11)
    features = pd.DataFrame({"origin": origins, "delivery_time": delivery,
        "max_history_available_at": origins-pd.Timedelta(hours=1),
        "rte_at_origin_mw": 10000+100*x, "cal_evening": 1,
        "country_lag": x, "regional_lag": x*.9,
        "all_missing": np.nan})
    labels = pd.DataFrame({"delivery_time": delivery, "actual_mw": 10100+1500*x,
        "label_available_at": delivery+pd.Timedelta(hours=3),
        "realized_temperature_c": 5+8*x})
    data = SimpleNamespace(features=features, labels=labels, contract={"timezone": "Europe/Paris"})
    columns = {"national": ["rte_at_origin_mw", "country_lag", "all_missing"],
               "regional_only": ["rte_at_origin_mw", "regional_lag", "all_missing"],
               "combined": ["rte_at_origin_mw", "country_lag", "regional_lag", "all_missing"]}
    data.columns = lambda arm, tier="engineered": columns[arm]
    return data


class LearnedFeatureTests(unittest.TestCase):
    def test_future_labels_do_not_change_earlier_oos_features(self):
        original = synthetic_data()
        cutoff = pd.Timestamp("2023-07-10 10:00", tz="UTC")
        before = build_learned_feature_bank(original)
        changed = deepcopy(original)
        future = changed.labels.label_available_at.ge(cutoff)
        changed.labels.loc[future, "actual_mw"] = 1e9
        changed.labels.loc[future, "realized_temperature_c"] = -100
        after = build_learned_feature_bank(changed)
        mask = before.origin.le(cutoff)
        pd.testing.assert_frame_equal(before.loc[mask], after.loc[mask])
        self.assertTrue(before.loc[before.origin.lt("2023-04-01"), learned_columns("national")].isna().all().all())
        self.assertTrue(before.loc[before.origin.ge("2023-05-01"), learned_columns("national")].notna().all().all())

    def test_fit_ledger_proves_label_and_feature_release_cutoffs_and_arm_scope(self):
        bank = build_learned_feature_bank(synthetic_data())
        for record in bank.attrs["fit_ledger"]:
            self.assertTrue(record["out_of_sample_bank"])
            self.assertTrue(record["original_feature_cutoffs_verified"])
            if record["training_rows"]:
                self.assertLess(pd.Timestamp(record["latest_label_available_at"]), pd.Timestamp(record["fit_origin"]))
                self.assertLess(pd.Timestamp(record["latest_training_origin"]), pd.Timestamp(record["fit_origin"]))
            if record["arm"] == "regional_only":
                self.assertNotIn("country_lag", record["predictor_columns"])
                self.assertIn("regional_lag", record["predictor_columns"])

    def test_one_class_probability_is_explicit_training_frequency(self):
        data = synthetic_data(160)
        data.labels["realized_temperature_c"] = 0.
        data.labels["actual_mw"] = data.features.rte_at_origin_mw
        bank = build_learned_feature_bank(data, arms=("regional_only",))
        cold, under = learned_columns("regional_only")[1:]
        self.assertTrue(bank[cold].dropna().eq(1).all())
        self.assertTrue(bank[under].dropna().eq(0).all())
        fitted = [r for r in bank.attrs["fit_ledger"] if r["status"] == "empirical_one_class"]
        self.assertTrue(fitted)
        self.assertEqual({r["event_probability"] for r in fitted}, {0., 1.})

    def test_explicit_late_temperature_release_blocks_cold_classifier(self):
        data = synthetic_data(160)
        data.labels["temperature_available_at"] = pd.Timestamp("2030-01-01", tz="UTC")
        bank = build_learned_feature_bank(data, arms=("national",))
        self.assertTrue(bank[learned_columns("national")[1]].isna().all())
        self.assertTrue(bank[learned_columns("national")[0]].notna().any())

    def test_original_row_history_cutoff_violation_is_rejected(self):
        data = synthetic_data(100)
        data.features.loc[5, "max_history_available_at"] = data.features.loc[5, "origin"]
        with self.assertRaisesRegex(AssertionError, "original origin"):
            build_learned_feature_bank(data)

    def test_label_release_before_observed_interval_end_is_rejected(self):
        data = synthetic_data(100)
        data.labels.loc[5, "label_available_at"] = data.features.loc[5, "origin"]
        with self.assertRaisesRegex(AssertionError, "interval end"):
            build_learned_feature_bank(data)

    def test_regional_only_rejects_national_history_predictor(self):
        data = synthetic_data(100)
        data.columns = lambda arm, tier="engineered": ["rte_at_origin_mw", "country_lag"]
        with self.assertRaisesRegex(AssertionError, "regional-only"):
            build_learned_feature_bank(data, arms=("regional_only",))

    def test_cache_reuse_and_tamper_detection(self):
        data = synthetic_data(130)
        with TemporaryDirectory() as directory:
            first = build_learned_feature_bank(data, cache_dir=directory, arms=("national",))
            second = build_learned_feature_bank(data, cache_dir=directory, arms=("national",))
            pd.testing.assert_frame_equal(first, second)
            cache = Path(first.attrs["cache_path"])
            (cache/"fit_ledger.json").write_text("[]")
            with self.assertRaisesRegex(RuntimeError, "Altered learned-feature cache"):
                build_learned_feature_bank(data, cache_dir=directory, arms=("national",))

    def test_cold_weights_use_forecast_probability_and_keep_missing_neutral(self):
        data = synthetic_data(3)
        bank = data.features[["origin", "delivery_time"]].copy()
        bank[learned_columns("national")[1]] = [np.nan, .25, 1.]
        weights = predicted_cold_evening_weights(data.features, bank, "national", multiplier=3)
        np.testing.assert_array_equal(weights, [1., 1.5, 3.])
        data.features["realized_temperature_c"] = [-100, 100, -50]
        np.testing.assert_array_equal(predicted_cold_evening_weights(data.features, bank, "national", 3), weights)


if __name__ == "__main__":
    unittest.main()
