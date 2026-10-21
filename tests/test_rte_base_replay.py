"""Selection timing and out-of-sample safeguards for cached base stacking."""
import unittest

import numpy as np
import pandas as pd

from src.rte_base_replay import causal_quarter_stream, select_released_trial


def forecasts(deliveries, predictions, origins=None):
    deliveries = pd.to_datetime(deliveries, utc=True)
    origins = pd.to_datetime(origins, utc=True) if origins is not None else deliveries - pd.Timedelta(hours=24)
    return pd.DataFrame({"origin": origins, "delivery_time": deliveries,
                         "prediction_mw": predictions, "raw_prediction_mw": predictions,
                         "fit_origin": origins - pd.Timedelta(days=7)})


class CausalBaseReplayTests(unittest.TestCase):
    def test_quarter_boundary_excludes_late_march_labels_and_exact_cutoff(self):
        cutoff = pd.Timestamp("2024-03-31 09:00Z")  # Apr1 delivery, D11 Paris
        times = ["2024-03-30 18:00Z", "2024-03-31 06:00Z", "2024-03-31 18:00Z"]
        labels = pd.DataFrame({"delivery_time": pd.to_datetime(times, utc=True),
                               "actual_mw": [100., 0., 0.],
                               "label_available_at": pd.to_datetime(["2024-03-30 21:00Z", "2024-03-31 09:00Z", "2024-03-31 21:00Z"], utc=True)})
        trials = {0: forecasts(times, [100., 10000., 10000.]), 1: forecasts(times, [101., 0., 0.])}
        trial, choices = select_released_trial(trials, labels, cutoff)
        self.assertEqual(trial, 0)
        self.assertTrue(all(item["eligible_targets"] == 1 for item in choices))
        self.assertTrue(all(pd.Timestamp(item["latest_label_available_at"]) < cutoff for item in choices))

    def test_future_outcome_changes_leave_earlier_selection_unchanged(self):
        times = ["2024-03-20 12:00Z", "2024-03-31 18:00Z"]
        labels = pd.DataFrame({"delivery_time": pd.to_datetime(times, utc=True), "actual_mw": [100., 0.],
                               "label_available_at": pd.to_datetime(times, utc=True) + pd.Timedelta(hours=3)})
        trials = {0: forecasts(times, [100., 1000.]), 1: forecasts(times, [101., 0.])}
        before = select_released_trial(trials, labels, pd.Timestamp("2024-03-31 09:00Z"))
        labels.loc[1, "actual_mw"] = 1e12
        after = select_released_trial(trials, labels, pd.Timestamp("2024-03-31 09:00Z"))
        self.assertEqual(before, after)

    def test_raw_availability_guard_uses_only_nonmissing_targets(self):
        times = ["2024-03-20 12:00Z", "2024-03-21 12:00Z"]
        labels = pd.DataFrame({"delivery_time": pd.to_datetime(times, utc=True), "actual_mw": [100., np.nan],
                               "label_available_at": pd.to_datetime(times, utc=True) + pd.Timedelta(hours=3)})
        trials = {0: forecasts(times, [100., 100.]), 1: forecasts(times, [101., 101.])}
        trials[0].loc[1, "raw_prediction_mw"] = np.nan
        trial, choices = select_released_trial(trials, labels, pd.Timestamp("2024-03-31 09:00Z"))
        self.assertEqual(trial, 0)
        self.assertEqual(choices[0]["raw_availability"], 1.)
        trials[0].loc[0, "raw_prediction_mw"] = np.nan
        trial, _ = select_released_trial(trials, labels, pd.Timestamp("2024-03-31 09:00Z"))
        self.assertEqual(trial, 1)

    def test_burnin_fixed_and_later_selection_is_causal(self):
        q1 = {0: forecasts(["2024-01-15 12:00Z"], [100.]), 1: forecasts(["2024-01-15 12:00Z"], [90.])}
        q2 = {0: forecasts(["2024-04-01 12:00Z"], [110.]), 1: forecasts(["2024-04-01 12:00Z"], [120.])}
        labels = pd.DataFrame({"delivery_time": pd.to_datetime(["2024-01-15 12:00Z", "2024-04-01 12:00Z"], utc=True),
                               "actual_mw": [90., 110.],
                               "label_available_at": pd.to_datetime(["2024-01-15 15:00Z", "2024-04-01 15:00Z"], utc=True)})
        stream, ledger = causal_quarter_stream([("Q1", q1), ("Q2", q2)], labels)
        self.assertEqual(list(stream.prediction_mw), [100., 120.])
        self.assertEqual([item["selected_trial"] for item in ledger], [0, 1])
        self.assertTrue(ledger[0]["burnin_only"])

    def test_rejects_outcome_columns_and_post_origin_fitting(self):
        clean = forecasts(["2024-01-15 12:00Z"], [100.])
        labels = pd.DataFrame({"delivery_time": clean.delivery_time, "actual_mw": [100.],
                               "label_available_at": clean.delivery_time + pd.Timedelta(hours=3)})
        leaked = clean.assign(actual_mw=100.)
        with self.assertRaisesRegex(ValueError, "Outcome columns"):
            causal_quarter_stream([("Q1", {0: leaked, 1: clean})], labels)
        future_fit = clean.assign(fit_origin=clean.origin + pd.Timedelta(hours=1))
        with self.assertRaisesRegex(ValueError, "In-sample or post-origin"):
            causal_quarter_stream([("Q1", {0: future_fit, 1: clean})], labels)

    def test_ties_choose_trial_zero(self):
        frame = forecasts(["2024-01-15 12:00Z"], [100.])
        labels = pd.DataFrame({"delivery_time": frame.delivery_time, "actual_mw": [99.],
                               "label_available_at": frame.delivery_time + pd.Timedelta(hours=3)})
        trial, _ = select_released_trial({1: frame, 0: frame}, labels, pd.Timestamp("2024-04-01", tz="UTC"))
        self.assertEqual(trial, 0)


if __name__ == "__main__":
    unittest.main()
