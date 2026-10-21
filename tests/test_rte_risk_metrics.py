"""Numerical contracts for cold-risk point losses and probabilistic scores."""
from types import SimpleNamespace
import unittest

import numpy as np
import pandas as pd

from src.rte_risk_lab import point_scores
from src.rte_risk_uncertainty import assemble_quantiles, interval_score, pinball, quantile_scores, signed_quantile


class RiskMetricTests(unittest.TestCase):
    def test_normalized_cold_weighted_mae_excludes_only_missing_targets(self):
        # Winter local18:00 is UTC17:00; the extreme but genuine observation
        # receives the predeclared fivefold weight and remains in the score.
        times = pd.to_datetime(["2025-01-01 10:00Z", "2025-01-01 17:00Z",
                                "2025-01-01 18:00Z", "2025-01-01 19:00Z"])
        p = pd.DataFrame({"delivery_time": times, "local_day": "2025-01-01",
                          "prediction_mw": [90., 970., 150., 0.],
                          "raw_prediction_mw": [90., 970., 150., 0.]})
        labels = pd.DataFrame({"delivery_time": times, "actual_mw": [100., 1000., 100., np.nan],
                               "realized_temperature_c": [2., 2., 10., 2.]})
        score = point_scores(p, labels)
        self.assertEqual(score["n"], 3)
        self.assertEqual(score["missing_target_n"], 1)
        self.assertAlmostEqual(score["MAE_MW"], (10+30+50)/3)
        self.assertAlmostEqual(score["weighted_MAE_MW"], (10+5*30+50)/7)
        self.assertEqual(score["cold_n"], 1)
        self.assertEqual(score["cold_days"], 1)
        self.assertEqual(score["cold_MAE_MW"], 30)

    def test_upper_quantile_pinball_penalizes_shortage_more(self):
        loss = pinball(np.array([100., 100.]), np.array([90., 110.]), .9)
        np.testing.assert_allclose(loss, [9., 1.])
        np.testing.assert_allclose(pinball([100.], [100.], .9), [0.])

    def test_winkler_penalizes_both_interval_exceedances(self):
        values = interval_score([100., 120., 80.], [90., 90., 90.], [110., 110., 110.], .1)
        np.testing.assert_allclose(values, [20., 220., 220.])

    def test_wis_uses_central_interval_weights_and_missing_targets_are_unscored(self):
        times = pd.date_range("2025-01-01", periods=3, freq="h", tz="UTC")
        p = pd.DataFrame({"delivery_time": times, "q05": 90., "q25": 95., "q50": 100., "q75": 105., "q95": 110.})
        labels = pd.DataFrame({"delivery_time": times, "actual_mw": [100., 120., np.nan]})
        result = quantile_scores(p, labels)
        # At100: IS90=20, IS50=10. At120: IS90=220, IS50=70.
        first = (.5*0+.05*20+.25*10)/2.5
        second = (.5*20+.05*220+.25*70)/2.5
        self.assertEqual(result["n"], 2)
        self.assertAlmostEqual(result["WIS_MW"], (first+second)/2)
        self.assertEqual(result["coverage_90"], .5)
        self.assertEqual(result["upper_exceedance_rate"], .5)

    def test_quantile_rearrangement_preserves_keys_and_does_not_hide_missing_raw_forecasts(self):
        delivery = pd.date_range("2025-10-26", periods=2, freq="h", tz="UTC")
        origin = pd.Timestamp("2025-10-25 09:00Z")
        values = {.05: [100., 90.], .5: [80., np.nan], .95: [120., 110.]}
        def replay(variant, params, start, end, phase, q, fixed):
            return pd.DataFrame({"model": "test", "origin": origin, "delivery_time": delivery,
                                 "local_day": "2025-10-26", "fit_origin": origin,
                                 "raw_prediction_mw": values[q], "prediction_mw": [100., 100.]})
        run = SimpleNamespace(spec={"quantiles": [.05, .5, .95]}, replay=replay)
        p = assemble_quantiles(run, {"variant": {}, "params": {}}, "a", "b", "synthetic")
        np.testing.assert_array_equal(p.loc[0, ["q05", "q50", "q95"]].astype(float), [80., 100., 120.])
        self.assertTrue(p.loc[1, ["q05", "q50", "q95"]].isna().all())
        self.assertTrue(p.loc[0, "quantile_crossing_repaired"])
        pd.testing.assert_index_equal(pd.DatetimeIndex(p.delivery_time), delivery, check_names=False)

    def test_conformal_order_statistic_is_finite_sample_rank_not_interpolated_quantile(self):
        self.assertEqual(signed_quantile(np.arange(1, 20), .1), 18.)
        self.assertTrue(np.isinf(signed_quantile(np.arange(1, 20), .025)))
        self.assertEqual(signed_quantile([1., 2., np.nan, 3.], .5), 2.)


if __name__ == "__main__":
    unittest.main()
