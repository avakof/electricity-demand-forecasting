"""Small numerical and causal checks for the RTE risk experiment."""
from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.common import forecast_origin, local_day_grid, project_root
from src.rte_risk_lab import RiskLabRun, WeightedAdapter, point_scores
from src.rte_risk_uncertainty import Q_NAMES, assemble_quantiles, interval_score, pinball, quantile_scores, signed_quantile, uncertainty_replay


class MeanAdapter:
    fits = []

    def __init__(self, *args, **kwargs):
        self.warnings = []

    def fit(self, X, y, weights):
        self.value = float(np.average(y, weights=weights))
        self.fits.append({"columns": list(X), "target": np.asarray(y).copy(), "weights": np.asarray(weights).copy()})
        return self

    def predict(self, X):
        return np.full(len(X), self.value)


def variant(mode="residual", tier="stacked", arm="national"):
    return dict(model=f"synthetic__{mode}__{tier}__{arm}", family="lightgbm", arm=arm,
                mode=mode, tier=tier, cold_weight=5, base=None, restrict_to_base_burnin=False)


def run_fixture(directory, day="2025-07-01"):
    run = RiskLabRun(project_root(), Path(directory)/"run")
    run.verify_manifest = lambda: True
    # The single synthetic forecast day belongs to development in these tests.
    run.spec["test"]["start"] = "2025-07-02"
    history = pd.date_range("2025-01-01", periods=24*100, freq="h", tz="UTC")
    future = local_day_grid(day)
    times = history.append(future)
    local = times.tz_convert("Europe/Paris")
    origins = [forecast_origin(x, run.contract) for x in local.strftime("%Y-%m-%d")]
    frame = pd.DataFrame({"origin": origins, "delivery_time": times,
        "local_day": local.strftime("%Y-%m-%d"), "hour": local.hour,
        "rte_at_origin_mw": 100., "rte_bias_recent_mw": 10., "x": 1.,
        "week_lag_1": 95., "comparable_mean": 96., "last_load": 97.})
    frame["max_history_available_at"] = frame.origin-pd.Timedelta(hours=1)
    for arm in ("national", "regional_only", "combined"):
        frame[f"learned_ridge_demand_mw__{arm}"] = 110.
        frame[f"learned_cold_probability__{arm}"] = .25
        frame[f"learned_underprediction_probability__{arm}"] = .1
    labels = pd.DataFrame({"delivery_time": times, "actual_mw": 110.,
        "label_available_at": times+pd.Timedelta(hours=3), "realized_temperature_c": 1.})
    data = SimpleNamespace(features=frame, labels=labels,
                           columns=lambda arm, tier="engineered": ["rte_at_origin_mw", "x"])
    run._data, run._frame = data, frame
    return run


class RiskLabTests(unittest.TestCase):
    def setUp(self):
        MeanAdapter.fits = []

    def test_weighted_mae_arithmetic_and_error_sign(self):
        times = pd.DatetimeIndex(["2025-01-01 17:00Z", "2025-01-01 20:00Z", "2025-01-02 17:00Z"])
        predictions = pd.DataFrame({"delivery_time": times, "local_day": ["2025-01-01", "2025-01-01", "2025-01-02"],
                                    "prediction_mw": [90., 130., 100.], "raw_prediction_mw": [90., 130., 100.]})
        labels = pd.DataFrame({"delivery_time": times, "actual_mw": [100., 100., 100.],
                               "realized_temperature_c": [0., 0., 10.]})
        score = point_scores(predictions, labels)
        self.assertAlmostEqual(score["weighted_MAE_MW"], 80/7)
        self.assertEqual(score["weighted_loss_sum"], 80.)
        self.assertEqual(score["weight_sum"], 7.)
        self.assertAlmostEqual(score["MAE_MW"], 40/3)
        self.assertAlmostEqual(score["bias_prediction_minus_actual_MW"], 20/3)
        self.assertEqual(score["cold_n"], 1)
        self.assertEqual(score["cold_days"], 1)

    def test_sample_weights_change_actual_boosting_fit(self):
        X = pd.DataFrame({"x": np.zeros(40)})
        y = np.r_[np.zeros(30), np.full(10, 10.)]
        params = dict(max_iter=2, max_leaf_nodes=3, min_samples_leaf=2, l2_regularization=0.)
        uniform = WeightedAdapter("hist_gradient_boosting", params).fit(X, y, np.ones(40)).predict(X.iloc[:1])[0]
        weighted = WeightedAdapter("hist_gradient_boosting", params).fit(X, y, np.r_[np.ones(30), np.full(10, 10.)]).predict(X.iloc[:1])[0]
        self.assertAlmostEqual(uniform, 0.)
        self.assertAlmostEqual(weighted, 10.)

    @patch("src.rte_risk_lab.WeightedAdapter", MeanAdapter)
    def test_direct_and_residual_add_rte_exactly_once(self):
        with TemporaryDirectory() as directory:
            run = run_fixture(directory)
            direct = run.replay(variant("direct"), {}, "2025-07-01", "2025-07-02", "validation")
            residual = run.replay(variant("residual"), {}, "2025-07-01", "2025-07-02", "validation")
            np.testing.assert_allclose(direct.prediction_mw, 110.)
            np.testing.assert_allclose(residual.prediction_mw, 110.)
            np.testing.assert_allclose(MeanAdapter.fits[0]["target"], 110.)
            np.testing.assert_allclose(MeanAdapter.fits[1]["target"], 10.)

    @patch("src.rte_risk_lab.WeightedAdapter", MeanAdapter)
    def test_training_weights_ignore_future_realized_weather_and_demand(self):
        with TemporaryDirectory() as directory:
            run = run_fixture(directory)
            a = run.replay(variant(), {}, "2025-07-01", "2025-07-02", "before_mutation")
            origin = forecast_origin("2025-07-01", run.contract)
            unavailable = run.data.labels.label_available_at.ge(origin)
            run.data.labels.loc[unavailable, "realized_temperature_c"] = -1000.
            run.data.labels.loc[unavailable, "actual_mw"] = 1e9
            b = run.replay(variant(), {}, "2025-07-01", "2025-07-02", "after_mutation")
            pd.testing.assert_frame_equal(a, b)
            np.testing.assert_array_equal(MeanAdapter.fits[0]["weights"], MeanAdapter.fits[1]["weights"])
            self.assertEqual(set(MeanAdapter.fits[0]["weights"]), {1., 2.})

    @patch("src.rte_risk_lab.WeightedAdapter", MeanAdapter)
    def test_cache_tamper_and_source_verification(self):
        with TemporaryDirectory() as directory:
            run = run_fixture(directory)
            run.replay(variant(), {}, "2025-07-01", "2025-07-02", "validation")
            count = len(MeanAdapter.fits)
            run.replay(variant(), {}, "2025-07-01", "2025-07-02", "validation")
            self.assertEqual(len(MeanAdapter.fits), count)
            run.verify_manifest = lambda: (_ for _ in ()).throw(AssertionError("Frozen source changed"))
            with self.assertRaisesRegex(AssertionError, "Frozen source changed"):
                run.replay(variant(), {}, "2025-07-01", "2025-07-02", "validation")
            run.verify_manifest = lambda: True
            next((run.path/"fits").glob("*.json")).write_text("[]")
            with self.assertRaisesRegex(AssertionError, "Altered forecast cache"):
                run.replay(variant(), {}, "2025-07-01", "2025-07-02", "validation")

    def test_first_freeze_rejects_native_quantile_test_artifact(self):
        with TemporaryDirectory() as directory:
            run = run_fixture(directory)
            (run.path/"forecasts/lab__synthetic__native_quantile_test__hash.json").write_text(json.dumps({"end": "2026-07-01"}))
            with self.assertRaisesRegex(AssertionError, "freeze|test"):
                run.freeze()

    def test_unfrozen_test_overlap_is_rejected_before_fit_or_artifact(self):
        with TemporaryDirectory() as directory:
            run = run_fixture(directory)
            with patch("src.rte_risk_lab.WeightedAdapter", MeanAdapter):
                with self.assertRaisesRegex(AssertionError, "Freeze model selection"):
                    run.replay(variant(), {}, "2025-07-01", "2025-07-03", "arbitrary_phase_name")
            self.assertFalse(MeanAdapter.fits)
            self.assertFalse(list((run.path/"forecasts").iterdir()))

    def test_crossing_repair_preserves_missing_quantile_rows(self):
        deliveries = pd.date_range("2025-01-02", periods=2, freq="h", tz="UTC")
        grid = pd.DataFrame({"model": "synthetic", "origin": pd.Timestamp("2025-01-01 10:00Z"),
            "delivery_time": deliveries, "local_day": "2025-01-02", "fit_origin": pd.Timestamp("2025-01-01 10:00Z")})
        def replay(*args, q=None, **kwargs):
            result = grid.copy()
            result["raw_prediction_mw"] = {0.05: [3., 1.], .5: [2., np.nan], .95: [1., 3.]}[q]
            return result
        run = SimpleNamespace(spec={"quantiles": [.05, .5, .95]}, replay=replay)
        result = assemble_quantiles(run, {"variant": {}, "params": {}}, "2025-01-02", "2025-01-03", "validation")
        np.testing.assert_allclose(result.loc[0, ["q05", "q50", "q95"]].to_numpy(dtype=float), [1., 2., 3.])
        self.assertTrue(result.loc[0, "quantile_crossing_repaired"])
        self.assertTrue(result.loc[1, ["q05", "q50", "q95"]].isna().all())

    def test_wis_interval_and_pinball_constants(self):
        times = pd.DatetimeIndex(["2025-01-01 00:00Z"])
        prediction = pd.DataFrame({"delivery_time": times, "q05": [0.], "q25": [4.],
                                    "q50": [6.], "q75": [8.], "q95": [12.]})
        labels = pd.DataFrame({"delivery_time": times, "actual_mw": [10.]})
        # IS90=12; IS50=4+4*2=12; median AE=4. WIS=(2+.6+3)/2.5.
        self.assertAlmostEqual(quantile_scores(prediction, labels)["WIS_MW"], 2.24)
        np.testing.assert_array_equal(interval_score([10.], [4.], [8.], .5), [12.])
        np.testing.assert_allclose(pinball([10., 0.], [8., 2.], .9), [1.8, .2])

    def test_finite_sample_tail_ranks_are_directionally_correct(self):
        scores = np.arange(40.)
        self.assertEqual(signed_quantile(scores, .075), 37.)
        self.assertEqual(signed_quantile(scores, .025), 39.)
        self.assertTrue(np.isinf(signed_quantile([1.], .025)))

    def test_cqr_fixed_mapping_and_strict_calibration_tail_release(self):
        with TemporaryDirectory() as directory:
            run = run_fixture(directory)
            run.spec["calibration"] = {"start": "2025-01-02", "end": "2025-06-30"}
            run.spec["test"] = {"start": "2025-07-01", "end": "2025-07-02"}
            entry = {"variant": variant(), "params": {}}
            run.freeze = lambda: {"selected_quantile": "candidate", "selections": {"candidate": entry}}
            cal_times = pd.date_range("2025-01-02", periods=81, freq="D", tz="UTC")
            test_time = pd.Timestamp("2025-07-01", tz="UTC")
            times = cal_times.append(pd.DatetimeIndex([test_time]))
            values = np.r_[np.tile(np.arange(-10., 30.), 2), 1e9, 30.]
            run.data.labels = pd.DataFrame({"delivery_time": times, "actual_mw": values,
                "label_available_at": times+pd.Timedelta(hours=3)})
            cutoff = forecast_origin("2025-07-01", run.contract)
            run.data.labels.loc[80, "label_available_at"] = cutoff
            calls = []
            def fake_assemble(actual_run, selected, start, end, phase, fixed=False):
                calls.append((deepcopy(selected), fixed))
                grid = times if fixed else pd.DatetimeIndex([test_time])
                result = pd.DataFrame({"origin": [forecast_origin(x.strftime("%Y-%m-%d"), run.contract) for x in grid],
                    "delivery_time": grid, "local_day": grid.strftime("%Y-%m-%d"),
                    "fit_origin": pd.Timestamp("2025-01-01 10:00Z"), "model": selected["variant"]["model"]})
                for name, value in zip(Q_NAMES.values(), [0., 2., 4., 8., 12., 14., 16., 18.]):
                    result[name] = value
                result["method"] = "native_quantile_lightgbm"
                return result
            with patch("src.rte_risk_uncertainty.assemble_quantiles", side_effect=fake_assemble):
                result = uncertainty_replay(run)
            self.assertEqual(calls[0][0]["variant"]["tier"], "stacked")
            self.assertFalse(calls[0][1])
            self.assertEqual(calls[1][0]["variant"]["tier"], "engineered")
            self.assertTrue(calls[1][1])
            self.assertFalse(any(c.startswith("learned_") for c in run.columns(calls[1][0]["variant"])))
            protocol = json.loads((run.path/"tables/uncertainty_protocol.json").read_text())
            self.assertEqual(protocol["calibration_n"], 80)
            self.assertLess(pd.Timestamp(protocol["calibration_latest_available_at"]), cutoff)
            self.assertEqual(protocol["CQR_lower_expansion_MW"], 10.)
            self.assertEqual(protocol["CQR_upper_expansion_MW"], 11.)
            cqr = result.loc[result.method.eq("fixed_engineered_CQR")].iloc[0]
            self.assertEqual(cqr.q075, -8.)
            self.assertEqual(cqr.q975, 29.)
            self.assertTrue(pd.isna(cqr.q25))


if __name__ == "__main__":
    unittest.main()
