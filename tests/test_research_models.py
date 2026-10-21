"""Behavioral checks for fixed-parameter updates and training-only transforms."""
import unittest

import numpy as np
import pandas as pd

from src.research_models import make_statistical, make_tabular, parameter_grid


class ResearchModelTests(unittest.TestCase):
    def setUp(self):
        self.index = pd.date_range("2024-03-25", periods=240, freq="h", tz="UTC")
        t = np.arange(len(self.index))
        self.y = pd.Series(50000 + 3000*np.sin(2*np.pi*t/24), index=self.index)
        self.x = pd.DataFrame({"hour": self.index.hour, "trend": t,
                               "constant": 1., "missing": np.nan}, index=self.index)

    def test_additive_transform_is_fitted_only_on_training_data(self):
        model = make_tabular("gam", {"n_knots": 4, "alpha": 10.}).fit(self.x.iloc[:168], self.y.iloc[:168])
        statistics = model.estimator_["imputer"].statistics_.copy()
        future = self.x.iloc[168:].copy()
        future["trend"] = 1e12
        model.predict(future)
        np.testing.assert_equal(model.estimator_["imputer"].statistics_, statistics)
        np.testing.assert_allclose(model.predict(self.x.iloc[168:]),
                                   model.predict(self.x.iloc[168:][list(reversed(self.x.columns))]))

    def test_missing_labels_are_not_silently_imputed(self):
        labels = self.y.iloc[:168].copy()
        labels.iloc[10] = np.nan
        with self.assertRaises(ValueError):
            make_tabular("gam").fit(self.x.iloc[:168], labels)

    def test_statistical_update_keeps_parameters_and_hourly_grid(self):
        observed = self.y.iloc[:168].drop(self.index[20])
        model = make_statistical("arima", {"order": [1, 0, 0], "maxiter": 10}).fit(observed)
        parameters = np.asarray(model.result_.params).copy()
        self.assertEqual(model.metadata_["missing_training_rows"], 1)
        update = self.y.iloc[168:192].drop(self.index[180])
        model.update(update)
        np.testing.assert_allclose(model.result_.params, parameters)
        self.assertEqual(model.last_timestamp_, self.index[191])
        self.assertEqual(model.metadata_["parameter_refits"], 1)
        self.assertEqual(len(model.forecast(48)), 48)
        self.assertTrue(np.isfinite(model.forecast(48)).all())

    def test_exogenous_future_must_cover_complete_bridge(self):
        model = make_statistical("sarimax", {"order": [1, 0, 0], "seasonal_order": [0, 0, 0, 0], "maxiter": 10})
        model.fit(self.y.iloc[:168], self.x[["hour"]].iloc[:168])
        with self.assertRaises(ValueError):
            model.forecast(48, self.x[["hour"]].iloc[168:192])
        forecast = model.forecast(48, self.x[["hour"]].iloc[168:216])
        self.assertTrue(np.isfinite(forecast).all())

    def test_random_early_stopping_is_rejected(self):
        with self.assertRaises(ValueError):
            make_tabular("hist_gradient_boosting", {"early_stopping": True}).fit(self.x, self.y)

    def test_quantile_models_have_explicit_level(self):
        self.assertEqual(make_tabular("lightgbm_quantile").quantile, .5)
        with self.assertRaises(ValueError):
            make_tabular("lightgbm_quantile", quantile=1.)
        self.assertEqual(len(parameter_grid("arima", "full")), 8)
        self.assertEqual(len(parameter_grid("gam", "full")), 12)


if __name__ == "__main__":
    unittest.main()
