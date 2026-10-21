"""Check same-fit aggregation and the reporting script's development-only gate."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from scripts.development_fit_diagnostics import build_diagnostics


class DevelopmentFitDiagnosticsTests(unittest.TestCase):
    def fixture(self, root):
        (root/"data").mkdir()
        (root/"data/france_electricity_hourly.csv").write_text(
            "datetime_utc,load_mw\n"
            "2023-01-01 00:00:00+0000,earlier_value_not_needed\n"
            "2024-01-01 00:00:00+0000,100\n"
            "2024-01-02 00:00:00+0000,100\n"
            "2024-01-03 00:00:00+0000,\n"
            "2024-01-04 00:00:00+0000,100\n"
            "2025-01-01 00:00:00+0000,test_value_must_not_be_parsed\n")
        run=root/"run"
        for name in ("forecasts", "fits", "tables"):
            (run/name).mkdir(parents=True, exist_ok=True)
        contract={"timezone":"Europe/Paris", "validation_folds":[
            {"name":"2024_Q1", "start":"2024-01-01", "end":"2024-04-01"}],
            "neural_finalist_seeds":[42,123,2026]}
        (run/"manifest.json").write_text(json.dumps({"contract":contract}))
        choice={"status":"selected", "family":"gam", "arm":"national", "trial":1,
                "params":{"alpha":20}}
        (run/"selected_parameters.json").write_text(json.dumps({"gam__national":choice}))
        stem="gam__national__2024_Q1_trial1__synthetic"
        frame=pd.DataFrame({
            "model":"gam__national", "origin":["2023-12-31T10:00:00Z", "2024-01-01T10:00:00Z",
                "2024-01-02T10:00:00Z", "2024-01-03T10:00:00Z"],
            "delivery_time":pd.date_range("2024-01-01",periods=4,freq="D",tz="UTC"),
            "fit_origin":["2023-12-31T10:00:00Z"]*3+["2024-01-03T10:00:00Z"],
            "raw_prediction_mw":[90,np.nan,95,120], "prediction_mw":[90,105,95,120]})
        prediction_path=run/"forecasts"/(stem+".csv.gz")
        frame.to_csv(prediction_path,index=False,compression="gzip")
        (run/"forecasts"/(stem+".json")).write_text(json.dumps({"phase":"2024_Q1_trial1", "params":choice["params"]}))
        records=[{"origin":"2023-12-31T10:00:00Z", "status":"fit", "train_MAE_MW":5,
                  "training_rows":100, "metadata":{"objective":"penalized squared error; conditional mean"}},
                 {"origin":"2024-01-03T10:00:00Z", "status":"fit", "train_MAE_MW":7,
                  "training_rows":200, "metadata":{"objective":"penalized squared error; conditional mean"}}]
        (run/"fits"/(stem+".json")).write_text(json.dumps(records))
        # These are intentionally invalid: the reporting function must never open
        # test caches or unselected-trial prediction files.
        (run/"forecasts/gam__national__test__do_not_read.csv.gz").write_bytes(b"not gzip")
        (run/"forecasts/gam__national__2024_Q1_trial0__do_not_read.csv.gz").write_bytes(b"not gzip")
        pd.DataFrame([{"model":"dlinear__regional", "sensitivity":"neural_seed", "seed":42, "MAE_MW":10},
                      {"model":"dlinear__regional", "sensitivity":"neural_seed", "seed":123, "MAE_MW":20},
                      {"model":"gam__national", "sensitivity":"matched_expanding", "seed":42, "MAE_MW":3}]
        ).to_csv(run/"tables/development_sensitivities.csv",index=False)
        return run,prediction_path

    def test_same_fit_raw_gap_counts_and_development_only_read(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);run,_=self.fixture(root)
            result=build_diagnostics(run,root)
            gaps=result["fit_gaps"].sort_values("fit_origin")
            first,second=gaps.iloc[0],gaps.iloc[1]
            self.assertEqual(first.n_labels,2)
            self.assertEqual(first.n_missing_labels,1)
            self.assertEqual(first.n_raw_scored,1)
            self.assertEqual(first.raw_availability,.5)
            self.assertEqual(first.oos_raw_MAE_MW,10)
            self.assertEqual(first.oos_policy_MAE_MW,7.5)
            self.assertEqual(first.same_fit_gap_MW,5)
            self.assertEqual(second.same_fit_gap_MW,13)
            self.assertTrue(gaps.point_functional.eq("conditional mean").all())
            quarter=result["quarter_scores"].iloc[0]
            self.assertEqual(quarter.n_labels,3)
            self.assertAlmostEqual(quarter.oos_policy_MAE_MW,35/3)
            self.assertAlmostEqual(quarter.oos_raw_MAE_MW,15)
            self.assertEqual(result["seed_stability"].iloc[0].seeds_recorded,2)
            self.assertEqual(result["seed_stability"].iloc[0].seeds_expected,3)
            self.assertEqual(len(result["figure_paths"]),2)

    def test_absent_unscaled_training_score_is_explicitly_unavailable(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);run,prediction=self.fixture(root)
            fit=run/"fits"/(prediction.name.removesuffix(".csv.gz")+".json")
            records=json.loads(fit.read_text());records[0].pop("train_MAE_MW")
            fit.write_text(json.dumps(records))
            result=build_diagnostics(run,root)
            first=result["fit_gaps"].sort_values("fit_origin").iloc[0]
            self.assertTrue(np.isnan(first.same_fit_gap_MW))
            self.assertTrue(first.gap_status.startswith("unavailable_"))

    def test_out_of_fold_prediction_cache_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);run,prediction=self.fixture(root)
            frame=pd.read_csv(prediction);frame.loc[0,"delivery_time"]="2025-07-01 00:00:00+00:00"
            frame.to_csv(prediction,index=False,compression="gzip")
            with self.assertRaisesRegex(ValueError,"non-development/fold"):
                build_diagnostics(run,root)


if __name__=="__main__":
    unittest.main()
