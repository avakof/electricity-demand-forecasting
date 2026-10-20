"""Behavioral checks: availability, DST, leakage, replay and calibration."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from src.common import load_contract, local_day_grid, forecast_origin, holiday_dates, project_root
from src.data import select_forecast_run, load_archive
from src.features import build_origin_features, prediction_fallback
from src.uncertainty import empirical_intervals


class ForecastIntegrityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.contract = load_contract(project_root())
        index = pd.date_range("2024-01-01", "2024-05-01", freq="h", tz="UTC")
        cls.data = pd.DataFrame({"load_mw": 50000+np.sin(np.arange(len(index))/24)*1000, "temperature_c": 10., "rte_forecast_d1_mw": 49000., "label_available_at_assumed": index+pd.Timedelta(hours=3)}, index=index)

    def test_dst_delivery_counts(self):
        self.assertEqual(len(local_day_grid("2024-03-31")),23)
        self.assertEqual(len(local_day_grid("2024-10-27")),25)
        self.assertEqual(len(local_day_grid("2024-03-30")),24)
        self.assertTrue(local_day_grid("2024-10-27").is_unique)

    def test_origin_remains_pre_noon_across_dst(self):
        for day in ("2024-03-31", "2024-04-01", "2024-10-27", "2024-10-28"):
            origin = forecast_origin(day,self.contract).tz_convert("Europe/Paris")
            self.assertEqual(origin.hour,11)

    def test_future_observations_cannot_change_features_or_baseline(self):
        day = "2024-04-05"; origin = forecast_origin(day,self.contract)
        before = build_origin_features(self.data,day,self.contract)
        changed = self.data.copy()
        changed.loc[changed.index >= origin,["load_mw","temperature_c"]] = 1e9
        after = build_origin_features(changed,day,self.contract)
        assert_frame_equal(before,after)
        p1,_=prediction_fallback(before,np.full(len(before),np.nan))
        p2,_=prediction_fallback(after,np.full(len(after),np.nan))
        np.testing.assert_array_equal(p1,p2)

    def test_delayed_publication_cannot_change_features(self):
        day="2024-04-05"; origin=forecast_origin(day,self.contract)
        changed=self.data.copy()
        mask=(changed.index < origin) & (changed.label_available_at_assumed >= origin)
        changed.loc[mask,"load_mw"]=1e9
        assert_frame_equal(build_origin_features(self.data,day,self.contract),build_origin_features(changed,day,self.contract))

    def test_archive_uses_only_accessible_issue_and_publication(self):
        origin=pd.Timestamp("2024-04-04 09:00",tz="UTC"); grid=local_day_grid("2024-04-05")
        records=[]
        for run,issued,available,value in (("old",origin-pd.Timedelta(hours=2),origin-pd.Timedelta(hours=1),10),("late",origin-pd.Timedelta(minutes=30),origin+pd.Timedelta(minutes=30),20),("future",origin+pd.Timedelta(hours=1),origin+pd.Timedelta(hours=2),30)):
            for t in grid:
                records.append({"run_id":run,"issued_at":issued,"available_at":available,"valid_time":t,"value":value})
        selected,meta=select_forecast_run(pd.DataFrame(records),origin,grid,24)
        self.assertTrue(selected.eq(10).all()); self.assertEqual(meta["run_id"],"old")

    def test_stale_and_partial_archive(self):
        origin=pd.Timestamp("2024-04-04 09:00",tz="UTC"); grid=local_day_grid("2024-04-05")
        archive=pd.DataFrame({"run_id":["old"],"issued_at":[origin-pd.Timedelta(hours=50)],"available_at":[origin-pd.Timedelta(hours=49)],"valid_time":[grid[0]],"value":[1.]})
        values,meta=select_forecast_run(archive,origin,grid,24)
        self.assertTrue(values.isna().all()); self.assertEqual(meta["reason"],"stale_or_unavailable")
        archive["issued_at"]=origin-pd.Timedelta(hours=2); archive["available_at"]=origin-pd.Timedelta(hours=1)
        values,meta=select_forecast_run(archive,origin,grid,24)
        self.assertEqual(values.notna().sum(),1); self.assertFalse(meta["available"])

    def test_no_silent_target_interpolation(self):
        from src.metrics import point_metrics
        result=point_metrics(pd.DataFrame({"actual_mw":[10,np.nan,30],"prediction_mw":[9,20,31]}))
        self.assertEqual(result["scored"],2); self.assertEqual(result["missing_targets"],1)
        self.assertEqual(result["MAE_MW"],1)

    def test_missing_source_interval_does_not_remove_delivery_interval(self):
        day="2024-04-05"
        expected=local_day_grid(day)
        changed=self.data.drop(index=expected[5])
        result=build_origin_features(changed,day,self.contract)
        self.assertEqual(len(result),len(expected))
        self.assertIn(expected[5],result.delivery_time.tolist())

    def test_fallback_is_a_complete_policy(self):
        frame=pd.DataFrame({"rte_at_origin_mw":[100,np.nan],"week_lag_1":[80,np.nan],"comparable_mean":[70,75],"last_load":[60,60]})
        p,source=prediction_fallback(frame,[np.nan,np.nan],True)
        self.assertEqual(p.tolist(),[100,75]); self.assertEqual(source.tolist(),["rte","comparable"])

    def test_repeated_source_hour_policy(self):
        index=pd.date_range("2024-10-01","2024-11-05",freq="h",tz="UTC")
        data=pd.DataFrame({"load_mw":100.,"temperature_c":10.,"rte_forecast_d1_mw":100.,"label_available_at_assumed":index+pd.Timedelta(hours=3)},index=index)
        repeated=(index.tz_convert("Europe/Paris").strftime("%Y-%m-%d %H")=="2024-10-27 02")
        data.loc[repeated,"load_mw"]=[80.,120.]
        result=build_origin_features(data,"2024-11-03",self.contract)
        self.assertEqual(result.loc[result.hour.eq(2),"week_lag_1"].iloc[0],100.)

    def test_fallback_for_nonexistent_source_hour(self):
        result=build_origin_features(self.data,"2024-04-07",self.contract)
        hour=result.loc[result.hour.eq(2)]
        self.assertTrue(hour.week_lag_1.isna().all())
        prediction,source=prediction_fallback(hour,[np.nan])
        self.assertTrue(prediction.notna().all())

    def test_quantile_calibration_does_not_use_future_labels(self):
        contract={**self.contract,"calibration_minimum_observations":1}
        origin=pd.Timestamp("2024-04-04 09:00",tz="UTC")
        hist=pd.DataFrame({"origin":[origin-pd.Timedelta(days=2)],"delivery_time":[origin-pd.Timedelta(days=1)],"model":["m"],"prediction_mw":[100.]})
        pred=pd.DataFrame({"origin":[origin],"delivery_time":[origin+pd.Timedelta(days=1)],"model":["m"],"prediction_mw":[100.]})
        labels=pd.DataFrame({"delivery_time":pd.concat([hist.delivery_time,pred.delivery_time],ignore_index=True),"actual_mw":[110.,120.],"label_available_at":[origin-pd.Timedelta(hours=1),origin+pd.Timedelta(days=2)]})
        first=empirical_intervals(pred,hist,labels,contract)
        labels.loc[1,"actual_mw"]=1e9
        second=empirical_intervals(pred,hist,labels,contract)
        assert_frame_equal(first,second); self.assertEqual(first.q05.iloc[0],110.)

    def test_archive_rejects_duplicate_versions(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"archive.csv"
            pd.DataFrame({"valid_time":["2024-01-01T00:00:00Z"]*2,"available_at":["2024-01-01T02:00:00Z"]*2,"value":[1,2]}).to_csv(path,index=False)
            with self.assertRaisesRegex(ValueError,"duplicate version"):
                load_archive(path,forecast=False)

    def test_calendar_holidays(self):
        self.assertIn(pd.Timestamp("2024-04-01").date(),holiday_dates(2024))

    def test_trained_forecast_and_preprocessor_ignore_future_data(self):
        from src.backtest import Experiment,fit_at_origin,feature_columns
        feature_frames=[build_origin_features(self.data,day,self.contract) for day in pd.date_range("2024-03-01","2024-03-15",freq="D")]
        features=pd.concat(feature_frames,ignore_index=True)
        labels=self.data[["load_mw","label_available_at_assumed"]].rename(columns={"load_mw":"actual_mw","label_available_at_assumed":"label_available_at"}).rename_axis("delivery_time").reset_index()
        contract={**self.contract,"minimum_training_days":1}
        exp=Experiment("test","future mutation invariance",max_iter=5)
        origin=forecast_origin("2024-03-15",contract)
        model,log=fit_at_origin(features,labels,origin,exp,contract)
        changed_labels=labels.copy(); changed_features=features.copy()
        changed_labels.loc[changed_labels.label_available_at.ge(origin),"actual_mw"]=1e9
        changed_features.loc[changed_features.origin.ge(origin),feature_columns(exp)]=1e9
        again,second_log=fit_at_origin(changed_features,changed_labels,origin,exp,contract)
        x=features.loc[features.origin.eq(origin),feature_columns(exp)]
        np.testing.assert_allclose(model.predict(x),again.predict(x),rtol=0,atol=0)
        np.testing.assert_array_equal(model[0].statistics_,again[0].statistics_)
        self.assertLess(log["latest_label_available_at"],origin)
        self.assertEqual(log["training_rows"],second_log["training_rows"])

    def test_paired_bootstrap_preserves_matched_loss_ratio(self):
        from src.metrics import paired_block_bootstrap
        rows=[]; truths=[]
        for day in pd.date_range("2024-05-01","2024-05-28",freq="D"):
            for t in local_day_grid(day):
                truths.append({"delivery_time":t,"actual_mw":1000.})
                for model,p in (("candidate",950.),("rte_proxy",900.)):
                    rows.append({"origin":forecast_origin(day,self.contract),"delivery_time":t,"model":model,"prediction_mw":p})
        result=paired_block_bootstrap(pd.DataFrame(rows),pd.DataFrame(truths),"candidate",block_days=[7,14],repetitions=30)
        np.testing.assert_allclose(result.relative_improvement_percent,50.)
        np.testing.assert_allclose(result.relative_low,50.)
        np.testing.assert_allclose(result.absolute_MAE_improvement_MW,50.)
        partial_truth=pd.DataFrame(truths)
        partial_truth.loc[0,"actual_mw"]=np.nan
        partial=paired_block_bootstrap(pd.DataFrame(rows),partial_truth,"candidate",block_days=[7],repetitions=10)
        self.assertEqual(partial.paired_hours.iloc[0],len(partial_truth)-1)
        self.assertEqual(partial.partially_observed_days_retained.iloc[0],1)

    def test_replay_is_reproducible(self):
        first=build_origin_features(self.data,"2024-04-05",self.contract)
        second=build_origin_features(self.data,"2024-04-05",self.contract)
        assert_frame_equal(first,second)

    def test_grouped_origins_do_not_cross_validation_boundary(self):
        features=pd.concat([build_origin_features(self.data,day,self.contract) for day in pd.date_range("2024-03-30","2024-04-02",freq="D")],ignore_index=True)
        assignment=features.local_day.ge("2024-04-01")
        self.assertEqual(assignment.groupby(features.origin).nunique().max(),1)


if __name__ == "__main__":
    unittest.main()
