import io
import unittest
import numpy as np
import pandas as pd
from core import prepare, predict, evaluate, evaluate_events, predict_latest, latest_state, excel_bytes, analyze
from config import HORIZONS, InputConfig, ModelConfig
from validation import assert_patient_disjoint, assert_purged_boundary
from synthetic import make_scenario, SCENARIOS
from site_compat import predict_site, evaluate_site


def frame(values=None,n=60):
    values=np.full(n,120.) if values is None else np.asarray(values)
    return pd.DataFrame({"time":pd.date_range("2026-01-01",periods=len(values),freq="5min",tz="UTC"),"glucose":values})


def compute(f,**kwargs):
    grid,meta=prepare(f,"time","glucose",**kwargs)
    a,summary=evaluate(predict(grid))
    return a,summary,meta


class InputTests(unittest.TestCase):
    def test_synthetic_scenarios_are_only_qa_inputs(self):
        for kind in SCENARIOS:
            f = make_scenario(kind)
            self.assertEqual(len(f), 48)
            self.assertTrue(f.time.dt.tz is not None)

    def test_site_compatibility_reference(self):
        f = pd.DataFrame({"time": pd.date_range("2026-01-01", periods=48, freq="5min", tz="UTC"),
                          "glucose": np.r_[np.full(40, 120.), np.linspace(100, 60, 8)]})
        g, _ = prepare(f, "time", "glucose")
        result = evaluate_site(predict_site(g))
        self.assertIn("event_recall", result)
    def test_sisensing(self):
        f=pd.DataFrame({"time":["04-12-2024 10:45 GMT+3:30"],"glucose":[100]})
        g,m=prepare(f,"time","glucose")
        self.assertEqual(g.index[0],pd.Timestamp("2024-12-04T07:15Z"))

    def test_iso_mixed_offsets_same_instant(self):
        f=pd.DataFrame({"time":["2026-01-01T03:30+03:30","2026-01-01T00:00Z"],"glucose":[100,100]})
        g,m=prepare(f,"time","glucose")
        self.assertEqual(m['duplicates_removed'],1)
        self.assertEqual(len(g),1)

    def test_ambiguous_date_requires_format(self):
        f=pd.DataFrame({"time":["04/05/2026 12:00"],"glucose":[100]})
        with self.assertRaises(ValueError): prepare(f,"time","glucose")
        g,_=prepare(f,"time","glucose",InputConfig(date_format="%d/%m/%Y %H:%M",timezone="UTC"))
        self.assertEqual(g.index[0].month,5)

    def test_numeric_epoch_explicit(self):
        f=pd.DataFrame({"time":[1767225600],"glucose":[100]})
        with self.assertRaises(ValueError): prepare(f,"time","glucose")
        g,_=prepare(f,"time","glucose",InputConfig(numeric_time_unit="s"))
        self.assertEqual(g.index[0],pd.Timestamp("2026-01-01T00:00Z"))

    def test_excel_serial(self):
        g,_=prepare(pd.DataFrame({'t':[46023.],'g':[100]}),'t','g',InputConfig(numeric_time_unit='excel',timezone='UTC'))
        self.assertEqual(g.index[0],pd.Timestamp('2026-01-01T00:00Z'))

    def test_unit_equivalence(self):
        f=frame(np.linspace(200,80,60))
        g,_=prepare(f,'time','glucose')
        f.glucose/=18
        h,_=prepare(f,'time','glucose',InputConfig(unit='mmol/L'))
        np.testing.assert_allclose(g.Raw,h.Raw)

    def test_primary_horizon_must_be_a_configured_horizon(self):
        self.assertEqual(ModelConfig(primary_horizon=20).primary_horizon,20)
        with self.assertRaises(ValueError):
            ModelConfig(primary_horizon=17)

    def test_conflicting_duplicates_abstain(self):
        f=frame(); f=pd.concat([f,pd.DataFrame({'time':[f.time.iloc[20]],'glucose':[55]})],ignore_index=True)
        g,m=prepare(f,'time','glucose')
        p=predict(g)
        self.assertEqual(m['conflicting_timestamps'],1)
        self.assertFalse(p.Prediction_Ready.iloc[20:33].any())

    def test_equivalent_numeric_duplicates(self):
        f=pd.DataFrame({'time':['2026-01-01']*2,'glucose':['100','100.0']})
        _,m=prepare(f,'time','glucose')
        self.assertEqual(m['conflicting_timestamps'],0)

    def test_invalid_tail_is_retained(self):
        f=frame(); f.loc[59,'glucose']=np.inf
        a,_,m=compute(f)
        self.assertEqual(len(a),60)
        self.assertFalse(a.Prediction_Ready.iloc[-1])
        self.assertEqual(latest_state(a,m)['status'],'sensor_check')

    def test_censored_values_not_invented(self):
        f=frame([100,'LOW','HIGH'])
        g,m=prepare(f,'time','glucose')
        self.assertTrue(g.Raw.iloc[1:].isna().all())
        self.assertEqual(m['latest_kind'],'censored_high')

    def test_low_without_numeric_bound(self):
        a,_,m=compute(frame(['LOW']))
        self.assertEqual(latest_state(a,m)['status'],'sensor_check')
        a,_,m=compute(frame(['LOW']),cfg=InputConfig(sensor_lower=40))
        self.assertEqual(latest_state(a,m)['status'],'current_low')

    def test_profile_limits_not_clipped(self):
        g,_=prepare(frame([19,100,401]),'time','glucose',InputConfig(sensor_lower=20,sensor_upper=400))
        self.assertTrue(g.Raw.iloc[[0,2]].isna().all())

    def test_multi_patient_rejected(self):
        f=frame();f['patient']=['a']*30+['b']*30
        with self.assertRaises(ValueError): prepare(f,'time','glucose',stream_columns=['patient'])

    def test_stream_ids_missing_rejected(self):
        f=frame();f['patient']='a';f.loc[4,'patient']=None
        with self.assertRaises(ValueError): prepare(f,'time','glucose',stream_columns=['patient'])

    def test_sort_order_invariance(self):
        f=frame(np.arange(60)+100)
        g,_=prepare(f,'time','glucose'); h,_=prepare(f.sample(frac=1,random_state=2),'time','glucose')
        pd.testing.assert_frame_equal(g,h)

    def test_invalid_unit_and_timezone(self):
        with self.assertRaises(ValueError): InputConfig(unit='guess')
        with self.assertRaises(Exception): prepare(frame(),'time','glucose',InputConfig(timezone='not/a/zone'))

    def test_dst_ambiguity_not_guessed(self):
        f=pd.DataFrame({'time':['2026-11-01T01:30:00'],'glucose':[100]})
        with self.assertRaises(Exception): prepare(f,'time','glucose',InputConfig(timezone='America/New_York'))


class CausalTests(unittest.TestCase):
    def test_all_complete_prefixes(self):
        rng=np.random.default_rng(731)
        f=frame(120+np.cumsum(rng.normal(0,3,60)))
        full,_,_=compute(f)
        cols=['Current_Glucose','Forecast_Slope','Low_Risk_Score']+[f'Alert_{h}m' for h in HORIZONS]
        for size in (1,12,13,20,47):
            short,_,_=compute(f.iloc[:size])
            pd.testing.assert_frame_equal(short[cols],full.loc[short.index,cols])

    def test_future_values_cannot_change_past(self):
        f=frame(); a,_,_=compute(f)
        f.loc[40:,'glucose']=40; b,_,_=compute(f)
        pd.testing.assert_frame_equal(a.iloc[:40].filter(regex='^Alert_|^Pred_Glucose_'),b.iloc[:40].filter(regex='^Alert_|^Pred_Glucose_'))

    def test_asof_filters_future_and_open_bin(self):
        f=frame(); cutoff=pd.Timestamp('2026-01-01T02:03Z')
        g,m=prepare(f,'time','glucose',as_of=cutoff)
        self.assertGreater(m['excluded_future_rows'],0)
        p=predict(g,method='robust')
        self.assertFalse(p.Prediction_Ready.iloc[-1])
        state,_=predict_latest(f,'time','glucose',as_of=cutoff)
        self.assertEqual(state['forecast_time_UTC'],'2026-01-01 02:00:00+00:00')

    def test_live_batch_replay_equivalent(self):
        f=frame(np.linspace(160,75,60))
        stamp=f.time.iloc[35]
        a,_=predict_latest(f,'time','glucose',as_of=stamp)
        b,_=predict_latest(f.iloc[:36],'time','glucose',as_of=stamp)
        self.assertEqual(a['forecasts'],b['forecasts'])

    def test_long_staleness(self):
        f=frame()
        s,m=predict_latest(f,'time','glucose',as_of='2029-01-01T00:00Z')
        self.assertEqual(s['status'],'stale')
        self.assertEqual(s['forecasts'],{})

    def test_latest_low_open_bin_not_suppressed(self):
        f=frame(); f.loc[60]=[f.time.iloc[-1]+pd.Timedelta(minutes=1),55]
        s,_=predict_latest(f,'time','glucose',as_of=f.time.iloc[-1])
        self.assertEqual(s['status'],'current_low')

    def test_stale_low_is_not_claimed_current(self):
        f=frame([55])
        s,_=predict_latest(f,'time','glucose',as_of='2026-01-01T01:00Z')
        self.assertEqual(s['status'],'stale')
        self.assertTrue(s['latest_reading_low'])

    def test_timestamp_required_for_live(self):
        with self.assertRaises(ValueError): predict_latest(frame(),'time','glucose',as_of='2026-01-01')


class ScenarioTests(unittest.TestCase):
    def test_preventive_near_boundary_rule(self):
        a,_,_=compute(frame(np.r_[np.full(56,120.),[74,73,72,71]]))
        self.assertTrue(a.Preventive_Alert.iloc[-1])
        self.assertIn('near_low',a.Alert_Reason.iloc[-1])

    def test_preventive_fast_drop_before_low(self):
        a,_,_=compute(frame(np.r_[np.full(55,160.),[136,117,100,84,73,61]]))
        self.assertTrue(a.Fast_Drop_Risk.iloc[-3])
        self.assertTrue(a.Preventive_Alert.iloc[-3])

    def test_guarded_policy_keeps_fast_drop_rescue_path(self):
        grid,_=prepare(frame(np.r_[np.full(55,160.),[136,117,100,84,73,61]]),"time","glucose")
        guarded=predict(grid,method="guarded")
        self.assertIn("Scenario_Votes_30m",guarded.columns)
        self.assertTrue(guarded.Fast_Drop_Risk.iloc[-3])
        self.assertEqual(guarded.Alert_30m.iloc[-3],1)

    def test_preventive_recent_low_after_recovery(self):
        a,_,_=compute(frame(np.r_[np.full(56,150.),[65,68,72,75]]))
        self.assertTrue(a.Recent_Low_Risk.iloc[-1])
        self.assertTrue(a.Preventive_Alert.iloc[-1])

    def test_preventive_rules_do_not_cross_gap(self):
        f=frame(np.r_[np.full(20,120.),[65,68,72,75,76,78]]).drop(index=21)
        a,_,_=compute(f)
        self.assertFalse(a.Preventive_Alert.iloc[-1])

    def test_one_minute_sensor_resampling(self):
        f=pd.DataFrame({'time':pd.date_range('2026-01-01',periods=91,freq='min',tz='UTC'),'glucose':np.full(91,120.)})
        a,_,_=compute(f)
        self.assertTrue(a.Prediction_Ready.iloc[-1])
        self.assertEqual(a.Pred_Glucose_30m.iloc[-1],120)

    def test_sparse_sensor_abstains_instead_of_interpolating(self):
        f=pd.DataFrame({'time':pd.date_range('2026-01-01',periods=20,freq='15min',tz='UTC'),'glucose':np.linspace(160,80,20)})
        a,_,_=compute(f)
        self.assertFalse(a.Prediction_Ready.any())
        self.assertTrue(a.Alert_30m.isna().all())

    def test_long_gap_no_cross_gap_alarm_credit(self):
        f=frame();f=f.drop(index=range(20,40));f.loc[40,'glucose']=60
        a,_,_=compute(f);ev,_=evaluate_events(a)
        self.assertEqual(len(ev),1)
        self.assertFalse(ev.Eligible_30.iloc[0])

    def test_empty_and_bad_grid(self):
        with self.assertRaises(ValueError): prepare(frame(n=0),'time','glucose')
        g,_=prepare(frame(),'time','glucose')
        with self.assertRaises(ValueError): predict(g.iloc[::2])

    def test_constant_normal(self):
        a,_,_=compute(frame())
        self.assertEqual(a.Alert_30m.dropna().sum(),0)
        self.assertTrue(a.Forecast_Slope.dropna().eq(0).all())

    def test_sustained_decline(self):
        a,_,_=compute(frame(np.linspace(180,75,60)))
        self.assertEqual(a.Alert_30m.iloc[-1],1)
        self.assertEqual(a.Downward_Persistence.iloc[-1],100)

    def test_sustained_rise(self):
        a,_,_=compute(frame(np.linspace(80,240,60)))
        self.assertEqual(a.Alert_30m.dropna().sum(),0)

    def test_current_low_first_reading(self):
        a,_,m=compute(frame([65]))
        self.assertTrue(pd.isna(a.Alert_30m.iloc[0]))
        self.assertEqual(latest_state(a,m)['status'],'current_low')

    def test_gap_restarts_history(self):
        a,_,_=compute(frame().drop(index=[20,21]))
        self.assertFalse(a.Prediction_Ready.iloc[20:34].any())
        self.assertTrue(a.Prediction_Ready.iloc[34])

    def test_noise_seeds_finite_and_nested(self):
        for seed in range(12):
            rng=np.random.default_rng(seed)
            f=frame(np.clip(120+np.cumsum(rng.normal(0,8,100)),30,350))
            a,_,_=compute(f)
            ready=a.loc[a.Prediction_Ready]
            for h in HORIZONS:
                self.assertTrue(np.isfinite(ready[f'Pred_Glucose_{h}m']).all())
                self.assertTrue(ready[f'Lower_Scenario_{h}m'].le(ready[f'Pred_Glucose_{h}m']).all())
                self.assertTrue(ready[f'Upper_Scenario_{h}m'].ge(ready[f'Pred_Glucose_{h}m']).all())
            self.assertTrue(ready.Alert_10m.isin([0,1]).all())

    def test_artifact_like_low_not_deleted_as_noise(self):
        f=frame(); f.loc[59,'glucose']=45
        a,_,m=compute(f)
        self.assertEqual(a.Raw.iloc[-1],45)
        self.assertEqual(latest_state(a,m)['status'],'current_low')

    def test_recovery_and_reversal(self):
        f=frame(np.r_[np.linspace(170,85,30),np.linspace(90,180,30)])
        a,_,_=compute(f)
        self.assertEqual(a.Alert_30m.iloc[-1],0)
        self.assertTrue(a.Trend_Disagreement.any())

    def test_same_history_can_have_different_futures(self):
        f=frame(); other=f.copy();other.loc[40:,'glucose']=55
        a,_,_=compute(f); b,_,_=compute(other)
        self.assertEqual(a.Alert_30m.iloc[38],b.Alert_30m.iloc[38])
        self.assertNotEqual(a.Y30_Actual.iloc[38],b.Y30_Actual.iloc[38])

    def test_score_does_not_activate_robust_alert(self):
        g,_=prepare(frame(np.r_[np.full(47,200.),np.linspace(200,150,13)]),'time','glucose')
        p=predict(g,method='robust')
        expected=(p.Current_Glucose+p.Forecast_Slope*30).le(70)&p.Forecast_Slope.lt(0)
        np.testing.assert_array_equal(p.Alert_30m.dropna().to_numpy(),expected[p.Prediction_Ready].to_numpy())

    def test_notification_cooldown_separate_from_flags(self):
        g,_=prepare(frame(np.linspace(240,72,80)),'time','glucose')
        p=predict(g)
        self.assertGreater(p.Alert_30m.sum(),p.Notification_30m.sum())
        times=p.index[p.Notification_30m]
        self.assertTrue(((times[1:]-times[:-1])>=pd.Timedelta(minutes=15)).all())


class EvaluationTests(unittest.TestCase):
    def test_future_tail_unknown_and_current_excluded(self):
        a,_,_=compute(frame(np.r_[65,np.full(59,120)]))
        self.assertEqual(a.Y30_Actual.iloc[0],0)
        self.assertTrue(a.Y30_Actual.tail(6).isna().all())
        self.assertEqual(a.Outcome_30m.iloc[0],'AlreadyLow')

    def test_equality_at_70(self):
        f=frame(); f.loc[20,'glucose']=70
        a,_,_=compute(f)
        self.assertEqual(a.Y30_Actual.iloc[14],1)
        self.assertEqual(a.Y30_Actual.iloc[13],0)

    def test_unknown_future_not_negative(self):
        a,_,_=compute(frame().drop(index=20))
        self.assertTrue(pd.isna(a.Y30_Actual.iloc[18]))

    def test_events_singleton_and_multi_reading(self):
        f=frame();f.loc[20:23,'glucose']=60;f.loc[40,'glucose']=65
        a,_,_=compute(f);ev,s=evaluate_events(a)
        self.assertEqual(len(ev),2)
        self.assertEqual(ev.Low_readings.tolist(),[4,1])
        self.assertEqual(s.All_events.tolist(),[2]*len(HORIZONS))

    def test_event_needs_sustained_recovery_to_close(self):
        f=frame(); f.loc[20:25,'glucose']=[60,69,74,68,62,75]
        a,_,_=compute(f);ev,_=evaluate_events(a)
        self.assertEqual(len(ev),1)
        self.assertEqual(ev.Low_readings.iloc[0],4)

    def test_no_events_undefined_recall(self):
        a,s,_=compute(frame());ev,e=evaluate_events(a)
        self.assertTrue(s.Recall.isna().all())
        self.assertTrue(e.Capture_all_events.isna().all())

    def test_ineligible_event_not_hidden(self):
        a,_,_=compute(frame(np.r_[120,65,np.full(58,120)]))
        _,s=evaluate_events(a)
        self.assertTrue(s.All_events.eq(1).all())
        self.assertTrue(s.No_opportunity.eq(1).all())
        self.assertTrue(s.Capture_all_events.eq(0).all())

    def test_excel_export_safe(self):
        a,s,e,es,m=analyze(frame(),'time','glucose')
        m['test']='=1+1'
        data=excel_bytes(a,s,m,e,es)
        sheets=pd.read_excel(io.BytesIO(data),sheet_name=None)
        self.assertEqual(set(sheets),{'Analysis','Events','Event_Evaluation','Row_Evaluation','Metadata'})
        import openpyxl
        w=openpyxl.load_workbook(io.BytesIO(data))
        self.assertTrue(all(c.data_type!='f' for ws in w for row in ws for c in row))

    def test_patient_leakage_guard(self):
        assert_patient_disjoint(['a'],['b'],['c'])
        with self.assertRaises(ValueError): assert_patient_disjoint(['a'],['b'],['a'])
        with self.assertRaises(ValueError): assert_patient_disjoint(['a'],[None],['c'])

    def test_time_boundary_guard(self):
        with self.assertRaises(ValueError): assert_purged_boundary('2026-01-01T00:00Z','2026-01-01T02:00Z')
        assert_purged_boundary('2026-01-01T00:00Z','2026-01-01T02:05Z')

    def test_baseline_features_match_frozen_v1(self):
        from baseline_v1 import predict as frozen
        g,_=prepare(frame(np.random.default_rng(67).uniform(60,230,200)),'time','glucose')
        a=predict(g,method='baseline'); b=frozen(g)
        for name in ['Current_Glucose','ROC_15m','Acceleration','Low_Risk_Score']:
            np.testing.assert_allclose(a[name],b[name],equal_nan=True)
        for h in (30,):
            pd.testing.assert_series_equal(a[f'Alert_{h}m'],b[f'Alert_{h}m'])


if __name__=='__main__':
    unittest.main()

