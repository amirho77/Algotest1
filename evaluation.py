"""Observed CGM labels; row, event and notification metrics are kept distinct."""
import numpy as np
import pandas as pd
from config import HORIZONS, ModelConfig


def ratio(a,b):
    return a/b if b else np.nan


def evaluate(predictions, cfg=None):
    cfg=cfg or ModelConfig()
    out=predictions.copy()
    metrics=[]
    for h in HORIZONS:
        future=pd.concat([out.Observed_Min.where(out.Label_Observed).shift(-k) for k in range(1,h//5+1)],axis=1)
        complete=future.notna().all(axis=1)
        out[f"Min_next_{h}m"]=future.min(axis=1).where(complete)
        y=out[f"Min_next_{h}m"].le(cfg.threshold).astype("Int64").where(complete)
        out[f"Y{h}_Actual"]=y
        eligible=complete & out.Alert_Ready & out.Raw.gt(cfg.threshold)
        outcome=pd.Series("NotEvaluable",index=out.index)
        for alert,target,label in ((1,1,"TP"),(1,0,"FP"),(0,1,"FN"),(0,0,"TN")):
            outcome.loc[(eligible & out[f"Alert_{h}m"].eq(alert)&y.eq(target)).fillna(False)]=label
        forecast_outcome=pd.Series("NotEvaluable",index=out.index)
        forecast_eligible=complete & out.Prediction_Ready & out.Raw.gt(cfg.threshold)
        for alert,target,label in ((1,1,"TP"),(1,0,"FP"),(0,1,"FN"),(0,0,"TN")):
            forecast_outcome.loc[(forecast_eligible & out[f"Forecast_Alert_{h}m"].eq(alert)&y.eq(target)).fillna(False)]=label
        outcome.loc[out.Current_Low.eq(1).fillna(False)]="AlreadyLow"
        out[f"Outcome_{h}m"]=outcome
        out[f"Forecast_Outcome_{h}m"]=forecast_outcome
        counts={k:int(outcome.eq(k).sum()) for k in ("TP","FP","FN","TN")}
        tp,fp,fn,tn=[counts[k] for k in ("TP","FP","FN","TN")]
        fcounts={f"Forecast_{k}":int(forecast_outcome.eq(k).sum()) for k in ("TP","FP","FN","TN")}
        notice=out[f"Notification_{h}m"]
        notice_tp=int((notice & eligible & y.eq(1)).sum())
        notice_fp=int((notice & eligible & y.eq(0)).sum())
        days=float(eligible.sum())*5/1440
        metrics.append({"Horizon_min":h, "Evaluable_rows":int(eligible.sum()), **counts,
                        "Precision":ratio(tp,tp+fp),"Recall":ratio(tp,tp+fn),"Specificity":ratio(tn,tn+fp),
                        "Notification_TP":notice_tp,"Notification_FP":notice_fp,
                        "Notification_Unknown":int((notice & ~eligible).sum()),
                        "False_notifications_per_evaluable_day":ratio(notice_fp,days),
                        **fcounts,
                        "Prediction_coverage":ratio(int(out.Prediction_Ready.sum()),int(out.Bin_Complete.sum())),
                        "Alert_coverage":ratio(int(out.Alert_Ready.sum()),int(out.Bin_Complete.sum()))})
    return out,pd.DataFrame(metrics)


def evaluate_events(analysis, cfg=None):
    """Group a low episode until a sustained recovery, then score causal alerts."""
    cfg=cfg or ModelConfig()
    observed=analysis.Label_Observed.to_numpy(bool)
    low=(analysis.Observed_Min.le(cfg.threshold)&analysis.Label_Observed).to_numpy(bool)
    values=analysis.Observed_Min.to_numpy(float)
    events=[]
    i=0
    event_id=0
    while i<len(low):
        if not low[i]:
            i+=1
            continue
        event_id+=1
        start=i
        last_low=i
        recovery_bins=0
        i+=1
        while i<len(low):
            # A missing bin ends an event. It must not join two low periods
            # whose continuity cannot be observed.
            if not observed[i]:
                break
            if low[i]:
                last_low=i
                recovery_bins=0
            elif values[i] >= cfg.threshold + cfg.event_recovery_margin:
                recovery_bins+=1
                if recovery_bins>=cfg.event_recovery_confirmations:
                    break
            else:
                # A shallow, transient crossing above the threshold does not
                # close a hypo episode.
                recovery_bins=0
            i+=1
        j=last_low
        item={"Event":event_id,"Start_UTC":analysis.index[start],"End_low_UTC":analysis.index[j],
              "Min_glucose":float(analysis.Observed_Min.iloc[start:j+1].min()),"Low_readings":int(low[start:j+1].sum()),
              "Onset_observed":bool(start>0 and observed[start-1] and not low[start-1])}
        for h in HORIZONS:
            candidates=[k for k in range(max(0,start-h//5),start)
                        if bool(analysis.Alert_Ready.iloc[k]) and pd.notna(analysis.Raw.iloc[k])
                        and analysis.Raw.iloc[k]>cfg.threshold and observed[k:start+1].all()]
            hits=[k for k in candidates if pd.notna(analysis[f"Alert_{h}m"].iloc[k])
                  and analysis[f"Alert_{h}m"].iloc[k] == 1]
            notices=[k for k in candidates if bool(analysis[f"Notification_{h}m"].iloc[k])]
            item[f"Eligible_{h}"]=bool(candidates)
            item[f"Captured_{h}"]=bool(hits)
            item[f"Notified_{h}"]=bool(notices)
            item[f"Lead_minutes_{h}"]=float((analysis.index[start]-analysis.index[min(hits)]).total_seconds()/60) if hits else np.nan
        events.append(item)
    table=pd.DataFrame(events)
    summary=[]
    for h in HORIZONS:
        eligible=sum(e[f"Eligible_{h}"] for e in events)
        captured=sum(e[f"Captured_{h}"] for e in events)
        notified=sum(e[f"Notified_{h}"] for e in events)
        leads=[e[f"Lead_minutes_{h}"] for e in events if e[f"Captured_{h}"]]
        summary.append({"Horizon_min":h,"All_events":len(events),"Eligible_events":eligible,
                        "Captured":captured,"Missed_eligible":eligible-captured,"No_opportunity":len(events)-eligible,
                        "Capture_all_events":ratio(captured,len(events)),"Recall_eligible_events":ratio(captured,eligible),
                        "Notified_events":notified,"Median_lead_minutes":float(np.median(leads)) if leads else np.nan})
    return table,pd.DataFrame(summary)
