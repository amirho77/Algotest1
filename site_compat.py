"""Compatibility implementation of the public site's Model 1 defaults.

This is intentionally separate from the production model so comparisons use
the same 5-point/10-minute/30-minute policy and strict <70 event definition.
"""
import numpy as np
import pandas as pd


def predict_site(grid, slope_threshold=-0.1, horizon=10, cooldown=30):
    out = grid.copy()
    values = out.Raw.to_numpy(float)
    valid = (out.Observed & out.Bin_Complete & np.isfinite(values)).to_numpy(bool)
    n = len(out)
    slope = np.full(n, np.nan); forecast = np.full(n, np.nan); alert = np.zeros(n, bool)
    x = np.arange(5) * 5.0
    for i in range(4, n):
        if not valid[i-4:i+1].all(): continue
        y = values[i-4:i+1]
        b = float(np.polyfit(x, y, 1)[0])
        slope[i] = b; forecast[i] = values[i] + b*horizon
        alert[i] = values[i] >= 70 and b <= slope_threshold and forecast[i] <= 70
    notification = np.zeros(n, bool); last = None
    for i in np.flatnonzero(alert):
        if last is None or (out.index[i]-last).total_seconds()/60 >= cooldown:
            notification[i] = True; last = out.index[i]
    out["Site_Slope"] = slope; out["Site_Predicted_10m"] = forecast
    out["Site_Alert"] = alert; out["Site_Notification"] = notification
    return out


def evaluate_site(predictions, threshold=70, verify_minutes=30):
    raw = predictions.Raw.to_numpy(float)
    observed = (predictions.Observed & predictions.Bin_Complete).to_numpy(bool)
    low = (predictions.Observed_Min.lt(threshold) & predictions.Label_Observed).to_numpy(bool)
    starts = np.flatnonzero(low & ~np.r_[False, low[:-1]])
    events=[]
    for i in starts:
        candidates=[k for k in range(max(0, i-verify_minutes//5), i)
                    if predictions.Site_Alert.iloc[k] and observed[k:i+1].all() and raw[k] >= threshold]
        events.append(bool(candidates))
    notices = np.flatnonzero(predictions.Site_Notification.to_numpy(bool) & (raw >= threshold))
    outcomes=[]
    for k in notices:
        future=low[k+1:k+1+verify_minutes//5]
        outcomes.append("correct" if future.any() else "wrong")
    return {"events": int(len(starts)), "captured_events": int(sum(events)),
            "event_recall": float(np.mean(events)) if events else np.nan,
            "alerts": int(len(notices)), "correct_alerts": int(outcomes.count("correct")),
            "wrong_alerts": int(outcomes.count("wrong")),
            "precision": outcomes.count("correct")/len(outcomes) if outcomes else np.nan}
