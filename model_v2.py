"""Causal multi-window trend baseline. No fitting on supplied evaluation files."""
import numpy as np
import pandas as pd
from config import HORIZONS, ModelConfig


def _median_slopes(values):
    """Theil-Sen slope across all pairs in each equally spaced window."""
    n = values.shape[1]
    i, j = np.triu_indices(n, 1)
    return np.median((values[:, j]-values[:, i])/((j-i)*5), axis=1)


def _validate_grid(grid):
    if not isinstance(grid.index, pd.DatetimeIndex) or grid.index.tz is None:
        raise ValueError("Grid must have timezone-aware datetime index.")
    if not grid.index.is_unique or not grid.index.is_monotonic_increasing:
        raise ValueError("Grid timestamps must be unique and sorted.")
    if len(grid)>1 and not ((grid.index[1:]-grid.index[:-1]) == pd.Timedelta(minutes=5)).all():
        raise ValueError("Use prepare(): the model requires an exact 5-minute grid.")
    if not grid.index.equals(grid.index.floor("5min")):
        raise ValueError("Use prepare(): timestamps must be aligned to UTC 5-minute boundaries.")


def predict(grid, cfg=None, *, method="robust"):
    cfg = cfg or ModelConfig()
    if method not in ("robust", "baseline"):
        raise ValueError("Choose robust or baseline.")
    _validate_grid(grid)
    out = grid.copy()
    n = len(out)
    columns = ["Current_Glucose", "ROC_15m", "Acceleration", "Distance_to_70", "Downward_Persistence",
               "Low_Risk_Score", "Slope_15m_Robust", "Slope_30m_Robust", "Forecast_Slope", "Residual_Spread"]
    features = np.full((n,len(columns)),np.nan)
    valid = out.Observed.to_numpy(bool) & out.Bin_Complete.to_numpy(bool) & np.isfinite(out.Raw.to_numpy(float))
    ready = np.zeros(n,dtype=bool)
    models = np.full((n,3),np.nan)
    disagree = np.zeros(n,dtype=bool)
    if n >= 13:
        windows = np.lib.stride_tricks.sliding_window_view(out.Raw.to_numpy(float),13).copy()
        good = np.lib.stride_tricks.sliding_window_view(valid,13).all(axis=1)
        indices = np.flatnonzero(good)+12
        ready[indices] = True
        y = windows[good]
        if len(y):
            smooth = y.copy()
            for k in range(1,13):
                smooth[:,k] = .5*y[:,k]+.5*smooth[:,k-1]
            current = smooth[:,-1]
            roc = (current-smooth[:,-4])/15
            acc = (roc-(smooth[:,-3]-smooth[:,-6])/15)/10
            persist = (np.diff(smooth,axis=1)<0).mean(axis=1)*100
            score = .35*np.clip((180-current)/110*100,0,100)+.35*np.clip(-roc/2*100,0,100)+.15*np.clip(-acc/.1*100,0,100)+.15*persist
            s15 = _median_slopes(smooth[:,-4:])
            s30 = _median_slopes(smooth[:,-7:])
            slopes = np.column_stack([roc,s15,s30])
            velocity = np.median(slopes,axis=1) if method=="robust" else roc
            x = np.arange(-30,1,5)
            residual = y[:,-7:]-(current[:,None]+velocity[:,None]*x)
            spread = np.maximum(cfg.uncertainty_floor,1.4826*np.median(np.abs(residual-np.median(residual,axis=1)[:,None]),axis=1))
            features[indices] = np.column_stack([current,roc,acc,current-cfg.threshold,persist,score,s15,s30,velocity,spread])
            models[indices] = slopes
            disagree[indices] = (slopes.min(axis=1)<0)&(slopes.max(axis=1)>0)
    for j,name in enumerate(columns):
        out[name] = features[:,j]
    out["Prediction_Ready"] = ready
    out["Trend_Disagreement"] = disagree
    out["Current_Low"] = out.Raw.le(cfg.threshold).astype("Int64").where(out.Observed & out.Reading_Kind.eq("numeric"))
    out["Quality_Status"] = np.select([~out.Bin_Complete, out.Invalid_In_Bin, ~out.Observed, ~out.Prediction_Ready],
                                      ["bin_not_closed", "invalid_or_censored", "missing", "insufficient_history"], default="ready")
    for h in HORIZONS:
        projections = out.Current_Glucose.to_numpy()[:,None]+models*h
        center = out.Current_Glucose+out.Forecast_Slope*h
        out[f"Pred_Glucose_{h}m"] = center.clip(20,400)
        out[f"Forecast_Clipped_{h}m"] = (center.lt(20)|center.gt(400)).where(out.Prediction_Ready)
        spread = out.Residual_Spread*np.sqrt(1+h/15)
        out[f"Lower_Scenario_{h}m"] = (np.min(projections,axis=1)-spread).clip(20,400)
        out[f"Upper_Scenario_{h}m"] = (np.max(projections,axis=1)+spread).clip(20,400)
        # No heuristic score can independently switch all horizons on in robust mode.
        # Robust mode abstains when slope windows disagree in direction. A
        # disagreement is a high-noise/reversal state where a single alert is
        # more likely to be false than useful. Baseline behavior is preserved.
        candidate = (center.le(cfg.threshold)&out.Forecast_Slope.lt(0)&~out.Trend_Disagreement) if method=="robust" else (center.le(cfg.threshold)|out.Low_Risk_Score.ge(65))
        out[f"Alert_{h}m"] = candidate.astype("Int64").where(out.Prediction_Ready)
        watch = out[f"Lower_Scenario_{h}m"].le(cfg.threshold) & ~candidate
        out[f"Watch_{h}m"] = watch.astype("Int64").where(out.Prediction_Ready)
        # Notification policy is distinct from model decisions; it never suppresses Current_Low.
        notifications = np.zeros(n,dtype=bool)
        last_sent = None
        for i in np.flatnonzero((candidate & out.Prediction_Ready & out.Raw.gt(cfg.threshold)).to_numpy()):
            if last_sent is None or (out.index[i]-last_sent).total_seconds()/60 >= cfg.notification_cooldown_minutes:
                notifications[i] = True
                last_sent = out.index[i]
        out[f"Notification_{h}m"] = notifications
    out.attrs["method"] = method
    out.attrs["model_config"] = cfg.to_dict()
    return out


def latest_state(predictions, metadata, cfg=None):
    """Snapshot with stale readings, censored low and open bins handled explicitly."""
    cfg = cfg or ModelConfig()
    kind, value = metadata["latest_kind"], metadata["latest_value"]
    age = metadata["latest_age_minutes"]
    numeric_low = kind=="numeric" and value is not None and value<=cfg.threshold
    bounded_low = kind=="censored_low" and metadata["sensor_lower"] is not None and metadata["sensor_lower"]<=cfg.threshold
    latest_low = numeric_low or bounded_low
    completed = predictions.loc[predictions.Bin_Complete]
    row = completed.iloc[-1] if len(completed) else None
    if age > cfg.stale_after_minutes:
        status = "stale"
    elif latest_low:
        status = "current_low"
    elif kind != "numeric":
        status = "sensor_check"
    elif row is None or not row.Prediction_Ready:
        status = "insufficient_data"
    elif row[f"Alert_{cfg.primary_horizon}m"]==1:
        status = "predicted_low"
    elif row.Trend_Disagreement or row[f"Watch_{cfg.primary_horizon}m"]==1:
        status = "uncertain_trend"
    else:
        status = "no_model_alert"
    usable = status not in ("stale","sensor_check","insufficient_data") and row is not None and bool(row.Prediction_Ready)
    return {"status":status,"latest_reading_low":latest_low,"reading_age_minutes":age,
            "forecast_time_UTC":str(row.name) if row is not None else None,
            "forecasts":{h:float(row[f"Pred_Glucose_{h}m"]) for h in HORIZONS} if usable else {},
            "row":row if usable else None}
