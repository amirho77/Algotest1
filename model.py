"""v3 adds explicitly labeled preventive rules; v2 numeric forecasts remain unchanged."""
import numpy as np
import pandas as pd
from config import HORIZONS, ModelConfig
from model_v2 import predict as predict_v2, latest_state as state_v2


def predict(grid, cfg=None, *, method="enhanced"):
    cfg = cfg or ModelConfig()
    if method not in ("enhanced", "guarded", "guarded_legacy", "filtered", "episode", "recovery", "high_precision", "precision", "robust", "baseline"):
        raise ValueError("Choose enhanced, guarded, guarded_legacy, filtered, episode, recovery, high_precision, precision, robust (v2), or baseline (v1).")
    out = predict_v2(grid, cfg, method="robust" if method in ("enhanced", "guarded", "guarded_legacy", "filtered", "episode", "recovery", "high_precision", "precision") else method)
    for h in HORIZONS:
        out[f"Forecast_Alert_{h}m"] = out[f"Alert_{h}m"].copy()
    out["Alert_Ready"] = out.Prediction_Ready.copy()
    out["Preventive_Alert"] = False
    out["Near_Low_Risk"] = False
    out["Recent_Low_Risk"] = False
    out["Fast_Drop_Risk"] = False
    out["Approaching_Low_Risk"] = False
    out["Curvature_Drop_Risk"] = False
    out["Recovery_Filter"] = False
    out["Fast_ROC_10m"] = np.nan
    out["Alert_Reason"] = ""
    if method not in ("enhanced", "guarded", "guarded_legacy", "filtered", "episode", "recovery", "high_precision", "precision"):
        out.attrs["method"] = method
        return out
    n = len(out)
    valid = (out.Observed & out.Bin_Complete & np.isfinite(out.Raw)).to_numpy(bool)
    raw = out.Raw.to_numpy(float)
    short_ready = np.zeros(n, dtype=bool)
    near = np.zeros(n, dtype=bool)
    recent = np.zeros(n, dtype=bool)
    fast = np.zeros(n, dtype=bool)
    approaching = np.zeros(n, dtype=bool)
    curvature = np.zeros(n, dtype=bool)
    rates = np.full(n, np.nan)
    if n >= 3:
        window = np.lib.stride_tricks.sliding_window_view(raw, 3)
        usable = np.lib.stride_tricks.sliding_window_view(valid, 3).all(axis=1)
        indices = np.flatnonzero(usable) + 2
        short_ready[indices] = True
        y = window[usable]
        if len(y):
            rate = np.median(np.column_stack([(y[:, 1]-y[:, 0])/5, (y[:, 2]-y[:, 1])/5, (y[:, 2]-y[:, 0])/10]), axis=1)
            rates[indices] = rate
            ema = .25*y[:, 0] + .25*y[:, 1] + .5*y[:, 2]
            above = y[:, -1] > cfg.threshold
            near[indices] = ((y > cfg.threshold) & (y <= cfg.threshold + cfg.near_low_margin)).all(axis=1) & (y[:, -1] <= y[:, 0])
            fast[indices] = (above & (np.diff(y, axis=1) < 0).all(axis=1) & (rate <= -cfg.fast_drop_rate)
                             & (y[:, -1] <= cfg.threshold + cfg.fast_drop_max_distance)
                             & (ema + rate*30 <= cfg.threshold))
            approaching[indices] = (above & (np.diff(y, axis=1) < 0).all(axis=1)
                                    & (rate <= -cfg.guarded_approach_rate)
                                    & (y[:, -1] <= cfg.threshold + cfg.guarded_approach_margin))
    # Some real falls begin with modest velocity but a clearly worsening
    # velocity. Admit this narrow rescue only near the boundary; the later
    # per-horizon quadratic crossing test decides whether it is actionable.
    if method == "guarded":
        curvature = (short_ready & (raw > cfg.threshold)
                     & (raw <= cfg.threshold + cfg.curvature_rescue_margin)
                     & out.ROC_15m.le(cfg.curvature_rescue_max_roc).fillna(False).to_numpy(bool)
                     & out.Acceleration.le(cfg.curvature_rescue_max_acceleration).fillna(False).to_numpy(bool))
    for lag in (1, 2, 3):
        if n <= lag:
            continue
        continuous = np.lib.stride_tricks.sliding_window_view(valid, lag+1).all(axis=1)
        recent[lag:] |= continuous & (raw[:-lag] <= cfg.threshold)
    recent &= short_ready & (raw > cfg.threshold) & (raw <= cfg.threshold + cfg.recurrence_margin)
    # Near-boundary and recurrence rules must persist across consecutive
    # bins. Fast-drop already requires three monotonic readings, so it remains
    # immediate. This is the precision-oriented false-alarm control.
    k_guard = int(cfg.preventive_confirmations)
    near_confirmed = np.zeros(n, dtype=bool)
    recent_confirmed = np.zeros(n, dtype=bool)
    if k_guard == 1:
        near_confirmed, recent_confirmed = near.copy(), recent.copy()
    elif n >= k_guard:
        near_confirmed[k_guard-1:] = np.lib.stride_tricks.sliding_window_view(near, k_guard).all(axis=1)
        recent_confirmed[k_guard-1:] = np.lib.stride_tricks.sliding_window_view(recent, k_guard).all(axis=1)
    guard = near_confirmed | recent_confirmed | fast | curvature
    out["Alert_Ready"] = short_ready
    out["Preventive_Alert"] = guard
    out["Near_Low_Risk"] = near_confirmed
    out["Recent_Low_Risk"] = recent_confirmed
    out["Fast_Drop_Risk"] = fast
    out["Approaching_Low_Risk"] = approaching
    out["Curvature_Drop_Risk"] = curvature
    out["Fast_ROC_10m"] = rates
    out["Alert_Reason"] = ["|".join(name for name, flags in (("near_low", near_confirmed), ("recent_low", recent_confirmed), ("fast_drop", fast), ("accelerating_drop", curvature)) if flags[i]) for i in range(n)]
    out["Notification_Event_Id"] = pd.Series(pd.NA, index=out.index, dtype="Int64")
    out["Quality_Status"] = out.Quality_Status.where(~(short_ready & ~out.Prediction_Ready), "short_history_guards_only")
    for h in HORIZONS:
        forecast = out[f"Forecast_Alert_{h}m"].fillna(0).eq(1).to_numpy(bool)
        combined = forecast | guard
        if method in ("guarded", "guarded_legacy"):
            # Borrow the safety architecture of predictive pump algorithms
            # without using dosing inputs: several causal CGM-only slope
            # scenarios vote on an impending low. A normal alert needs two
            # scenario votes and two consecutive decision points. A fast,
            # monotonic approach to the boundary stays an immediate rescue
            # path so that confirmation does not hide rapid lows.
            scenario_slopes = np.column_stack([
                out.ROC_15m.to_numpy(float),
                out.Slope_15m_Robust.to_numpy(float),
                out.Slope_30m_Robust.to_numpy(float),
            ])
            current = out.Current_Glucose.to_numpy(float)
            crossings = (current[:, None] + scenario_slopes * h <= cfg.threshold) & (scenario_slopes < 0)
            consensus = np.isfinite(scenario_slopes).all(axis=1) & (crossings.sum(axis=1) >= 2)
            delta = np.diff(raw, prepend=np.nan)
            two_up = np.zeros(n, dtype=bool)
            if n >= 3:
                two_up[2:] = np.lib.stride_tricks.sliding_window_view(delta[1:], 2).min(axis=1) > 0
            recovery = two_up | out.Acceleration.ge(0.05).fillna(False).to_numpy(bool)
            persistence = out.Downward_Persistence.ge(50).fillna(False).to_numpy(bool)
            quadratic_prediction = (raw + out.ROC_15m.to_numpy(float) * h
                                    + 0.5 * out.Acceleration.to_numpy(float) * h * h)
            curvature_crossing = curvature & np.isfinite(quadratic_prediction) & (quadratic_prediction <= cfg.threshold)
            base = consensus & persistence & ~recovery
            confirmed = np.zeros(n, dtype=bool)
            k = int(cfg.guarded_confirmations)
            if k == 1:
                confirmed = base
            elif n >= k:
                confirmed[k - 1:] = np.lib.stride_tricks.sliding_window_view(base, k).all(axis=1)
            combined = (confirmed | fast | (curvature_crossing & ~recovery)
                        | (approaching & ~recovery) | (near_confirmed & ~recovery)
                        | (recent_confirmed & ~recovery))
            combined &= short_ready
            out[f"Scenario_Votes_{h}m"] = crossings.sum(axis=1)
            out[f"Scenario_Consensus_{h}m"] = consensus
            out[f"Curvature_Crossing_{h}m"] = curvature_crossing
            out["Recovery_Filter"] = recovery
        if method == "episode":
            # Three-factor episode gate: persistent descent, no recent
            # rebound, then 2-of-3-bin confirmation before opening an event.
            d = np.diff(raw, prepend=np.nan)
            neg = np.isfinite(d) & (d < -0.1)
            sustained = np.zeros(n, dtype=bool)
            if n >= 4:
                sustained[3:] = np.lib.stride_tricks.sliding_window_view(neg, 4).sum(axis=1) >= 3
            rebound = np.zeros(n, dtype=bool)
            if n >= 2:
                rebound[1:] = np.lib.stride_tricks.sliding_window_view(d, 2).sum(axis=1) > 2.0
            base = forecast & sustained & ~rebound & ~out.Trend_Disagreement.to_numpy(bool)
            base |= fast
            confirmed = np.zeros(n, dtype=bool)
            if n >= 3:
                confirmed[2:] = np.lib.stride_tricks.sliding_window_view(base, 3).sum(axis=1) >= 2
            combined = confirmed
        if method == "recovery":
            # Suppress a forecast that is already losing downward momentum.
            # Keep rapid-drop and near-boundary preventive signals available.
            delta = np.diff(raw, prepend=np.nan)
            recent_rebound = np.zeros(n, dtype=bool)
            if n >= 2:
                recent_rebound[1:] = np.lib.stride_tricks.sliding_window_view(delta, 2).sum(axis=1) > 2.0
            recovering = out.Acceleration.ge(0.05).fillna(False).to_numpy(bool) | recent_rebound
            combined = (forecast & ~recovering) | fast | (guard & (raw <= cfg.threshold + 5))
        if method == "filtered":
            # A single forecast crossing is not enough to open a user-facing
            # episode. Require corroboration from persistent descent, or use
            # the explicit three-point rapid-drop rule.
            persistence = out.Downward_Persistence.ge(60).fillna(False).to_numpy(bool)
            slope = out.Forecast_Slope.lt(-0.10).fillna(False).to_numpy(bool)
            corroborated = persistence & slope & ~out.Trend_Disagreement.to_numpy(bool)
            combined = (forecast & corroborated) | fast
            combined &= short_ready
        out[f"Alert_{h}m"] = pd.Series(combined, index=out.index).astype("Int64").where(short_ready)
        # Group a continuous risk episode into one user-facing event. A
        # notification is emitted only on entry into a confirmed episode (or
        # when it becomes urgent), rather than once per horizon/bin.
        notice = np.zeros(n, dtype=bool)
        event_ids = np.full(n, np.nan)
        active = combined & short_ready & (raw > cfg.threshold)
        confirmed_signal = np.zeros(n, dtype=bool)
        if n >= 3:
            confirmed_signal[2:] = np.lib.stride_tricks.sliding_window_view(active, 3).sum(axis=1) >= 2
        confirmed_signal |= fast
        urgent = active & ((raw <= cfg.threshold + 2) | fast)
        in_episode = False
        event_id = 0
        was_confirmed = False
        was_urgent = False
        last_sent = None
        for i in range(n):
            if not active[i]:
                in_episode = False; was_confirmed = False; was_urgent = False
                continue
            if not in_episode:
                event_id += 1; in_episode = True; was_confirmed = False; was_urgent = False
            event_ids[i] = event_id
            confirmed_now = bool(confirmed_signal[i])
            urgent_now = bool(urgent[i])
            transition = (confirmed_now and not was_confirmed) or (urgent_now and not was_urgent)
            elapsed = (out.index[i]-last_sent).total_seconds()/60 if last_sent is not None else np.inf
            if transition and (last_sent is None or elapsed >= cfg.notification_cooldown_minutes):
                notice[i] = True; last_sent = out.index[i]
            was_confirmed |= confirmed_now
            was_urgent |= urgent_now
        out["Notification_Event_Id"] = pd.Series(event_ids, index=out.index).astype("Int64")
        out[f"Notification_{h}m"] = notice
    if method in ("high_precision", "precision"):
        # Conservative abstention policy: only repeated numeric forecast flags
        # can notify. This is a precision-oriented gate, not a calibrated 95%
        # probability; calibration must be learned on held-out patients.
        out["Preventive_Alert"] = False
        out["Confidence_Status"] = "strict_repeated_forecast_uncalibrated"
        for h in HORIZONS:
            raw_forecast = out[f"Forecast_Alert_{h}m"].fillna(0).eq(1).to_numpy(bool)
            confirmed = np.zeros(n, dtype=bool)
            k = int(cfg.strict_confirmations)
            if k == 1:
                confirmed = raw_forecast.copy()
            elif n >= k:
                confirmed[k-1:] = np.lib.stride_tricks.sliding_window_view(raw_forecast, k).all(axis=1)
            confirmed &= out.Prediction_Ready.to_numpy(bool)
            # A precision notification needs independent evidence that the
            # descent is persistent and the trend windows agree. This keeps
            # isolated noisy forecast crossings out of the user-facing alert.
            if method == "precision":
                confirmed &= out.Downward_Persistence.ge(60).fillna(False).to_numpy(bool)
                confirmed &= ~out.Trend_Disagreement.to_numpy(bool)
            out[f"Alert_{h}m"] = pd.Series(confirmed, index=out.index).astype("Int64").where(out.Prediction_Ready)
            out[f"High_Confidence_Alert_{h}m"] = out[f"Alert_{h}m"]
            notice = np.zeros(n, dtype=bool)
            last_sent = None
            for i in np.flatnonzero(confirmed & (raw > cfg.threshold)):
                if last_sent is None or (out.index[i]-last_sent).total_seconds()/60 >= cfg.notification_cooldown_minutes:
                    notice[i] = True
                    last_sent = out.index[i]
            out[f"Notification_{h}m"] = notice
    out.attrs["method"] = method
    out.attrs["model_config"] = cfg.to_dict()
    return out


def latest_state(predictions, metadata, cfg=None):
    cfg = cfg or ModelConfig()
    state = state_v2(predictions, metadata, cfg)
    state["preventive_reasons"] = []
    state["alert_row"] = None
    if state["status"] in ("stale", "current_low", "sensor_check"):
        return state
    completed = predictions.loc[predictions.Bin_Complete]
    if len(completed):
        row = completed.iloc[-1]
        if row.Alert_Ready:
            state["alert_row"] = row
            forecast_active = row[f"Forecast_Alert_{cfg.primary_horizon}m"] == 1
            if row.Preventive_Alert and not forecast_active:
                state["status"] = "preventive_risk"
                state["preventive_reasons"] = row.Alert_Reason.split("|")
            elif not row.Prediction_Ready:
                state["status"] = "short_history"
    return state
