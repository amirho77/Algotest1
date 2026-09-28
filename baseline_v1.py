"""Research baseline. Scores are heuristic, not calibrated probabilities."""
import io
import re
import numpy as np
import pandas as pd

HORIZONS = (30, 45, 60)


def prepare(frame, time_col, glucose_col, timezone="Asia/Tehran", unit="mg/dL"):
    """One patient per call. Right-closed bins never timestamp a reading earlier."""
    if time_col == glucose_col:
        raise ValueError("Time and glucose columns must differ.")
    def timestamp(value):
        if pd.isna(value):
            return pd.NaT
        if isinstance(value, (int, float, np.number)):
            raise ValueError("Numeric timestamps are ambiguous; use ISO date/time.")
        # SiSensing exports DD-MM-YYYY and a literal GMT+H:MM offset.
        # Generic parsers can swap day/month AND invert GMT offset semantics.
        if isinstance(value, str):
            match = re.fullmatch(r"\s*(\d{2}-\d{2}-\d{4} \d{2}:\d{2})(:\d{2})?\s+GMT([+-])(\d{1,2}):(\d{2})\s*", value)
            if match:
                date, seconds, sign, hours, minutes = match.groups()
                stamp = f"{date}{seconds or ''} {sign}{int(hours):02d}{minutes}"
                fmt = "%d-%m-%Y %H:%M:%S %z" if seconds else "%d-%m-%Y %H:%M %z"
                return pd.to_datetime(stamp, format=fmt, errors="coerce", utc=True)
        try:
            t = pd.Timestamp(value)
        except (ValueError, TypeError):
            return pd.NaT
        return t.tz_localize(timezone, ambiguous="raise", nonexistent="raise").tz_convert("UTC") if t.tzinfo is None else t.tz_convert("UTC")
    times = pd.to_datetime(frame[time_col].map(timestamp), utc=True)
    values = pd.to_numeric(frame[glucose_col], errors="coerce")
    if unit == "mmol/L":
        values = values * 18.0
    elif unit != "mg/dL":
        raise ValueError("Unsupported glucose unit.")
    valid = times.notna() & values.notna() & np.isfinite(values) & values.gt(0)
    clean = pd.DataFrame({"Time_UTC": times[valid], "Raw": values[valid]})
    clean = clean.sort_values("Time_UTC", kind="stable")
    duplicate_count = int(clean.Time_UTC.duplicated().sum())
    clean = clean.drop_duplicates("Time_UTC", keep="last")
    if clean.empty:
        raise ValueError("No valid numeric glucose readings with valid timestamps.")
    s = clean.set_index("Time_UTC").Raw
    if s.index[-1] - s.index[0] > pd.Timedelta(days=366):
        raise ValueError("Upload at most 366 days for one patient per run.")
    # Last sample for features; minimum observed sample for outcome detection.
    resampler = s.resample("5min", closed="right", label="right", origin="epoch")
    out = pd.DataFrame({"Raw": resampler.last(), "Observed_Min": resampler.min()})
    out["Observed"] = out.Raw.notna()
    missing = out.Raw.isna()
    groups = missing.ne(missing.shift()).cumsum()
    lengths = missing.groupby(groups).transform("sum")
    # Offline visualization only; NEVER consumed by features or ground truth.
    out["Offline_Interpolated"] = out.Raw.where(~(missing & lengths.le(2)), out.Raw.interpolate(method="time", limit_area="inside"))
    info = {"input_rows": len(frame), "dropped_invalid": int((~valid).sum()),
            "duplicates_removed": duplicate_count, "missing_bins": int(missing.sum()),
            "timezone_for_naive_input": timezone, "input_unit": unit,
            "latest_input_UTC": str(s.index[-1]), "grid_policy": "right-closed, 5min; last sample; offline interpolation excluded"}
    return out, info


def predict(grid):
    out = grid.copy()
    out["Current_Glucose"] = np.nan
    # Each prediction uses only the most recent 13 readings (60 min, 12 intervals).
    out["Prediction_Ready"] = out.Raw.notna().rolling(13).sum().eq(13)
    for name in ("ROC_15m", "Acceleration", "Distance_to_70", "Downward_Persistence", "Low_Risk_Score"):
        out[name] = np.nan
    for i in range(len(out)):
        if not out.Prediction_Ready.iloc[i]:
            continue
        smooth = out.Raw.iloc[i-12:i+1].ewm(span=3, adjust=False).mean()
        current = smooth.iloc[-1]
        roc = (current - smooth.iloc[-4]) / 15
        previous_roc = (smooth.iloc[-3] - smooth.iloc[-6]) / 15
        acc = (roc - previous_roc) / 10
        persistence = smooth.diff().iloc[1:].lt(0).mean() * 100
        score = (.35*np.clip((180-current)/110*100, 0, 100)
                 + .35*np.clip(-roc/2*100, 0, 100)
                 + .15*np.clip(-acc/.1*100, 0, 100) + .15*persistence)
        out.loc[out.index[i], ["Current_Glucose", "ROC_15m", "Acceleration", "Distance_to_70", "Downward_Persistence", "Low_Risk_Score"]] = [current, roc, acc, current-70, persistence, score]
    out["Current_Low"] = out.Raw.le(70).astype("Int64").where(out.Observed)
    for h in HORIZONS:
        out[f"Pred_Glucose_{h}m"] = (out.Current_Glucose + out.ROC_15m*h).clip(20, 400)
        out[f"Alert_{h}m"] = (out[f"Pred_Glucose_{h}m"].le(70) | out.Low_Risk_Score.ge(65)).astype("Int64").where(out.Prediction_Ready)
    return out


def evaluate(predictions):
    out = predictions.copy()
    rows = []
    for h in HORIZONS:
        steps = h//5
        future = pd.concat([out.Observed_Min.shift(-k) for k in range(1, steps+1)], axis=1)
        complete = future.notna().all(axis=1)
        out[f"Min_next_{h}m"] = future.min(axis=1).where(complete)
        y = out[f"Min_next_{h}m"].le(70).astype("Int64").where(complete)
        out[f"Y{h}_Actual"] = y
        alert = out[f"Alert_{h}m"]
        eligible = complete & out.Prediction_Ready & out.Raw.gt(70)
        outcome = pd.Series("NotEvaluable", index=out.index)
        for a, b, label in ((1, 1, "TP"), (1, 0, "FP"), (0, 1, "FN"), (0, 0, "TN")):
            outcome.loc[(eligible & alert.eq(a) & y.eq(b)).fillna(False)] = label
        outcome.loc[out.Current_Low.eq(1).fillna(False)] = "AlreadyLow"
        out[f"Outcome_{h}m"] = outcome
        counts = {x: int(outcome.eq(x).sum()) for x in ("TP", "FP", "FN", "TN")}
        tp, fp, fn, tn = (counts[x] for x in ("TP", "FP", "FN", "TN"))
        rows.append({"Horizon_min": h, "Evaluable_rows": int(eligible.sum()), **counts,
                     "Precision": tp/(tp+fp) if tp+fp else np.nan,
                     "Recall": tp/(tp+fn) if tp+fn else np.nan,
                     "Specificity": tn/(tn+fp) if tn+fp else np.nan})
    return out, pd.DataFrame(rows)


def excel_bytes(analysis, summary, metadata):
    from openpyxl.styles import Font, PatternFill
    data = analysis.reset_index()
    data["Time_UTC"] = data.Time_UTC.dt.tz_convert("UTC").dt.tz_localize(None)
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        data.to_excel(writer, sheet_name="Analysis", index=False)
        summary.to_excel(writer, sheet_name="Evaluation", index=False)
        notes = {**metadata, "timestamp_export": "UTC, timezone removed only for Excel serialization",
                 "score": "0-100 heuristic; NOT probability; larger means greater risk",
                 "labels": "observed CGM minima; full future coverage required; not blood reference",
                 "interpolation": "Offline_Interpolated is retrospective only; excluded from prediction and labels",
                 "evaluation": "row-level only; no clinical validation; <=70 target; near-miss counts as FP"}
        pd.DataFrame(list(notes.items()), columns=["Setting", "Value"]).to_excel(writer, sheet_name="Metadata", index=False)
        for sheet in writer.book:
            sheet.freeze_panes = "A2"
            sheet.auto_filter.ref = sheet.dimensions
            for cell in sheet[1]:
                cell.font = Font(bold=True, color="FFFFFF")
                cell.fill = PatternFill("solid", fgColor="164E63")
                sheet.column_dimensions[cell.column_letter].width = 25
            # User-controlled strings must not become spreadsheet formulas.
            for row in sheet:
                for cell in row:
                    if cell.data_type == "f":
                        cell.data_type = "s"
    return buffer.getvalue()
