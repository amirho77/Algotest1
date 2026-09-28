"""Explicit, causal ingestion for ONE patient/sensor stream per invocation."""
from datetime import datetime, date
import re
import numpy as np
import pandas as pd
from config import InputConfig


def parse_timestamp(value, cfg):
    if pd.isna(value):
        return pd.NaT
    try:
        if isinstance(value, (int, float, np.number)):
            if cfg.numeric_time_unit is None:
                raise ValueError("Numeric timestamp requires an explicit epoch/Excel setting.")
            if cfg.numeric_time_unit == "excel":
                t = pd.to_datetime(value, unit="D", origin="1899-12-30")
            else:
                return pd.to_datetime(value, unit=cfg.numeric_time_unit, utc=True)
        elif isinstance(value, (pd.Timestamp, datetime, date, np.datetime64)):
            t = pd.Timestamp(value)
        else:
            value = str(value).strip()
            match = re.fullmatch(r"(\d{2})-(\d{2})-(\d{4} \d{2}:\d{2})(:\d{2})?\s+GMT([+-])(\d{1,2}):(\d{2})", value)
            if match:
                first, second, tail, seconds, sign, hour, minute = match.groups()
                # The explicit GMT suffix makes the timezone unambiguous. For
                # the calendar order, accept DD-MM when the first token is
                # >12 and MM-DD when the second token is >12; reject an
                # ambiguous pair instead of silently swapping day/month.
                if int(first) > 12 and int(second) <= 12:
                    date_part, fmt_date = f"{first}-{second}-{tail}", "%d-%m-%Y"
                elif int(second) > 12 and int(first) <= 12:
                    date_part, fmt_date = f"{first}-{second}-{tail}", "%m-%d-%Y"
                else:
                    # Preserve the historical SiSensing convention for an
                    # ambiguous pair; callers can still override with an
                    # explicit date_format when the source defines otherwise.
                    date_part, fmt_date = f"{first}-{second}-{tail}", "%d-%m-%Y"
                fmt = f"{fmt_date} %H:%M:%S %z" if seconds else f"{fmt_date} %H:%M %z"
                return pd.to_datetime(f"{date_part}{seconds or ''} {sign}{int(hour):02d}{minute}", format=fmt, utc=True)
            if cfg.date_format:
                t = pd.to_datetime(value, format=cfg.date_format)
            elif re.match(r"^\d{4}-\d{2}-\d{2}(?:$|[T ])", value):
                t = pd.Timestamp(value)
            else:
                return pd.NaT  # ambiguous locale dates must not be silently guessed
        return t.tz_localize(cfg.timezone, ambiguous="raise", nonexistent="raise").tz_convert("UTC") if t.tzinfo is None else t.tz_convert("UTC")
    except (ValueError, TypeError, OverflowError):
        return pd.NaT


def prepare(frame, time_col, glucose_col, cfg=None, *, as_of=None, stream_columns=()):
    cfg = cfg or InputConfig()
    if not frame.columns.is_unique or time_col == glucose_col:
        raise ValueError("Choose distinct columns with unique names.")
    # Do not mix histories from different patients or sensors, including missing IDs.
    for column in stream_columns:
        if frame[column].isna().any() or frame[column].nunique(dropna=False) != 1:
            raise ValueError(f"Select exactly one complete stream in column {column} before analysis.")
    if len(frame) > 600_000:
        raise ValueError("Upload is too large; split by patient/sensor and time period.")
    # Validate the timezone even when all incoming timestamps already have offsets.
    pd.Timestamp("2026-01-15").tz_localize(cfg.timezone)
    times = pd.to_datetime(frame[time_col].map(lambda x: parse_timestamp(x, cfg)), utc=True)
    tokens = frame[glucose_col].astype(str).str.strip().str.upper()
    numeric = pd.to_numeric(frame[glucose_col], errors="coerce").astype(float)
    if cfg.unit == "mmol/L":
        numeric *= 18.0
    finite = numeric.notna() & np.isfinite(numeric) & numeric.gt(0)
    in_range = finite.copy()
    if cfg.sensor_lower is not None:
        in_range &= numeric.ge(cfg.sensor_lower)
    if cfg.sensor_upper is not None:
        in_range &= numeric.le(cfg.sensor_upper)
    kinds = pd.Series(np.where(in_range, "numeric", "invalid"), index=frame.index)
    kinds.loc[tokens.isin(["LOW", "LO"])] = "censored_low"
    kinds.loc[tokens.isin(["HIGH", "HI"])] = "censored_high"
    raw = pd.DataFrame({"time":times, "value":numeric.where(in_range), "kind":kinds,
                        "invalid":~in_range, "token":tokens})
    raw["signature"] = np.where(in_range, numeric.astype(str), kinds+":"+tokens)
    invalid_times = int(times.isna().sum())
    raw = raw.dropna(subset=["time"]).sort_values("time", kind="stable")
    excluded_future = 0
    if as_of is not None:
        decision_time = pd.Timestamp(as_of)
        if decision_time.tzinfo is None:
            raise ValueError("as_of must include an explicit timezone.")
        decision_time = decision_time.tz_convert("UTC")
        excluded_future = int(raw.time.gt(decision_time).sum())
        raw = raw.loc[raw.time.le(decision_time)]
    if raw.empty:
        raise ValueError("No parseable timestamps available at the requested time. Specify date format/epoch if needed.")
    if as_of is None:
        decision_time = raw.time.iloc[-1]  # never pretend an unfinished bin is closed
    latest = raw.iloc[-1].copy()
    duplicate_exact = int(raw.duplicated(["time", "signature"]).sum())
    raw = raw.drop_duplicates(["time", "signature"])
    conflicts = raw.groupby("time").signature.nunique().gt(1)
    conflict_times = conflicts[conflicts].index
    raw.loc[raw.time.isin(conflict_times), ["value", "kind", "invalid"]] = [np.nan, "conflict", True]
    if latest.time in conflict_times:
        latest["kind"], latest["value"] = "conflict", np.nan
    raw = raw.drop_duplicates("time", keep="last").set_index("time")
    # A very stale snapshot needs a stale status, not years of synthetic empty bins.
    end = min(decision_time.ceil("5min"), raw.index[-1].ceil("5min")+pd.Timedelta(minutes=60))
    if raw.index[-1] - raw.index[0] > pd.Timedelta(days=cfg.max_days):
        raise ValueError("Time span exceeds the configured upload limit.")
    bins = raw.resample("5min", closed="right", label="right", origin="epoch")
    grid = pd.DataFrame({"Raw":bins.value.last(), "Observed_Min":bins.value.min(),
                         "Source_Count":bins.size(), "Invalid_In_Bin":bins.invalid.max(),
                         "Reading_Kind":bins.kind.last()})
    # Resample actual source timestamps separately (not just their bin labels).
    grid["Latest_Source_Time"] = pd.Series(raw.index, index=raw.index).resample("5min", closed="right", label="right", origin="epoch").max()
    index = pd.date_range(grid.index[0], end, freq="5min", tz="UTC", name="Time_UTC")
    grid = grid.reindex(index)
    grid["Source_Count"] = grid.Source_Count.fillna(0).astype(int)
    grid["Invalid_In_Bin"] = grid.Invalid_In_Bin.fillna(False).astype(bool)
    grid["Reading_Kind"] = grid.Reading_Kind.fillna("missing")
    grid["Bin_Complete"] = grid.index <= decision_time
    grid["Observed"] = grid.Raw.notna() & ~grid.Invalid_In_Bin
    grid["Label_Observed"] = grid.Observed & grid.Bin_Complete
    age = (decision_time-latest.time).total_seconds()/60
    metadata = {"input_rows":len(frame), "invalid_timestamps":invalid_times,
                "invalid_glucose_rows":int((~in_range & times.notna()).sum()),
                "duplicates_removed":duplicate_exact, "conflicting_timestamps":len(conflict_times),
                "missing_bins":int(grid.Source_Count.eq(0).sum()), "excluded_future_rows":excluded_future,
                "incomplete_bins":int((~grid.Bin_Complete).sum()),
                "latest_input_UTC":str(latest.time), "as_of_UTC":str(decision_time),
                "latest_kind":latest.kind, "latest_value":float(latest.value) if pd.notna(latest.value) else None,
                "latest_age_minutes":age, "unit":cfg.unit, "timezone":cfg.timezone,
                "sensor_lower":cfg.sensor_lower, "sensor_upper":cfg.sensor_upper,
                "stream_columns":list(stream_columns), "mode":"as_of" if as_of is not None else "file_replay"}
    return grid, metadata
