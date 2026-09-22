from __future__ import annotations

from datetime import datetime, timedelta
import io
import re
from typing import Iterable

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ============================================================
# CGM LOW-GLUCOSE EARLY-WARNING MVP v0.5
# ============================================================
# Fixes vs v0.4:
#   1) Robust timestamp dtype handling in LIVE mode.
#   2) Empty-dataframe schema has explicit datetime/float dtypes.
#   3) Graceful UI while the 60-minute buffer is filling.
#   4) Safer evaluation for short LIVE streams.
#   5) Clearer LIVE dashboard and alert panel.
#   6) Same v0.4 mathematical algorithm; this version fixes runtime/UI.
#
# This remains a research/prototype algorithm, not a clinical device.
# ============================================================

CFG = {
    "LOW": 70.0,
    "LOOKBACK_MIN": 60,
    "EXPECTED_SAMPLE_MIN": 5,
    "MAX_ALLOWED_GAP_MIN": 10,
    "EMA_SPAN": 3,
    "RISK_THRESHOLD": 68.0,
    "FORECAST_THRESHOLD": 80.0,
    "TIME_RISK_HORIZON_MIN": 45.0,
    "TIME_RISK_TARGET": 75.0,
    "W_PROX": 0.25,
    "W_TIME": 0.45,
    "W_VEL": 0.10,
    "W_LONG_VEL": 0.05,
    "W_PERSIST": 0.05,
    "W_ACCEL": 0.10,
    "MIN_ROC_30": -0.20,
    "ALERT_COOLDOWN_MIN": 20,
    "HORIZONS": [30, 45, 60],
}


st.set_page_config(
    page_title="CGM Low-Glucose Early Warning MVP v0.5",
    page_icon="🩸",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main { direction: rtl; }
    .block-container { padding-top: 1.2rem; }
    .live-card {
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        padding: 16px;
        background: #ffffff;
        margin-bottom: 12px;
    }
    .alert-card {
        border-radius: 14px;
        padding: 18px;
        margin: 8px 0 16px 0;
        text-align: center;
        font-size: 1.05rem;
        font-weight: 700;
    }
    .small-note {
        color: #64748b;
        font-size: 0.85rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SCHEMA / INPUT HELPERS
# ============================================================


def empty_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Timestamp": pd.Series([], dtype="datetime64[ns]"),
            "Glucose": pd.Series([], dtype="float64"),
        }
    )


def ensure_schema(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if "Timestamp" not in out.columns or "Glucose" not in out.columns:
        raise ValueError("داده باید ستون‌های Timestamp و Glucose را داشته باشد.")

    # THIS is the critical runtime fix:
    # always force Timestamp to true datetime dtype before .dt operations.
    out["Timestamp"] = pd.to_datetime(
        out["Timestamp"],
        errors="coerce",
    )
    out["Glucose"] = pd.to_numeric(
        out["Glucose"],
        errors="coerce",
    ).astype("float64")

    out = (
        out.dropna(subset=["Timestamp", "Glucose"])
        .sort_values("Timestamp")
        .drop_duplicates("Timestamp", keep="last")
        .reset_index(drop=True)
    )

    return out


def parse_glucose(value) -> float:
    s = str(value).strip().upper()
    if s == "LOW":
        return 39.0
    if s == "HIGH":
        return 401.0
    return float(value)


def parse_timestamp(value) -> pd.Timestamp:
    s = re.sub(r"\s*GMT.*$", "", str(value).strip())
    return pd.to_datetime(s, dayfirst=True, errors="coerce")


def normalize_dataframe(raw: pd.DataFrame) -> pd.DataFrame:
    if raw.shape[1] < 2:
        raise ValueError("فایل باید حداقل دو ستون داشته باشد: زمان و قند")

    out = pd.DataFrame(
        {
            "Timestamp": raw.iloc[:, 0].map(parse_timestamp),
            "Glucose": raw.iloc[:, 1].map(parse_glucose),
        }
    )

    return ensure_schema(out)


def read_sensor_file(uploaded_file) -> pd.DataFrame:
    name = uploaded_file.name.lower()
    raw = pd.read_csv(uploaded_file) if name.endswith(".csv") else pd.read_excel(uploaded_file)
    return normalize_dataframe(raw)


def split_segments(df: pd.DataFrame) -> list[pd.DataFrame]:
    """
    Split on large timestamp gaps.

    Important: v0.5 explicitly converts Timestamp to datetime64[ns]
    before using .dt, fixing the LIVE-mode crash under newer pandas.
    """
    out = ensure_schema(df)

    if len(out) <= 1:
        return [out]

    gaps = (
        out["Timestamp"].diff().dt.total_seconds().div(60.0)
    )

    segment_id = gaps.gt(CFG["MAX_ALLOWED_GAP_MIN"]).cumsum()
    out = out.copy()
    out["Segment"] = segment_id.astype(int)

    segments: list[pd.DataFrame] = []
    for _, part in out.groupby("Segment", sort=True):
        segments.append(
            part.drop(columns=["Segment"]).reset_index(drop=True)
        )

    return segments


def sample_df(
    values: Iterable[float],
    start: str = "2025-01-01 00:00",
) -> pd.DataFrame:
    values = [float(v) for v in values]
    return ensure_schema(
        pd.DataFrame(
            {
                "Timestamp": pd.date_range(
                    start=start,
                    periods=len(values),
                    freq="5min",
                ),
                "Glucose": values,
            }
        )
    )


# ============================================================
# CORE FEATURE ENGINE
# ============================================================


def calculate_features(segment: pd.DataFrame) -> pd.DataFrame:
    out = ensure_schema(segment)
    g = out["Glucose"]

    smooth = g.ewm(
        span=CFG["EMA_SPAN"],
        adjust=False,
    ).mean()

    out["Smooth"] = smooth

    out["ROC_10m"] = (smooth - smooth.shift(2)) / 10.0
    out["ROC_30m"] = (smooth - smooth.shift(6)) / 30.0
    out["ROC_60m"] = (smooth - smooth.shift(12)) / 60.0
    out["Raw_Net_10m"] = g - g.shift(2)

    out["Acceleration"] = (
        out["ROC_10m"] - out["ROC_30m"]
    )

    moves = g.diff()

    out["Persistence_30m"] = (
        moves.lt(0)
        .rolling(6, min_periods=6)
        .mean()
        * 100.0
    )

    out["Persistence_60m"] = (
        moves.lt(0)
        .rolling(12, min_periods=12)
        .mean()
        * 100.0
    )

    # Adaptive trend: when glucose is low, recent slope matters more.
    trend = np.full(len(out), np.nan, dtype=float)

    for i in range(len(out)):
        g_now = float(g.iloc[i])
        s10 = out["ROC_10m"].iloc[i]
        s30 = out["ROC_30m"].iloc[i]

        if not (np.isfinite(s10) and np.isfinite(s30)):
            continue

        if g_now <= 90:
            w10 = 0.70
        elif g_now <= 110:
            w10 = 0.55
        elif g_now <= 140:
            w10 = 0.35
        else:
            w10 = 0.20

        trend[i] = w10 * s10 + (1.0 - w10) * s30

    out["Trend_Slope"] = trend

    for h in CFG["HORIZONS"]:
        out[f"Pred_{h}m"] = np.clip(
            smooth + trend * h,
            20.0,
            400.0,
        )

    effective_slope = (
        0.60 * out["ROC_10m"]
        + 0.40 * out["ROC_30m"]
    )
    out["Effective_Slope"] = effective_slope

    out["Time_to_75m"] = np.where(
        effective_slope < 0,
        (CFG["TIME_RISK_TARGET"] - smooth) / effective_slope,
        np.inf,
    )

    # Risk score components (0-100)
    s_prox = np.clip(
        (130.0 - smooth) / 60.0 * 100.0,
        0.0,
        100.0,
    )

    s_time = np.clip(
        (
            CFG["TIME_RISK_HORIZON_MIN"]
            - out["Time_to_75m"]
        )
        / CFG["TIME_RISK_HORIZON_MIN"]
        * 100.0,
        0.0,
        100.0,
    )
    s_time = np.where(
        out["Time_to_75m"] <= 0,
        100.0,
        s_time,
    )

    s_vel = np.clip(
        -out["ROC_10m"] / 1.5 * 100.0,
        0.0,
        100.0,
    )

    s_long_vel = np.clip(
        -out["ROC_30m"] / 1.0 * 100.0,
        0.0,
        100.0,
    )

    s_persist = out["Persistence_60m"].fillna(0.0)

    s_acc = np.clip(
        -out["Acceleration"] / 0.5 * 100.0,
        0.0,
        100.0,
    )

    out["Risk_Score"] = (
        CFG["W_PROX"] * s_prox
        + CFG["W_TIME"] * s_time
        + CFG["W_VEL"] * s_vel
        + CFG["W_LONG_VEL"] * s_long_vel
        + CFG["W_PERSIST"] * s_persist
        + CFG["W_ACCEL"] * s_acc
    )

    out["Risk_Proximity"] = s_prox
    out["Risk_Time"] = s_time
    out["Risk_Velocity"] = s_vel
    out["Risk_Long_Velocity"] = s_long_vel
    out["Risk_Persistence"] = s_persist
    out["Risk_Acceleration"] = s_acc

    return out


# ============================================================
# LABELS
# ============================================================


def add_future_labels(segment: pd.DataFrame) -> pd.DataFrame:
    out = ensure_schema(segment)

    times = out["Timestamp"].to_numpy(dtype="datetime64[ns]")
    glucose = out["Glucose"].to_numpy(dtype=float)

    for h in CFG["HORIZONS"]:
        mins = np.full(len(out), np.nan)
        y = np.full(len(out), np.nan)
        horizon = np.timedelta64(h, "m")

        for i, now in enumerate(times):
            end = now + horizon

            if end > times[-1]:
                continue

            j = np.searchsorted(times, end, side="right")
            future = glucose[i + 1 : j]

            if len(future) == 0:
                continue

            mins[i] = float(np.min(future))
            y[i] = float(mins[i] <= CFG["LOW"])

        out[f"Min_next_{h}m"] = mins
        out[f"Y{h}"] = y

    return out


# ============================================================
# ALERT LOGIC
# ============================================================


def apply_alert_logic(segment: pd.DataFrame) -> pd.DataFrame:
    out = ensure_schema(segment)

    has_history = (
        out["Timestamp"] - out["Timestamp"].iloc[0]
        >= pd.Timedelta(minutes=CFG["LOOKBACK_MIN"])
    )

    common = (
        has_history
        & (out["Glucose"] > CFG["LOW"])
        & (out["ROC_30m"] <= CFG["MIN_ROC_30"])
        & (out["Risk_Score"] >= CFG["RISK_THRESHOLD"])
    )

    for h in CFG["HORIZONS"]:
        out[f"Alert_{h}m"] = (
            common
            & (out[f"Pred_{h}m"] <= CFG["FORECAST_THRESHOLD"])
        )

    raw_alert = out["Alert_30m"].fillna(False).to_numpy(bool)

    cooldown_steps = max(
        1,
        int(
            CFG["ALERT_COOLDOWN_MIN"]
            / CFG["EXPECTED_SAMPLE_MIN"]
        ),
    )

    alert = raw_alert.copy()
    last_alert_idx = -10**9

    for i in np.where(raw_alert)[0]:
        if i - last_alert_idx <= cooldown_steps:
            alert[i] = False
        else:
            last_alert_idx = i

    out["Alert"] = alert
    out["Current_Low"] = out["Glucose"] <= CFG["LOW"]
    out["Any_Alert"] = out["Alert"]

    out["Status"] = np.select(
        [
            out["Current_Low"],
            out["Any_Alert"],
            has_history,
        ],
        [
            "🚨 Low فعلی",
            "⚠️ هشدار افت",
            "✅ بدون هشدار",
        ],
        default="⏳ در حال تکمیل بافر",
    )

    return out


# ============================================================
# PIPELINE
# ============================================================


def process_one_segment(segment: pd.DataFrame) -> pd.DataFrame:
    return apply_alert_logic(
        add_future_labels(
            calculate_features(segment)
        )
    )


def process_dataset(df: pd.DataFrame) -> pd.DataFrame:
    clean = ensure_schema(df)

    if clean.empty:
        return empty_data()

    parts = [
        process_one_segment(segment)
        for segment in split_segments(clean)
        if not segment.empty
    ]

    if not parts:
        return empty_data()

    return (
        pd.concat(parts, ignore_index=True)
        .sort_values("Timestamp")
        .reset_index(drop=True)
    )


# ============================================================
# EVALUATION
# ============================================================


def low_episode_starts(df: pd.DataFrame) -> list[int]:
    if df.empty:
        return []

    low = df["Glucose"].to_numpy(dtype=float) <= CFG["LOW"]
    return np.where(low & ~np.r_[False, low[:-1]])[0].tolist()


def alert_runs(alert: np.ndarray) -> list[tuple[int, int]]:
    runs = []
    i = 0

    while i < len(alert):
        if alert[i]:
            start = i
            while i + 1 < len(alert) and alert[i + 1]:
                i += 1
            runs.append((start, i))
        i += 1

    return runs


def classification_metrics(
    df: pd.DataFrame,
    alert_column: str,
    horizon: int,
) -> dict:
    if df.empty or f"Y{horizon}" not in df.columns:
        return {
            "N": 0,
            "TP": 0,
            "FP": 0,
            "FN": 0,
            "TN": 0,
            "Precision": np.nan,
            "Recall": np.nan,
            "Specificity": np.nan,
            "F1": np.nan,
        }

    y = df[f"Y{horizon}"].to_numpy(dtype=float)
    pred = df[alert_column].fillna(False).to_numpy(dtype=bool)
    glucose = df["Glucose"].to_numpy(dtype=float)

    mask = ~np.isnan(y) & (glucose > CFG["LOW"])
    yy = y[mask].astype(int)
    pp = pred[mask].astype(int)

    tp = int(((yy == 1) & (pp == 1)).sum())
    fp = int(((yy == 0) & (pp == 1)).sum())
    fn = int(((yy == 1) & (pp == 0)).sum())
    tn = int(((yy == 0) & (pp == 0)).sum())

    precision = tp / (tp + fp) if tp + fp else np.nan
    recall = tp / (tp + fn) if tp + fn else np.nan
    specificity = tn / (tn + fp) if tn + fp else np.nan
    f1 = (
        2 * precision * recall / (precision + recall)
        if np.isfinite(precision)
        and np.isfinite(recall)
        and precision + recall > 0
        else np.nan
    )

    return {
        "N": len(yy),
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "TN": tn,
        "Precision": precision,
        "Recall": recall,
        "Specificity": specificity,
        "F1": f1,
    }


def event_metrics(
    df: pd.DataFrame,
    alert_column: str = "Alert",
    min_lead_min: int = 10,
    max_lead_min: int = 60,
) -> dict:
    if df.empty:
        return {
            "Low_Episodes": 0,
            "Detected_any_<=60m": 0,
            "Event_Recall_any": np.nan,
            "Detected_with_>=10m_lead": 0,
            "Event_Recall_>=10m": np.nan,
            "False_Alert_Runs": 0,
            "False_Alerts_per_Day": np.nan,
            "Median_Lead_any": np.nan,
            "Median_Lead_>=10m": np.nan,
            "Lead_Times_any": [],
            "Lead_Times_early": [],
        }

    alerts = df[alert_column].fillna(False).to_numpy(dtype=bool)
    starts = low_episode_starts(df)
    runs = alert_runs(alerts)

    detected_any = 0
    detected_early = 0
    leads_any = []
    leads_early = []
    matched_alert_runs = set()

    for low_start in starts:
        candidates = []

        for j, (run_start, _) in enumerate(runs):
            lead = (low_start - run_start) * CFG["EXPECTED_SAMPLE_MIN"]
            if 0 < lead <= max_lead_min:
                candidates.append((j, lead))

        if not candidates:
            continue

        j, lead = candidates[-1]
        matched_alert_runs.add(j)
        detected_any += 1
        leads_any.append(lead)

        if lead >= min_lead_min:
            detected_early += 1
            leads_early.append(lead)

    false_alert_runs = max(
        0,
        len(runs) - len(matched_alert_runs),
    )

    episode_count = len(starts)
    duration_days = max(
        (
            df["Timestamp"].iloc[-1]
            - df["Timestamp"].iloc[0]
        ).total_seconds() / 86400.0,
        1 / 1440,
    )

    return {
        "Low_Episodes": episode_count,
        "Detected_any_<=60m": detected_any,
        "Event_Recall_any": (
            detected_any / episode_count
            if episode_count
            else np.nan
        ),
        "Detected_with_>=10m_lead": detected_early,
        "Event_Recall_>=10m": (
            detected_early / episode_count
            if episode_count
            else np.nan
        ),
        "False_Alert_Runs": false_alert_runs,
        "False_Alerts_per_Day": false_alert_runs / duration_days,
        "Median_Lead_any": (
            float(np.median(leads_any)) if leads_any else np.nan
        ),
        "Median_Lead_>=10m": (
            float(np.median(leads_early)) if leads_early else np.nan
        ),
        "Lead_Times_any": leads_any,
        "Lead_Times_early": leads_early,
    }


def evaluate_dataset(processed: pd.DataFrame) -> dict:
    metrics = {}

    for h in CFG["HORIZONS"]:
        metrics[f"{h}m_point"] = classification_metrics(
            processed,
            f"Alert_{h}m",
            h,
        )

    metrics["event_30m"] = event_metrics(
        processed,
        "Alert",
    )

    return metrics


def temporal_split_evaluation(processed: pd.DataFrame) -> pd.DataFrame:
    if len(processed) < 5:
        return pd.DataFrame()

    n = len(processed)
    cut1 = int(n * 0.60)
    cut2 = int(n * 0.80)

    rows = []

    for name, start, end in [
        ("Train/Development", 0, cut1),
        ("Validation", cut1, cut2),
        ("Holdout/Test", cut2, n),
        ("All", 0, n),
    ]:
        part = processed.iloc[start:end].copy()
        cm = classification_metrics(
            part,
            "Alert",
            30,
        )
        em = event_metrics(part, "Alert")

        rows.append(
            {
                "Split": name,
                **cm,
                "Low Episodes": em["Low_Episodes"],
                "Event Recall >=10m": em["Event_Recall_>=10m"],
                "Event Recall any <=60m": em["Event_Recall_any"],
                "False Alert Runs": em["False_Alert_Runs"],
                "Median Lead >=10m": em["Median_Lead_>=10m"],
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# EXPORT
# ============================================================


def make_export(
    processed: pd.DataFrame,
    eval_df: pd.DataFrame,
) -> bytes:
    buffer = io.BytesIO()

    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        processed.to_excel(
            writer,
            index=False,
            sheet_name="CGM_MVP",
        )

        eval_df.to_excel(
            writer,
            index=False,
            sheet_name="Evaluation",
        )

        starts = low_episode_starts(processed)
        low_rows = []

        for i, start in enumerate(starts, 1):
            next_starts = [x for x in starts if x > start]
            end = (
                next_starts[0] - 1
                if next_starts
                else len(processed) - 1
            )

            low_rows.append(
                {
                    "Episode": i,
                    "Start": processed.loc[start, "Timestamp"],
                    "Start_Glucose": processed.loc[start, "Glucose"],
                    "End": processed.loc[end, "Timestamp"],
                    "Minimum": processed.loc[start:end, "Glucose"].min(),
                }
            )

        pd.DataFrame(low_rows).to_excel(
            writer,
            index=False,
            sheet_name="Low_Episodes",
        )

    return buffer.getvalue()


# ============================================================
# BUILT-IN TESTS
# ============================================================

APRIL_DATASET_1 = [
    190,181,178,174,171,169,160,149,140,136,133,129,126,122,120,117,
    113,111,115,111,106,100,97,90,88,86,86,84,82,82,82,84,86,86,86,86,
    88,88,79,82,90,90,97,97,97,97,97,99,97,95,95,95,77,72,72,91,90,93,
    93,93,88,86,86,90,90,86,84,84,84,84,81,72,81,91,90,86,84,82,86,82,
    81,79,77,73,73,72,70,72,72,79,82,81,82,82,82,79,77,81,79,82,90,91,
    91,93,95,95,
]

APRIL_DATASET_2 = [
    329,324,299,273,250,243,243,230,228,203,194,183,176,167,156,140,
    129,118,111,106,100,97,99,93,93,91,84,86,88,88,84,81,73,68,
]


# ============================================================
# SESSION STATE
# ============================================================

if "data" not in st.session_state:
    st.session_state.data = empty_data()

if "live_base" not in st.session_state:
    st.session_state.live_base = pd.Timestamp.now().floor("min")


# ============================================================
# SIDEBAR / INPUT UI
# ============================================================

st.title("🩸 CGM Low-Glucose Early Warning — MVP v0.5")
st.caption(
    "CGM-only mathematical early-warning prototype — Low threshold = 70 mg/dL"
)

with st.sidebar:
    st.header("ورودی داده")

    mode = st.radio(
        "حالت ورود",
        [
            "LIVE — هر ۵ دقیقه",
            "آپلود Excel / CSV",
            "تست داخلی",
        ],
    )

    if mode == "LIVE — هر ۵ دقیقه":
        st.subheader("ثبت Reading جدید")

        new_glucose = st.number_input(
            "قند (mg/dL)",
            min_value=20.0,
            max_value=450.0,
            value=115.0,
            step=1.0,
            key="live_glucose",
        )

        if st.button(
            "➕ ثبت Reading",
            use_container_width=True,
            type="primary",
        ):
            if st.session_state.data.empty:
                timestamp = pd.Timestamp(
                    st.session_state.live_base
                ).floor("min")
            else:
                timestamp = (
                    st.session_state.data.iloc[-1]["Timestamp"]
                    + pd.Timedelta(minutes=5)
                )

            new_row = pd.DataFrame(
                {
                    "Timestamp": [timestamp],
                    "Glucose": [float(new_glucose)],
                }
            )

            st.session_state.data = ensure_schema(
                pd.concat(
                    [
                        st.session_state.data,
                        new_row,
                    ],
                    ignore_index=True,
                )
            )

            st.rerun()

        if st.button(
            "🗑 پاک‌سازی Live",
            use_container_width=True,
        ):
            st.session_state.data = empty_data()
            st.session_state.live_base = pd.Timestamp.now().floor("min")
            st.rerun()

        count = len(st.session_state.data)
        buffer_needed = int(CFG["LOOKBACK_MIN"] / CFG["EXPECTED_SAMPLE_MIN"]) + 1
        progress = min(count / buffer_needed, 1.0)

        st.progress(
            progress,
            text=f"بافر: {min(count, buffer_needed)}/{buffer_needed} reading",
        )

        if count < buffer_needed:
            st.info(
                f"برای تحلیل کامل حداقل {buffer_needed} reading لازم است "
                f"(حدود ۶۰ دقیقه + نقطه فعلی)."
            )

    elif mode == "آپلود Excel / CSV":
        uploaded = st.file_uploader(
            "فایل داده CGM",
            type=["xls", "xlsx", "csv"],
        )

        if uploaded is not None:
            try:
                st.session_state.data = read_sensor_file(uploaded)
                st.success(
                    f"{len(st.session_state.data):,} رکورد بارگذاری شد."
                )
            except Exception as exc:
                st.error(f"خطا در خواندن فایل: {exc}")

    else:
        if st.button(
            "تست April #1",
            use_container_width=True,
        ):
            st.session_state.data = sample_df(
                APRIL_DATASET_1,
                "2025-04-20 00:00",
            )
            st.rerun()

        if st.button(
            "تست April #2",
            use_container_width=True,
        ):
            st.session_state.data = sample_df(
                APRIL_DATASET_2,
                "2025-04-20 00:00",
            )
            st.rerun()


# ============================================================
# EMPTY STATE
# ============================================================

if st.session_state.data.empty:
    st.info(
        "هنوز داده‌ای ثبت نشده. از Sidebar یک Reading وارد کن یا فایل CGM را آپلود کن."
    )
    st.stop()


# ============================================================
# PROCESS
# ============================================================

processed = process_dataset(
    st.session_state.data
)

eval_all = evaluate_dataset(processed)
eval_split = temporal_split_evaluation(processed)
last = processed.iloc[-1]


# ============================================================
# BUFFER / ALERT BANNER
# ============================================================

history_minutes = (
    last["Timestamp"]
    - processed["Timestamp"].iloc[0]
).total_seconds() / 60.0

buffer_ready = (
    history_minutes >= CFG["LOOKBACK_MIN"]
)

if not buffer_ready:
    remaining = max(
        0,
        CFG["LOOKBACK_MIN"] - history_minutes,
    )

    st.warning(
        f"⏳ بافر اولیه هنوز کامل نشده است. حدود {remaining:.0f} دقیقه دیگر برای محاسبه کامل Risk/Alert لازم است."
    )

elif bool(last["Current_Low"]):
    st.markdown(
        '<div class="alert-card" style="background:#fee2e2;color:#991b1b;">'
        "🚨 Glucose فعلی در محدوده Low است (≤70)"
        "</div>",
        unsafe_allow_html=True,
    )

elif bool(last["Alert"]):
    st.markdown(
        '<div class="alert-card" style="background:#fef3c7;color:#92400e;">'
        "⚠️ هشدار افت قند — مسیر ۳۰ دقیقه آینده پرریسک تشخیص داده شد"
        "</div>",
        unsafe_allow_html=True,
    )

else:
    st.success(
        "✅ در این لحظه Alert فعال نیست."
    )


# ============================================================
# TOP METRICS
# ============================================================

c1, c2, c3, c4, c5 = st.columns(5)

c1.metric(
    "قند فعلی",
    f"{last['Glucose']:.0f} mg/dL",
)

c2.metric(
    "EMA",
    f"{last['Smooth']:.1f}",
)

c3.metric(
    "ROC 10m",
    (
        "—"
        if pd.isna(last["ROC_10m"])
        else f"{last['ROC_10m']:.2f}"
    ),
)

c4.metric(
    "Risk Score",
    (
        "—"
        if pd.isna(last["Risk_Score"])
        else f"{last['Risk_Score']:.1f}/100"
    ),
)

c5.metric(
    "وضعیت",
    last["Status"],
)


# ============================================================
# FORECAST CARDS
# ============================================================

st.subheader("🔮 پیش‌بینی")
fc = st.columns(3)

for col, h in zip(fc, CFG["HORIZONS"]):
    with col:
        pred = last[f"Pred_{h}m"]
        alert = bool(last[f"Alert_{h}m"])

        st.metric(
            f"+{h} دقیقه",
            "—" if pd.isna(pred) else f"{pred:.1f} mg/dL",
            "⚠️ Alert" if alert else "No alert",
        )


# ============================================================
# CHARTS
# ============================================================

st.subheader("📈 روند CGM")

fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=processed["Timestamp"],
        y=processed["Glucose"],
        name="CGM",
        mode="lines+markers",
    )
)

fig.add_trace(
    go.Scatter(
        x=processed["Timestamp"],
        y=processed["Smooth"],
        name="EMA",
        mode="lines",
    )
)

fig.add_hline(
    y=70,
    line_dash="dash",
    annotation_text="Low = 70",
)

if pd.notna(last["Pred_30m"]):
    future_x = [
        last["Timestamp"] + timedelta(minutes=h)
        for h in CFG["HORIZONS"]
    ]
    future_y = [
        last[f"Pred_{h}m"]
        for h in CFG["HORIZONS"]
    ]

    fig.add_trace(
        go.Scatter(
            x=future_x,
            y=future_y,
            name="Forecast",
            mode="lines+markers",
        )
    )

fig.update_layout(
    height=420,
    hovermode="x unified",
    margin=dict(l=20, r=20, t=30, b=20),
)

st.plotly_chart(
    fig,
    use_container_width=True,
)


# ============================================================
# TABS
# ============================================================

tab_live, tab_details, tab_eval, tab_params = st.tabs(
    [
        "🩸 LIVE",
        "📋 جزئیات",
        "📊 ارزیابی",
        "⚙️ پارامترها",
    ]
)


# ------------------------------------------------------------
# LIVE TAB
# ------------------------------------------------------------
with tab_live:
    st.subheader("آخرین وضعیت")

    live_cols = [
        "Timestamp",
        "Glucose",
        "Smooth",
        "ROC_10m",
        "ROC_30m",
        "Persistence_60m",
        "Risk_Score",
        "Pred_30m",
        "Alert",
        "Status",
    ]

    recent = processed[live_cols].tail(12).copy()
    st.dataframe(
        recent.sort_values(
            "Timestamp",
            ascending=False,
        ),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown(
        '<div class="small-note">'
        "در LIVE mode، Evaluation تا زمانی که داده‌ی آینده وجود نداشته باشد کامل نیست؛ "
        "Y30/Y45/Y60 نیاز به readingهای آینده دارند."
        "</div>",
        unsafe_allow_html=True,
    )


# ------------------------------------------------------------
# DETAILS TAB
# ------------------------------------------------------------
with tab_details:
    columns = [
        "Timestamp",
        "Glucose",
        "Smooth",
        "ROC_10m",
        "ROC_30m",
        "ROC_60m",
        "Persistence_60m",
        "Acceleration",
        "Effective_Slope",
        "Time_to_75m",
        "Risk_Score",
        "Pred_30m",
        "Pred_45m",
        "Pred_60m",
        "Alert",
        "Current_Low",
        "Status",
    ]

    st.dataframe(
        processed[columns].sort_values(
            "Timestamp",
            ascending=False,
        ),
        use_container_width=True,
        hide_index=True,
    )

    export = make_export(
        processed,
        eval_split,
    )

    st.download_button(
        "📥 دانلود Excel پردازش‌شده + Evaluation",
        export,
        file_name="CGM_MVP_v05_results.xlsx",
        mime=(
            "application/vnd.openxmlformats-officedocument."
            "spreadsheetml.sheet"
        ),
    )


# ------------------------------------------------------------
# EVALUATION TAB
# ------------------------------------------------------------
with tab_eval:
    st.subheader("Point-wise Evaluation")

    metric_rows = []
    for h in CFG["HORIZONS"]:
        m = eval_all[f"{h}m_point"]
        metric_rows.append(
            {
                "Horizon": f"{h} min",
                "N": m["N"],
                "TP": m["TP"],
                "FP": m["FP"],
                "FN": m["FN"],
                "TN": m["TN"],
                "Precision": m["Precision"],
                "Recall": m["Recall"],
                "Specificity": m["Specificity"],
                "F1": m["F1"],
            }
        )

    metric_df = pd.DataFrame(metric_rows)

    for col in [
        "Precision",
        "Recall",
        "Specificity",
        "F1",
    ]:
        metric_df[col] = metric_df[col].map(
            lambda x: f"{x * 100:.1f}%"
            if pd.notna(x)
            else "—"
        )

    st.dataframe(
        metric_df,
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Episode-level Evaluation")
    em = eval_all["event_30m"]

    event_rows = [
        {
            "Metric": "Low episodes",
            "Value": em["Low_Episodes"],
        },
        {
            "Metric": "Event recall — any alert <=60m",
            "Value": (
                f"{em['Event_Recall_any'] * 100:.1f}%"
                if pd.notna(em["Event_Recall_any"])
                else "—"
            ),
        },
        {
            "Metric": "Event recall — >=10m lead",
            "Value": (
                f"{em['Event_Recall_>=10m'] * 100:.1f}%"
                if pd.notna(em["Event_Recall_>=10m"])
                else "—"
            ),
        },
        {
            "Metric": "False alert runs",
            "Value": em["False_Alert_Runs"],
        },
        {
            "Metric": "False alerts / day",
            "Value": (
                f"{em['False_Alerts_per_Day']:.2f}"
                if pd.notna(em["False_Alerts_per_Day"])
                else "—"
            ),
        },
        {
            "Metric": "Median lead >=10m",
            "Value": (
                f"{em['Median_Lead_>=10m']:.0f} min"
                if pd.notna(em["Median_Lead_>=10m"])
                else "—"
            ),
        },
    ]

    st.dataframe(
        pd.DataFrame(event_rows),
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Temporal Holdout")

    if eval_split.empty:
        st.info(
            "برای Holdout حداقل چند نقطه زمانی بیشتر لازم است."
        )
    else:
        split_display = eval_split.copy()

        for col in [
            "Precision",
            "Recall",
            "Specificity",
            "F1",
            "Event Recall >=10m",
            "Event Recall any <=60m",
        ]:
            split_display[col] = split_display[col].map(
                lambda x: f"{x * 100:.1f}%"
                if pd.notna(x)
                else "—"
            )

        st.dataframe(
            split_display,
            use_container_width=True,
            hide_index=True,
        )


# ------------------------------------------------------------
# PARAMETERS TAB
# ------------------------------------------------------------
with tab_params:
    st.write(
        "این مقادیر پارامترهای توسعه‌ای MVP هستند و threshold بالینی محسوب نمی‌شوند."
    )

    params = pd.DataFrame(
        [
            ("Low threshold", CFG["LOW"], "mg/dL"),
            ("Risk threshold", CFG["RISK_THRESHOLD"], "0–100"),
            ("Forecast threshold", CFG["FORECAST_THRESHOLD"], "mg/dL"),
            ("Lookback", CFG["LOOKBACK_MIN"], "min"),
            ("Risk horizon", CFG["TIME_RISK_HORIZON_MIN"], "min"),
            ("Time-to target", CFG["TIME_RISK_TARGET"], "mg/dL"),
            ("ROC30 minimum", CFG["MIN_ROC_30"], "mg/dL/min"),
            ("Alert cooldown", CFG["ALERT_COOLDOWN_MIN"], "min"),
            ("EMA span", CFG["EMA_SPAN"], "samples"),
        ],
        columns=["Parameter", "Value", "Unit"],
    )

    st.dataframe(
        params,
        use_container_width=True,
        hide_index=True,
    )
