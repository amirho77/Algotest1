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
# CGM LOW-GLUCOSE EARLY-WARNING MVP v0.4
# ============================================================
# Purpose:
#   Rule-based / mathematical early warning using CGM only.
#   No ML, insulin, meals, activity, age, diabetes type, etc.
#
# Low definition:
#   glucose <= 70 mg/dL
#
# Primary product horizon:
#   30 minutes
#
# Important:
#   This is a research/prototype algorithm, not a clinical device
#   or a medical diagnosis/treatment system.
#
# Development tuning performed on the uploaded 14-day CGM dataset
#   (4,021 readings, 5-minute sampling, 25 low episodes).
# Parameters are intentionally explicit and inspectable.
# ============================================================

CFG = {
    "LOW": 70.0,

    # History / sampling assumptions
    "LOOKBACK_MIN": 60,
    "EXPECTED_SAMPLE_MIN": 5,
    "MAX_ALLOWED_GAP_MIN": 10,
    "EMA_SPAN": 3,

    # Forecast / risk score parameters
    "RISK_THRESHOLD": 68.0,
    "FORECAST_THRESHOLD": 80.0,
    "TIME_RISK_HORIZON_MIN": 45.0,
    "TIME_RISK_TARGET": 75.0,

    # Risk components
    # proximity + time-to-75 + velocity + long trend + persistence + acceleration
    "W_PROX": 0.25,
    "W_TIME": 0.45,
    "W_VEL": 0.10,
    "W_LONG_VEL": 0.05,
    "W_PERSIST": 0.05,
    "W_ACCEL": 0.10,

    # Guard against short, noisy reversals / high-glucose transient drops
    "MIN_ROC_30": -0.20,

    # User-facing alert refractory period
    "ALERT_COOLDOWN_MIN": 20,

    "HORIZONS": [30, 45, 60],
}


st.set_page_config(
    page_title="CGM Low-Glucose Early Warning MVP v0.4",
    page_icon="🩸",
    layout="wide",
)

st.markdown(
    """
    <style>
    .main { direction: rtl; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# INPUT / PARSING
# ============================================================

def parse_glucose(value) -> float:
    s = str(value).strip().upper()

    if s == "LOW":
        return 39.0

    if s == "HIGH":
        return 401.0

    return float(value)


def parse_timestamp(value) -> pd.Timestamp:
    s = re.sub(
        r"\s*GMT.*$",
        "",
        str(value).strip(),
    )

    return pd.to_datetime(
        s,
        dayfirst=True,
        errors="coerce",
    )


def normalize_dataframe(
    df: pd.DataFrame,
) -> pd.DataFrame:

    if df.shape[1] < 2:
        raise ValueError(
            "فایل باید حداقل دو ستون داشته باشد: زمان و قند"
        )

    out = pd.DataFrame(
        {
            "Timestamp": df.iloc[:, 0].map(
                parse_timestamp
            ),
            "Glucose": df.iloc[:, 1].map(
                parse_glucose
            ),
        }
    )

    out = (
        out
        .dropna()
        .sort_values("Timestamp")
        .drop_duplicates(
            "Timestamp",
            keep="last",
        )
        .reset_index(drop=True)
    )

    if out.empty:
        raise ValueError(
            "هیچ رکورد معتبر زمان/قند پیدا نشد"
        )

    return out


def read_sensor_file(
    uploaded_file,
) -> pd.DataFrame:

    name = uploaded_file.name.lower()

    if name.endswith(".csv"):
        raw = pd.read_csv(
            uploaded_file
        )

    else:
        raw = pd.read_excel(
            uploaded_file
        )

    return normalize_dataframe(
        raw
    )


def split_segments(
    df: pd.DataFrame,
) -> list[pd.DataFrame]:

    """
    Split on large timestamp gaps so history/labels
    never jump across missing data.
    """

    out = (
        df
        .sort_values("Timestamp")
        .reset_index(drop=True)
        .copy()
    )

    gaps = (
        out["Timestamp"]
        .diff()
        .dt.total_seconds()
        .div(60)
    )

    segment_id = (
        gaps
        .gt(CFG["MAX_ALLOWED_GAP_MIN"])
        .cumsum()
    )

    out["Segment"] = (
        segment_id.astype(int)
    )

    segments = []

    for _, part in out.groupby(
        "Segment",
        sort=True,
    ):
        segments.append(
            part
            .drop(columns=["Segment"])
            .reset_index(drop=True)
        )

    return segments


def sample_df(
    values: Iterable[float],
    start: str = "2025-01-01 00:00",
) -> pd.DataFrame:

    values = [
        float(v)
        for v in values
    ]

    return pd.DataFrame(
        {
            "Timestamp": pd.date_range(
                start=start,
                periods=len(values),
                freq="5min",
            ),
            "Glucose": values,
        }
    )


# ============================================================
# CORE FEATURE ENGINE
# ============================================================

def calculate_features(
    segment: pd.DataFrame,
) -> pd.DataFrame:

    out = (
        segment
        .copy()
        .reset_index(drop=True)
    )

    g = out["Glucose"].astype(
        float
    )

    smooth = (
        g
        .ewm(
            span=CFG["EMA_SPAN"],
            adjust=False,
        )
        .mean()
    )

    out["Smooth"] = smooth

    # --------------------------------------------------------
    # Trend features
    # --------------------------------------------------------

    out["ROC_10m"] = (
        smooth
        - smooth.shift(2)
    ) / 10.0

    out["ROC_30m"] = (
        smooth
        - smooth.shift(6)
    ) / 30.0

    out["ROC_60m"] = (
        smooth
        - smooth.shift(12)
    ) / 60.0

    out["Raw_Net_10m"] = (
        g
        - g.shift(2)
    )

    # Recent slope minus long slope
    # More negative = stronger downward acceleration.
    out["Acceleration"] = (
        out["ROC_10m"]
        - out["ROC_30m"]
    )

    moves = g.diff()

    out["Persistence_30m"] = (
        moves
        .lt(0)
        .rolling(
            6,
            min_periods=6,
        )
        .mean()
        * 100.0
    )

    out["Persistence_60m"] = (
        moves
        .lt(0)
        .rolling(
            12,
            min_periods=12,
        )
        .mean()
        * 100.0
    )

    # --------------------------------------------------------
    # Adaptive trend slope
    # --------------------------------------------------------
    #
    # Near Low:
    #   recent 10m trend gets more weight.
    #
    # At high glucose:
    #   30m trend gets more weight.
    #
    # This reduces sensitivity to a short high-glucose
    # drop that quickly reverses.
    # --------------------------------------------------------

    trend = np.full(
        len(out),
        np.nan,
        dtype=float,
    )

    for i in range(
        len(out)
    ):

        g_now = g.iloc[i]

        s10 = (
            out["ROC_10m"].iloc[i]
        )

        s30 = (
            out["ROC_30m"].iloc[i]
        )

        if not (
            np.isfinite(s10)
            and np.isfinite(s30)
        ):
            continue

        if g_now <= 90:
            w10 = 0.70

        elif g_now <= 110:
            w10 = 0.55

        elif g_now <= 140:
            w10 = 0.35

        else:
            w10 = 0.20

        trend[i] = (
            w10 * s10
            +
            (1.0 - w10) * s30
        )

    out["Trend_Slope"] = trend

    # --------------------------------------------------------
    # Exact-horizon forecasts
    # --------------------------------------------------------

    for h in CFG["HORIZONS"]:

        out[f"Pred_{h}m"] = np.clip(
            smooth
            +
            trend * h,
            20.0,
            400.0,
        )

    # --------------------------------------------------------
    # Effective slope for Time-to-75
    # --------------------------------------------------------

    effective_slope = (
        0.60 * out["ROC_10m"]
        +
        0.40 * out["ROC_30m"]
    )

    out["Effective_Slope"] = (
        effective_slope
    )

    out["Time_to_75m"] = np.where(
        effective_slope < 0,

        (
            CFG["TIME_RISK_TARGET"]
            - smooth
        )
        / effective_slope,

        np.inf,
    )

    # --------------------------------------------------------
    # Risk components
    # --------------------------------------------------------

    # 1) Proximity
    s_prox = np.clip(
        (
            130.0
            - smooth
        )
        / 60.0
        * 100.0,

        0.0,
        100.0,
    )

    # 2) Time-to-75
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

    # 3) Recent velocity
    s_vel = np.clip(
        -out["ROC_10m"]
        / 1.5
        * 100.0,

        0.0,
        100.0,
    )

    # 4) Long-term velocity
    s_long_vel = np.clip(
        -out["ROC_30m"]
        / 1.0
        * 100.0,

        0.0,
        100.0,
    )

    # 5) Persistence
    s_persist = (
        out["Persistence_60m"]
        .fillna(0.0)
    )

    # 6) Acceleration
    s_acc = np.clip(
        -out["Acceleration"]
        / 0.5
        * 100.0,

        0.0,
        100.0,
    )

    # --------------------------------------------------------
    # Final Risk Score
    # --------------------------------------------------------

    out["Risk_Score"] = (

        CFG["W_PROX"]
        * s_prox

        +

        CFG["W_TIME"]
        * s_time

        +

        CFG["W_VEL"]
        * s_vel

        +

        CFG["W_LONG_VEL"]
        * s_long_vel

        +

        CFG["W_PERSIST"]
        * s_persist

        +

        CFG["W_ACCEL"]
        * s_acc
    )

    # Keep components for debugging.
    out["Risk_Proximity"] = s_prox
    out["Risk_Time"] = s_time
    out["Risk_Velocity"] = s_vel
    out["Risk_Long_Velocity"] = s_long_vel
    out["Risk_Persistence"] = s_persist
    out["Risk_Acceleration"] = s_acc

    return out


# ============================================================
# FUTURE LABELS
# ============================================================

def add_future_labels(
    segment: pd.DataFrame,
) -> pd.DataFrame:

    out = (
        segment
        .copy()
        .reset_index(drop=True)
    )

    times = out[
        "Timestamp"
    ].to_numpy(
        dtype="datetime64[ns]"
    )

    glucose = out[
        "Glucose"
    ].to_numpy(
        dtype=float
    )

    for h in CFG["HORIZONS"]:

        min_future = np.full(
            len(out),
            np.nan,
        )

        y = np.full(
            len(out),
            np.nan,
        )

        horizon = np.timedelta64(
            h,
            "m",
        )

        for i, now in enumerate(
            times
        ):

            end = (
                now
                +
                horizon
            )

            if end > times[-1]:
                continue

            j = np.searchsorted(
                times,
                end,
                side="right",
            )

            future = glucose[
                i + 1 : j
            ]

            if len(future) == 0:
                continue

            min_future[i] = float(
                np.min(future)
            )

            y[i] = float(
                min_future[i]
                <= CFG["LOW"]
            )

        out[
            f"Min_next_{h}m"
        ] = min_future

        out[
            f"Y{h}"
        ] = y

    return out


# ============================================================
# ALERT LOGIC
# ============================================================

def apply_alert_logic(
    segment: pd.DataFrame,
) -> pd.DataFrame:

    out = (
        segment
        .copy()
        .reset_index(drop=True)
    )

    has_history = (
        out["Timestamp"]
        -
        out["Timestamp"].iloc[0]
        >=
        pd.Timedelta(
            minutes=CFG["LOOKBACK_MIN"]
        )
    )

    common = (
        has_history
        &
        (out["Glucose"] > CFG["LOW"])
        &
        (
            out["ROC_30m"]
            <= CFG["MIN_ROC_30"]
        )
        &
        (
            out["Risk_Score"]
            >= CFG["RISK_THRESHOLD"]
        )
    )

    for h in CFG["HORIZONS"]:

        out[
            f"Alert_{h}m"
        ] = (
            common
            &
            (
                out[
                    f"Pred_{h}m"
                ]
                <=
                CFG["FORECAST_THRESHOLD"]
            )
        )

    # --------------------------------------------------------
    # Primary user-facing alert = 30m
    # --------------------------------------------------------

    raw_alert = (
        out["Alert_30m"]
        .fillna(False)
        .to_numpy(bool)
    )

    # --------------------------------------------------------
    # Cooldown / refractory period
    # --------------------------------------------------------

    cooldown_steps = max(
        1,
        int(
            CFG["ALERT_COOLDOWN_MIN"]
            /
            CFG["EXPECTED_SAMPLE_MIN"]
        ),
    )

    alert = raw_alert.copy()

    last_alert_idx = -10**9

    for i in np.where(
        raw_alert
    )[0]:

        if (
            i - last_alert_idx
            <= cooldown_steps
        ):

            alert[i] = False

        else:

            last_alert_idx = i

    out["Alert"] = alert

    out["Current_Low"] = (
        out["Glucose"]
        <= CFG["LOW"]
    )

    out["Any_Alert"] = (
        out["Alert"]
    )

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
        default="⏳ بافر ۶۰ دقیقه‌ای",
    )

    return out


# ============================================================
# COMPLETE PIPELINE
# ============================================================

def process_one_segment(
    segment: pd.DataFrame,
) -> pd.DataFrame:

    out = calculate_features(
        segment
    )

    out = add_future_labels(
        out
    )

    out = apply_alert_logic(
        out
    )

    return out


def process_dataset(
    df: pd.DataFrame,
) -> pd.DataFrame:

    parts = []

    for segment in split_segments(
        df
    ):

        if not segment.empty:

            parts.append(
                process_one_segment(
                    segment
                )
            )

    if not parts:
        return pd.DataFrame()

    return (
        pd.concat(
            parts,
            ignore_index=True,
        )
        .sort_values(
            "Timestamp"
        )
        .reset_index(drop=True)
    )


# ============================================================
# EVALUATION
# ============================================================

def low_episode_starts(
    df: pd.DataFrame,
) -> list[int]:

    low = (
        df["Glucose"]
        .to_numpy(float)
        <= CFG["LOW"]
    )

    starts = np.where(
        low
        &
        ~np.r_[
            False,
            low[:-1]
        ]
    )[0]

    return starts.tolist()


def alert_runs(
    alert: np.ndarray,
) -> list[tuple[int, int]]:

    runs = []

    i = 0

    while i < len(alert):

        if alert[i]:

            start = i

            while (
                i + 1 < len(alert)
                and alert[i + 1]
            ):
                i += 1

            runs.append(
                (
                    start,
                    i,
                )
            )

        i += 1

    return runs


def classification_metrics(
    df: pd.DataFrame,
    alert_column: str,
    horizon: int,
    row_mask: np.ndarray | None = None,
) -> dict:

    y = (
        df[f"Y{horizon}"]
        .to_numpy(float)
    )

    pred = (
        df[alert_column]
        .fillna(False)
        .to_numpy(bool)
    )

    glucose = (
        df["Glucose"]
        .to_numpy(float)
    )

    mask = (
        ~np.isnan(y)
        &
        (glucose > CFG["LOW"])
    )

    if row_mask is not None:
        mask &= row_mask

    yy = y[mask].astype(int)
    pp = pred[mask].astype(int)

    tp = int(
        (
            (yy == 1)
            &
            (pp == 1)
        ).sum()
    )

    fp = int(
        (
            (yy == 0)
            &
            (pp == 1)
        ).sum()
    )

    fn = int(
        (
            (yy == 1)
            &
            (pp == 0)
        ).sum()
    )

    tn = int(
        (
            (yy == 0)
            &
            (pp == 0)
        ).sum()
    )

    precision = (
        tp / (tp + fp)
        if tp + fp
        else np.nan
    )

    recall = (
        tp / (tp + fn)
        if tp + fn
        else np.nan
    )

    specificity = (
        tn / (tn + fp)
        if tn + fp
        else np.nan
    )

    f1 = (
        2
        * precision
        * recall
        / (
            precision
            +
            recall
        )
        if (
            np.isfinite(
                precision
            )
            and
            np.isfinite(
                recall
            )
            and
            precision
            +
            recall
        )
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

    alerts = (
        df[alert_column]
        .fillna(False)
        .to_numpy(bool)
    )

    starts = low_episode_starts(
        df
    )

    runs = alert_runs(
        alerts
    )

    detected_any = 0
    detected_early = 0

    leads_any = []
    leads_early = []

    matched_alert_runs = set()

    for low_start in starts:

        candidates = []

        for j, (
            run_start,
            _,
        ) in enumerate(runs):

            lead = (
                low_start
                -
                run_start
            ) * CFG[
                "EXPECTED_SAMPLE_MIN"
            ]

            if (
                0 < lead
                <= max_lead_min
            ):

                candidates.append(
                    (
                        j,
                        lead,
                    )
                )

        if not candidates:
            continue

        # Latest alert before the Low.
        j, lead = candidates[-1]

        matched_alert_runs.add(j)

        detected_any += 1
        leads_any.append(lead)

        if lead >= min_lead_min:

            detected_early += 1
            leads_early.append(
                lead
            )

    false_alert_runs = max(
        0,
        len(runs)
        -
        len(matched_alert_runs),
    )

    episode_count = len(
        starts
    )

    any_recall = (
        detected_any
        /
        episode_count
        if episode_count
        else np.nan
    )

    early_recall = (
        detected_early
        /
        episode_count
        if episode_count
        else np.nan
    )

    duration_days = (
        df["Timestamp"].iloc[-1]
        -
        df["Timestamp"].iloc[0]
    ).total_seconds() / 86400

    false_alerts_per_day = (
        false_alert_runs
        /
        duration_days
        if duration_days > 0
        else np.nan
    )

    return {
        "Low_Episodes": episode_count,

        "Detected_any_<=60m":
            detected_any,

        "Event_Recall_any":
            any_recall,

        "Detected_with_>=10m_lead":
            detected_early,

        "Event_Recall_>=10m":
            early_recall,

        "False_Alert_Runs":
            false_alert_runs,

        "False_Alerts_per_Day":
            false_alerts_per_day,

        "Median_Lead_any":
            (
                float(
                    np.median(
                        leads_any
                    )
                )
                if leads_any
                else np.nan
            ),

        "Median_Lead_>=10m":
            (
                float(
                    np.median(
                        leads_early
                    )
                )
                if leads_early
                else np.nan
            ),

        "Lead_Times_any":
            leads_any,

        "Lead_Times_early":
            leads_early,
    }


def evaluate_dataset(
    processed: pd.DataFrame,
) -> dict:

    metrics = {}

    for h in CFG["HORIZONS"]:

        metrics[
            f"{h}m_point"
        ] = classification_metrics(
            processed,
            f"Alert_{h}m",
            h,
        )

    metrics[
        "event_30m"
    ] = event_metrics(
        processed,
        alert_column="Alert",
    )

    return metrics


# ============================================================
# TEMPORAL HOLDOUT
# ============================================================

def temporal_split_evaluation(
    processed: pd.DataFrame,
) -> pd.DataFrame:

    """
    Descriptive time split:
        60% development
        20% validation
        20% holdout

    No retuning occurs here.
    The final fixed parameters are evaluated as-is.
    """

    n = len(processed)

    cut1 = int(
        n * 0.60
    )

    cut2 = int(
        n * 0.80
    )

    rows = []

    for (
        name,
        start,
        end,
    ) in [

        (
            "Train/Development",
            0,
            cut1,
        ),

        (
            "Validation",
            cut1,
            cut2,
        ),

        (
            "Holdout/Test",
            cut2,
            n,
        ),

        (
            "All",
            0,
            n,
        ),
    ]:

        part = (
            processed
            .iloc[start:end]
            .copy()
        )

        cm = classification_metrics(
            processed.iloc[
                start:end
            ],
            "Alert",
            30,
        )

        em = (
            event_metrics(
                part,
                "Alert",
            )
            if len(part)
            else {}
        )

        rows.append(
            {
                "Split": name,

                **cm,

                "Low Episodes":
                    em.get(
                        "Low_Episodes",
                        np.nan,
                    ),

                "Event Recall >=10m":
                    em.get(
                        "Event_Recall_>=10m",
                        np.nan,
                    ),

                "Event Recall any <=60m":
                    em.get(
                        "Event_Recall_any",
                        np.nan,
                    ),

                "False Alert Runs":
                    em.get(
                        "False_Alert_Runs",
                        np.nan,
                    ),

                "Median Lead >=10m":
                    em.get(
                        "Median_Lead_>=10m",
                        np.nan,
                    ),
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# EXPORT
# ============================================================

def make_export(
    processed: pd.DataFrame,
    eval_df: pd.DataFrame,
) -> bytes:

    buffer = io.BytesIO()

    with pd.ExcelWriter(
        buffer,
        engine="openpyxl",
    ) as writer:

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

        low_rows = []

        starts = low_episode_starts(
            processed
        )

        for i, start in enumerate(
            starts,
            start=1,
        ):

            next_starts = [
                s
                for s in starts
                if s > start
            ]

            end = (
                next_starts[0] - 1
                if next_starts
                else len(processed) - 1
            )

            low_rows.append(
                {
                    "Episode":
                        i,

                    "Start":
                        processed.loc[
                            start,
                            "Timestamp",
                        ],

                    "Start_Glucose":
                        processed.loc[
                            start,
                            "Glucose",
                        ],

                    "End":
                        processed.loc[
                            end,
                            "Timestamp",
                        ],

                    "Minimum":
                        processed.loc[
                            start:end,
                            "Glucose",
                        ].min(),
                }
            )

        pd.DataFrame(
            low_rows
        ).to_excel(
            writer,
            index=False,
            sheet_name="Low_Episodes",
        )

    return buffer.getvalue()


# ============================================================
# BUILT-IN DEVELOPMENT TESTS
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
# APP
# ============================================================

if "data" not in st.session_state:

    st.session_state.data = pd.DataFrame(
        columns=[
            "Timestamp",
            "Glucose",
        ]
    )


st.title(
    "🩸 CGM Low-Glucose Early Warning — MVP v0.4"
)

st.caption(
    "مدل ریاضی CGM-only؛ "
    "آستانه Low = 70 mg/dL. "
    "این پروتوتایپ ابزار تشخیص یا درمان پزشکی نیست."
)


with st.sidebar:

    st.header(
        "ورودی داده"
    )

    mode = st.radio(
        "حالت",
        [
            "Excel / CSV",
            "Live 5-minute",
            "Development tests",
        ],
    )

    # --------------------------------------------------------
    # EXCEL / CSV
    # --------------------------------------------------------

    if mode == "Excel / CSV":

        uploaded = st.file_uploader(
            "فایل CGM",
            type=[
                "xls",
                "xlsx",
                "csv",
            ],
        )

        if uploaded is not None:

            try:

                st.session_state.data = (
                    read_sensor_file(
                        uploaded
                    )
                )

                st.success(
                    f"{len(st.session_state.data):,} رکورد بارگذاری شد."
                )

            except Exception as exc:

                st.error(
                    str(exc)
                )

    # --------------------------------------------------------
    # DEVELOPMENT TESTS
    # --------------------------------------------------------

    elif mode == "Development tests":

        if st.button(
            "تست April #1",
            use_container_width=True,
        ):

            st.session_state.data = (
                sample_df(
                    APRIL_DATASET_1,
                    "2025-04-20 00:00",
                )
            )

            st.rerun()

        if st.button(
            "تست April #2",
            use_container_width=True,
        ):

            st.session_state.data = (
                sample_df(
                    APRIL_DATASET_2,
                    "2025-04-20 00:00",
                )
            )

            st.rerun()

    # --------------------------------------------------------
    # LIVE
    # --------------------------------------------------------

    else:

        if "live_base" not in st.session_state:

            st.session_state.live_base = (
                datetime.now()
                .replace(
                    second=0,
                    microsecond=0,
                )
            )

        glucose = st.number_input(
            "قند (mg/dL)",
            min_value=20.0,
            max_value=450.0,
            value=115.0,
            step=1.0,
        )

        if st.button(
            "➕ ثبت ۵ دقیقه",
            use_container_width=True,
        ):

            if st.session_state.data.empty:

                t = (
                    st.session_state.live_base
                )

            else:

                t = (
                    st.session_state.data.iloc[-1][
                        "Timestamp"
                    ]
                    +
                    timedelta(
                        minutes=5
                    )
                )

            st.session_state.data = (
                pd.concat(
                    [
                        st.session_state.data,
                        pd.DataFrame(
                            {
                                "Timestamp": [t],
                                "Glucose": [glucose],
                            }
                        ),
                    ],
                    ignore_index=True,
                )
            )

            st.rerun()

        if st.button(
            "🗑 پاک‌سازی",
            use_container_width=True,
        ):

            st.session_state.data = (
                pd.DataFrame(
                    columns=[
                        "Timestamp",
                        "Glucose",
                    ]
                )
            )

            st.rerun()


if st.session_state.data.empty:

    st.info(
        "از Sidebar داده را وارد کن."
    )

    st.stop()


processed = process_dataset(
    st.session_state.data
)

eval_all = evaluate_dataset(
    processed
)

eval_split = (
    temporal_split_evaluation(
        processed
    )
)

last = processed.iloc[-1]


# ============================================================
# TOP METRICS
# ============================================================

c1, c2, c3, c4, c5 = st.columns(5)


c1.metric(
    "آخرین قند",
    f"{last['Glucose']:.0f}",
)


c2.metric(
    "EMA",
    f"{last['Smooth']:.1f}",
)


c3.metric(
    "ROC 10m",
    (
        "-"
        if pd.isna(
            last["ROC_10m"]
        )
        else
        f"{last['ROC_10m']:.2f}"
    ),
)


c4.metric(
    "Risk Score",
    (
        "-"
        if pd.isna(
            last["Risk_Score"]
        )
        else
        f"{last['Risk_Score']:.1f}/100"
    ),
)


c5.metric(
    "وضعیت",
    last["Status"],
)


# ============================================================
# CHART
# ============================================================

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


if pd.notna(
    last["Pred_30m"]
):

    future_x = [
        last["Timestamp"]
        +
        timedelta(
            minutes=h
        )
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
)


st.plotly_chart(
    fig,
    use_container_width=True,
)


# ============================================================
# FORECAST CARDS
# ============================================================

fc = st.columns(3)


for col, h in zip(
    fc,
    CFG["HORIZONS"],
):

    with col:

        pred = last[
            f"Pred_{h}m"
        ]

        alert = bool(
            last[
                f"Alert_{h}m"
            ]
        )

        st.metric(
            f"پیش‌بینی +{h} دقیقه",
            (
                "-"
                if pd.isna(pred)
                else
                f"{pred:.1f} mg/dL"
            ),
            (
                "⚠️ Alert"
                if alert
                else
                "No alert"
            ),
        )


# ============================================================
# TABS
# ============================================================

tab_details, tab_eval, tab_params = (
    st.tabs(
        [
            "جزئیات",
            "ارزیابی",
            "پارامترهای MVP",
        ]
    )
)


# ============================================================
# DETAILS
# ============================================================

with tab_details:

    columns = [

        "Timestamp",
        "Glucose",
        "Smooth",

        "ROC_10m",
        "ROC_30m",

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
        processed[
            columns
        ].sort_values(
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
        file_name=(
            "CGM_MVP_v04_results.xlsx"
        ),
        mime=(
            "application/vnd.openxmlformats-officedocument."
            "spreadsheetml.sheet"
        ),
    )


# ============================================================
# EVALUATION
# ============================================================

with tab_eval:

    st.subheader(
        "Point-wise Evaluation"
    )


    metric_rows = []

    for h in CFG["HORIZONS"]:

        m = eval_all[
            f"{h}m_point"
        ]

        metric_rows.append(
            {
                "Horizon":
                    f"{h} min",

                "N":
                    m["N"],

                "TP":
                    m["TP"],

                "FP":
                    m["FP"],

                "FN":
                    m["FN"],

                "TN":
                    m["TN"],

                "Precision":
                    m["Precision"],

                "Recall":
                    m["Recall"],

                "Specificity":
                    m["Specificity"],

                "F1":
                    m["F1"],
            }
        )


    metric_df = pd.DataFrame(
        metric_rows
    )


    for col in [
        "Precision",
        "Recall",
        "Specificity",
        "F1",
    ]:

        metric_df[col] = (
            metric_df[col]
            .map(
                lambda x:
                    f"{x * 100:.1f}%"
                    if pd.notna(x)
                    else "-"
            )
        )


    st.dataframe(
        metric_df,
        use_container_width=True,
        hide_index=True,
    )


    st.subheader(
        "Episode-level Evaluation"
    )


    em = eval_all[
        "event_30m"
    ]


    event_rows = [

        {
            "Metric":
                "Low episodes",

            "Value":
                em["Low_Episodes"],
        },

        {
            "Metric":
                "Event recall — any alert <=60m",

            "Value":
                (
                    f"{em['Event_Recall_any'] * 100:.1f}%"
                    if pd.notna(
                        em[
                            "Event_Recall_any"
                        ]
                    )
                    else "-"
                ),
        },

        {
            "Metric":
                "Event recall — >=10m lead",

            "Value":
                (
                    f"{em['Event_Recall_>=10m'] * 100:.1f}%"
                    if pd.notna(
                        em[
                            "Event_Recall_>=10m"
                        ]
                    )
                    else "-"
                ),
        },

        {
            "Metric":
                "False alert runs",

            "Value":
                em[
                    "False_Alert_Runs"
                ],
        },

        {
            "Metric":
                "False alerts / day",

            "Value":
                (
                    f"{em['False_Alerts_per_Day']:.2f}"
                    if pd.notna(
                        em[
                            "False_Alerts_per_Day"
                        ]
                    )
                    else "-"
                ),
        },

        {
            "Metric":
                "Median lead >=10m",

            "Value":
                (
                    f"{em['Median_Lead_>=10m']:.0f} min"
                    if pd.notna(
                        em[
                            "Median_Lead_>=10m"
                        ]
                    )
                    else "-"
                ),
        },
    ]


    st.dataframe(
        pd.DataFrame(
            event_rows
        ),
        use_container_width=True,
        hide_index=True,
    )


    st.subheader(
        "Temporal Holdout"
    )


    split_display = (
        eval_split.copy()
    )


    for col in [
        "Precision",
        "Recall",
        "Specificity",
        "F1",
        "Event Recall >=10m",
        "Event Recall any <=60m",
    ]:

        split_display[col] = (
            split_display[col]
            .map(
                lambda x:
                    f"{x * 100:.1f}%"
                    if pd.notna(x)
                    else "-"
            )
        )


    st.dataframe(
        split_display,
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# PARAMETER TAB
# ============================================================

with tab_params:

    st.write(
        "پارامترهای زیر نسخه‌ی توسعه‌ای MVP هستند؛ "
        "این اعداد threshold بالینی نیستند."
    )


    param_rows = [

        (
            "Low threshold",
            CFG["LOW"],
            "mg/dL",
        ),

        (
            "Risk threshold",
            CFG["RISK_THRESHOLD"],
            "0–100",
        ),

        (
            "Forecast threshold",
            CFG["FORECAST_THRESHOLD"],
            "mg/dL",
        ),

        (
            "Lookback",
            CFG["LOOKBACK_MIN"],
            "min",
        ),

        (
            "Risk horizon",
            CFG["TIME_RISK_HORIZON_MIN"],
            "min",
        ),

        (
            "Time-to target",
            CFG["TIME_RISK_TARGET"],
            "mg/dL",
        ),

        (
            "ROC30 minimum",
            CFG["MIN_ROC_30"],
            "mg/dL/min",
        ),

        (
            "Alert cooldown",
            CFG["ALERT_COOLDOWN_MIN"],
            "min",
        ),

        (
            "EMA span",
            CFG["EMA_SPAN"],
            "samples",
        ),
    ]


    st.dataframe(
        pd.DataFrame(
            param_rows,
            columns=[
                "Parameter",
                "Value",
                "Unit",
            ],
        ),
        use_container_width=True,
        hide_index=True,
    )
