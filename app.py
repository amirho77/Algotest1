from datetime import datetime, timedelta
import io
import re
import numpy as np
import openpyxl
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ==============================================================================
# ۱. تنظیمات صفحه و استایل
# ==============================================================================
st.set_page_config(
    page_title="CGM Hypo Predictor Pro",
    page_icon="🩸",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Vazirmatn:wght@300;400;500;700;900&display=swap');
    * { font-family: 'Vazirmatn', Tahoma, sans-serif !important; }
    .main { direction: rtl; }
    .metric-card {
        background: #ffffff; border: 1px solid #e2e8f0; border-radius: 12px;
        padding: 16px; text-align: center; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
    }
    .status-badge-safe { background-color: #d1fae5; color: #065f46; padding: 6px 12px; border-radius: 20px; font-weight: 700; }
    .status-badge-warn { background-color: #fee2e2; color: #991b1b; padding: 6px 12px; border-radius: 20px; font-weight: 700; animation: pulse 2s infinite; }
    .status-badge-low { background-color: #7f1d1d; color: #ffffff; padding: 6px 12px; border-radius: 20px; font-weight: 700; }
    .status-badge-buff { background-color: #fef3c7; color: #92400e; padding: 6px 12px; border-radius: 20px; font-weight: 700; }
    @keyframes pulse { 0% { transform: scale(1); } 50% { transform: scale(1.02); } 100% { transform: scale(1); } }
    </style>
""",
    unsafe_allow_html=True,
)

# ==============================================================================
# ۲. پیکربندی اصول فیزیولوژیک و کینماتیک (بدون شروط هاردکد)
# ==============================================================================
CFG = {
    "hypo_threshold": 70.0,
    "alert_margin": 5.0,  # حاشیه پیش‌بینی (75 mg/dL)
    "risk_score_threshold": 75.0,
    "tau_minutes": 45.0,  # زمان مشخصه میرایی فیزیولوژیک
    "horizons": [30, 45, 60],
    "max_phys_drop_rate": -2.5,  # حداکثر شیب فیزیولوژیک پایدار (مهار نویز و فشار)
    "w_dist": 0.35,
    "w_vel": 0.35,
    "w_acc": 0.15,
    "w_persist": 0.15,
}


def calculate_metrics_for_stream(records):
    """محاسبه کینماتیک ترمز، شیب فیزیولوژیک و فیلتر تایید پیوستگی"""
    if not records:
        return []

    df = pd.DataFrame(records)
    # ۱. فیلتر علّی هموارسازی (EMA span=3)
    df["Smooth"] = df["Raw"].ewm(span=3, adjust=False).mean()

    n = len(df)
    processed = []

    for i in range(n):
        item = records[i].copy()
        item["Smooth"] = round(float(df.loc[i, "Smooth"]), 1)

        # دوره پر شدن بافر پایه (۱۲ خوانش = ۱ ساعت)
        if i < 11:
            item["ROC_15m"] = "-"
            item["Risk_Score"] = "-"
            for h in CFG["horizons"]:
                item[f"Pred_{h}m"] = "-"
                item[f"Alert_{h}m"] = False
            item["Status"] = f"بافر ({i+1}/12)"
            processed.append(item)
            continue

        current_smooth = df.loc[i, "Smooth"]
        current_raw = df.loc[i, "Raw"]

        # ۲. محاسبه نرخ تغییرات (ROC) در ۱۵ دقیقه اخیر
        raw_roc = (current_smooth - df.loc[i - 3, "Smooth"]) / 15.0
        item["ROC_15m"] = round(float(raw_roc), 2)

        # ۳. کران‌دار کردن فیزیولوژیک نرخ سقوط (مهار نویز خوابیدن روی سنسور)
        roc_phys = np.clip(raw_roc, CFG["max_phys_drop_rate"], 3.0)

        # ۴. محاسبه شتاب (مشتق دوم - ترمز یا شتاب‌گیری)
        if i >= 5:
            roc_prev = (df.loc[i - 2, "Smooth"] - df.loc[i - 5, "Smooth"]) / 15.0
            acc = (raw_roc - roc_prev) / 10.0
        else:
            acc = 0.0

        # ۵. تداوم نزول در ۱۲ نقطه اخیر
        smooth_slice = df.loc[max(0, i - 11) : i, "Smooth"]
        persistence = float((smooth_slice.diff() < 0).mean() * 100.0)

        # ۶. شاخص ریسک تجمیعی
        s_dist = np.clip((180.0 - current_smooth) / 110.0 * 100.0, 0.0, 100.0)
        s_vel = np.clip(-raw_roc / 2.0 * 100.0, 0.0, 100.0)
        s_acc = np.clip(-acc / 0.1 * 100.0, 0.0, 100.0)
        risk_score = (
            CFG["w_dist"] * s_dist
            + CFG["w_vel"] * s_vel
            + CFG["w_acc"] * s_acc
            + CFG["w_persist"] * persistence
        )
        item["Risk_Score"] = round(float(risk_score), 1)

        # ۷. پیش‌بینی کینماتیک با احتساب اثر ترمز (Deceleration Damping)
        has_alert = False

        for h in CFG["horizons"]:
            tau = CFG["tau_minutes"]
            h_eff = tau * (1.0 - np.exp(-h / tau))

            # اگر قند رو به پایین است ولی شتاب مثبت است (یعنی سرعت افت در حال کند شدن است)
            # شیب مؤثر با ترمز تعدیل می‌شود تا سقوط کاذب پیش‌بینی نشود
            if roc_phys < 0 and acc > 0:
                braking_adjustment = np.clip(
                    0.5 * acc * h_eff, 0.0, -0.7 * roc_phys
                )
                v_eff = roc_phys + braking_adjustment
            else:
                v_eff = roc_phys

            pred_g = float(
                np.clip(current_smooth + v_eff * h_eff, 20.0, 400.0)
            )
            raw_alert = (
                pred_g <= (CFG["hypo_threshold"] + CFG["alert_margin"])
            ) or (risk_score >= CFG["risk_score_threshold"])

            # ۸. فیلتر تایید پیوستگی (Confirmation / Anti-Chattering)
            # هشدار با ۱ نقطه نویز شلیک نمی‌شود؛ تداوم در ۲ نقطه یا افت شدید الزامی است
            if i >= 1:
                prev_raw_alert = processed[i - 1].get(f"_raw_alert_{h}", False)
            else:
                prev_raw_alert = False

            severe_drop = (raw_roc < -1.5) and (
                pred_g <= (CFG["hypo_threshold"] + CFG["alert_margin"])
            )
            confirmed_alert = bool(
                (raw_alert and prev_raw_alert) or severe_drop
            )

            item[f"_raw_alert_{h}"] = raw_alert
            item[f"Pred_{h}m"] = round(pred_g, 1)
            item[f"Alert_{h}m"] = confirmed_alert

            if confirmed_alert:
                has_alert = True

        if current_raw <= CFG["hypo_threshold"]:
            item["Status"] = "🚨 افت لحظه‌ای"
        elif has_alert:
            item["Status"] = "⚠️ هشدار افت"
        else:
            item["Status"] = "✅ ایمن"

        processed.append(item)

    return processed


# ==============================================================================
# ۳. مدیریت State
# ==============================================================================
if "records" not in st.session_state:
    st.session_state.records = []

if "start_time" not in st.session_state:
    st.session_state.start_time = datetime.now().replace(
        second=0, microsecond=0
    )

# ==============================================================================
# ۴. نوار کناری (Sidebar)
# ==============================================================================
with st.sidebar:
    st.title("🩸 کنترل‌پنل CGM")
    st.caption("سیستم پیش‌بین تعمیم‌پذیر با فیلتر کینماتیک")
    st.markdown("---")

    mode = st.radio(
        "حالت ورود داده:",
        ["ورود زنده (شبیه‌ساز ۵ دقیقه‌ای)", "آپلود فایل اکسل/CSV سنسور"],
    )

    if mode == "ورود زنده (شبیه‌ساز ۵ دقیقه‌ای)":
        new_glucose = st.number_input(
            "عدد قند خون (mg/dL):",
            min_value=20.0,
            max_value=450.0,
            value=115.0,
            step=1.0,
        )
        c1, c2 = st.columns(2)
        with c1:
            if st.button("➕ ثبت ۵ دقیقه", use_container_width=True):
                cur_step = len(st.session_state.records)
                sim_t = st.session_state.start_time + timedelta(
                    minutes=cur_step * 5
                )
                st.session_state.records.append(
                    {
                        "Step": cur_step + 1,
                        "Time": sim_t.strftime("%H:%M"),
                        "Raw": float(new_glucose),
                    }
                )
                st.rerun()
        with c2:
            if st.button("🗑️ پاکسازی", use_container_width=True):
                st.session_state.records = []
                st.rerun()

    else:
        uploaded_file = st.file_uploader(
            "فایل اکسل سنسور:", type=["xls", "xlsx", "csv"]
        )
        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith(".csv"):
                    df_up = pd.read_csv(uploaded_file)
                else:
                    df_up = pd.read_excel(uploaded_file)

                time_col, gluc_col = df_up.columns[0], df_up.columns[1]

                def parse_g(v):
                    s = str(v).strip().upper()
                    return (
                        39.0
                        if s == "LOW"
                        else (401.0 if s == "HIGH" else float(v))
                    )

                df_up["Raw"] = df_up[gluc_col].apply(parse_g)
                c_times = (
                    df_up[time_col]
                    .astype(str)
                    .apply(lambda x: re.sub(r"\s*GMT.*", "", x).strip())
                )
                dt_series = pd.to_datetime(
                    c_times, format="%d-%m-%Y %H:%M", errors="coerce"
                )
                if dt_series.isna().all():
                    dt_series = pd.to_datetime(
                        c_times, dayfirst=True, errors="coerce"
                    )

                df_up["Time_dt"] = dt_series
                df_up = df_up.dropna(subset=["Time_dt", "Raw"]).sort_values(
                    "Time_dt"
                )

                st.session_state.records = []
                for i, row in enumerate(df_up.itertuples()):
                    st.session_state.records.append(
                        {
                            "Step": i + 1,
                            "Time": row.Time_dt.strftime("%d-%m %H:%M"),
                            "Raw": float(row.Raw),
                        }
                    )
                st.success(f"{len(df_up)} رکورد با موفقیت بارگذاری شد.")
            except Exception as e:
                st.error(f"خطا در خواندن فایل: {e}")

# ==============================================================================
# ۵. داشبورد و مانیتورینگ
# ==============================================================================
st.title("🩸 مانیتورینگ پیش‌بینی افت قند خون (نسخه تعمیم‌پذیر)")

processed_data = calculate_metrics_for_stream(st.session_state.records)

if not processed_data:
    st.info("👈 داده‌ای ثبت نشده است. عددی وارد کنید یا فایل اکسل را آپلود کنید.")
else:
    last = processed_data[-1]
    count = len(processed_data)

    c1, c2, c3, c4 = st.columns([1, 1, 1, 1.2])
    with c1:
        st.markdown(
            f"""<div class="metric-card">
            <div style="color: #64748b; font-size: 13px;">زمان ثبت</div>
            <div style="font-size: 26px; font-weight: 700; color: #1e293b;">{last['Time']}</div>
            <div style="color: #0284c7; font-size: 12px;">گام #{count}</div>
        </div>""",
            unsafe_allow_html=True,
        )

    with c2:
        smooth_val = last["Smooth"] if last["Smooth"] != "-" else last["Raw"]
        st.markdown(
            f"""<div class="metric-card">
            <div style="color: #64748b; font-size: 13px;">قند خام (هموار)</div>
            <div style="font-size: 26px; font-weight: 700; color: #1e293b;">{int(last['Raw'])} <span style="font-size: 15px; color: #64748b;">({smooth_val})</span></div>
            <div style="color: #64748b; font-size: 12px;">mg/dL</div>
        </div>""",
            unsafe_allow_html=True,
        )

    with c3:
        roc_disp = (
            f"{last['ROC_15m']} mg/dL/min" if last["ROC_15m"] != "-" else "-"
        )
        risk_disp = (
            f"{last['Risk_Score']} / 100" if last["Risk_Score"] != "-" else "-"
        )
        st.markdown(
            f"""<div class="metric-card">
            <div style="color: #64748b; font-size: 13px;">نرخ افت و ریسک</div>
            <div style="font-size: 22px; font-weight: 700; color: #1e293b;">{roc_disp}</div>
            <div style="color: #ea580c; font-size: 12px;">نمره ریسک: {risk_disp}</div>
        </div>""",
            unsafe_allow_html=True,
        )

    with c4:
        st_text = last["Status"]
        badge_cls = (
            "status-badge-safe"
            if "ایمن" in st_text
            else (
                "status-badge-warn"
                if "هشدار" in st_text
                else (
                    "status-badge-low"
                    if "افت" in st_text
                    else "status-badge-buff"
                )
            )
        )
        st.markdown(
            f"""<div class="metric-card">
            <div style="color: #64748b; font-size: 13px; margin-bottom: 6px;">وضعیت هشدار زودهنگام</div>
            <span class="{badge_cls}">{st_text}</span>
        </div>""",
            unsafe_allow_html=True,
        )

    tab_chart, tab_table = st.tabs(["📈 نمودار ترند", "📋 جدول داده‌ها"])

    with tab_chart:
        df_p = pd.DataFrame(processed_data)
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=df_p["Time"],
                y=df_p["Raw"],
                mode="lines+markers",
                name="قند خام",
                line=dict(color="#3b82f6", width=2),
                marker=dict(size=4),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=df_p["Time"],
                y=df_p["Smooth"],
                mode="lines",
                name="هموارشده",
                line=dict(color="#10b981", width=2, dash="dot"),
            )
        )
        fig.add_hline(
            y=70,
            line_dash="dash",
            line_color="#ef4444",
            annotation_text="مرز هیپو (70)",
        )
        fig.update_layout(
            height=400,
            margin=dict(l=20, r=20, t=20, b=20),
            hovermode="x unified",
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab_table:
        display_rows = []
        for r in reversed(processed_data):
            display_rows.append(
                {
                    "گام": f"#{r['Step']}",
                    "زمان": r["Time"],
                    "قند خام": r["Raw"],
                    "هموار": r["Smooth"],
                    "نرخ افت": r["ROC_15m"],
                    "نمره ریسک": r["Risk_Score"],
                    "پیش‌بینی ۳۰ دقیقه": f"⚠️ {r['Pred_30m']}"
                    if r.get("Alert_30m")
                    else r["Pred_30m"],
                    "پیش‌بینی ۴۵ دقیقه": f"⚠️ {r['Pred_45m']}"
                    if r.get("Alert_45m")
                    else r["Pred_45m"],
                    "پیش‌بینی ۶۰ دقیقه": f"⚠️ {r['Pred_60m']}"
                    if r.get("Alert_60m")
                    else r["Pred_60m"],
                    "وضعیت": r["Status"],
                }
            )
        st.dataframe(
            pd.DataFrame(display_rows), use_container_width=True, hide_index=True
        )

        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
            pd.DataFrame(processed_data).to_excel(
                writer, index=False, sheet_name="CGM_Predictions"
            )

        st.download_button(
            label="📥 دانلود فایل اکسل تحلیل کامل",
            data=buffer.getvalue(),
            file_name="CGM_Predictions_Validated.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
