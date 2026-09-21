from datetime import datetime, timedelta
import io
import re
import numpy as np
import openpyxl
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ==============================================================================
# ۱. تنظیمات صفحه و تم گرافیکی
# ==============================================================================
st.set_page_config(
    page_title="CGM Hypo Predictor MVP",
    page_icon="🩸",
    layout="wide",
    initial_sidebar_state="expanded",
)

# استایل‌های CSS سفارشی (RTL و فونت یکپارچه)
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Vazirmatn:wght@300;400;500;700;900&display=swap');
    
    * {
        font-family: 'Vazirmatn', Tahoma, sans-serif !important;
    }
    
    .main {
        direction: rtl;
    }
    
    .metric-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 16px;
        text-align: center;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
    }
    
    .status-badge-safe {
        background-color: #d1fae5;
        color: #065f46;
        padding: 6px 12px;
        border-radius: 20px;
        font-weight: 700;
        display: inline-block;
    }
    
    .status-badge-warn {
        background-color: #fee2e2;
        color: #991b1b;
        padding: 6px 12px;
        border-radius: 20px;
        font-weight: 700;
        display: inline-block;
        animation: pulse 2s infinite;
    }
    
    .status-badge-low {
        background-color: #7f1d1d;
        color: #ffffff;
        padding: 6px 12px;
        border-radius: 20px;
        font-weight: 700;
        display: inline-block;
    }
    
    .status-badge-buff {
        background-color: #fef3c7;
        color: #92400e;
        padding: 6px 12px;
        border-radius: 20px;
        font-weight: 700;
        display: inline-block;
    }
    
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.03); }
        100% { transform: scale(1); }
    }
    </style>
""",
    unsafe_allow_html=True,
)

# ==============================================================================
# ۲. هسته محاسباتی الگوریتم (Core Engine)
# ==============================================================================
CFG = {
    "hypo_threshold": 70.0,
    "alert_margin": 5.0,  # حاشیه پیش‌بینی (70 + 5 = 75 mg/dL)
    "risk_score_threshold": 75.0,  # آستانه ریسک
    "tau_minutes": 45.0,  # پارامتر میرایی افق زمانی
    "horizons": [30, 45, 60],
    "w_dist": 0.35,
    "w_vel": 0.35,
    "w_acc": 0.15,
    "w_persist": 0.15,
}


def calculate_metrics_for_stream(records):
    """محاسبه ویژگی‌های ریاضی و پیش‌بینی‌ها روی بافر زنده"""
    if not records:
        return []

    df = pd.DataFrame(records)
    df["Smooth"] = df["Raw"].ewm(span=3, adjust=False).mean()

    processed = []
    for i in range(len(df)):
        item = records[i].copy()
        item["Smooth"] = round(float(df.loc[i, "Smooth"]), 1)

        # در ۱۱ گام اول (کمتر از ۱ ساعت)، سیستم در حالت بافر است
        if i < 11:
            item["ROC_15m"] = "-"
            item["Risk_Score"] = "-"
            item["Pred_30m"] = "-"
            item["Alert_30m"] = False
            item["Pred_45m"] = "-"
            item["Alert_45m"] = False
            item["Pred_60m"] = "-"
            item["Alert_60m"] = False
            item["Status"] = f"بافر ({i+1}/12)"
            processed.append(item)
            continue

        current_smooth = df.loc[i, "Smooth"]
        current_raw = df.loc[i, "Raw"]

        # ROC در ۱۵ دقیقه (۳ گام قبلی)
        roc_15m = (current_smooth - df.loc[i - 3, "Smooth"]) / 15.0
        item["ROC_15m"] = round(float(roc_15m), 2)

        # شتاب تغییرات
        if i >= 5:
            roc_prev = (df.loc[i - 2, "Smooth"] - df.loc[i - 5, "Smooth"]) / 15.0
            acc = (roc_15m - roc_prev) / 10.0
        else:
            acc = 0.0

        # تداوم نزول در ۱۲ خوانش اخیر
        smooth_slice = df.loc[max(0, i - 11) : i, "Smooth"]
        persistence = float((smooth_slice.diff() < 0).mean() * 100.0)

        # نمره ریسک
        s_dist = np.clip((180.0 - current_smooth) / 110.0 * 100.0, 0.0, 100.0)
        s_vel = np.clip(-roc_15m / 2.0 * 100.0, 0.0, 100.0)
        s_acc = np.clip(-acc / 0.1 * 100.0, 0.0, 100.0)
        risk_score = (
            CFG["w_dist"] * s_dist
            + CFG["w_vel"] * s_vel
            + CFG["w_acc"] * s_acc
            + CFG["w_persist"] * persistence
        )
        item["Risk_Score"] = round(float(risk_score), 1)

        # محاسبه پیش‌بینی‌ها
        has_alert = False
        for h in CFG["horizons"]:
            tau = CFG["tau_minutes"]
            h_eff = tau * (1.0 - np.exp(-h / tau)) if tau else float(h)
            pred_g = float(
                np.clip(current_smooth + roc_15m * h_eff, 20.0, 400.0)
            )
            alert_flag = bool(
                pred_g <= (CFG["hypo_threshold"] + CFG["alert_margin"])
                or risk_score >= CFG["risk_score_threshold"]
            )

            item[f"Pred_{h}m"] = round(pred_g, 1)
            item[f"Alert_{h}m"] = alert_flag
            if alert_flag:
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
# ۳. مدیریت State و حافظه برنامه
# ==============================================================================
if "records" not in st.session_state:
    st.session_state.records = []

if "start_time" not in st.session_state:
    st.session_state.start_time = datetime.now().replace(
        second=0, microsecond=0
    )

# ==============================================================================
# ۴. نوار کناری (Sidebar): تنظیمات و ورودی داده
# ==============================================================================
with st.sidebar:
    st.image(
        "https://img.icons8.com/fluency/96/blood-drop.png",
        width=60,
    )
    st.title("کنترل‌پنل CGM")
    st.caption("سیستم هوشمند پیش‌هشدار هیپوگلایسمی")
    st.markdown("---")

    mode = st.radio(
        "حالت ورود داده:",
        ["ورود زنده (شبیه‌ساز ۵ دقیقه‌ای)", "آپلود فایل اکسل/CSV سنسور"],
    )

    if mode == "ورود زنده (شبیه‌ساز ۵ دقیقه‌ای)":
        st.subheader("ثبت قند جدید")
        new_glucose = st.number_input(
            "عدد قند خون (mg/dL):",
            min_value=20.0,
            max_value=450.0,
            value=115.0,
            step=1.0,
        )

        col_b1, col_b2 = st.columns(2)
        with col_b1:
            if st.button("➕ ثبت ۵ دقیقه", use_container_width=True):
                current_step = len(st.session_state.records)
                sim_time = st.session_state.start_time + timedelta(
                    minutes=current_step * 5
                )
                st.session_state.records.append(
                    {
                        "Step": current_step + 1,
                        "Time": sim_time.strftime("%H:%M"),
                        "Raw": float(new_glucose),
                    }
                )
                st.rerun()

        with col_b2:
            if st.button("🗑️ پاکسازی", use_container_width=True):
                st.session_state.records = []
                st.rerun()

        st.markdown("---")
        st.subheader("تست‌های آماده (تزریق سریع)")
        if st.button("⚡ تزریق نمونه اول (افت روزانه)", use_container_width=True):
            # دیتای تست اول
            sample_1 = [
                117,
                115,
                113,
                117,
                115,
                115,
                115,
                113,
                120,
                124,
                129,
                129,
                124,
                124,
                117,
                109,
                106,
                99,
                93,
                88,
                81,
                73,
                68,
            ]
            st.session_state.records = []
            base_t = datetime.strptime("14:22", "%H:%M")
            for i, val in enumerate(sample_1):
                t = base_t + timedelta(minutes=i * 5)
                st.session_state.records.append(
                    {"Step": i + 1, "Time": t.strftime("%H:%M"), "Raw": float(val)}
                )
            st.rerun()

        if st.button("⚡ تزریق نمونه دوم (سقوط شبانه)", use_container_width=True):
            # دیتای تست دوم
            sample_2 = [
                158,
                162,
                156,
                151,
                140,
                135,
                131,
                136,
                138,
                136,
                136,
                124,
                126,
                117,
                108,
                99,
                77,
                64,
            ]
            st.session_state.records = []
            base_t = datetime.strptime("23:07", "%H:%M")
            for i, val in enumerate(sample_2):
                t = base_t + timedelta(minutes=i * 5)
                st.session_state.records.append(
                    {"Step": i + 1, "Time": t.strftime("%H:%M"), "Raw": float(val)}
                )
            st.rerun()

    else:
        uploaded_file = st.file_uploader(
            "فایل اکسل سنسور (.xls, .xlsx, .csv):", type=["xls", "xlsx", "csv"]
        )
        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith(".csv"):
                    df_up = pd.read_csv(uploaded_file)
                else:
                    df_up = pd.read_excel(uploaded_file)

                time_col = df_up.columns[0]
                glucose_col = df_up.columns[1]

                # پردازش اولیه مقادیر قند
                def parse_g(v):
                    if str(v).strip().upper() == "LOW":
                        return 39.0
                    if str(v).strip().upper() == "HIGH":
                        return 401.0
                    return float(v)

                df_up["Raw"] = df_up[glucose_col].apply(parse_g)

                # استخراج زمان
                clean_times = (
                    df_up[time_col]
                    .astype(str)
                    .apply(lambda x: re.sub(r"\s*GMT.*", "", x).strip())
                )
                dt_series = pd.to_datetime(
                    clean_times, format="%d-%m-%Y %H:%M", errors="coerce"
                )
                if dt_series.isna().all():
                    dt_series = pd.to_datetime(
                        clean_times, dayfirst=True, errors="coerce"
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
                st.success(f"تعداد {len(df_up)} رکورد بارگذاری شد!")
            except Exception as e:
                st.error(f"خطا در خواندن فایل: {e}")

# ==============================================================================
# ۵. صفحه اصلی داشبورد
# ==============================================================================
st.title("🩸 داشبورد زنده پیش‌بینی افت قند خون (CGM Alert)")

processed_data = calculate_metrics_for_stream(st.session_state.records)

if not processed_data:
    st.info(
        "👈 داده‌ای وجود ندارد. از منوی سمت راست عدد قند را ثبت کنید یا یکی از دکمه‌های «تست سریع» را بزنید."
    )
else:
    last = processed_data[-1]
    count = len(processed_data)

    # کارت‌های متریک بالای صفحه
    c1, c2, c3, c4 = st.columns([1, 1, 1, 1.2])

    with c1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div style="color: #64748b; font-size: 13px;">آخرین زمان ثبت</div>
                <div style="font-size: 26px; font-weight: 700; color: #1e293b; margin: 4px 0;">{last['Time']}</div>
                <div style="color: #0284c7; font-size: 12px;">گام ۵ دقیقه‌ای #{count}</div>
            </div>
        """,
            unsafe_allow_html=True,
        )

    with c2:
        smooth_val = last["Smooth"] if last["Smooth"] != "-" else last["Raw"]
        st.markdown(
            f"""
            <div class="metric-card">
                <div style="color: #64748b; font-size: 13px;">قند لحظه‌ای (خام / هموار)</div>
                <div style="font-size: 26px; font-weight: 700; color: #1e293b; margin: 4px 0;">
                    {int(last['Raw'])} <span style="font-size: 16px; color: #64748b;">({smooth_val})</span>
                </div>
                <div style="color: #64748b; font-size: 12px;">mg/dL</div>
            </div>
        """,
            unsafe_allow_html=True,
        )

    with c3:
        roc_disp = f"{last['ROC_15m']} mg/dL/min" if last["ROC_15m"] != "-" else "-"
        risk_disp = f"{last['Risk_Score']} / 100" if last["Risk_Score"] != "-" else "-"
        st.markdown(
            f"""
            <div class="metric-card">
                <div style="color: #64748b; font-size: 13px;">نرخ افت (ROC) و ریسک</div>
                <div style="font-size: 22px; font-weight: 700; color: #1e293b; margin: 4px 0;">{roc_disp}</div>
                <div style="color: #ea580c; font-size: 12px; font-weight: 600;">نمره ریسک: {risk_disp}</div>
            </div>
        """,
            unsafe_allow_html=True,
        )

    with c4:
        st_text = last["Status"]
        if "ایمن" in st_text:
            badge = f'<div class="status-badge-safe">{st_text}</div>'
            sub = "بدون خطر افت در ۶۰ دقیقه بعد"
        elif "هشدار" in st_text:
            badge = f'<div class="status-badge-warn">{st_text}</div>'
            sub = "احتمال وقوع افت قند (≤ 70)"
        elif "افت" in st_text:
            badge = f'<div class="status-badge-low">{st_text}</div>'
            sub = "قند هم‌اکنون در بازه خطرناک است"
        else:
            badge = f'<div class="status-badge-buff">{st_text}</div>'
            sub = f"{12 - count} نقطه تا تکمیل یک ساعت"

        st.markdown(
            f"""
            <div class="metric-card" style="background: #fafafa;">
                <div style="color: #64748b; font-size: 13px; margin-bottom: 6px;">وضعیت هشدار زودهنگام</div>
                {badge}
                <div style="color: #64748b; font-size: 11px; margin-top: 6px;">{sub}</div>
            </div>
        """,
            unsafe_allow_html=True,
        )

    # --------------------------------------------------------------------------
    # تب‌های نمایش: نمودار و جدول داده‌ها
    # --------------------------------------------------------------------------
    tab_chart, tab_table = st.tabs(["📈 نمودار ترند و پیش‌بینی", "📋 جدول جزئیات گام‌ها"])

    with tab_chart:
        df_p = pd.DataFrame(processed_data)

        fig = go.Figure()

        # خط قند خام
        fig.add_trace(
            go.Scatter(
                x=df_p["Time"],
                y=df_p["Raw"],
                mode="lines+markers",
                name="قند خام سنسور",
                line=dict(color="#3b82f6", width=2),
                marker=dict(size=6),
            )
        )

        # خط قند هموارشده
        fig.add_trace(
            go.Scatter(
                x=df_p["Time"],
                y=df_p["Smooth"],
                mode="lines",
                name="هموارشده (EMA)",
                line=dict(color="#10b981", width=2, dash="dot"),
            )
        )

        # خط قرمز مرز هیپوگلایسمی (۷۰)
        fig.add_hline(
            y=70,
            line_dash="dash",
            line_color="#ef4444",
            annotation_text="مرز افت قند (70 mg/dL)",
            annotation_position="bottom right",
        )

        # نقاط پیش‌بینی آخرین گام (در صورت وجود)
        if last["Pred_30m"] != "-":
            last_t_idx = len(df_p) - 1
            pred_times = ["+30m", "+45m", "+60m"]
            pred_vals = [last["Pred_30m"], last["Pred_45m"], last["Pred_60m"]]

            fig.add_trace(
                go.Scatter(
                    x=pred_times,
                    y=pred_vals,
                    mode="markers+lines",
                    name="مسیر پیش‌بینی آینده",
                    line=dict(color="#f59e0b", width=2, dash="dash"),
                    marker=dict(size=10, symbol="diamond", color="#d97706"),
                )
            )

        fig.update_layout(
            height=420,
            margin=dict(l=20, r=20, t=30, b=20),
            hovermode="x unified",
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
            ),
            plot_bgcolor="#f8fafc",
            paper_bgcolor="#ffffff",
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab_table:
        # ساخت جدول شکیل برای نمایش ردیف‌ها
        display_rows = []
        for r in reversed(processed_data):

            def style_alert(pred, alt):
                if pred == "-":
                    return "-"
                if alt:
                    return f"⚠️ {pred}"
                return f"{pred}"

            display_rows.append(
                {
                    "گام": f"#{r['Step']}",
                    "زمان": r["Time"],
                    "قند خام": r["Raw"],
                    "هموار (EMA)": r["Smooth"],
                    "نرخ افت (ROC)": r["ROC_15m"],
                    "نمره ریسک": r["Risk_Score"],
                    "پیش‌بینی ۳۰ دقیقه": style_alert(
                        r["Pred_30m"], r["Alert_30m"]
                    ),
                    "پیش‌بینی ۴۵ دقیقه": style_alert(
                        r["Pred_45m"], r["Alert_45m"]
                    ),
                    "پیش‌بینی ۶۰ دقیقه": style_alert(
                        r["Pred_60m"], r["Alert_60m"]
                    ),
                    "وضعیت": r["Status"],
                }
            )

        table_df = pd.DataFrame(display_rows)
        st.dataframe(table_df, use_container_width=True, hide_index=True)

        # دانلود جدول به صورت اکسل
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
            pd.DataFrame(processed_data).to_excel(
                writer, index=False, sheet_name="CGM_Stream_Data"
            )

        st.download_button(
            label="📥 دانلود کل داده‌های پردازش‌شده (اکسل)",
            data=buffer.getvalue(),
            file_name="CGM_Stream_Predictions.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
