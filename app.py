"""CGM hypoglycaemia MVP dashboard: user-facing Streamlit surface."""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from config import HORIZONS, InputConfig, ModelConfig, VERSION
from core import analyze, excel_bytes, latest_state

st.set_page_config(page_title="CGM Hypo", page_icon="🩸", layout="wide", initial_sidebar_state="expanded")
st.markdown("""
<style>
  .block-container{max-width:1320px;padding-top:1.25rem}
  [data-testid="stMetric"]{background:#fff;border:1px solid #e5e7eb;border-radius:12px;padding:12px}
  .topbar{padding:18px 22px;background:linear-gradient(110deg,#0f766e,#115e59);border-radius:16px;color:#fff;margin-bottom:18px}
  .topbar h1{margin:0;font-size:1.7rem}.topbar p{margin:6px 0 0;opacity:.9}
  .status{padding:14px 16px;border-radius:12px;margin:12px 0 18px}.status.good{background:#f0fdf4;border:1px solid #86efac}.status.warn{background:#fffbeb;border:1px solid #fcd34d}.status.danger{background:#fef2f2;border:1px solid #fca5a5}.status.neutral{background:#f8fafc;border:1px solid #cbd5e1}
  .horizon{border:1px solid #e5e7eb;border-top:4px solid #0f766e;border-radius:12px;padding:14px;background:#fff}.horizon.alert{border-top-color:#dc2626;background:#fffafa}
  .muted{color:#64748b;font-size:.9rem}
</style>
""", unsafe_allow_html=True)


def is_true(value):
    return pd.notna(value) and bool(value)


def plain_message(row):
    if is_true(row.get("Notification_30m", False)):
        if is_true(row.get("Fast_Drop_Risk", False)):
            return "افت سریع دیده شده است؛ روند قند را بررسی کنید."
        if is_true(row.get("Preventive_Alert", False)):
            return "روند نزولی پایدار است؛ قند را زودتر بررسی کنید."
        return "ادامه روند فعلی ممکن است به افت قند منجر شود؛ روند را بررسی کنید."
    if is_true(row.get("Alert_30m", False)):
        return "روند نیازمند پایش است؛ اعلان اصلی هنوز ارسال نشده است."
    return "اعلان فعالی نیست."


def trend_label(roc):
    if pd.isna(roc):
        return "داده کافی نیست"
    if roc < -0.10:
        return "نزولی"
    if roc > 0.10:
        return "صعودی"
    return "تقریباً ثابت"


st.markdown("""<div class="topbar"><h1>پلتفرم پایش افت قند</h1><p>روند قند را بخوانید، هشدارهای مدل را ببینید و عملکرد همان فایل را ارزیابی کنید.</p></div>""", unsafe_allow_html=True)

with st.sidebar:
    st.header("ورود داده")
    source = st.radio("روش ورود", ["فایل CGM", "ورود دستی", "نمونه آموزشی"])
    if source == "فایل CGM":
        upload = st.file_uploader("فایل Excel یا CSV", type=["xls", "xlsx", "csv"])
    else:
        upload = None
    mode_label = st.selectbox("اولویت مدل", ["شناسایی بیشترین افت", "تعادل با کنترل برگشت", "هشدار محافظه‌کارانه"])
    method = {"شناسایی بیشترین افت":"enhanced", "تعادل با کنترل برگشت":"recovery", "هشدار محافظه‌کارانه":"precision"}[mode_label]
    st.caption("حالت اول برای اولویت فعلی پروژه انتخاب شده است.")

manual_mode = source == "ورود دستی"
if manual_mode:
    st.subheader("ورود دستی خوانش‌ها")
    st.caption("هر ردیف یک خوانش است. سیستم بین همه ردیف‌ها ۵ دقیقه فاصله می‌گذارد.")
    count = st.number_input("تعداد خوانش", 12, 288, 24, 1)
    entered = pd.DataFrame({"ردیف": range(1, int(count) + 1), "قند mg/dL": [140.0] * int(count)})
    entered = st.data_editor(entered, hide_index=True, num_rows="fixed", use_container_width=True,
        column_config={"ردیف": st.column_config.NumberColumn(disabled=True), "قند mg/dL": st.column_config.NumberColumn(min_value=20, max_value=400, step=1)})
    glucose = pd.to_numeric(entered["قند mg/dL"], errors="coerce")
    if glucose.isna().any():
        st.error("همه ردیف‌ها باید مقدار عددی قند داشته باشند.")
        st.stop()
    last_time = pd.Timestamp.now(tz="UTC").floor("5min")
    frame = pd.DataFrame({"time": pd.date_range(last_time - pd.Timedelta(minutes=5 * (len(glucose)-1)), last_time, freq="5min"), "glucose": glucose})
    time_col, glucose_col, stream_columns, original_count = "time", "glucose", [], len(frame)
elif source == "نمونه آموزشی":
    frame = pd.DataFrame({"time": pd.date_range("2026-01-01", periods=48, freq="5min", tz="UTC"), "glucose": list(range(160, 112, -1))})
    time_col, glucose_col, stream_columns, original_count = "time", "glucose", [], len(frame)
else:
    if upload is None:
        st.info("یک فایل CGM انتخاب کنید یا ورود دستی را فعال کنید.")
        st.stop()
    try:
        frame = pd.read_csv(upload) if upload.name.lower().endswith(".csv") else pd.read_excel(upload)
    except Exception as exc:
        st.error(f"فایل خوانده نشد: {exc}")
        st.stop()
    if frame.empty or len(frame.columns) < 2:
        st.error("فایل باید حداقل ستون زمان و ستون قند داشته باشد.")
        st.stop()
    original_count = len(frame)
    with st.sidebar:
        st.header("معرفی ستون‌ها")
        time_col = st.selectbox("ستون زمان", frame.columns)
        glucose_col = st.selectbox("ستون قند", frame.columns, index=1)
        candidates = [c for c in frame.columns if c not in (time_col, glucose_col)]
        stream_columns = st.multiselect("شناسه بیمار یا سنسور (در صورت وجود)", candidates)
        if not stream_columns and not st.checkbox("فایل فقط یک بیمار و یک سنسور دارد"):
            st.info("برای جلوگیری از ترکیب تاریخچه‌ها، تک‌جریانی بودن فایل را تأیید کنید.")
            st.stop()

with st.sidebar:
    st.header("تنظیمات")
    timezone = st.text_input("منطقه زمانی", "Asia/Tehran")
    unit = st.selectbox("واحد قند", ["mg/dL", "mmol/L"])

try:
    inputs = InputConfig(timezone=timezone, unit=unit)
    config = ModelConfig()
    analysis, row_metrics, events, event_metrics, metadata = analyze(frame, time_col, glucose_col, inputs, config, stream_columns=stream_columns, method=method)
    state = latest_state(analysis, metadata, config)
except Exception as exc:
    st.error(f"تحلیل انجام نشد: {exc}")
    st.stop()

tabs = st.tabs(["داشبورد", "جدول پیش‌بینی", "ارزیابی فایل", "روش محاسبه"])

with tabs[0]:
    status = {
        "current_low": ("danger", "قند فعلی پایین است", "آخرین خوانش در محدوده افت قرار دارد."),
        "predicted_low": ("danger", "هشدار روند فعال است", "مدل ادامه روند را نزدیک یا پایین‌تر از مرز افت می‌بیند."),
        "preventive_risk": ("warn", "روند نزولی نیازمند توجه است", "هنوز افت قطعی ثبت نشده، اما جهت حرکت قند هشداردهنده است."),
        "uncertain_trend": ("warn", "روند نامطمئن است", "برای تصمیم بهتر، خوانش‌های بعدی را بررسی کنید."),
        "no_model_alert": ("good", "هشدار فعالی نیست", "در داده فعلی شواهد کافی برای هشدار وجود ندارد."),
        "insufficient_data": ("neutral", "سابقه کافی نیست", "برای پیش‌بینی کامل به حدود یک ساعت داده پیوسته نیاز است."),
        "short_history": ("neutral", "سابقه کوتاه است", "فعلاً فقط روند کوتاه‌مدت قابل بررسی است."),
        "stale": ("neutral", "داده قدیمی است", "پیش‌بینی جاری با داده قدیمی معتبر نیست."),
        "sensor_check": ("neutral", "خوانش سنسور نامعتبر است", "آخرین مقدار قابل اعتماد نیست.")}
    cls, title, description = status.get(state["status"], ("neutral", "وضعیت نامشخص", ""))
    st.markdown(f'<div class="status {cls}"><b>{title}</b><br>{description}</div>', unsafe_allow_html=True)
    if state.get("row") is not None:
        row = state["row"]
        left, mid, right = st.columns(3)
        left.metric("قند فعلی", f"{row.Current_Glucose:.0f} mg/dL")
        mid.metric("روند ۱۵ دقیقه اخیر", trend_label(row.ROC_15m), f"شیب {row.ROC_15m:.2f}")
        right.metric("تداوم نزول", f"{row.Downward_Persistence:.0f}%")
        st.subheader("پیش‌بینی کوتاه‌مدت")
        cols = st.columns(len(HORIZONS))
        for column, horizon in zip(cols, HORIZONS):
            alert = is_true(row[f"Alert_{horizon}m"])
            css = "alert" if alert else ""
            label = "هشدار" if alert else "بدون هشدار"
            column.markdown(f'<div class="horizon {css}"><b>{horizon} دقیقه بعد</b><h3>{row[f"Pred_Glucose_{horizon}m"]:.0f} mg/dL</h3><span class="muted">{label}</span></div>', unsafe_allow_html=True)
    fig = go.Figure()
    fig.add_scatter(x=analysis.index, y=analysis["Raw"], name="قند ثبت‌شده", connectgaps=False, line={"color":"#94a3b8"})
    fig.add_scatter(x=analysis.index, y=analysis["Current_Glucose"], name="روند هموار", connectgaps=False, line={"color":"#0f766e", "width":3})
    fig.add_hline(y=70, line_dash="dash", line_color="#dc2626", annotation_text="مرز افت ۷۰")
    fig.update_layout(height=390, hovermode="x unified", yaxis_title="mg/dL", xaxis_title="زمان", margin={"l":10,"r":10,"t":20,"b":10})
    st.plotly_chart(fig, use_container_width=True)

with tabs[1]:
    st.subheader("خوانش‌ها، پیش‌بینی و پیام قابل ارسال")
    st.caption("ردیف‌های هشدار با پیام قابل ارسال مشخص‌اند. در حالت ورود دستی، زمان‌ها با فاصله ۵ دقیقه ساخته می‌شوند.")
    display = pd.DataFrame({"زمان": analysis.index.astype(str), "قند ثبت‌شده": analysis["Raw"].round(1), "قند هموار": analysis["Current_Glucose"].round(1), "روند": analysis["ROC_15m"].map(trend_label), **{f"پیش‌بینی {h} دقیقه": analysis[f"Pred_Glucose_{h}m"].round(1) for h in HORIZONS}, "هشدار": analysis["Alert_30m"].map({1:"بله",0:"خیر"}), "پیام کاربر": [plain_message(analysis.iloc[i]) for i in range(len(analysis))]})
    alerts_only = st.toggle("فقط ردیف‌های هشدار را نمایش بده")
    if alerts_only:
        display = display.loc[analysis["Alert_30m"].fillna(0).eq(1).to_numpy()]
    st.dataframe(display, use_container_width=True, hide_index=True, height=560)

with tabs[2]:
    st.subheader("عملکرد مدل روی همین فایل")
    if manual_mode:
        st.info("در ورود دستی، آینده واقعی هنوز ثبت نشده است؛ بنابراین درست یا غلط بودن هشدارها قابل ارزیابی نیست. پیش‌بینی‌ها در جدول قبلی نمایش داده شده‌اند.")
    else:
        event30 = event_metrics.loc[event_metrics["Horizon_min"].eq(30)].iloc[0]
        row30 = row_metrics.loc[row_metrics["Horizon_min"].eq(30)].iloc[0]
        actual = int(event30["All_events"])
        detected = int(event30["Captured"])
        missed = int(event30["Missed_eligible"] + event30["No_opportunity"])
        false_alerts = int(row30["Notification_FP"])
        precision = float(row30["Precision"]) if pd.notna(row30["Precision"]) else 0.0
        recall = float(event30["Recall_eligible_events"]) if pd.notna(event30["Recall_eligible_events"]) else 0.0
        false_alert_component = max(0, 1 - false_alerts / max(1, false_alerts + int(row30["Notification_TP"])))
        score = round(100 * (0.60 * recall + 0.25 * precision + 0.15 * false_alert_component))
        c1,c2,c3,c4,c5,c6=st.columns(6)
        c1.metric("افت واقعی فایل", actual)
        c2.metric("افت شناسایی‌شده", detected)
        c3.metric("افت شناسایی‌نشده", missed)
        c4.metric("هشدار کاذب", false_alerts)
        c5.metric("دقت هشدار", f"{precision*100:.1f}%")
        c6.metric("امتیاز فایل", f"{max(0,min(100,score))}/100")
        feedback = []
        if recall < .80: feedback.append("مدل بخشی از افت‌های واقعی را ندیده است؛ حساسیت این فایل پایین‌تر از هدف است.")
        if precision < .50: feedback.append("هشدارهای کاذب در این فایل قابل توجه‌اند؛ این حالت معمولاً در روندهای برگشتی یا ناپایدار رخ می‌دهد.")
        if not feedback: feedback.append("عملکرد این فایل از نظر پوشش افت و دقت هشدار متعادل است.")
        st.write("**نقد خودکار:** " + " ".join(feedback))
        st.dataframe(events, use_container_width=True, hide_index=True)

with tabs[3]:
    st.subheader("مدل با چه منطقی محاسبه می‌کند؟")
    st.markdown("""
1. داده‌ها پاک‌سازی و روی فاصله‌های منظم ۵ دقیقه‌ای مرتب می‌شوند.
2. نوسان‌های کوتاه سنسور با هموارسازی کاهش می‌یابند.
3. قند فعلی، شیب ۱۵ دقیقه، شتاب، فاصله تا ۷۰ و تداوم نزول محاسبه می‌شود.
4. پیش‌بینی هر افق از فرمول «قند فعلی + شیب پایدار × زمان آینده» ساخته می‌شود.
5. مدل Enhanced برای بیشترین شناسایی افت استفاده می‌شود. حالت Recovery فقط روندهایی را که نشانه برگشت دارند محدود می‌کند.
6. افت واقعی یعنی کمترین قند پنجره آینده به ۷۰ یا کمتر برسد. هشدار کاذب یعنی اعلان صادر شود ولی در پنجره آینده افت رخ ندهد.
    """)
    with st.expander("معنی شاخص‌ها"):
        st.markdown("**شیب:** سرعت تغییر قند؛ عدد منفی یعنی افت.  \n**شتاب:** افت در حال سریع‌ترشدن یا کندشدن است.  \n**تداوم نزول:** درصد تغییرات نزولی اخیر.  \n**دقت:** سهم اعلان‌های درست.  \n**حساسیت:** سهم افت‌های واقعی که شناسایی شده‌اند.")

st.download_button("دانلود گزارش کامل Excel", excel_bytes(analysis, row_metrics, metadata, events, event_metrics), "CGM_hypo_report.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
