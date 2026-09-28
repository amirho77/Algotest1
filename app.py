import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from config import HORIZONS, InputConfig, ModelConfig, VERSION
from core import analyze, excel_bytes, latest_state

st.set_page_config(page_title="CGM Hypo MVP", page_icon="🩸", layout="wide")
st.markdown("""
<style>
.block-container {max-width: 1400px; padding-top: 2rem;}
[data-testid="stMetric"] {background:#f7fafc; border:1px solid #e2e8f0; padding:12px; border-radius:12px;}
.step {background:#f8fafc; border-left:4px solid #0f766e; padding:10px 14px; border-radius:8px; margin:5px 0;}
</style>
""", unsafe_allow_html=True)

st.title("🩸 پایش و پیش‌بینی افت قند")
st.caption(f"MVP پژوهشی • نسخه {VERSION} • مدل پیش‌فرض: Recovery-filtered Enhanced")
st.warning("این ابزار تشخیصی یا درمانی نیست، دوز انسولین پیشنهاد نمی‌دهد و احتمال کالیبره‌شده تولید نمی‌کند. خروجی فایل تاریخی برای اعلان زنده استفاده نمی‌شود.")

with st.sidebar:
    st.header("ورودی")
    demo = st.checkbox("نمونه آموزشی ساختگی")
    upload = st.file_uploader("فایل CSV یا Excel", type=["csv", "xlsx", "xls"])
    method_label = st.selectbox("حالت تحلیل", ["Recovery (پیشنهاد MVP)", "Enhanced (حساسیت بیشتر)", "Precision (محافظه‌کارانه)"])
    method = {"Recovery (پیشنهاد MVP)":"recovery", "Enhanced (حساسیت بیشتر)":"enhanced", "Precision (محافظه‌کارانه)":"precision"}[method_label]

if demo:
    frame = pd.DataFrame({"time": pd.date_range("2026-01-01", periods=60, freq="5min", tz="UTC"), "glucose": [180 - i * 1.7 for i in range(60)]})
    st.info("این نمونه ساختگی است و برای نمایش مسیر محاسبه استفاده می‌شود.")
elif upload:
    if upload.size > 50_000_000:
        st.error("حجم فایل نباید بیشتر از ۵۰ مگابایت باشد."); st.stop()
    try: frame = pd.read_csv(upload) if upload.name.lower().endswith(".csv") else pd.read_excel(upload)
    except Exception as exc: st.error(f"خواندن فایل ممکن نشد: {exc}"); st.stop()
else:
    st.info("برای شروع، یک فایل CGM بارگذاری کنید یا نمونه آموزشی را فعال کنید."); st.stop()

if frame.empty or len(frame.columns) < 2:
    st.error("فایل باید حداقل یک ردیف و دو ستون داشته باشد."); st.stop()
with st.sidebar:
    time_col = st.selectbox("ستون زمان", frame.columns)
    glucose_col = st.selectbox("ستون قند", frame.columns, index=1)
    candidates = [c for c in frame.columns if c not in (time_col, glucose_col)]
    known = [c for c in candidates if str(c).lower() in ("patient_id", "subject_id", "sensor_id", "session_id")]
    ids = st.multiselect("شناسه بیمار/سنسور/جلسه", candidates, default=known)
    original_count = len(frame)
    for column in ids:
        choices = frame[column].dropna().unique().tolist()
        if not choices: st.error(f"شناسه معتبر در {column} نیست."); st.stop()
        chosen = st.selectbox(f"انتخاب {column}", choices); frame = frame.loc[frame[column].eq(chosen)]
    unit = st.selectbox("واحد قند", ["mg/dL", "mmol/L"])
    timezone = st.text_input("منطقه زمانی", "Asia/Tehran")
    with st.expander("تنظیمات ورودی"):
        date_format = st.text_input("قالب تاریخ، اختیاری", "")
        numeric_time = st.selectbox("واحد زمان عددی", ["غیرفعال", "excel", "s", "ms"])
        use_bounds = st.checkbox("حدود سنسور را می‌دانم")
        lower = st.number_input("حد پایین mg/dL", min_value=1., value=40.) if use_bounds else None
        upper = st.number_input("حد بالای mg/dL", min_value=2., value=400.) if use_bounds else None
if not ids and not demo and not st.sidebar.checkbox("فایل فقط یک جریان بیمار/سنسور دارد"):
    st.info("برای جلوگیری از ترکیب تاریخچه بیماران، شناسه را انتخاب یا تک‌جریانی بودن فایل را تأیید کنید."); st.stop()

try:
    inputs = InputConfig(timezone=timezone, unit=unit, date_format=date_format or None, numeric_time_unit=None if numeric_time == "غیرفعال" else numeric_time, sensor_lower=lower, sensor_upper=upper)
    cfg = ModelConfig()
    analysis, rows, events, event_summary, metadata = analyze(frame, time_col, glucose_col, inputs, cfg, stream_columns=ids, method=method)
    state = latest_state(analysis, metadata, cfg)
except Exception as exc:
    st.error(f"پردازش انجام نشد: {exc}"); st.stop()

st.subheader("وضعیت آخرین داده")
st.caption(f"آخرین زمان ورودی (UTC): {metadata['latest_input_UTC']} • {len(frame):,} ردیف از {original_count:,} ردیف")
status_text = {"current_low":"قند فعلی در محدوده افت است.", "predicted_low":"روند مدل در یک یا چند افق به محدوده افت می‌رسد.", "preventive_risk":"روند نزولی نیازمند توجه است.", "uncertain_trend":"روند نامطمئن است؛ داده بیشتری لازم است.", "stale":"داده قدیمی است؛ پیش‌بینی جاری ارائه نمی‌شود.", "sensor_check":"آخرین خوانش عدد معتبر ندارد.", "insufficient_data":"سابقه پیوسته کافی نیست.", "short_history":"سابقه برای پیش‌بینی کامل کافی نیست.", "no_model_alert":"هشدار فعال نیست؛ این تضمین ایمنی نیست."}
if state["status"] in ("current_low", "predicted_low"): st.error(status_text[state["status"]])
elif state["status"] in ("preventive_risk", "uncertain_trend"): st.warning(status_text[state["status"]])
else: st.info(status_text.get(state["status"], state["status"]))

if state.get("row") is not None:
    row = state["row"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("قند هموار فعلی", f"{row.Current_Glucose:.1f} mg/dL")
    c2.metric("شیب ۱۵ دقیقه", f"{row.ROC_15m:.2f} mg/dL/min")
    c3.metric("تداوم نزول", f"{row.Downward_Persistence:.0f}%")
    c4.metric("امتیاز ریسک", f"{row.Low_Risk_Score:.0f}/100")
    st.markdown("**پیش‌بینی افق‌های زمانی**")
    cols = st.columns(3)
    for col, h in zip(cols, HORIZONS):
        alert = bool(row[f"Alert_{h}m"] == 1)
        col.metric(f"{h} دقیقه بعد", f"{row[f'Pred_Glucose_{h}m']:.1f} mg/dL", "هشدار" if alert else "بدون هشدار")
        col.caption(f"سناریو: {row[f'Lower_Scenario_{h}m']:.0f} تا {row[f'Upper_Scenario_{h}m']:.0f}")

with st.expander("روند محاسبه را ببینید"):
    st.markdown("""
    <div class="step"><b>۱. پاک‌سازی:</b> حذف زمان/قند نامعتبر، مرتب‌سازی، مدیریت timezone و بازنمونه‌گیری ۵ دقیقه‌ای.</div>
    <div class="step"><b>۲. هموارسازی:</b> EMA با span=3 برای کاهش نویز کوتاه‌مدت.</div>
    <div class="step"><b>۳. ویژگی‌ها:</b> قند فعلی، شیب ۱۵ دقیقه، شتاب، فاصله تا ۷۰، تداوم نزول و پراکندگی روند.</div>
    <div class="step"><b>۴. پیش‌بینی:</b> برون‌یابی خطی پایدار برای ۳۰، ۴۵ و ۶۰ دقیقه با محدودیت ۲۰ تا ۴۰۰.</div>
    <div class="step"><b>۵. فیلتر Recovery:</b> اگر افت در حال خنثی‌شدن یا برگشت باشد، هشدار معمولی حذف می‌شود؛ افت سریع و نزدیک به مرز حفظ می‌شود.</div>
    <div class="step"><b>۶. رویداد و اعلان:</b> روندهای پیوسته گروه‌بندی می‌شوند و یک افت واحد چند هشدار تکراری تولید نمی‌کند.</div>
    """, unsafe_allow_html=True)

fig = go.Figure(); fig.add_scatter(x=analysis.index, y=analysis["Raw"], name="خوانش سنسور", connectgaps=False); fig.add_scatter(x=analysis.index, y=analysis["Current_Glucose"], name="قند هموار", connectgaps=False); fig.add_hline(y=cfg.threshold, line_color="red", line_dash="dash", annotation_text="مرز ۷۰")
fig.update_layout(height=430, hovermode="x unified", xaxis_title="زمان (UTC)", yaxis_title="mg/dL", legend_title="سری داده")
st.plotly_chart(fig, use_container_width=True)

tab1, tab2, tab3 = st.tabs(["رویدادهای افت", "ارزیابی مدل", "جزئیات فنی"])
with tab1:
    st.caption("هر رویداد یک بازه پیوسته از قند پایین است؛ هشدارهای تکراری یک رویداد جداگانه محسوب نمی‌شوند."); st.dataframe(event_summary, use_container_width=True, hide_index=True); st.dataframe(events, use_container_width=True, hide_index=True)
with tab2:
    st.caption("Precision و Recall در سطح زمان تصمیم مدل هستند؛ Capture در سطح رویداد محاسبه می‌شود."); st.dataframe(rows, use_container_width=True, hide_index=True)
with tab3:
    st.json(metadata); st.dataframe(analysis.tail(300), use_container_width=True)
st.download_button("دانلود گزارش Excel", excel_bytes(analysis, rows, metadata, events, event_summary), "CGM_recovery_analysis.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
