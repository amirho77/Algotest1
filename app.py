import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from config import HORIZONS, InputConfig, ModelConfig, VERSION
from core import analyze, excel_bytes, latest_state

st.set_page_config(page_title="CGM | پایش افت قند", page_icon="🩸", layout="wide", initial_sidebar_state="expanded")
st.markdown("""
<style>
.block-container {max-width: 1380px; padding-top: 1.6rem;}
.hero {padding: 1.2rem 1.4rem; border-radius: 16px; background: linear-gradient(120deg,#ecfeff,#f8fafc); border:1px solid #cbd5e1; margin-bottom:1rem;}
.card {padding: 1rem; border-radius: 14px; border:1px solid #e2e8f0; background:#fff; min-height:150px;}
.card h4 {margin:0 0 .4rem 0; color:#0f172a;}
.card p {margin:.25rem 0; color:#475569; font-size:.92rem;}
.good {border-top:5px solid #16a34a;} .warn {border-top:5px solid #f59e0b;} .danger {border-top:5px solid #dc2626;} .neutral {border-top:5px solid #64748b;}
.step {padding:.65rem .8rem; border-radius:9px; background:#f8fafc; border:1px solid #e2e8f0; margin:.35rem 0;}
.small {font-size:.86rem; color:#64748b;}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="hero"><h1>🩸 پایش روند قند و هشدار افت</h1><p>این صفحه به زبان ساده نشان می‌دهد روند قند بیمار به کدام سمت می‌رود و آیا در ۳۰، ۴۵ یا ۶۰ دقیقه آینده احتمال رسیدن به محدوده افت وجود دارد.</p></div>', unsafe_allow_html=True)
st.caption(f"نسخه MVP پژوهشی {VERSION} | مدل پیش‌فرض: Recovery")
st.warning("این ابزار برای اطلاع‌رسانی روند است؛ درمان، دوز انسولین یا تصمیم پزشکی پیشنهاد نمی‌کند و جایگزین هشدار دستگاه نیست.")

with st.sidebar:
    st.header("۱) ورود داده")
    source = st.radio("روش ورود داده", ["فایل CGM", "ورود دستی", "نمونه آموزشی"], index=0)
    demo = source == "نمونه آموزشی"
    upload = st.file_uploader("فایل داده CGM را انتخاب کنید", type=["csv", "xlsx", "xls"]) if source == "فایل CGM" else None
    method_label = st.selectbox("نوع بررسی", ["پیشنهاد MVP (متعادل)", "حساسیت بیشتر", "هشدارهای محافظه‌کارانه"])
    method = {"پیشنهاد MVP (متعادل)":"recovery", "حساسیت بیشتر":"enhanced", "هشدارهای محافظه‌کارانه":"precision"}[method_label]

if source == "ورود دستی":
    st.subheader("ورود دستی سری زمانی")
    st.caption("هر ردیف یک خوانش است؛ فاصله زمانی خوانش‌ها به‌صورت خودکار ۵ دقیقه در نظر گرفته می‌شود.")
    n_manual = st.number_input("تعداد خوانش‌ها", min_value=12, max_value=288, value=24, step=1)
    manual = pd.DataFrame({"شماره خوانش": range(1, int(n_manual) + 1), "قند (mg/dL)": [140.0] * int(n_manual)})
    manual = st.data_editor(manual, num_rows="fixed", use_container_width=True, hide_index=True,
                            column_config={"شماره خوانش": st.column_config.NumberColumn("خوانش", disabled=True),
                                           "قند (mg/dL)": st.column_config.NumberColumn("قند (mg/dL)", min_value=20, max_value=400, step=1, format="%.0f")})
    values = pd.to_numeric(manual["قند (mg/dL)"], errors="coerce")
    if values.isna().any(): st.error("همه ردیف‌ها باید عدد قند داشته باشند."); st.stop()
    end = pd.Timestamp.now(tz="UTC").floor("5min")
    frame = pd.DataFrame({"زمان": pd.date_range(end - pd.Timedelta(minutes=5 * (len(values)-1)), end, freq="5min"), "قند": values.to_numpy()})
    time_col, glucose_col, ids, original_count = "زمان", "قند", [], len(values)
elif demo:
    frame = pd.DataFrame({"زمان":pd.date_range("2026-01-01", periods=60, freq="5min", tz="UTC"), "قند": [180-i*1.7 for i in range(60)]})
    st.info("این فقط نمونه آموزشی است و برای تصمیم درباره بیمار استفاده نمی‌شود.")
elif upload:
    if upload.size > 50_000_000: st.error("حجم فایل بیشتر از ۵۰ مگابایت است."); st.stop()
    try: frame = pd.read_csv(upload) if upload.name.lower().endswith(".csv") else pd.read_excel(upload)
    except Exception as exc: st.error(f"فایل خوانده نشد: {exc}"); st.stop()
else:
    st.info("از نوار کناری یک فایل CGM انتخاب کنید تا تحلیل شروع شود."); st.stop()

if frame.empty or len(frame.columns)<2: st.error("فایل باید حداقل دو ستون داشته باشد."); st.stop()
with st.sidebar:
    st.header("۲) معرفی ستون‌ها")
    if source != "ورود دستی":
        time_col = st.selectbox("کدام ستون زمان است؟", frame.columns)
        glucose_col = st.selectbox("کدام ستون عدد قند است؟", frame.columns, index=1)
    other = [c for c in frame.columns if c not in (time_col, glucose_col)]
    known = [c for c in other if str(c).lower() in ("patient_id","subject_id","sensor_id","session_id")]
    ids = st.multiselect("شناسه بیمار یا سنسور (اختیاری)", other, default=known)
    original_count=len(frame)
    for column in ids:
        values=frame[column].dropna().unique().tolist()
        if not values: st.error(f"ستون {column} شناسه معتبر ندارد."); st.stop()
        selected=st.selectbox(f"انتخاب {column}", values); frame=frame.loc[frame[column].eq(selected)]
    if not ids and not demo and source != "ورود دستی":
        st.checkbox("فایل فقط مربوط به یک بیمار/سنسور است", key="single_stream")
    unit=st.selectbox("واحد عدد قند", ["mg/dL","mmol/L"])
    timezone=st.text_input("منطقه زمانی داده", "Asia/Tehran")
    with st.expander("تنظیمات پیشرفته ورود داده"):
        date_format=st.text_input("قالب تاریخ (در صورت نیاز)", "")
        numeric_time=st.selectbox("واحد زمان عددی", ["غیرفعال","excel","s","ms"])
        use_bounds=st.checkbox("حد پایین/بالای سنسور مشخص است")
        lower=st.number_input("حد پایین mg/dL", min_value=1., value=40.) if use_bounds else None
        upper=st.number_input("حد بالا mg/dL", min_value=2., value=400.) if use_bounds else None
if not ids and not demo and source != "ورود دستی" and not st.session_state.get("single_stream",False):
    st.info("برای جلوگیری از ترکیب داده چند نفر، تک‌جریانی بودن فایل را تأیید کنید یا شناسه انتخاب کنید."); st.stop()

try:
    inputs=InputConfig(timezone=timezone,unit=unit,date_format=date_format or None,numeric_time_unit=None if numeric_time=="غیرفعال" else numeric_time,sensor_lower=lower,sensor_upper=upper)
    cfg=ModelConfig(); analysis,rows,events,event_summary,metadata=analyze(frame,time_col,glucose_col,inputs,cfg,stream_columns=ids,method=method); state=latest_state(analysis,metadata,cfg)
except Exception as exc: st.error(f"تحلیل انجام نشد: {exc}"); st.stop()

status_map={"current_low":("danger","قند فعلی پایین است","آخرین عدد ثبت‌شده در محدوده افت قرار دارد."),"predicted_low":("danger","احتمال افت در روند دیده می‌شود","روند فعلی در یکی از افق‌های زمانی به محدوده افت می‌رسد."),"preventive_risk":("warn","روند نزولی نیازمند توجه است","هنوز الزاماً افت رخ نداده، اما جهت حرکت قند هشداردهنده است."),"uncertain_trend":("warn","روند نامطمئن است","داده‌ها جهت یکسانی ندارند؛ بررسی خوانش بعدی مهم است."),"stale":("neutral","داده قدیمی است","برای وضعیت فعلی داده تازه کافی نیست."),"sensor_check":("neutral","خوانش سنسور قابل اتکا نیست","آخرین خوانش عدد معتبر ندارد."),"insufficient_data":("neutral","سابقه کافی نیست","برای پیش‌بینی به داده پیوسته بیشتری نیاز است."),"short_history":("neutral","سابقه کوتاه است","فعلاً فقط روند کوتاه‌مدت بررسی شده است."),"no_model_alert":("good","هشدار فعالی دیده نشد","در داده فعلی نشانه کافی برای هشدار وجود ندارد.")}
kind,title,desc=status_map.get(state["status"],("neutral",state["status"],""))
st.markdown(f'<div class="card {kind}"><h3>{title}</h3><p>{desc}</p><p class="small">آخرین زمان داده: {metadata["latest_input_UTC"]} (UTC)</p></div>', unsafe_allow_html=True)

if state.get("row") is not None:
    row=state["row"]
    st.subheader("خلاصه قابل فهم از وضعیت")
    a,b,c,d=st.columns(4)
    a.metric("قند فعلی پس از صاف‌سازی",f"{row.Current_Glucose:.1f} mg/dL")
    b.metric("جهت حرکت در ۱۵ دقیقه",f"{row.ROC_15m:.2f}",help="عدد منفی یعنی قند در حال پایین آمدن است.")
    c.metric("چند درصد مسیر نزولی بوده؟",f"{row.Downward_Persistence:.0f}%",help="درصد خوانش‌های نزولی در یک ساعت اخیر.")
    d.metric("امتیاز هشدار داخلی",f"{row.Low_Risk_Score:.0f} از ۱۰۰",help="امتیاز فنی برای مقایسه روندهاست، نه احتمال پزشکی.")
    st.subheader("پیش‌بینی سه بازه زمانی")
    cols=st.columns(3)
    for col,h in zip(cols,HORIZONS):
        alert=_is_true(row[f"Alert_{h}m"] == 1); cls="danger" if alert else "good"; label="هشدار روند" if alert else "بدون هشدار"
        col.markdown(f'<div class="card {cls}"><h4>{h} دقیقه بعد</h4><h2>{row[f"Pred_Glucose_{h}m"]:.1f} <small>mg/dL</small></h2><p>{label}</p><p>مرز افت: ۷۰ mg/dL</p></div>',unsafe_allow_html=True)

with st.expander("چطور به این نتیجه رسیدیم؟", expanded=False):
    steps=[("۱. مرتب‌سازی","زمان‌ها پاک‌سازی و داده روی فاصله‌های منظم ۵ دقیقه‌ای قرار می‌گیرد."),("۲. کاهش نویز","نوسان‌های لحظه‌ای سنسور نرم می‌شوند."),("۳. اندازه‌گیری روند","سرعت، شتاب، فاصله تا ۷۰ و تداوم نزول محاسبه می‌شوند."),("۴. نگاه به آینده","روند فعلی برای ۳۰، ۴۵ و ۶۰ دقیقه جلو برده می‌شود."),("۵. جلوگیری از هشدار اشتباه","اگر قند در حال برگشت باشد، هشدار معمولی حذف می‌شود؛ افت سریع حفظ می‌شود."),("۶. یک افت = یک رویداد","هشدارهای پشت‌سرهم برای یک روند واحد یکی حساب می‌شوند.")]
    for name,text in steps: st.markdown(f'<div class="step"><b>{name}</b><br>{text}</div>',unsafe_allow_html=True)

fig=go.Figure(); fig.add_scatter(x=analysis.index,y=analysis["Raw"],name="عدد خام سنسور",connectgaps=False); fig.add_scatter(x=analysis.index,y=analysis["Current_Glucose"],name="روند صاف‌شده",connectgaps=False); fig.add_hline(y=70,line_color="red",line_dash="dash",annotation_text="مرز افت ۷۰"); fig.update_layout(height=420,hovermode="x unified",xaxis_title="زمان",yaxis_title="mg/dL",legend_title="توضیح نمودار")
st.plotly_chart(fig,use_container_width=True)

st.header("جدول تصمیم‌های مدل")
st.caption("هر ردیف یک خوانش پنج‌دقیقه‌ای است. ستون «پیام کاربر» نشان می‌دهد اگر همان لحظه اعلان ارسال شود، متن قابل نمایش چه خواهد بود.")
def _trend(v):
    if pd.isna(v): return "سابقه کافی نیست"
    if v < -0.10: return "نزولی"
    if v > 0.10: return "صعودی"
    return "تقریباً ثابت"
def _is_true(v):
    return pd.notna(v) and bool(v)
def _message(r):
    if _is_true(r.get("Notification_30m", False)):
        if _is_true(r.get("Fast_Drop_Risk", False)): return "افت سریع دیده شد؛ روند قند را فوراً بررسی کنید."
        if _is_true(r.get("Preventive_Alert", False)): return "روند نزولی پایدار است؛ قند را زودتر بررسی کنید."
        return "ادامه روند فعلی می‌تواند به افت قند منجر شود؛ روند را بررسی کنید."
    alert = r.get("Alert_30m", 0)
    if pd.notna(alert) and alert == 1: return "روند نیازمند پایش است؛ هنوز اعلان اصلی ارسال نشده است."
    return "اعلان فعالی نیست."
view = analysis.copy()
table = pd.DataFrame({
    "زمان": view.index.astype(str),
    "قند ثبت‌شده": view["Raw"].round(1),
    "قند هموارشده": view["Current_Glucose"].round(1),
    "روند ۱۵ دقیقه": view["ROC_15m"].map(_trend),
    "شیب (واحد/دقیقه)": view["ROC_15m"].round(2),
    "پیش‌بینی ۳۰ دقیقه": view["Pred_Glucose_30m"].round(1),
    "پیش‌بینی ۴۵ دقیقه": view["Pred_Glucose_45m"].round(1),
    "پیش‌بینی ۶۰ دقیقه": view["Pred_Glucose_60m"].round(1),
    "هشدار ۳۰ دقیقه": view["Alert_30m"].map({1:"بله",0:"خیر"}),
    "پیام کاربر": [_message(view.iloc[i]) for i in range(len(view))],
}, index=view.index)
st.dataframe(table, use_container_width=True, hide_index=True)

st.header("ارزیابی همین فایل")
if source == "ورود دستی":
    st.info("در ورود دستی آینده واقعی وجود ندارد؛ بنابراین درست/غلط بودن هشدار و امتیاز عملکرد تا زمانی که خوانش‌های بعدی وارد نشوند قابل محاسبه نیست.")
else:
    r30 = rows.loc[rows["Horizon_min"].eq(30)].iloc[0]
    e30 = event_summary.loc[event_summary["Horizon_min"].eq(30)].iloc[0]
    captured = int(e30["Captured"]); eligible = int(e30["Eligible_events"]); missed = int(e30["Missed_eligible"] + e30["No_opportunity"])
    false_alerts = int(r30["Notification_FP"])
    precision = float(r30["Precision"]) if pd.notna(r30["Precision"]) else 0.0
    recall = float(e30["Recall_eligible_events"]) if pd.notna(e30["Recall_eligible_events"]) else 0.0
    score = max(0, min(100, round(100 * (0.55 * recall + 0.35 * precision + 0.10 * max(0, 1 - false_alerts / max(1, int(r30["Notification_TP"] + false_alerts)))))))
    m1,m2,m3,m4,m5=st.columns(5)
    m1.metric("افت واقعی شناسایی‌شده", captured)
    m2.metric("افت قابل تشخیص از دست‌رفته", missed)
    m3.metric("هشدار کاذب", false_alerts)
    m4.metric("دقت هشدار", f"{precision*100:.1f}%")
    m5.metric("امتیاز این فایل", f"{score}/100")
    notes=[]
    if recall < .80: notes.append("بخشی از افت‌های واقعی قبل از هشدار از دست رفته‌اند؛ حساسیت باید بررسی شود.")
    if precision < .50: notes.append("هشدار کاذب زیاد است؛ برگشت قند و روندهای ناپایدار علت محتمل هستند.")
    if not notes: notes.append("تعادل حساسیت و دقت در این فایل قابل قبول است، اما نتیجه جایگزین اعتبارسنجی بالینی نیست.")
    st.markdown("**نقد خودکار فایل:** " + " ".join(notes))

with st.expander("فرمول‌ها و معنی شاخص‌ها"):
    st.markdown("""
**مرز افت:** قند ۷۰ mg/dL یا کمتر.  
**شیب:** تغییر متوسط قند در ۱۵ دقیقه اخیر؛ عدد منفی یعنی روند نزولی.  
**شتاب:** تغییر خودِ شیب؛ مثبت‌شدن آن می‌تواند نشانه کندشدن افت یا برگشت قند باشد.  
**تداوم نزول:** درصد تغییرات نزولی در حدود یک ساعت اخیر.  
**پیش‌بینی:** قند فعلی + شیب پایدار × تعداد دقیقه آینده.  
**افت واقعی:** کمترین قند ثبت‌شده در پنجره آینده به ۷۰ یا کمتر برسد.  
**حساسیت:** چند درصد افت‌های واقعی شناسایی شده‌اند.  
**دقت:** چند درصد اعلان‌ها واقعاً با افت همراه بوده‌اند.  
**هشدار کاذب:** اعلان صادر شده اما در پنجره بررسی افت واقعی رخ نداده است.  
**امتیاز فایل:** ترکیبی از حساسیت، دقت و جریمه هشدار کاذب است؛ احتمال پزشکی نیست.
    """)

with st.expander("جزئیات فنی و ارزیابی مدل (اختیاری)"):
    st.caption("این بخش برای بررسی فنی است. جدول‌های اصلی عمداً مخفی هستند تا استفاده روزمره ساده بماند.")
    st.write("خلاصه رویدادهای افت"); st.dataframe(event_summary,use_container_width=True,hide_index=True)
    st.write("ارزیابی خوانش‌ها"); st.dataframe(rows,use_container_width=True,hide_index=True)
    st.write("داده فنی آخرین ۳۰۰ ردیف"); st.dataframe(analysis.tail(300),use_container_width=True)
    st.json(metadata)
st.download_button("دانلود گزارش کامل Excel",excel_bytes(analysis,rows,metadata,events,event_summary),"CGM_recovery_analysis.xlsx","application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
