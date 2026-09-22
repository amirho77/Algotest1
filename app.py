from datetime import datetime, timedelta
import io
import re
import numpy as np
import openpyxl
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ==============================================================================
# ۱. تنظیمات صفحه و تم گرافیکی
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
    .status-badge-acute { background-color: #b91c1c; color: #ffffff; padding: 6px 12px; border-radius: 20px; font-weight: 700; animation: pulse 1s infinite; }
    .status-badge-near { background-color: #ffedd5; color: #c2410c; padding: 6px 12px; border-radius: 20px; font-weight: 700; }
    .status-badge-low { background-color: #581c87; color: #ffffff; padding: 6px 12px; border-radius: 20px; font-weight: 700; }
    .status-badge-buff { background-color: #fef3c7; color: #92400e; padding: 6px 12px; border-radius: 20px; font-weight: 700; }
    @keyframes pulse { 0% { transform: scale(1); } 50% { transform: scale(1.02); } 100% { transform: scale(1); } }
    </style>
""",
    unsafe_allow_html=True,
)

# ==============================================================================
# ۲. پیکربندی و هسته محاسباتی پیشرفته
# ==============================================================================
CFG = {
    "hypo_threshold": 70.0,
    "near_miss_threshold": 75.0,  # مرز تعریف رویداد نزدیک به افت
    "alert_margin": 5.0,  # حاشیه پیش‌بینی (75 mg/dL)
    "risk_score_threshold": 75.0,
    "tau_minutes": 45.0,
    "horizons": [30, 45, 60],
    "max_phys_drop_rate": -2.5,  # سقف فیزیولوژیک پاکسازی گلوکز
    "w_dist": 0.35,
    "w_vel": 0.35,
    "w_acc": 0.15,
    "w_persist": 0.15,
}


def calculate_metrics_for_stream(records):
    """محاسبه کینماتیک، بای‌پس سقوط آزاد و برچسب‌های بالینی چندسطحی"""
    if not records:
        return []

    df = pd.DataFrame(records)
    df["Smooth"] = df["Raw"].ewm(span=3, adjust=False).mean()

    n = len(df)
    processed = []

    for i in range(n):
        item = records[i].copy()
        item["Smooth"] = round(float(df.loc[i, "Smooth"]), 1)

        # دوره پر شدن بافر پایه
        if i < 11:
            item["ROC_15m"] = "-"
            item["Risk_Score"] = "-"
            item["Is_Acute_Drop"] = False
            for h in CFG["horizons"]:
                item[f"Pred_{h}m"] = "-"
                item[f"Alert_{h}m"] = False
                item[f"Outcome_{h}m"] = "-"
            item["Status"] = f"بافر ({i+1}/12)"
            processed.append(item)
            continue

        current_smooth = df.loc[i, "Smooth"]
        current_raw = df.loc[i, "Raw"]
        prev_raw = df.loc[i - 1, "Raw"] if i >= 1 else current_raw
        diff_5m = current_raw - prev_raw

        # ۱. نرخ تغییرات و مهار نویز
        raw_roc = (current_smooth - df.loc[i - 3, "Smooth"]) / 15.0
        item["ROC_15m"] = round(float(raw_roc), 2)
        roc_phys = np.clip(raw_roc, CFG["max_phys_drop_rate"], 3.0)

        # ۲. شتاب (مشتق دوم)
        if i >= 5:
            roc_prev = (df.loc[i - 2, "Smooth"] - df.loc[i - 5, "Smooth"]) / 15.0
            acc = (raw_roc - roc_prev) / 10.0
        else:
            acc = 0.0

        # ۳. تداوم نزول
        smooth_slice = df.loc[max(0, i - 11) : i, "Smooth"]
        persistence = float((smooth_slice.diff() < 0).mean() * 100.0)

        # ۴. نمره ریسک تجمیعی
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

        # ۵. پیش‌بینی کینماتیک با ترمز و بررسی شروط آلارم
        has_alert = False
        has_acute = False

        for h in CFG["horizons"]:
            tau = CFG["tau_minutes"]
            h_eff = tau * (1.0 - np.exp(-h / tau))

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

            # --- گام ۱: بای‌پس هوشمند سقوط آزاد تک‌گام (Acute Plunge Bypass) ---
            # اگر در ۵ دقیقه بیش از ۲۰ واحد افت رخ دهد یا قند <= ۸۰ با پیش‌بینی زیر ۷۰ باشد
            is_acute_plunge = (
                (raw_roc < -1.5)
                or (diff_5m <= -20.0)
                or (current_raw <= 80.0 and pred_g <= 70.0)
            )

            prev_raw_alert = (
                processed[i - 1].get(f"_raw_alert_{h}", False) if i >= 1 else False
            )
            confirmed_alert = bool(
                (raw_alert and prev_raw_alert) or (raw_alert and is_acute_plunge)
            )

            item[f"_raw_alert_{h}"] = raw_alert
            item[f"Pred_{h}m"] = round(pred_g, 1)
            item[f"Alert_{h}m"] = confirmed_alert

            if is_acute_plunge and raw_alert:
                has_acute = True
            if confirmed_alert:
                has_alert = True

        item["Is_Acute_Drop"] = has_acute

        # وضعیت بصری لحظه‌ای
        if current_raw <= CFG["hypo_threshold"]:
            item["Status"] = "🚨 افت لحظه‌ای"
        elif has_acute:
            item["Status"] = "⚡ هشدار سقوط پرشتاب"
        elif has_alert:
            item["Status"] = "⚠️ هشدار افت"
        else:
            item["Status"] = "✅ ایمن"

        processed.append(item)

    # --- گام ۲: محاسبه برچسب‌های بالینی چندسطحی (Ground Truth & Outcomes) ---
    # در صورتی که نقاط بعدی در بافر وجود داشته باشند (مثلاً هنگام آپلود فایل)
    for h in CFG["horizons"]:
        steps = int(h / 5)
        for i in range(n):
            if i + steps < n:
                window_vals = [processed[k]["Raw"] for k in range(i + 1, i + steps + 1)]
                min_next = min(window_vals)
                processed[i][f"Min_next_{h}m"] = round(min_next, 1)

                cur_r = processed[i]["Raw"]
                alt = processed[i][f"Alert_{h}m"]

                if cur_r <= CFG["hypo_threshold"]:
                    outcome = "AlreadyLow"
                else:
                    if alt:
                        if min_next <= CFG["hypo_threshold"]:
                            outcome = "TP"
                        elif min_next <= CFG["near_miss_threshold"]:
                            outcome = "Near-Miss"
                        else:
                            outcome = "FP"
                    else:
                        if min_next <= CFG["hypo_threshold"]:
                            outcome = "FN"
                        else:
                            outcome = "TN"
                processed[i][f"Outcome_{h}m"] = outcome
            else:
                processed[i][f"Min_next_{h}m"] = None
                processed[i][f"Outcome_{h}m"] = "-"

    return processed


# ==============================================================================
# ۳. تولید اکسل اختصاصی با شیت‌های Analysis و Clinical_Summary
# ==============================================================================
def create_multilevel_excel_report(processed_records):
    wb = openpyxl.Workbook()

    # استایل‌ها
    header_fill = PatternFill(start_color="1F497D", end_color="1F497D", fill_type="solid")
    header_font = Font(bold=True, color="FFFFFF", size=11)
    center_align = Alignment(horizontal="center", vertical="center")

    fills = {
        "TP": PatternFill(start_color="C6EFCE", end_color="C6EFCE", fill_type="solid"),
        "Near-Miss": PatternFill(start_color="FFE6CC", end_color="FFE6CC", fill_type="solid"),
        "FP": PatternFill(start_color="FFC7CE", end_color="FFC7CE", fill_type="solid"),
        "FN": PatternFill(start_color="FFEB9C", end_color="FFEB9C", fill_type="solid"),
        "TN": PatternFill(start_color="F2F2F2", end_color="F2F2F2", fill_type="solid"),
        "AlreadyLow": PatternFill(start_color="D9D9D9", end_color="D9D9D9", fill_type="solid"),
    }

    # ۱. شیت ردیفی Analysis
    ws_analysis = wb.active
    ws_analysis.title = "Analysis"
    ws_analysis.views.sheetView[0].showGridLines = True

    ordered_cols = [
        ("Step", "گام"),
        ("Time", "زمان"),
        ("Raw", "قند خام"),
        ("Smooth", "هموار (EMA)"),
        ("ROC_15m", "نرخ تغییر (ROC)"),
        ("Risk_Score", "نمره ریسک"),
        ("Is_Acute_Drop", "افت پرشتاب"),
        ("Min_next_30m", "حداقل ۳۰ دقیقه بعد"),
        ("Pred_30m", "پیش‌بینی ۳۰ دقیقه"),
        ("Alert_30m", "هشدار ۳۰ دقیقه"),
        ("Outcome_30m", "نتیجه بالینی ۳۰ دقیقه"),
        ("Min_next_45m", "حداقل ۴۵ دقیقه بعد"),
        ("Pred_45m", "پیش‌بینی ۴۵ دقیقه"),
        ("Alert_45m", "هشدار ۴۵ دقیقه"),
        ("Outcome_45m", "نتیجه بالینی ۴۵ دقیقه"),
        ("Min_next_60m", "حداقل ۶۰ دقیقه بعد"),
        ("Pred_60m", "پیش‌بینی ۶۰ دقیقه"),
        ("Alert_60m", "هشدار ۶۰ دقیقه"),
        ("Outcome_60m", "نتیجه بالینی ۶۰ دقیقه"),
        ("Status", "وضعیت سیستم"),
    ]

    for c_idx, (_, col_name) in enumerate(ordered_cols, start=1):
        cell = ws_analysis.cell(row=1, column=c_idx, value=col_name)
        cell.fill = header_fill
        cell.font = header_font
        cell.alignment = center_align

    for r_idx, r in enumerate(processed_records, start=2):
        for c_idx, (col_key, _) in enumerate(ordered_cols, start=1):
            val = r.get(col_key, None)
            cell = ws_analysis.cell(row=r_idx, column=c_idx)

            if col_key == "Is_Acute_Drop":
                cell.value = "بله" if val else "-"
            elif val is None or val == "-":
                cell.value = "-"
            elif isinstance(val, (int, float, np.integer, np.floating)):
                cell.value = round(float(val), 2)
            else:
                cell.value = str(val)

            cell.alignment = center_align

            # رنگ‌آمیزی شرطی ستون‌های نتایج بالینی
            if "Outcome_" in col_key and str(val) in fills:
                cell.fill = fills[str(val)]
                cell.font = Font(bold=True)

    ws_analysis.freeze_panes = "A2"
    ws_analysis.auto_filter.ref = ws_analysis.dimensions

    # ۲. شیت ارزیابی بالینی چندسطحی (Clinical_Summary)
    ws_summary = wb.create_sheet(title="Clinical_Summary")
    ws_summary.views.sheetView[0].showGridLines = True

    # محاسبه شاخص‌های رویداد و ردیف
    df_all = pd.DataFrame(processed_records)
    hypos = df_all[df_all["Raw"] <= CFG["hypo_threshold"]].copy()
    episodes = []
    if len(hypos) > 0 and "dt" in df_all.columns:
        hypos["time_diff"] = hypos["dt"].diff().dt.total_seconds()
        hypos["new_episode"] = (hypos["time_diff"].isna() | (hypos["time_diff"] > 1800)).astype(int)
        episodes = hypos[hypos["new_episode"] == 1]["dt"].tolist()

    total_episodes = len(episodes)

    summary_rows = [
        ["گزارش ارزیابی بالینی چندسطحی پیش‌بینی افت قند خون (CGM)", ""],
        ["شاخص کلی پایگاه داده", "مقدار"],
        ["تعداد کل خوانش‌های ۵ دقیقه‌ای", len(processed_records)],
        ["تعداد کل رویدادهای مستقل افت قند خون (<= 70 mg/dL)", total_episodes],
        [],
        [
            "افق پیش‌بینی",
            "پوشش رویداد (Capture Rate)",
            "میانگین پیش‌هشدار (دقیقه)",
            "دقت بالینی با Near-Miss (قند <= 75)",
            "دقت سخت‌گیرانه (قند <= 70)",
            "بازیابی (Recall)",
            "مثبت قطعی (TP)",
            "نزدیک افت (Near-Miss: 71-75)",
            "هشدار کاذب خالص (FP: >75)",
            "افت کشف‌نشده (FN)",
            "منفی درست (TN)",
        ],
    ]

    for h in CFG["horizons"]:
        eval_rows = [r for r in processed_records if r.get(f"Outcome_{h}m") not in ["AlreadyLow", "-", None]]
        tp = sum(1 for r in eval_rows if r[f"Outcome_{h}m"] == "TP")
        near_miss = sum(1 for r in eval_rows if r[f"Outcome_{h}m"] == "Near-Miss")
        fp = sum(1 for r in eval_rows if r[f"Outcome_{h}m"] == "FP")
        fn = sum(1 for r in eval_rows if r[f"Outcome_{h}m"] == "FN")
        tn = sum(1 for r in eval_rows if r[f"Outcome_{h}m"] == "TN")

        strict_prec = (tp / (tp + near_miss + fp) * 100.0) if (tp + near_miss + fp) > 0 else 0.0
        clin_prec = ((tp + near_miss) / (tp + near_miss + fp) * 100.0) if (tp + near_miss + fp) > 0 else 0.0
        recall = (tp / (tp + fn) * 100.0) if (tp + fn) > 0 else 0.0

        # محاسبه لیدتایم رویداد
        captured = 0
        lead_times = []
        if total_episodes > 0 and "dt" in df_all.columns:
            for ep in episodes:
                w_start = ep - pd.Timedelta(minutes=h)
                w_end = ep - pd.Timedelta(minutes=5)
                alts = df_all[(df_all["dt"] >= w_start) & (df_all["dt"] <= w_end) & (df_all[f"Alert_{h}m"] == True) & (df_all["Raw"] > 70.0)]
                if len(alts) > 0:
                    captured += 1
                    lead_times.append((ep - alts["dt"].min()).total_seconds() / 60.0)

        cap_str = f"{captured} از {total_episodes} ({(captured/total_episodes*100.0):.1f}%)" if total_episodes > 0 else "-"
        lead_str = f"{np.median(lead_times):.1f}" if lead_times else "-"

        summary_rows.append([
            f"{h} دقیقه",
            cap_str,
            lead_str,
            f"{clin_prec:.1f}%",
            f"{strict_prec:.1f}%",
            f"{recall:.1f}%",
            tp,
            near_miss,
            fp,
            fn,
            tn,
        ])

    summary_rows.extend([
        [],
        ["تعاریف و استانداردهای بالینی ارزیابی:", ""],
        ["۱. مثبت قطعی (TP):", "قند خون در بازه آینده به کمتر یا مساوی ۷۰ mg/dL رسیده است."],
        ["۲. نزدیک به افت (Near-Miss):", "هشدار داده شده و قند به ۷۱ تا ۷۵ mg/dL افت کرده است (مفید بالینی برای پیشگیری)."],
        ["۳. هشدار کاذب خالص (FP):", "هشدار شلیک شده اما قند خون بالای ۷۵ mg/dL مانده است."],
        ["۴. دقت بالینی (Clinical Precision):", "سهم هشدارهای به موقع که به افت یا لبه افت ختم شده‌اند: (TP + NearMiss) / کل هشدارها"],
    ])

    for r_idx, row_vals in enumerate(summary_rows, start=1):
        for c_idx, val in enumerate(row_vals, start=1):
            cell = ws_summary.cell(row=r_idx, column=c_idx, value=val)
            if r_idx in [1, 2, 6]:
                cell.font = Font(bold=True)
                cell.fill = PatternFill(start_color="DCE6F1", end_color="DCE6F1", fill_type="solid")

    for ws in [ws_analysis, ws_summary]:
        for col in ws.columns:
            col_letter = get_column_letter(col[0].column)
            max_len = max(len(str(c.value or "")) for c in col[:50])
            ws.column_dimensions[col_letter].width = max(max_len + 3, 14)

    buffer = io.BytesIO()
    wb.save(buffer)
    return buffer.getvalue()


# ==============================================================================
# ۴. مدیریت State و سایدبار
# ==============================================================================
if "records" not in st.session_state:
    st.session_state.records = []

if "start_time" not in st.session_state:
    st.session_state.start_time = datetime.now().replace(second=0, microsecond=0)

with st.sidebar:
    st.title("🩸 کنترل‌پنل CGM")
    st.caption("الگوریتم کینماتیک با بای‌پس سقوط آزاد و تفکیک بالینی")
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
                sim_t = st.session_state.start_time + timedelta(minutes=cur_step * 5)
                st.session_state.records.append(
                    {
                        "Step": cur_step + 1,
                        "Time": sim_t.strftime("%H:%M"),
                        "Raw": float(new_glucose),
                        "dt": sim_t,
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
                    return 39.0 if s == "LOW" else (401.0 if s == "HIGH" else float(v))

                df_up["Raw"] = df_up[gluc_col].apply(parse_g)
                c_times = df_up[time_col].astype(str).apply(lambda x: re.sub(r"\s*GMT.*", "", x).strip())
                dt_series = pd.to_datetime(c_times, format="%d-%m-%Y %H:%M", errors="coerce")
                if dt_series.isna().all():
                    dt_series = pd.to_datetime(c_times, dayfirst=True, errors="coerce")

                df_up["Time_dt"] = dt_series
                df_up = df_up.dropna(subset=["Time_dt", "Raw"]).sort_values("Time_dt")

                st.session_state.records = []
                for i, row in enumerate(df_up.itertuples()):
                    st.session_state.records.append(
                        {
                            "Step": i + 1,
                            "Time": row.Time_dt.strftime("%d-%m %H:%M"),
                            "Raw": float(row.Raw),
                            "dt": row.Time_dt,
                        }
                    )
                st.success(f"{len(df_up)} رکورد بارگذاری شد.")
            except Exception as e:
                st.error(f"خطا در خواندن فایل: {e}")

# ==============================================================================
# ۵. صفحه اصلی و مانیتورینگ
# ==============================================================================
st.title("🩸 مانیتورینگ هوشمند پیش‌بینی افت قند خون")

processed_data = calculate_metrics_for_stream(st.session_state.records)

if not processed_data:
    st.info("👈 داده‌ای ثبت نشده است. عددی وارد کنید یا فایل اکسل را آپلود نمایید.")
else:
    last = processed_data[-1]
    count = len(processed_data)

    c1, c2, c3, c4 = st.columns([1, 1, 1, 1.2])
    with c1:
        st.markdown(
            f"""<div class="metric-card">
            <div style="color: #64748b; font-size: 13px;">آخرین زمان ثبت</div>
            <div style="font-size: 26px; font-weight: 700; color: #1e293b;">{last['Time']}</div>
            <div style="color: #0284c7; font-size: 12px;">گام #{count}</div>
        </div>""",
            unsafe_allow_html=True,
        )

    with c2:
        smooth_val = last["Smooth"] if last["Smooth"] != "-" else last["Raw"]
        st.markdown(
            f"""<div class="metric-card">
            <div style="color: #64748b; font-size: 13px;">قند لحظه‌ای (خام / هموار)</div>
            <div style="font-size: 26px; font-weight: 700; color: #1e293b;">{int(last['Raw'])} <span style="font-size: 16px; color: #64748b;">({smooth_val})</span></div>
            <div style="color: #64748b; font-size: 12px;">mg/dL</div>
        </div>""",
            unsafe_allow_html=True,
        )

    with c3:
        roc_disp = f"{last['ROC_15m']} mg/dL/min" if last["ROC_15m"] != "-" else "-"
        risk_disp = f"{last['Risk_Score']} / 100" if last["Risk_Score"] != "-" else "-"
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
                "status-badge-acute"
                if "پرشتاب" in st_text
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
        )
        st.markdown(
            f"""<div class="metric-card">
            <div style="color: #64748b; font-size: 13px; margin-bottom: 6px;">وضعیت هشدار زودهنگام</div>
            <span class="{badge_cls}">{st_text}</span>
        </div>""",
            unsafe_allow_html=True,
        )

    tab_chart, tab_table, tab_clinical = st.tabs(
        ["📈 نمودار ترند", "📋 جدول داده‌ها و خروجی", "📊 ارزیابی بالینی چندسطحی"]
    )

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
        fig.add_hline(
            y=75,
            line_dash="dot",
            line_color="#f97316",
            annotation_text="مرز Near-Miss (75)",
        )
        fig.update_layout(
            height=420,
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
                    "سقوط پرشتاب": "⚡ بله" if r.get("Is_Acute_Drop") else "-",
                    "پیش‌بینی ۳۰ دقیقه": f"⚠️ {r['Pred_30m']}" if r.get("Alert_30m") else r["Pred_30m"],
                    "نتیجه ۳۰ دقیقه": r.get("Outcome_30m", "-"),
                    "پیش‌بینی ۴۵ دقیقه": f"⚠️ {r['Pred_45m']}" if r.get("Alert_45m") else r["Pred_45m"],
                    "نتیجه ۴۵ دقیقه": r.get("Outcome_45m", "-"),
                    "پیش‌بینی ۶۰ دقیقه": f"⚠️ {r['Pred_60m']}" if r.get("Alert_60m") else r["Pred_60m"],
                    "نتیجه ۶۰ دقیقه": r.get("Outcome_60m", "-"),
                    "وضعیت": r["Status"],
                }
            )
        st.dataframe(pd.DataFrame(display_rows), use_container_width=True, hide_index=True)

    with tab_clinical:
        st.subheader("تحلیل تفکیکی هشدارهای بالینی (چندسطحی)")
        st.caption("تمایز میان افت‌های قطعی (TP)، موارد نزدیک به افت مفید (Near-Miss: 71-75) و هشدارهای اشتباه خالص (FP: >75)")

        clin_summary = []
        for h in [30, 45, 60]:
            eval_rows = [r for r in processed_data if r.get(f"Outcome_{h}m") not in ["AlreadyLow", "-", None]]
            tp = sum(1 for r in eval_rows if r[f"Outcome_{h}m"] == "TP")
            nm = sum(1 for r in eval_rows if r[f"Outcome_{h}m"] == "Near-Miss")
            fp = sum(1 for r in eval_rows if r[f"Outcome_{h}m"] == "FP")
            fn = sum(1 for r in eval_rows if r[f"Outcome_{h}m"] == "FN")
            tn = sum(1 for r in eval_rows if r[f"Outcome_{h}m"] == "TN")

            strict_p = (tp / (tp + nm + fp) * 100.0) if (tp + nm + fp) > 0 else 0.0
            clin_p = ((tp + nm) / (tp + nm + fp) * 100.0) if (tp + nm + fp) > 0 else 0.0
            rec = (tp / (tp + fn) * 100.0) if (tp + fn) > 0 else 0.0

            clin_summary.append({
                "افق": f"{h} دقیقه",
                "دقت بالینی (<=75)": f"{clin_p:.1f}%",
                "دقت سخت‌گیرانه (<=70)": f"{strict_p:.1f}%",
                "بازیابی (Recall)": f"{rec:.1f}%",
                "مثبت قطعی (TP)": tp,
                "نزدیک به افت (Near-Miss)": nm,
                "هشدار کاذب خالص (FP)": fp,
                "افت جامانده (FN)": fn,
                "منفی درست (TN)": tn,
            })

        st.table(pd.DataFrame(clin_summary))

    # دکمه دانلود اکسل دو شیته کامل
    excel_bytes = create_multilevel_excel_report(processed_data)
    st.download_button(
        label="📥 دانلود فایل اکسل جامع (همراه با شیت تحلیل بالینی دوگانه)",
        data=excel_bytes,
        file_name="CGM_Multilevel_Clinical_Evaluation.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
