"""Fair descriptive benchmark of previous, current, and Farir Model 1 rules."""
from dataclasses import replace
from pathlib import Path
import argparse
import hashlib
import json
import math
import time
import warnings
import pandas as pd

from config import InputConfig, ModelConfig, VERSION
from data import prepare
from model import predict
from site_compat import predict_site, evaluate_alert_policy


POLICIES = (
    ("previous_guarded_v3_5", "نسخه قبلی: Guarded v3.5"),
    ("current_guarded_v3_6", "نسخه فعلی: Guarded v3.6"),
    ("farir_model_1", "Farir: مدل ۱"),
)


def _json_safe(value):
    """Convert undefined numeric metrics to JSON null, never non-standard NaN."""
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _load(path):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Workbook contains no default style")
        return pd.read_csv(path) if path.suffix.lower() == ".csv" else pd.read_excel(path)


def _score_one_stream(grid, cfg):
    previous = predict(grid, cfg, method="guarded_legacy")
    current = predict(grid, cfg, method="guarded")
    farir = predict_site(grid, slope_threshold=-0.1, horizon=10, cooldown=30)
    policies = {
        "previous_guarded_v3_5": (previous, "Alert_30m", "Notification_30m"),
        "current_guarded_v3_6": (current, "Alert_30m", "Notification_30m"),
        "farir_model_1": (farir, "Site_Alert", "Site_Notification"),
    }
    return {
        name: evaluate_alert_policy(prediction, alert_column, notification_column, threshold=70, verify_minutes=30)
        for name, (prediction, alert_column, notification_column) in policies.items()
    }


def _aggregate(records):
    totals=[]
    for name, label in POLICIES:
        subset=[item for item in records if item["policy"] == name]
        fields=("events", "captured_events", "alerts", "correct_alerts", "wrong_alerts", "observed_days")
        item={field: sum(row[field] for row in subset) for field in fields}
        item.update({
            "policy": name,
            "label_fa": label,
            "missed_events": item["events"] - item["captured_events"],
            "event_recall": item["captured_events"] / item["events"] if item["events"] else None,
            "precision": item["correct_alerts"] / item["alerts"] if item["alerts"] else None,
            "false_alerts_per_day": item["wrong_alerts"] / item["observed_days"] if item["observed_days"] else None,
        })
        totals.append(item)
    return totals


def run(paths, output, time_col=None, glucose_col=None):
    output=Path(output)
    output.mkdir(parents=True, exist_ok=True)
    cfg=replace(ModelConfig(), notification_cooldown_minutes=30)
    records=[]; sources=[]; duplicates=[]; seen={}
    started=time.monotonic()
    for name in paths:
        path=Path(name)
        digest=hashlib.sha256(path.read_bytes()).hexdigest()
        if digest in seen:
            duplicates.append({"file":path.name, "duplicate_of":seen[digest]})
            continue
        seen[digest]=path.name
        data=_load(path)
        tcol=time_col or data.columns[0]
        gcol=glucose_col or data.columns[1]
        grid, metadata=prepare(data, tcol, gcol, InputConfig())
        observed_days=float((grid.Observed & grid.Bin_Complete).sum()) * 5 / 1440
        sources.append({"file":path.name, "sha256":digest, "metadata":metadata})
        for policy, score in _score_one_stream(grid, cfg).items():
            records.append({"file":path.name, "policy":policy, "observed_days":observed_days, **score})
        print(f"Finished {path.name}", flush=True)
    totals=_aggregate(records)
    protocol={
        "threshold_mg_dL":70,
        "event_definition":"first reading in a consecutive strict glucose <70 mg/dL episode",
        "event_capture":"at least one alert in the 30 minutes before event onset",
        "alert_correctness":"strict glucose <70 mg/dL within 30 minutes after notification",
        "notification_cooldown_minutes":30,
        "farir_model_1":"5-point linear regression, 10-minute forecast, slope <= -0.1 mg/dL/min",
        "current_vs_previous":"same Guarded policy except v3.6 adds the constrained accelerating-drop rescue rule",
    }
    result={"version":VERSION, "protocol":protocol, "model_config":cfg.to_dict(),
            "sources":sources, "duplicates":duplicates, "per_file":records, "totals":totals,
            "elapsed_seconds":time.monotonic()-started,
            "interpretation":"Descriptive development-data comparison only. Parameters were not tuned by this script; results are not an independent clinical validation."}
    (output / "benchmark.json").write_text(json.dumps(_json_safe(result), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    lines=["# بنچمارک مشترک سه سیاست هشدار", "", "هر سه سیاست با مرز ۷۰، تعریف سخت‌گیرانهٔ افت (`<70`)، پنجرهٔ ۳۰ دقیقه و فاصلهٔ اعلان ۳۰ دقیقه سنجیده شده‌اند.", "", "|سیاست|افت گرفته‌شده / کل|افت ازدست‌رفته|هشدار|هشدار درست|هشدار نادرست|دقت اعلان|هشدار نادرست در روز|", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in totals:
        recall="—" if row["event_recall"] is None else f"{row['event_recall']*100:.1f}٪"
        precision="—" if row["precision"] is None else f"{row['precision']*100:.1f}٪"
        false_day="—" if row["false_alerts_per_day"] is None else f"{row['false_alerts_per_day']:.2f}"
        lines.append(f"|{row['label_fa']}|{row['captured_events']} / {row['events']} ({recall})|{row['missed_events']}|{row['alerts']}|{row['correct_alerts']}|{row['wrong_alerts']}|{precision}|{false_day}|")
    lines += ["", "`Farir: مدل ۱` بازسازی مستقلی از فرمول و تنظیمات منتشرشده در صفحهٔ بنچمارک Farir است. پیش از استفاده، بازسازی روی فایل `AA240307VW` با صفحهٔ سایت تطبیق داده شد: ۲۸ رویداد، ۲۰ رویداد شناسایی‌شده، ۲۸ هشدار و ۱۵ هشدار درست.", "", "این گزارش، دادهٔ توسعه را آزمون مستقل معرفی نمی‌کند. برای انتخاب نسخهٔ MVP باید همین پروتکل روی فایل‌های بیمارانی اجرا شود که در طراحی قواعد حضور نداشته‌اند."]
    (output / "BENCHMARK_FA.md").write_text("\n".join(lines), encoding="utf-8")
    # PowerShell sessions may still use a legacy console encoding; the files
    # above preserve Persian text, while console progress stays portable.
    print(json.dumps(totals, ensure_ascii=True), flush=True)
    return result


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="+")
    parser.add_argument("--output", required=True)
    parser.add_argument("--time-col")
    parser.add_argument("--glucose-col")
    args=parser.parse_args()
    run(args.files, args.output, args.time_col, args.glucose_col)
