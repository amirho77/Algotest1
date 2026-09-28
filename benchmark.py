"""Descriptive benchmark; no optimization, fitting, threshold search or model selection."""
from pathlib import Path
import argparse
import hashlib
import json
import time
import warnings
import numpy as np
import pandas as pd
from core import prepare,predict,evaluate,evaluate_events
from config import InputConfig, ModelConfig, HORIZONS, VERSION


def run(paths, output, time_col=None, glucose_col=None):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    records=[];duplicates=[];seen={};sources=[]
    cfg=ModelConfig()
    started=time.monotonic()
    source_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')}
    freeze=Path(__file__).with_name('DESIGN_FREEZE.json')
    freeze_hash=hashlib.sha256(freeze.read_bytes()).hexdigest()
    for name in paths:
        path=Path(name);digest=hashlib.sha256(path.read_bytes()).hexdigest()
        if digest in seen:
            duplicates.append({'file':path.name,'duplicate_of':seen[digest]});continue
        seen[digest]=path.name
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore',message='Workbook contains no default style')
            data=pd.read_csv(path) if path.suffix.lower()=='.csv' else pd.read_excel(path)
        tcol=time_col or data.columns[0];gcol=glucose_col or data.columns[1]
        grid,meta=prepare(data,tcol,gcol,InputConfig())
        sources.append({'file':path.name,'sha256':digest,'metadata':meta})
        for method in ('baseline','robust','enhanced','high_precision'):
            a,rows=evaluate(predict(grid,cfg,method=method),cfg)
            _,events=evaluate_events(a,cfg)
            for row,event in zip(rows.to_dict('records'),events.to_dict('records')):
                records.append({'file':path.name,'method':method,**row,**event})
        print(f'Finished {path.name}',flush=True)
    total=[]
    for method in ('baseline','robust','enhanced','high_precision'):
        for h in HORIZONS:
            subset=[r for r in records if r['method']==method and r['Horizon_min']==h]
            fields=['TP','FP','FN','TN','Evaluable_rows','All_events','Eligible_events','Captured','Missed_eligible','No_opportunity','Notified_events','Notification_TP','Notification_FP','Notification_Unknown']
            item={k:sum(r[k] for r in subset) for k in fields}
            tp,fp,fn=item['TP'],item['FP'],item['FN']
            item.update({'method':method,'horizon':h,'precision':tp/(tp+fp) if tp+fp else None,
                         'row_recall':tp/(tp+fn) if tp+fn else None,
                         'event_recall':item['Captured']/item['Eligible_events'] if item['Eligible_events'] else None,
                         'capture_all_events':item['Captured']/item['All_events'] if item['All_events'] else None,
                         'false_notifications_per_evaluable_day':item['Notification_FP']/(item['Evaluable_rows']*5/1440) if item['Evaluable_rows'] else None})
            total.append(item)
    result={'version':VERSION,'design_freeze_sha256':freeze_hash,'source_hashes':source_hashes,
            'model_config':cfg.to_dict(),'sources':sources,'duplicates':duplicates,
            'per_file':records,'totals':total,'elapsed_seconds':time.monotonic()-started,
            'interpretation':'Previously seen exports, descriptive regression comparison only. No patient ID mapping, no independent external validation, no tuning performed. Both methods use the same v2 ingestion/quality policy; baseline formula is unchanged.'}
    # JSON null rather than non-standard NaN for undefined metrics.
    clean=json.loads(pd.Series([result]).to_json(orient='values'))[0]
    (output/'benchmark.json').write_text(json.dumps(clean,ensure_ascii=False,indent=2),encoding='utf-8')
    text=['# مقایسه توصیفی؛ بدون تنظیم پارامتر بر فایل‌های ارزیابی','',
          'این داده‌ها قبلاً دیده شده‌اند؛ این جدول آزمون تعمیم به بیمار جدید نیست. هر دو روش از ورودی‌خوان و سیاست کیفیت نسخه ۲ استفاده می‌کنند. به همین دلیل تفاوت جزئی با گزارش تاریخی قبلی ممکن است ناشی از کنارگذاشتن بازه ناتمام انتهای فایل باشد. baseline همان فرمول قبلی است و robust روش پیشنهادی چندپنجره‌ای است. پارامترها پس از مشاهده این جدول تغییر داده نشده‌اند.','',
          '|روش|افق|افت شناسایی‌شده / قابل ارزیابی|افت ازدست‌رفته|هشدار کاذب ردیفی|Precision ردیفی|رویداد دارای اعلان پس از فاصله اعلان‌ها|',
          '|---|---:|---:|---:|---:|---:|---:|']
    for r in total:
        precision=f"{100*r['precision']:.1f}٪" if r['precision'] is not None else 'تعریف‌نشده'
        text.append(f"|{r['method']}|{r['horizon']}|{r['Captured']} / {r['Eligible_events']}|{r['Missed_eligible']}|{r['FP']}|{precision}|{r['Notified_events']}|")
    text+=['','FP این جدول تعداد زمان‌های تصمیم است؛ چند FP می‌تواند مربوط به یک دوره هشدار باشد. تعداد اعلان و رویداد جداگانه محاسبه شده‌اند. فاصله ۱۵ دقیقه‌ای اعلان، فقط شبیه‌سازی سیاست نمایش است و اتصال زنده یا اعلان واقعی ارسال نمی‌شود.',
           '','کاهش هشدار کاذب همراه با افت حساسیت، به‌تنهایی اثبات برتری بالینی نیست. معیار پذیرش بالینی و هزینه افت ازدست‌رفته باید قبل از تنظیم مدل مشخص شوند.',
           '',f"تعداد فایل یکتا: {len(sources)}؛ فایل تکراری حذف‌شده: {len(duplicates)}.",
           '','فایل benchmark.json شامل شمارش‌های هر سنسور، هش کد/ورودی، سیاست‌ها و معیارهای کامل است. شناسه‌های فایل صرفاً محلی‌اند؛ این خروجی را بدون بررسی حریم خصوصی منتشر نکنید.']
    (output/'BENCHMARK_FA.md').write_text('\n'.join(text),encoding='utf-8')
    print(json.dumps(total,ensure_ascii=False),flush=True)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('files',nargs='+')
    parser.add_argument('--output',required=True)
    parser.add_argument('--time-col')
    parser.add_argument('--glucose-col')
    args=parser.parse_args()
    run(args.files,args.output,args.time_col,args.glucose_col)
