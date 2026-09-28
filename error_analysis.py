import warnings,pandas as pd,numpy as np,hashlib,json
from pathlib import Path
from data import prepare
from config import InputConfig,ModelConfig
from model import predict
ROOT=Path(__file__).parent; seen=set();paths=[]
for line in open(ROOT/'current_paths.txt'):
 p=Path(line.strip());h=hashlib.sha256(p.read_bytes()).hexdigest()
 if h not in seen:seen.add(h);paths.append(p)
rows=[]
for p in paths:
 with warnings.catch_warnings():warnings.simplefilter('ignore');d=pd.read_excel(p)
 g,_=prepare(d,d.columns[0],d.columns[1],InputConfig());a=predict(g,ModelConfig(),method='enhanced'); raw=a.Raw.to_numpy(float); idx=a.index
 notices=np.flatnonzero(a.Notification_30m.to_numpy(bool)&(raw>=70))
 for i in notices:
  fut=raw[i+1:i+7]; fut=fut[np.isfinite(fut)]
  if len(fut)==0: continue
  actual_low=bool(np.nanmin(fut)<=70)
  if actual_low: continue
  first5=float(fut[0]); last=float(fut[-1]); rebound=(last-first5)>=5
  max_up=float(np.nanmax(np.diff(fut))) if len(fut)>1 else 0
  if rebound or max_up>=5: cat='reversal_after_alert'
  elif np.nanmin(fut)<=80: cat='near_miss_70_80'
  elif float(a.iloc[i].Downward_Persistence)>=70: cat='persistent_but_no_hypo'
  elif float(a.iloc[i].Residual_Spread)>=12: cat='high_noise'
  else: cat='flat_or_slow_recovery'
  rows.append({'file':p.name,'time_utc':str(idx[i]),'category':cat,'current':float(raw[i]),'pred30':float(a.iloc[i].Pred_Glucose_30m),'roc15':float(a.iloc[i].ROC_15m),'forecast_slope':float(a.iloc[i].Forecast_Slope),'persistence':float(a.iloc[i].Downward_Persistence),'spread':float(a.iloc[i].Residual_Spread),'future_min_30':float(np.nanmin(fut)),'future_last_30':last})
out=pd.DataFrame(rows); out.to_csv(ROOT/'false_alerts_enhanced.csv',index=False)
summary=out.groupby('category').agg(count=('category','size'),mean_current=('current','mean'),mean_pred30=('pred30','mean'),mean_roc15=('roc15','mean'),mean_persistence=('persistence','mean'),mean_future_min=('future_min_30','mean')).sort_values('count',ascending=False)
summary.to_csv(ROOT/'false_alert_summary.csv'); print(json.dumps({'false_alerts':len(out),'categories':summary.reset_index().to_dict('records')},ensure_ascii=False))
