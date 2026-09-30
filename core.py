"""Public API; no Streamlit dependency in the computation layer."""
import io
import json
import pandas as pd
from config import VERSION, HORIZONS, InputConfig, ModelConfig
from data import prepare
from model import predict, latest_state
from evaluation import evaluate, evaluate_events


def analyze(frame, time_col, glucose_col, input_config=None, model_config=None, *, as_of=None, stream_columns=(), method="guarded"):
    cfg=model_config or ModelConfig()
    grid, metadata=prepare(frame,time_col,glucose_col,input_config,as_of=as_of,stream_columns=stream_columns)
    predictions=predict(grid,cfg,method=method)
    analysis, rows=evaluate(predictions,cfg)
    events, event_summary=evaluate_events(analysis,cfg)
    metadata.update({"version":VERSION,"method":method,"model_config":cfg.to_dict(),
                     "scenario_band":"Uncalibrated model spread; NOT a confidence interval or probability",
                     "v3_alert":"Guarded is the product default: normal alerts require causal trend agreement and dual confirmation, while rapid descent keeps an immediate rescue path. Enhanced and recovery remain available for comparison.",
                     "primary_horizon_minutes":cfg.primary_horizon,
                     "low_threshold_mg_dL":cfg.threshold})
    return analysis, rows, events, event_summary, metadata


def predict_latest(frame,time_col,glucose_col,*,as_of,input_config=None,model_config=None,stream_columns=()):
    """as_of is mandatory. Future rows and unfinished bins cannot enter a forecast."""
    cfg=model_config or ModelConfig()
    grid,metadata=prepare(frame,time_col,glucose_col,input_config,as_of=as_of,stream_columns=stream_columns)
    predictions=predict(grid,cfg,method="guarded")
    return latest_state(predictions,metadata,cfg), metadata


def excel_bytes(analysis, row_summary, metadata, events=None, event_summary=None):
    """Application export; actual computation lives in data/model/evaluation modules."""
    from openpyxl.styles import Font, PatternFill
    def excel_safe(df):
        result=df.copy()
        for col in result:
            if isinstance(result[col].dtype,pd.DatetimeTZDtype):
                result[col]=result[col].dt.tz_convert("UTC").dt.tz_localize(None)
        return result
    sheets={"Row_Evaluation":row_summary}
    if event_summary is not None:
        sheets["Event_Evaluation"]=event_summary
    if events is not None:
        sheets["Events"]=events
    sheets["Analysis"]=analysis.reset_index()
    notes={**metadata,"excel_time":"All exported datetimes are UTC without timezone for Excel compatibility",
           "event_definition":"A low episode starts at an observed value <=threshold and ends only after the configured number of consecutive observed values are >= threshold plus the recovery margin; gaps split an episode.",
           "current_low":"Independent of forecast readiness; LOW/HIGH have no invented numeric values",
           "notification":"Cooldown affects notifications only; model Alert and current-low flags are retained"}
    sheets["Metadata"]=pd.DataFrame([(k,json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list)) else v) for k,v in notes.items()],columns=["Setting","Value"])
    buffer=io.BytesIO()
    with pd.ExcelWriter(buffer,engine="openpyxl") as writer:
        for name,df in sheets.items():
            excel_safe(df).to_excel(writer,sheet_name=name,index=False)
        for sheet in writer.book:
            sheet.freeze_panes="A2"
            sheet.auto_filter.ref=sheet.dimensions
            for cell in sheet[1]:
                cell.font=Font(bold=True,color="FFFFFF")
                cell.fill=PatternFill("solid",fgColor="164E63")
                sheet.column_dimensions[cell.column_letter].width=26
            for row in sheet:
                for cell in row:
                    if cell.data_type=="f":
                        cell.data_type="s"
    return buffer.getvalue()
