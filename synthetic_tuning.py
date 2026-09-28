"""Tune preventive-rule parameters on synthetic QA streams only.

The selected values are a starting point. Real-file evaluation remains the
acceptance gate and must be run after tuning.
"""
from itertools import product
import numpy as np
import pandas as pd
from synthetic import make_scenario, SCENARIOS
from data import prepare
from model import predict
from config import ModelConfig, HORIZONS
from evaluation import evaluate, evaluate_events


def _stream(kind, seed):
    f = make_scenario(kind, n=72, seed=seed)
    return prepare(f, "time", "glucose")[0]


def score(params):
    cfg = ModelConfig(near_low_margin=params[0], recurrence_margin=params[1],
                      fast_drop_max_distance=params[2], fast_drop_rate=params[3])
    recall = []; fp = []; eligible = 0
    for kind in SCENARIOS:
        for seed in range(4):
            a, _ = evaluate(predict(_stream(kind, seed), cfg, method="enhanced"), cfg)
            _, es = evaluate_events(a, cfg)
            for h in HORIZONS:
                row = a.loc[a[f"Outcome_{h}m"].isin(["TP", "FP"])]
                tp = int((row[f"Outcome_{h}m"] == "TP").sum())
                fpos = int((row[f"Outcome_{h}m"] == "FP").sum())
                summary = es.loc[es.Horizon_min.eq(h)].iloc[0]
                recall.append(float(summary.Capture_all_events) if pd.notna(summary.Capture_all_events) else 0.0)
                fp.append(fpos / max(len(row), 1))
                eligible += len(row)
    event_recall = float(np.mean(recall))
    false_rate = float(np.mean(fp))
    # Event recall is primary; false-rate is a tie breaker.
    return event_recall - 0.35 * false_rate, event_recall, false_rate


if __name__ == "__main__":
    best = []
    for p in product((3.0, 5.0), (5.0, 10.0), (20.0, 30.0), (0.8, 1.0, 1.2)):
        best.append((score(p), p))
    for result, params in sorted(best, reverse=True)[:10]:
        print({"params": params, "objective": result[0], "event_recall": result[1], "false_rate": result[2]})
