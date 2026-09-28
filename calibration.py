"""Precision-gated threshold selection for patient-disjoint validation.

This module deliberately reports an empirical lower confidence bound rather
than pretending a heuristic score is a probability. A gate is deployable only
when it passes on an independent validation set.
"""
from math import sqrt
import numpy as np


def wilson_lower_bound(tp, fp, z=1.96):
    n = int(tp + fp)
    if n == 0:
        return 0.0
    p = tp / n
    den = 1 + z*z/n
    return (p + z*z/(2*n) - z*sqrt(p*(1-p)/n + z*z/(4*n*n))) / den


def choose_threshold(scores, labels, target_precision=0.95, min_alerts=20):
    """Choose the highest-recall score threshold with Wilson precision >= target.

    ``scores`` must be out-of-sample and higher means more risk. The result is
    marked deployable only if at least ``min_alerts`` positives are observed.
    """
    s = np.asarray(scores, dtype=float)
    y = np.asarray(labels, dtype=bool)
    ok = np.isfinite(s)
    s, y = s[ok], y[ok]
    candidates = np.unique(s[np.argsort(s)[::-1]])
    best = None
    for t in candidates:
        pred = s >= t
        tp = int((pred & y).sum()); fp = int((pred & ~y).sum())
        if tp + fp < min_alerts:
            continue
        lb = wilson_lower_bound(tp, fp)
        if lb >= target_precision:
            recall = tp / max(int(y.sum()), 1)
            if best is None or recall > best["recall"]:
                best = {"threshold": float(t), "tp": tp, "fp": fp,
                        "precision": tp/(tp+fp), "precision_lower_95": lb,
                        "recall": recall, "deployable": True}
    return best or {"threshold": None, "deployable": False,
                    "reason": "No independent validation threshold met target precision."}
