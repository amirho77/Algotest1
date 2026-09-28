"""Deterministic synthetic CGM scenarios for QA and stress testing only.

Synthetic streams must never be mixed into the clinical performance report.
They exercise noise, gaps, reversals and abrupt drops that are sparse in the
available exports.
"""
import numpy as np
import pandas as pd


def make_scenario(kind="gradual_drop", n=48, seed=7):
    if n < 16:
        raise ValueError("n must provide at least one hour of history")
    rng = np.random.default_rng(seed)
    t = pd.date_range("2026-01-01", periods=n, freq="5min", tz="UTC")
    x = np.arange(n)
    if kind == "gradual_drop":
        y = 150 - 1.7 * x
    elif kind == "noisy_drop":
        y = 150 - 1.7 * x + rng.normal(0, 5, n)
    elif kind == "abrupt_drop":
        y = np.full(n, 145.0); y[max(0, n-6):] -= np.arange(min(6, n))*16
    elif kind == "recovery":
        y = np.r_[150 - 2.0*np.arange(n//2), 70 + 2.5*np.arange(n-n//2)]
    elif kind == "flat":
        y = np.full(n, 120.0) + rng.normal(0, 2, n)
    elif kind == "near_low":
        y = np.r_[np.full(n-8, 110.0), np.linspace(85, 68, 8)]
    else:
        raise ValueError("Unknown scenario")
    return pd.DataFrame({"time": t, "glucose": np.clip(y, 35, 300)})


SCENARIOS = ("gradual_drop", "noisy_drop", "abrupt_drop", "recovery", "flat", "near_low")
