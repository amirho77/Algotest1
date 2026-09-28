"""Guardrails for future fitting/calibration; this baseline has no training phase."""
import pandas as pd


def assert_patient_disjoint(train_ids, validation_ids, test_ids):
    groups=[set(x) for x in (train_ids,validation_ids,test_ids)]
    if any(any(pd.isna(x) or str(x).strip()=="" for x in g) for g in groups):
        raise ValueError("Every partition needs real patient identifiers.")
    if not all(groups):
        raise ValueError("Training, validation and test groups must all be nonempty.")
    if groups[0]&groups[1] or groups[0]&groups[2] or groups[1]&groups[2]:
        raise ValueError("Patient leakage across partitions.")


def assert_purged_boundary(last_train_decision, first_test_decision, lookback_minutes=60, horizon_minutes=60):
    left,right=pd.Timestamp(last_train_decision),pd.Timestamp(first_test_decision)
    if left.tzinfo is None or right.tzinfo is None:
        raise ValueError("Split timestamps must be timezone-aware.")
    if lookback_minutes<0 or horizon_minutes<0:
        raise ValueError("Window lengths cannot be negative.")
    if left+pd.Timedelta(minutes=horizon_minutes)>=right-pd.Timedelta(minutes=lookback_minutes):
        raise ValueError("Feature/label windows overlap across the time split.")
