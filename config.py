"""Engineering defaults, not patient-fitted or clinically calibrated constants."""
from dataclasses import dataclass, asdict
import math

HORIZONS = (10, 15, 20, 25, 30)
VERSION = "3.5.0-guarded-alert-policy"


@dataclass(frozen=True)
class ModelConfig:
    threshold: float = 70.0
    uncertainty_floor: float = 5.0
    notification_cooldown_minutes: int = 15
    stale_after_minutes: int = 10
    near_low_margin: float = 5.0
    recurrence_margin: float = 5.0
    fast_drop_max_distance: float = 20.0
    fast_drop_rate: float = 1.2
    preventive_confirmations: int = 2
    strict_confirmations: int = 2
    guarded_confirmations: int = 2
    guarded_approach_margin: float = 12.0
    guarded_approach_rate: float = 0.4
    event_recovery_margin: float = 5.0
    event_recovery_confirmations: int = 2
    primary_horizon: int = 20
    target_precision: float = 0.95

    def __post_init__(self):
        if not math.isfinite(self.threshold) or self.threshold <= 0:
            raise ValueError("Threshold must be positive and finite.")
        if not math.isfinite(self.uncertainty_floor) or self.uncertainty_floor < 0:
            raise ValueError("Uncertainty floor must be nonnegative and finite.")
        if self.notification_cooldown_minutes < 0 or self.stale_after_minutes < 0:
            raise ValueError("Timing settings cannot be negative.")
        for value in (self.near_low_margin, self.recurrence_margin, self.fast_drop_max_distance, self.fast_drop_rate):
            if not math.isfinite(value) or value <= 0:
                raise ValueError("Guard settings must be positive and finite.")
        if self.strict_confirmations < 1:
            raise ValueError("strict_confirmations must be at least one.")
        if self.preventive_confirmations < 1:
            raise ValueError("preventive_confirmations must be at least one.")
        if self.guarded_confirmations < 1:
            raise ValueError("guarded_confirmations must be at least one.")
        if self.event_recovery_confirmations < 1:
            raise ValueError("event_recovery_confirmations must be at least one.")
        for value in (self.guarded_approach_margin, self.guarded_approach_rate, self.event_recovery_margin):
            if not math.isfinite(value) or value <= 0:
                raise ValueError("Guarded approach settings must be positive and finite.")
        if self.primary_horizon not in HORIZONS:
            raise ValueError(f"primary_horizon must be one of {HORIZONS}.")
        if not math.isfinite(self.target_precision) or not 0 < self.target_precision <= 1:
            raise ValueError("target_precision must be in (0, 1].")

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class InputConfig:
    timezone: str = "Asia/Tehran"
    unit: str = "mg/dL"
    date_format: str | None = None
    numeric_time_unit: str | None = None  # excel, s, ms; never guess
    sensor_lower: float | None = None  # specified in mg/dL
    sensor_upper: float | None = None
    max_days: int = 366

    def __post_init__(self):
        if self.unit not in ("mg/dL", "mmol/L"):
            raise ValueError("Unit must be explicitly mg/dL or mmol/L.")
        if self.numeric_time_unit not in (None, "excel", "s", "ms"):
            raise ValueError("Numeric time unit must be excel, s or ms.")
        for bound in (self.sensor_lower, self.sensor_upper):
            if bound is not None and (not math.isfinite(bound) or bound <= 0):
                raise ValueError("Sensor bounds must be positive finite mg/dL values.")
        if self.sensor_lower is not None and self.sensor_upper is not None and self.sensor_lower >= self.sensor_upper:
            raise ValueError("Lower sensor bound must be less than upper bound.")
        if self.max_days <= 0 or self.max_days > 366:
            raise ValueError("Upload limit must be between 1 and 366 days.")
