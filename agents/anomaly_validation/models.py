from dataclasses import dataclass

@dataclass(frozen=True)
class ValidationConfig:
    z_threshold: float = 3.5
    robust_z_threshold: float = 4.0
    min_history_points: int = 28
    low_confidence_threshold: float = 0.60
    forecast_deviation_warn_pct: float = 50.0
    forecast_deviation_fail_pct: float = 100.0
    stockout_ratio_high: float = 0.75
    stockout_ratio_medium: float = 0.95
    overstock_ratio_high: float = 2.5
    overstock_ratio_medium: float = 1.75
    max_missing_rate: float = 0.05
