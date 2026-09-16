import numpy as np
import pandas as pd

GROUP_KEYS = ["restaurant_id", "menu_item_id"]


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create leakage-safe foundational features for downstream forecasting."""
    data = df.copy()
    data = data.sort_values(GROUP_KEYS + ["date"]).reset_index(drop=True)

    # Calendar features are regenerated from date so they are internally consistent.
    data["year"] = data["date"].dt.year.astype("int16")
    data["month"] = data["date"].dt.month.astype("int8")
    data["day"] = data["date"].dt.day.astype("int8")
    data["day_of_week_num"] = data["date"].dt.dayofweek.astype("int8")
    data["day_of_week"] = data["date"].dt.day_name()
    data["is_weekend"] = data["day_of_week_num"].isin([5, 6]).astype("int8")
    data["week_of_year"] = data["date"].dt.isocalendar().week.astype("int16")
    data["quarter"] = data["date"].dt.quarter.astype("int8")

    # Revenue is descriptive, not a forecast target.
    data["revenue"] = data["unit_price"] * data["quantity"]

    grouped = data.groupby(GROUP_KEYS, sort=False)["quantity"]
    for lag in (1, 7, 14, 28):
        data[f"quantity_lag_{lag}"] = grouped.shift(lag)

    # Shift first to prevent today's quantity leaking into today's rolling feature.
    shifted = grouped.shift(1)
    for window in (7, 14, 28):
        data[f"rolling_mean_{window}"] = (
            shifted.groupby([data[k] for k in GROUP_KEYS], sort=False)
            .rolling(window, min_periods=1).mean()
            .reset_index(level=[0, 1], drop=True)
        )
        data[f"rolling_std_{window}"] = (
            shifted.groupby([data[k] for k in GROUP_KEYS], sort=False)
            .rolling(window, min_periods=2).std()
            .reset_index(level=[0, 1], drop=True)
        )

    # Simple demand anomaly flag using past 28-day history only.
    mean = data["rolling_mean_28"]
    std = data["rolling_std_28"]
    data["demand_outlier_flag"] = (
        std.notna() & (std > 0) & ((data["quantity"] - mean).abs() > 3 * std)
    ).astype("int8")

    return data
