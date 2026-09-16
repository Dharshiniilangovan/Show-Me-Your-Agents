import numpy as np
import pandas as pd


def clean_demand_data(df: pd.DataFrame, drop_missing_target: bool = True) -> tuple[pd.DataFrame, dict]:
    """Clean QSR demand data without silently inventing target values."""
    data = df.copy()
    before = len(data)

    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    invalid_dates = int(data["date"].isna().sum())
    data = data.dropna(subset=["date"])

    # Standardize identifiers/text.
    for col in ["restaurant_id", "menu_item_id", "restaurant_name", "menu_item_name", "city", "state", "category"]:
        if col in data.columns:
            data[col] = data[col].astype("string").str.strip()

    # Contextual missingness means 'no named event/holiday/precipitation' in this dataset.
    fill_none = ["holiday_name", "special_event_name", "precip_type"]
    for col in fill_none:
        if col in data.columns:
            data[col] = data[col].fillna("None").astype("string")

    # Numeric coercion.
    numeric = ["unit_price", "quantity", "avg_temp_f", "precip_inches"]
    for col in numeric:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors="coerce")

    # Flags make quality decisions visible to downstream agents.
    data["quantity_was_missing"] = data["quantity"].isna().astype("int8")
    data["invalid_quantity"] = ((data["quantity"].notna()) & (data["quantity"] < 0)).astype("int8")
    data["invalid_price"] = ((data["unit_price"].isna()) | (data["unit_price"] <= 0)).astype("int8")

    missing_target = int(data["quantity"].isna().sum())
    negative_target = int((data["quantity"] < 0).fillna(False).sum())
    invalid_price = int(data["invalid_price"].sum())

    # Negative demand is not meaningful for this forecasting target.
    data.loc[data["quantity"] < 0, "quantity"] = np.nan
    if drop_missing_target:
        data = data.dropna(subset=["quantity"])

    # One observation should be restaurant + item + date.
    duplicate_keys = int(data.duplicated(["date", "restaurant_id", "menu_item_id"]).sum())
    data = data.drop_duplicates(["date", "restaurant_id", "menu_item_id"], keep="last")

    data = data.sort_values(["restaurant_id", "menu_item_id", "date"]).reset_index(drop=True)

    report = {
        "rows_input": before,
        "rows_output": len(data),
        "invalid_dates": invalid_dates,
        "missing_quantity": missing_target,
        "negative_quantity": negative_target,
        "invalid_price": invalid_price,
        "duplicate_business_keys": duplicate_keys,
    }
    return data, report
