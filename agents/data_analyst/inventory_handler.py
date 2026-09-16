import pandas as pd


def integrate_inventory(demand: pd.DataFrame, inventory: pd.DataFrame | None = None) -> pd.DataFrame:
    """Attach real inventory data when available; otherwise preserve demand data unchanged.

    Expected inventory keys: restaurant_id, menu_item_id. If inventory has a date column,
    the merge is performed on date too. Typical useful columns are current_stock,
    expiry_date, lead_time_days and stock_status.
    """
    if inventory is None:
        result = demand.copy()
        result["inventory_available"] = 0
        return result

    inv = inventory.copy()
    required = {"restaurant_id", "menu_item_id"}
    missing = required - set(inv.columns)
    if missing:
        raise ValueError(f"Inventory data missing merge keys: {sorted(missing)}")

    keys = ["restaurant_id", "menu_item_id"]
    if "date" in inv.columns:
        inv["date"] = pd.to_datetime(inv["date"], errors="coerce")
        keys.append("date")
    if "expiry_date" in inv.columns:
        inv["expiry_date"] = pd.to_datetime(inv["expiry_date"], errors="coerce")

    if inv.duplicated(keys).any():
        raise ValueError(f"Inventory contains duplicate rows for keys {keys}; resolve before merging.")

    result = demand.merge(inv, on=keys, how="left", validate="many_to_one")
    result["inventory_available"] = 1
    return result


def inventory_snapshot(df: pd.DataFrame) -> dict:
    """Small inventory summary for the orchestrator/dashboard."""
    if "current_stock" not in df.columns:
        return {"inventory_available": False}
    stock = pd.to_numeric(df["current_stock"], errors="coerce")
    return {
        "inventory_available": True,
        "rows_with_stock": int(stock.notna().sum()),
        "zero_stock_rows": int((stock == 0).sum()),
        "total_stock": float(stock.sum(skipna=True)),
    }
