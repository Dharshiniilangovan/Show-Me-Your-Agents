from pathlib import Path
import pandas as pd

REQUIRED_COLUMNS = {
    "date", "restaurant_id", "menu_item_id", "unit_price", "quantity"
}


def load_demand_data(path: str | Path) -> pd.DataFrame:
    """Load the QSR demand CSV and validate the minimum schema."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    df = pd.read_csv(path, low_memory=False)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")
    return df


def load_inventory_data(path: str | Path) -> pd.DataFrame:
    """Load optional real inventory data. No synthetic inventory is invented."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Inventory file not found: {path}")
    return pd.read_csv(path, low_memory=False)
