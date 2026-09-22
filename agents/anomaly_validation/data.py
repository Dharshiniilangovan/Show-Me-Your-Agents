from __future__ import annotations
from pathlib import Path
import os
import pandas as pd

REQUIRED_COLUMNS = {"date", "restaurant_id", "menu_item_id", "quantity"}

def get_dataset_path(state: dict) -> Path:
    shared_path = (
        state
        .get("shared_data", {})
        .get("unified_demand_path")
    )

    if shared_path:
        path = Path(shared_path)

    else:
        env_path = os.getenv("QSR_DATASET_PATH")

        if env_path:
            path = Path(env_path)
        else:
            path = (
                Path(__file__).resolve().parents[2]
                / "data"
                / "processed"
                / "unified_demand.csv"
            )

    if not path.exists():
        raise FileNotFoundError(
            f"Anomaly/Validation dataset not found: {path}"
        )

    return path

def load_dataset(path: str | Path | None = None) -> pd.DataFrame:
    p = Path(path) if path else get_dataset_path()
    if not p.exists():
        raise FileNotFoundError(f"Dataset not found: {p}. Set QSR_DATASET_PATH or pass dataset_path in state/request.")
    df = pd.read_csv(p, low_memory=False)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce")
    return df

def select_history(df: pd.DataFrame, request: dict) -> pd.DataFrame:
    out = df
    rid = request.get("restaurant_id")
    mid = request.get("menu_item_id")
    if request.get("restaurant_scope", "single") == "single" and rid is not None:
        out = out[out["restaurant_id"].astype(str) == str(rid)]
    if mid is not None:
        out = out[out["menu_item_id"].astype(str) == str(mid)]
    return out.sort_values("date")
