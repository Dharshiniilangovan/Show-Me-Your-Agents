"""
Data loading utilities for the Anomaly / Validation Agent.

The integrated multi-agent system stores the unified dataset path at:

    state["shared_context"]["data"]["unified_demand_path"]

This module uses that Phase-1 contract as the primary data source.

Fallbacks are retained for standalone development/testing:
    1. QSR_DATASET_PATH environment variable
    2. Local data/processed/unified_demand.csv
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pandas as pd


# ============================================================
# 1. DATASET PATH RESOLUTION
# ============================================================

def get_dataset_path(state: dict[str, Any] | None = None) -> Path:
    """
    Resolve the unified demand dataset path.

    Priority
    --------
    1. Phase-1 shared context:
       state["shared_context"]["data"]["unified_demand_path"]

    2. QSR_DATASET_PATH environment variable.

    3. Repository-local fallback:
       data/processed/unified_demand.csv

    Parameters
    ----------
    state:
        Common multi-agent state dictionary.

    Returns
    -------
    Path
        Resolved dataset path.

    Raises
    ------
    FileNotFoundError
        If the resolved dataset does not exist.
    """

    state = state or {}

    # --------------------------------------------------------
    # Phase-1 integration path
    # --------------------------------------------------------

    shared_context = state.get(
        "shared_context",
        {},
    )

    shared_data = shared_context.get(
        "data",
        {},
    )

    unified_path = shared_data.get(
        "unified_demand_path"
    )

    if unified_path:
        path = Path(unified_path)

        if path.exists():
            return path

        raise FileNotFoundError(
            "Unified demand dataset specified by Phase-1 "
            f"shared context does not exist: {path.resolve()}"
        )

    # --------------------------------------------------------
    # Standalone environment-variable fallback
    # --------------------------------------------------------

    env_path = os.getenv("QSR_DATASET_PATH")

    if env_path:
        path = Path(env_path)

        if path.exists():
            return path

        raise FileNotFoundError(
            "Dataset specified by QSR_DATASET_PATH "
            f"does not exist: {path.resolve()}"
        )

    # --------------------------------------------------------
    # Local repository fallback
    # --------------------------------------------------------

    fallback_path = (
        Path(__file__).resolve().parents[2]
        / "data"
        / "processed"
        / "unified_demand.csv"
    )

    if fallback_path.exists():
        return fallback_path

    raise FileNotFoundError(
        "No unified demand dataset could be found. "
        "Expected the integrated system to provide "
        "state['shared_context']['data']"
        "['unified_demand_path']. "
        f"Local fallback checked: {fallback_path.resolve()}"
    )


# ============================================================
# 2. LOAD DATASET
# ============================================================

def load_dataset(
    state: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """
    Load the unified demand dataset used by the
    Anomaly / Validation Agent.

    Parameters
    ----------
    state:
        Common multi-agent state.

    Returns
    -------
    pd.DataFrame
        Unified demand dataset.
    """

    dataset_path = get_dataset_path(state)

    try:
        df = pd.read_csv(dataset_path)

    except Exception as exc:
        raise RuntimeError(
            "Failed to load unified demand dataset from "
            f"{dataset_path}: {exc}"
        ) from exc

    if df.empty:
        raise ValueError(
            "Unified demand dataset was loaded successfully "
            "but contains no rows."
        )

    return df


# ============================================================
# 3. FILTER RESTAURANT / MENU ITEM HISTORY
# ============================================================

def select_history(
    df: pd.DataFrame,
    restaurant_id: str | None = None,
    menu_item_id: str | None = None,
) -> pd.DataFrame:
    """
    Filter historical demand for a restaurant/menu-item pair.
    """

    filtered = df.copy()

    if (
        restaurant_id is not None
        and "restaurant_id" in filtered.columns
    ):
        filtered = filtered[
            filtered["restaurant_id"].astype(str)
            == str(restaurant_id)
        ]

    if (
        menu_item_id is not None
        and "menu_item_id" in filtered.columns
    ):
        filtered = filtered[
            filtered["menu_item_id"].astype(str)
            == str(menu_item_id)
        ]

    return filtered.copy()


# ============================================================
# 4. DATASET INFORMATION
# ============================================================

def dataset_info(
    state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Return basic dataset information.

    Useful during integration testing.
    """

    path = get_dataset_path(state)
    df = load_dataset(state)

    return {
        "dataset_path": str(path),
        "dataset_exists": path.exists(),
        "row_count": len(df),
        "column_count": len(df.columns),
        "columns": df.columns.tolist(),
    }


# ============================================================
# 5. BACKWARD-COMPATIBILITY FUNCTION
# ============================================================

def default_dataset_path() -> Path:
    """
    Standalone compatibility helper.

    This function does NOT have access to the shared state.
    Integrated execution should use get_dataset_path(state).
    """

    env_path = os.getenv("QSR_DATASET_PATH")

    if env_path:
        return Path(env_path)

    return (
        Path(__file__).resolve().parents[2]
        / "data"
        / "processed"
        / "unified_demand.csv"
    )