from pathlib import Path

import pandas as pd


# Columns required by the Context & Seasonality Agent.
#
# The Data Analyst Agent is responsible for cleaning and
# preprocessing these fields before this agent receives them.
REQUIRED = [
    "date",
    "restaurant_id",
    "menu_item_id",
    "category",
    "quantity",
    "day_of_week_num",
    "month",
    "avg_temp_f",
    "precip_inches",
    "is_holiday",
    "is_special_event",
    "is_promotion",
]


def load_unified_data(path: str) -> pd.DataFrame:
    """
    Load the unified/processed demand dataset produced by
    the Data Analyst Agent.

    This function performs NO general preprocessing.
    It only:
        1. validates the path,
        2. loads the CSV,
        3. verifies the required schema.

    Parameters
    ----------
    path : str
        Path supplied through:
        state["shared_data"]["unified_demand_path"]

    Returns
    -------
    pd.DataFrame
        Processed demand dataset.

    Raises
    ------
    ValueError
        If the path is missing or required columns are absent.

    FileNotFoundError
        If the supplied path does not exist.
    """

    if not path:
        raise ValueError(
            "Unified demand dataset path was not provided. "
            "Expected state['shared_data']"
            "['unified_demand_path']."
        )

    file_path = Path(path)

    if not file_path.exists():
        raise FileNotFoundError(
            f"Unified demand dataset not found: {file_path}"
        )

    df = pd.read_csv(file_path)

    missing = [
        column
        for column in REQUIRED
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            "Unified demand dataset is missing columns "
            "required by Context & Seasonality Agent: "
            f"{missing}"
        )

    return df