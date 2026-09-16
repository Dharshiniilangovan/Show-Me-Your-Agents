import numpy as np
import pandas as pd


def add_context_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create calendar, seasonal, weather, contextual,
    and interaction features.

    The function is intentionally defensive:
    optional descriptive columns such as holiday_name,
    special_event_name and precip_type are created
    automatically when they are absent.
    """

    x = df.copy()

    # --------------------------------------------------
    # VALIDATE REQUIRED COLUMNS
    # --------------------------------------------------

    required = [
        "date",
        "day_of_week_num",
        "month",
        "avg_temp_f",
        "precip_inches",
        "is_holiday",
        "is_special_event",
        "is_promotion",
    ]

    missing = [
        col
        for col in required
        if col not in x.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    # --------------------------------------------------
    # DATE NORMALIZATION
    # --------------------------------------------------

    x["date"] = pd.to_datetime(
        x["date"],
        errors="coerce"
    )

    if x["date"].isna().any():
        raise ValueError(
            "Invalid date values detected."
        )

    d = x["date"]

    # --------------------------------------------------
    # OPTIONAL CONTEXT COLUMNS
    # --------------------------------------------------

    optional_text_columns = {
        "holiday_name": "None",
        "special_event_name": "None",
        "precip_type": "None",
    }

    for col, default in optional_text_columns.items():

        if col not in x.columns:
            x[col] = default
        else:
            x[col] = (
                x[col]
                .fillna(default)
                .astype(str)
            )

    # --------------------------------------------------
    # WEEKEND
    # --------------------------------------------------

    if "is_weekend" not in x.columns:
        x["is_weekend"] = (
            x["day_of_week_num"] >= 5
        ).astype("int8")
    else:
        x["is_weekend"] = (
            x["is_weekend"]
            .fillna(0)
            .astype("int8")
        )

    # --------------------------------------------------
    # NORMALIZE BINARY FLAGS
    # --------------------------------------------------

    binary_columns = [
        "is_holiday",
        "is_special_event",
        "is_promotion",
    ]

    for col in binary_columns:

        x[col] = (
            pd.to_numeric(
                x[col],
                errors="coerce"
            )
            .fillna(0)
            .clip(0, 1)
            .astype("int8")
        )

    # --------------------------------------------------
    # CALENDAR FEATURES
    # --------------------------------------------------

    x["week_of_year"] = (
        d.dt.isocalendar()
        .week
        .astype("int16")
    )

    x["quarter"] = (
        d.dt.quarter
        .astype("int8")
    )

    x["day_of_month"] = (
        d.dt.day
        .astype("int8")
    )

    x["day_of_year"] = (
        d.dt.dayofyear
        .astype("int16")
    )

    x["is_month_start"] = (
        d.dt.is_month_start
        .astype("int8")
    )

    x["is_month_end"] = (
        d.dt.is_month_end
        .astype("int8")
    )

    # --------------------------------------------------
    # CYCLICAL CALENDAR FEATURES
    # --------------------------------------------------

    x["dow_sin"] = np.sin(
        2
        * np.pi
        * x["day_of_week_num"]
        / 7
    )

    x["dow_cos"] = np.cos(
        2
        * np.pi
        * x["day_of_week_num"]
        / 7
    )

    x["month_sin"] = np.sin(
        2
        * np.pi
        * (x["month"] - 1)
        / 12
    )

    x["month_cos"] = np.cos(
        2
        * np.pi
        * (x["month"] - 1)
        / 12
    )

    x["doy_sin"] = np.sin(
        2
        * np.pi
        * (x["day_of_year"] - 1)
        / 365.25
    )

    x["doy_cos"] = np.cos(
        2
        * np.pi
        * (x["day_of_year"] - 1)
        / 365.25
    )

    # --------------------------------------------------
    # WEATHER FEATURES
    # --------------------------------------------------

    x["avg_temp_f"] = pd.to_numeric(
        x["avg_temp_f"],
        errors="coerce"
    )

    x["precip_inches"] = (
        pd.to_numeric(
            x["precip_inches"],
            errors="coerce"
        )
        .fillna(0)
        .clip(lower=0)
    )

    x["has_precipitation"] = (
        x["precip_inches"] > 0
    ).astype("int8")

    x["temp_c"] = (
        x["avg_temp_f"] - 32
    ) * 5 / 9

    x["temp_band"] = pd.cut(
        x["avg_temp_f"],
        bins=[
            -np.inf,
            32,
            50,
            68,
            86,
            np.inf,
        ],
        labels=[
            "freezing",
            "cold",
            "mild",
            "warm",
            "hot",
        ],
        include_lowest=True,
    )

    # --------------------------------------------------
    # CONTEXT INTENSITY
    # --------------------------------------------------

    x["context_event_count"] = (
        x[
            [
                "is_holiday",
                "is_special_event",
                "is_promotion",
            ]
        ]
        .sum(axis=1)
        .astype("int8")
    )

    x["is_context_day"] = (
        x["context_event_count"] > 0
    ).astype("int8")

    # --------------------------------------------------
    # INTERACTION FEATURES
    # --------------------------------------------------

    x["weekend_promotion"] = (
        x["is_weekend"]
        * x["is_promotion"]
    ).astype("int8")

    x["holiday_promotion"] = (
        x["is_holiday"]
        * x["is_promotion"]
    ).astype("int8")

    x["event_promotion"] = (
        x["is_special_event"]
        * x["is_promotion"]
    ).astype("int8")

    x["precipitation_weekend"] = (
        x["has_precipitation"]
        * x["is_weekend"]
    ).astype("int8")

    return x