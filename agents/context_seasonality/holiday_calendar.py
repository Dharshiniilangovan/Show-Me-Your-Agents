from datetime import date, datetime

import holidays
import pandas as pd


def _normalize_date(value):
    """
    Convert supported date representations into
    a Python date.
    """

    if value is None:
        return None

    if isinstance(value, datetime):
        return value.date()

    if isinstance(value, date):
        return value

    parsed = pd.to_datetime(
        value,
        errors="coerce",
    )

    if pd.isna(parsed):
        return None

    return parsed.date()


def get_us_holiday(
    date_value,
    state_code=None,
):
    """
    Return holiday information for a date.

    Parameters
    ----------
    date_value:
        Date/string/Timestamp.

    state_code:
        Optional US state code such as:
        IL, MO, WI, MN.

    Returns
    -------
    dict
    """

    target_date = _normalize_date(
        date_value
    )

    if target_date is None:
        return {
            "is_holiday": False,
            "holiday_name": None,
        }

    calendar = holidays.US(
        years=[target_date.year],
        subdiv=state_code,
        observed=True,
    )

    holiday_name = calendar.get(
        target_date
    )

    return {
        "is_holiday":
            holiday_name is not None,

        "holiday_name":
            holiday_name,
    }

def enrich_forecast_with_holidays(
    forecast_rows,
    state_code=None,
):
    """
    Add holiday metadata to forecast rows without
    modifying predicted quantities.
    """

    enriched = []

    for row in forecast_rows:

        new_row = dict(row)

        holiday = get_us_holiday(
            row.get("date"),
            state_code=state_code,
        )

        new_row["is_holiday"] = (
            holiday["is_holiday"]
        )

        new_row["holiday_name"] = (
            holiday["holiday_name"]
        )

        enriched.append(
            new_row
        )

    return enriched