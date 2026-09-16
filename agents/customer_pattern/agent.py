# agents/customer_pattern/agent.py

import pandas as pd
import numpy as np


def agent(state, sales_data):
    """
    Customer Pattern Agent

    Input:
        state["request"]
        state["agent_results"]  # optional
        sales_data              # sales dataset for analysis

    Required request fields:
        restaurant_scope
        restaurant_id
        restaurant_name
        menu_item_id
        menu_item_name
        time_period

    Output:
        {
            "customer_demand_trend": "...",
            "pattern_strength": 0.82
        }
    """

    # --------------------------------------------------
    # 1. Read request from shared state
    # --------------------------------------------------

    request = state["request"]

    restaurant_scope = request.get(
        "restaurant_scope"
    )

    restaurant_id = request.get(
        "restaurant_id"
    )

    menu_item_id = request.get(
        "menu_item_id"
    )

    # --------------------------------------------------
    # 2. Convert sales data to DataFrame
    # --------------------------------------------------

    if not isinstance(sales_data, pd.DataFrame):

        sales_data = pd.DataFrame(
            sales_data
        )

    df = sales_data.copy()

    # --------------------------------------------------
    # 3. Find required columns
    # --------------------------------------------------

    date_col = find_column(
        df,
        [
            "date",
            "order_date",
            "transaction_date",
            "sales_date"
        ]
    )

    quantity_col = find_column(
        df,
        [
            "quantity",
            "sales_quantity",
            "qty",
            "units_sold"
        ]
    )

    restaurant_col = find_column(
        df,
        [
            "restaurant_id",
            "store_id",
            "outlet_id"
        ]
    )

    item_col = find_column(
        df,
        [
            "menu_item_id",
            "product_id",
            "item_id",
            "sku"
        ]
    )

    # --------------------------------------------------
    # 4. Validate required columns
    # --------------------------------------------------

    if date_col is None:

        return {
            "customer_demand_trend": "unknown",
            "pattern_strength": 0.0
        }

    if quantity_col is None:

        return {
            "customer_demand_trend": "unknown",
            "pattern_strength": 0.0
        }

    # --------------------------------------------------
    # 5. Clean data
    # --------------------------------------------------

    df[date_col] = pd.to_datetime(
        df[date_col],
        errors="coerce"
    )

    df[quantity_col] = pd.to_numeric(
        df[quantity_col],
        errors="coerce"
    )

    df = df.dropna(
        subset=[
            date_col,
            quantity_col
        ]
    )

    # --------------------------------------------------
    # 6. Filter restaurant
    # --------------------------------------------------

    if (
        restaurant_scope == "single"
        and restaurant_id is not None
        and restaurant_col is not None
    ):

        df = df[
            df[restaurant_col].astype(str)
            == str(restaurant_id)
        ]

    # --------------------------------------------------
    # 7. Filter menu item
    # --------------------------------------------------

    if (
        menu_item_id is not None
        and item_col is not None
    ):

        df = df[
            df[item_col].astype(str)
            == str(menu_item_id)
        ]

    # --------------------------------------------------
    # 8. Check data
    # --------------------------------------------------

    if df.empty:

        return {
            "customer_demand_trend": "unknown",
            "pattern_strength": 0.0
        }

    # --------------------------------------------------
    # 9. Aggregate demand by day
    # --------------------------------------------------

    daily_demand = (
        df.groupby(
            df[date_col].dt.date
        )[quantity_col]
        .sum()
        .reset_index()
    )

    daily_demand.columns = [
        "date",
        "quantity"
    ]

    daily_demand["date"] = pd.to_datetime(
        daily_demand["date"]
    )

    daily_demand = daily_demand.sort_values(
        "date"
    )

    # --------------------------------------------------
    # 10. Calculate demand trend
    # --------------------------------------------------

    customer_demand_trend = calculate_trend(
        daily_demand["quantity"]
    )

    # --------------------------------------------------
    # 11. Calculate pattern strength
    # --------------------------------------------------

    pattern_strength = calculate_pattern_strength(
        daily_demand["quantity"]
    )

    # --------------------------------------------------
    # 12. Return contract-compatible output
    # --------------------------------------------------

    return {
        "customer_demand_trend":
            customer_demand_trend,

        "pattern_strength":
            round(pattern_strength, 2)
    }


# ======================================================
# Helper Functions
# ======================================================

def find_column(df, possible_columns):

    for column in possible_columns:

        if column in df.columns:
            return column

    return None


def calculate_trend(quantity):

    # Need enough observations
    if len(quantity) < 4:

        return "insufficient_data"

    x = np.arange(
        len(quantity)
    )

    y = quantity.to_numpy()

    # Linear regression slope
    slope = np.polyfit(
        x,
        y,
        1
    )[0]

    average = np.mean(y)

    if average == 0:

        return "stable"

    # Normalize slope
    relative_slope = (
        slope / average
    )

    if relative_slope > 0.01:

        return "increasing"

    elif relative_slope < -0.01:

        return "decreasing"

    else:

        return "stable"


def calculate_pattern_strength(quantity):

    if len(quantity) < 4:

        return 0.0

    x = np.arange(
        len(quantity)
    )

    y = quantity.to_numpy()

    correlation = np.corrcoef(
        x,
        y
    )[0, 1]

    if np.isnan(correlation):

        return 0.0

    return abs(correlation)