# agents/customer_pattern/agent.py

import pandas as pd
import numpy as np
from pathlib import Path


# ============================================================
# CUSTOMER PATTERN AGENT
# ============================================================

def agent(state):
    """
    Customer Pattern Agent

    Supports two modes:

    1. Single Menu Item Mode
       Used when the user asks about a specific restaurant/menu item.

       Output:
       {
           "customer_demand_trend": "increasing",
           "pattern_strength": 0.82
       }

    2. Dataset-Level Mode
       Used when pattern information is required for the complete
       sales dataset.

       Detailed results are saved to:
           data/customer_patterns.csv

       Output returned to the Orchestrator:
       {
           "patterns_file": "data/customer_patterns.csv",
           "pattern_count": 150
       }

    Input:
        state["request"]
        state["agent_results"]  # optional
    """

    # --------------------------------------------------------
    # 1. Read request from shared state
    # --------------------------------------------------------

    request = state.get("request", {})

    restaurant_scope = request.get(
        "restaurant_scope"
    )

    restaurant_id = request.get(
        "restaurant_id"
    )

    menu_item_id = request.get(
        "menu_item_id"
    )

    # --------------------------------------------------------
    # 2. Load sales data
    # --------------------------------------------------------

    sales_data = load_sales_data(state)

    if sales_data is None:

        return {
            "customer_demand_trend": "unknown",
            "pattern_strength": 0.0
        }

    # --------------------------------------------------------
    # 3. Convert to DataFrame
    # --------------------------------------------------------

    if not isinstance(
        sales_data,
        pd.DataFrame
    ):

        sales_data = pd.DataFrame(
            sales_data
        )

    df = sales_data.copy()

    # --------------------------------------------------------
    # 4. Find required columns
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # 5. Validate required columns
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # 6. Clean data
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # 7. Determine analysis mode
    # --------------------------------------------------------

    # Single menu item analysis
    if (
        restaurant_scope == "single"
        and restaurant_id is not None
        and menu_item_id is not None
    ):

        return analyze_single_item(
            df,
            date_col,
            quantity_col,
            restaurant_col,
            item_col,
            restaurant_id,
            menu_item_id
        )

    # Dataset-level analysis
    return analyze_all_items(
        df,
        date_col,
        quantity_col,
        restaurant_col,
        item_col
    )


# ============================================================
# LOAD SALES DATA
# ============================================================

def load_sales_data(state):
    """
    Loads sales data.

    Priority:

    1. sales_data already available in state
    2. CSV path provided in request
    3. Default project CSV
    """

    request = state.get(
        "request",
        {}
    )

    # --------------------------------------------------------
    # Option 1: Sales data already available in state
    # --------------------------------------------------------

    sales_data = state.get(
        "sales_data"
    )

    if sales_data is not None:

        return sales_data

    # --------------------------------------------------------
    # Option 2: CSV path provided in request
    # --------------------------------------------------------

    sales_data_path = request.get(
        "sales_data_path"
    )

    if sales_data_path:

        path = Path(
            sales_data_path
        )

        if path.exists():

            return pd.read_csv(
                path
            )

    # --------------------------------------------------------
    # Option 3: Default project location
    # --------------------------------------------------------

    default_path = (
        Path(__file__).resolve()
        .parents[2]
        / "data"
        / "sales_data.csv"
    )

    if default_path.exists():

        return pd.read_csv(
            default_path
        )

    # --------------------------------------------------------
    # No data available
    # --------------------------------------------------------

    return None


# ============================================================
# SINGLE MENU ITEM ANALYSIS
# ============================================================

def analyze_single_item(
    df,
    date_col,
    quantity_col,
    restaurant_col,
    item_col,
    restaurant_id,
    menu_item_id
):
    """
    Analyze demand pattern for one restaurant/menu item.
    """

    # --------------------------------------------------------
    # Filter restaurant
    # --------------------------------------------------------

    if restaurant_col is None:

        return {
            "customer_demand_trend": "unknown",
            "pattern_strength": 0.0
        }

    df = df[
        df[restaurant_col].astype(str)
        == str(restaurant_id)
    ]

    # --------------------------------------------------------
    # Filter menu item
    # --------------------------------------------------------

    if item_col is None:

        return {
            "customer_demand_trend": "unknown",
            "pattern_strength": 0.0
        }

    df = df[
        df[item_col].astype(str)
        == str(menu_item_id)
    ]

    # --------------------------------------------------------
    # Check data
    # --------------------------------------------------------

    if df.empty:

        return {
            "customer_demand_trend": "unknown",
            "pattern_strength": 0.0
        }

    # --------------------------------------------------------
    # Aggregate demand by day
    # --------------------------------------------------------

    daily_demand = aggregate_daily_demand(
        df,
        date_col,
        quantity_col
    )

    if daily_demand.empty:

        return {
            "customer_demand_trend": "unknown",
            "pattern_strength": 0.0
        }

    # --------------------------------------------------------
    # Calculate trend
    # --------------------------------------------------------

    trend = calculate_trend(
        daily_demand["quantity"]
    )

    # --------------------------------------------------------
    # Calculate pattern strength
    # --------------------------------------------------------

    pattern_strength = (
        calculate_pattern_strength(
            daily_demand["quantity"]
        )
    )

    # --------------------------------------------------------
    # Return contract output
    # --------------------------------------------------------

    return {
        "customer_demand_trend": trend,
        "pattern_strength": round(
            pattern_strength,
            2
        )
    }


# ============================================================
# DATASET-LEVEL ANALYSIS
# ============================================================

def analyze_all_items(
    df,
    date_col,
    quantity_col,
    restaurant_col,
    item_col
):
    """
    Calculate demand patterns for every
    restaurant/menu-item combination.

    Detailed results are saved to:
        data/customer_patterns.csv

    A small dictionary is returned to the Orchestrator.
    """

    # --------------------------------------------------------
    # Required grouping columns
    # --------------------------------------------------------

    if (
        restaurant_col is None
        or item_col is None
    ):

        return {
            "patterns_file": "data/customer_patterns.csv",
            "pattern_count": 0
        }

    # --------------------------------------------------------
    # Store results
    # --------------------------------------------------------

    patterns = []

    # --------------------------------------------------------
    # Group by restaurant + menu item
    # --------------------------------------------------------

    grouped = df.groupby(
        [
            restaurant_col,
            item_col
        ]
    )

    for (
        restaurant,
        menu_item
    ), group in grouped:

        # ----------------------------------------------
        # Aggregate daily demand
        # ----------------------------------------------

        daily_demand = aggregate_daily_demand(
            group,
            date_col,
            quantity_col
        )

        if daily_demand.empty:
            continue

        # ----------------------------------------------
        # Calculate trend
        # ----------------------------------------------

        trend = calculate_trend(
            daily_demand["quantity"]
        )

        # ----------------------------------------------
        # Calculate pattern strength
        # ----------------------------------------------

        pattern_strength = (
            calculate_pattern_strength(
                daily_demand["quantity"]
            )
        )

        # ----------------------------------------------
        # Store result
        # ----------------------------------------------

        patterns.append(
            {
                "restaurant_id":
                    str(restaurant),

                "menu_item_id":
                    str(menu_item),

                "customer_demand_trend":
                    trend,

                "pattern_strength":
                    round(
                        pattern_strength,
                        2
                    )
            }
        )

    # --------------------------------------------------------
    # Save dataset-level results to CSV
    # --------------------------------------------------------

    output_path = (
        Path(__file__).resolve().parents[2]
        / "data"
        / "customer_patterns.csv"
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    patterns_df = pd.DataFrame(
        patterns,
        columns=[
            "restaurant_id",
            "menu_item_id",
            "customer_demand_trend",
            "pattern_strength"
        ]
    )

    patterns_df.to_csv(
        output_path,
        index=False
    )

    # --------------------------------------------------------
    # Return dictionary to Orchestrator
    # --------------------------------------------------------

    return {
        "patterns_file": "data/customer_patterns.csv",
        "pattern_count": len(patterns)
    }


# ============================================================
# DAILY DEMAND AGGREGATION
# ============================================================

def aggregate_daily_demand(
    df,
    date_col,
    quantity_col
):
    """
    Aggregate sales quantity by date.
    """

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

    daily_demand = (
        daily_demand
        .sort_values("date")
        .reset_index(drop=True)
    )

    return daily_demand


# ============================================================
# FIND COLUMN
# ============================================================

def find_column(
    df,
    possible_columns
):

    for column in possible_columns:

        if column in df.columns:

            return column

    return None


# ============================================================
# CALCULATE TREND
# ============================================================

def calculate_trend(quantity):

    # --------------------------------------------------------
    # Need enough observations
    # --------------------------------------------------------

    if len(quantity) < 4:

        return "insufficient_data"

    x = np.arange(
        len(quantity)
    )

    y = quantity.to_numpy(
        dtype=float
    )

    # --------------------------------------------------------
    # Linear regression
    # --------------------------------------------------------

    slope = np.polyfit(
        x,
        y,
        1
    )[0]

    average = np.mean(y)

    if average == 0:

        return "stable"

    # --------------------------------------------------------
    # Normalize slope
    # --------------------------------------------------------

    relative_slope = (
        slope / average
    )

    # --------------------------------------------------------
    # Classify trend
    # --------------------------------------------------------

    if relative_slope > 0.01:

        return "increasing"

    elif relative_slope < -0.01:

        return "decreasing"

    else:

        return "stable"


# ============================================================
# CALCULATE PATTERN STRENGTH
# ============================================================

def calculate_pattern_strength(
    quantity
):

    if len(quantity) < 4:

        return 0.0

    x = np.arange(
        len(quantity)
    )

    y = quantity.to_numpy(
        dtype=float
    )

    # --------------------------------------------------------
    # Constant demand
    # --------------------------------------------------------

    if np.std(y) == 0:

        return 0.0

    # --------------------------------------------------------
    # Correlation between time and demand
    # --------------------------------------------------------

    correlation = np.corrcoef(
        x,
        y
    )[0, 1]

    if np.isnan(
        correlation
    ):

        return 0.0

    return abs(
        correlation
    )