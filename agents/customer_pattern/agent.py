# agents/customer_pattern/agent.py

from pathlib import Path

import numpy as np
import pandas as pd


def agent(state, sales_data):
    """
    Customer Pattern Agent

    Calculates pattern strength for each:
        restaurant_id x menu_item_id

    Pattern strength:
        Absolute Pearson correlation between time and daily demand.

        0 -> weak/no linear trend
        1 -> strong linear trend

    CSV Output:
        data/outputs/customer_pattern_features.csv

    CSV columns:
        restaurant_id
        menu_item_id
        customer_demand_trend
        pattern_strength

    Terminal:
        If the user requests a specific restaurant/item,
        only the matching pattern information is printed.
    """

    # ============================================================
    # 1. READ REQUEST
    # ============================================================

    request = state.get(
        "request",
        {}
    )

    restaurant_scope = request.get(
        "restaurant_scope"
    )

    restaurant_id = request.get(
        "restaurant_id"
    )

    menu_item_id = request.get(
        "menu_item_id"
    )

    # ============================================================
    # 2. CONVERT SALES DATA TO DATAFRAME
    # ============================================================

    if not isinstance(
        sales_data,
        pd.DataFrame
    ):

        sales_data = pd.DataFrame(
            sales_data
        )

    df = sales_data.copy()

    # ============================================================
    # 3. FIND REQUIRED COLUMNS
    # ============================================================

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

    # ============================================================
    # 4. VALIDATE COLUMNS
    # ============================================================

    if date_col is None:

        return {
            "status": "error",
            "message": "Date column not found.",
            "pattern_features": []
        }

    if quantity_col is None:

        return {
            "status": "error",
            "message": "Quantity column not found.",
            "pattern_features": []
        }

    if restaurant_col is None:

        return {
            "status": "error",
            "message": "Restaurant column not found.",
            "pattern_features": []
        }

    if item_col is None:

        return {
            "status": "error",
            "message": "Menu item column not found.",
            "pattern_features": []
        }

    # ============================================================
    # 5. CLEAN DATA
    # ============================================================

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
            quantity_col,
            restaurant_col,
            item_col
        ]
    )

    # Demand cannot be negative
    df = df[
        df[quantity_col] >= 0
    ].copy()

    if df.empty:

        return {
            "status": "success",
            "pattern_features": []
        }

    # ============================================================
    # 6. CALCULATE PATTERN STRENGTH FOR ALL RESTAURANT × ITEMS
    #
    # IMPORTANT:
    # We calculate everything first so the CSV always contains
    # the complete set of pattern features.
    # ============================================================

    pattern_rows = []

    grouped = df.groupby(
        [
            restaurant_col,
            item_col
        ],
        sort=False
    )

    for (
        current_restaurant,
        current_item
    ), group in grouped:

        # --------------------------------------------------------
        # Aggregate demand by date
        # --------------------------------------------------------

        daily_demand = (
            group.groupby(
                group[
                    date_col
                ].dt.date
            )[
                quantity_col
            ]
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
            .sort_values(
                "date"
            )
            .reset_index(
                drop=True
            )
        )

        # --------------------------------------------------------
        # Trend
        # --------------------------------------------------------

        customer_demand_trend = (
            calculate_trend(
                daily_demand[
                    "quantity"
                ]
            )
        )

        # --------------------------------------------------------
        # Pattern strength
        # --------------------------------------------------------

        pattern_strength = (
            calculate_pattern_strength(
                daily_demand[
                    "quantity"
                ]
            )
        )

        # --------------------------------------------------------
        # Store feature
        # --------------------------------------------------------

        pattern_rows.append(
            {
                "restaurant_id":
                    str(
                        current_restaurant
                    ),

                "menu_item_id":
                    str(
                        current_item
                    ),

                "customer_demand_trend":
                    customer_demand_trend,

                "pattern_strength":
                    round(
                        float(
                            pattern_strength
                        ),
                        4
                    )
            }
        )

    # ============================================================
    # 7. CREATE COMPLETE PATTERN DATAFRAME
    # ============================================================

    pattern_df = pd.DataFrame(
        pattern_rows
    )

    if pattern_df.empty:

        return {
            "status": "success",
            "pattern_features": []
        }

    # ============================================================
    # 8. SAVE COMPLETE CSV
    # ============================================================

    output_dir = (
        Path(__file__)
        .resolve()
        .parents[2]
        / "data"
        / "outputs"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    output_path = (
        output_dir
        / "customer_pattern_features.csv"
    )

    pattern_df.to_csv(
        output_path,
        index=False
    )

    # ============================================================
    # 9. GET ONLY USER-REQUESTED RESULT FOR TERMINAL
    # ============================================================

    requested_pattern = (
        pattern_df.copy()
    )

    # --------------------------------------------------------
    # Restaurant filter
    # --------------------------------------------------------

    if (
        restaurant_scope == "single"
        and restaurant_id is not None
    ):

        requested_pattern = (
            requested_pattern[
                requested_pattern[
                    "restaurant_id"
                ].astype(str)
                ==
                str(
                    restaurant_id
                )
            ]
        )

    # --------------------------------------------------------
    # Menu-item filter
    # --------------------------------------------------------

    if menu_item_id is not None:

        requested_pattern = (
            requested_pattern[
                requested_pattern[
                    "menu_item_id"
                ].astype(str)
                ==
                str(
                    menu_item_id
                )
            ]
        )

    # ============================================================
    # 10. TERMINAL OUTPUT
    # ============================================================

    print(
        "\n--- CUSTOMER PATTERN ---"
    )

    # --------------------------------------------------------
    # Restaurant + Item requested
    # --------------------------------------------------------

    if (
        restaurant_id is not None
        and menu_item_id is not None
    ):

        if requested_pattern.empty:

            print(
                "No customer pattern found for "
                f"restaurant {restaurant_id}, "
                f"item {menu_item_id}."
            )

        else:

            row = (
                requested_pattern
                .iloc[0]
            )

            print(
                "Restaurant:",
                row[
                    "restaurant_id"
                ]
            )

            print(
                "Menu Item:",
                row[
                    "menu_item_id"
                ]
            )

            print(
                "Demand Trend:",
                row[
                    "customer_demand_trend"
                ]
            )

            print(
                "Pattern Strength:",
                round(
                    float(
                        row[
                            "pattern_strength"
                        ]
                    ),
                    4
                )
            )

    # --------------------------------------------------------
    # Only item requested
    # --------------------------------------------------------

    elif menu_item_id is not None:

        if requested_pattern.empty:

            print(
                "No customer pattern found for "
                f"item {menu_item_id}."
            )

        else:

            print(
                "Menu Item:",
                menu_item_id
            )

            # If item exists in multiple restaurants,
            # show one concise row per restaurant.

            for _, row in (
                requested_pattern
                .iterrows()
            ):

                print(
                    f"Restaurant "
                    f"{row['restaurant_id']}: "
                    f"Pattern Strength = "
                    f"{float(row['pattern_strength']):.4f}, "
                    f"Trend = "
                    f"{row['customer_demand_trend']}"
                )

    # --------------------------------------------------------
    # Only restaurant requested
    # --------------------------------------------------------

    elif (
        restaurant_scope == "single"
        and restaurant_id is not None
    ):

        print(
            "Restaurant:",
            restaurant_id
        )

        print(
            "Pattern strengths calculated for",
            len(
                requested_pattern
            ),
            "menu items."
        )

        # Do not dump all items into terminal.

    # --------------------------------------------------------
    # All restaurants/items requested
    # --------------------------------------------------------

    else:

        print(
            "Pattern strengths calculated for",
            len(
                pattern_df
            ),
            "restaurant-item combinations."
        )

    # ============================================================
    # 11. RETURN TO ORCHESTRATOR
    # ============================================================

    return {
        "status":
            "success",

        # Full feature set for Demand Forecasting Agent
        "pattern_features":
            pattern_df.to_dict(
                orient="records"
            ),

        # User-specific result
        "result":
            requested_pattern.to_dict(
                orient="records"
            )
    }


# ================================================================
# HELPER: FIND COLUMN
# ================================================================

def find_column(
    df,
    possible_columns
):

    for column in possible_columns:

        if column in df.columns:

            return column

    return None


# ================================================================
# HELPER: CALCULATE TREND
# ================================================================

def calculate_trend(
    quantity
):

    if len(
        quantity
    ) < 4:

        return (
            "insufficient_data"
        )

    x = np.arange(
        len(
            quantity
        )
    )

    y = quantity.to_numpy(
        dtype=float
    )

    # ------------------------------------------------------------
    # Linear regression slope
    # ------------------------------------------------------------

    slope = np.polyfit(
        x,
        y,
        1
    )[0]

    average = np.mean(
        y
    )

    if average == 0:

        return "stable"

    relative_slope = (
        slope
        / average
    )

    if relative_slope > 0.01:

        return "increasing"

    elif relative_slope < -0.01:

        return "decreasing"

    else:

        return "stable"


# ================================================================
# HELPER: CALCULATE PATTERN STRENGTH
# ================================================================

def calculate_pattern_strength(
    quantity
):

    if len(
        quantity
    ) < 4:

        return 0.0

    x = np.arange(
        len(
            quantity
        )
    )

    y = quantity.to_numpy(
        dtype=float
    )

    # ------------------------------------------------------------
    # Constant demand has no measurable linear trend
    # ------------------------------------------------------------

    if np.std(
        y
    ) == 0:

        return 0.0

    # ------------------------------------------------------------
    # Correlation between time and demand
    # ------------------------------------------------------------

    correlation = np.corrcoef(
        x,
        y
    )[0, 1]

    if np.isnan(
        correlation
    ):

        return 0.0

    # Direction is handled separately by calculate_trend().
    # Pattern strength only represents strength.
    return float(
        abs(
            correlation
        )
    )