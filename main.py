# ============================================================
# main.py
# ============================================================

import pandas as pd
from pathlib import Path

from agents.orchestrator.agent import OrchestratorAgent

from agents.data_analyst.agent import DataAnalystAgent

from agents.context_seasonality.agent import (
    ContextSeasonalityAgent
)

from agents.customer_pattern.agent import (
    agent as customer_pattern_agent
)

from agents.demand_forecast.agent import DemandForecastingAgent

# ============================================================
# DATASET PATH
# ============================================================
#
# Change ONLY this path if your CSV is somewhere else.
#
# ============================================================

DATASET_PATH = (
    r"C:\Users\saaral\Desktop\hackathon\data"
    r"\2026-havi-niu-hackathon"
    r"\qsr_demand_dataset.csv"
)


# ============================================================
# DATA ANALYST PROCESSED OUTPUT
# ============================================================

DATA_ANALYST_OUTPUT_PATH = (
    Path(__file__).resolve().parent
    / "data"
    / "processed"
    / "unified_demand.csv"
)
# ============================================================
# CONTEXT / SEASONALITY OUTPUT DIRECTORY
# ============================================================

CONTEXT_OUTPUT_DIR = (
    Path(__file__).resolve().parent
    / "data"
    / "outputs"
)


# ============================================================
# LOAD DATASET
# ============================================================

print("\nLoading QSR demand dataset...")

df = pd.read_csv(DATASET_PATH)

print(
    f"Dataset loaded successfully: "
    f"{len(df):,} rows"
)


# ============================================================
# VALIDATE IMPORTANT COLUMNS
# ============================================================

required_columns = [
    "date",
    "restaurant_id",
    "menu_item_id",
    "quantity"
]

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:

    raise ValueError(
        "Dataset is missing required columns: "
        + ", ".join(missing_columns)
    )


# ============================================================
# CREATE RESTAURANT LOOKUP
# ============================================================

if "restaurant_name" in df.columns:

    restaurant_lookup = (
        df[
            [
                "restaurant_id",
                "restaurant_name"
            ]
        ]
        .drop_duplicates()
        .reset_index(drop=True)
    )

else:

    restaurant_lookup = (
        df[
            [
                "restaurant_id"
            ]
        ]
        .drop_duplicates()
        .reset_index(drop=True)
    )


# ============================================================
# CREATE MENU ITEM LOOKUP
# ============================================================

if "menu_item_name" in df.columns:

    menu_lookup = (
        df[
            [
                "menu_item_id",
                "menu_item_name"
            ]
        ]
        .drop_duplicates()
        .reset_index(drop=True)
    )

else:

    menu_lookup = (
        df[
            [
                "menu_item_id"
            ]
        ]
        .drop_duplicates()
        .reset_index(drop=True)
    )


# ============================================================
# RESOLVE RESTAURANT
# ============================================================

def resolve_restaurant(request):
    """
    Convert restaurant ID/name from the orchestrator request
    into the restaurant ID used by the dataset.

    Returns None when the request is for all restaurants.
    """

    restaurant_scope = request.get(
        "restaurant_scope"
    )

    restaurant_id = request.get(
        "restaurant_id"
    )

    restaurant_name = request.get(
        "restaurant_name"
    )

    # --------------------------------------------------------
    # ALL RESTAURANTS
    # --------------------------------------------------------

    if restaurant_scope == "all":

        return None

    # --------------------------------------------------------
    # RESTAURANT ID PROVIDED
    # --------------------------------------------------------

    if restaurant_id is not None:

        match = restaurant_lookup[
            restaurant_lookup[
                "restaurant_id"
            ]
            .astype(str)
            .str.lower()
            ==
            str(restaurant_id)
            .strip()
            .lower()
        ]

        if not match.empty:

            return match.iloc[0][
                "restaurant_id"
            ]

        raise ValueError(
            f"Restaurant ID "
            f"'{restaurant_id}' "
            f"was not found in the dataset."
        )

    # --------------------------------------------------------
    # RESTAURANT NAME PROVIDED
    # --------------------------------------------------------

    if (
        restaurant_name is not None
        and
        "restaurant_name"
        in restaurant_lookup.columns
    ):

        # Exact match first

        match = restaurant_lookup[
            restaurant_lookup[
                "restaurant_name"
            ]
            .astype(str)
            .str.lower()
            ==
            str(restaurant_name)
            .strip()
            .lower()
        ]

        if not match.empty:

            return match.iloc[0][
                "restaurant_id"
            ]

        # Partial match

        match = restaurant_lookup[
            restaurant_lookup[
                "restaurant_name"
            ]
            .astype(str)
            .str.lower()
            .str.contains(
                str(restaurant_name)
                .strip()
                .lower(),
                regex=False,
                na=False
            )
        ]

        if len(match) == 1:

            return match.iloc[0][
                "restaurant_id"
            ]

        if len(match) > 1:

            names = (
                match[
                    "restaurant_name"
                ]
                .astype(str)
                .tolist()
            )

            raise ValueError(
                "Multiple restaurants matched: "
                + ", ".join(names)
            )

        raise ValueError(
            f"Restaurant "
            f"'{restaurant_name}' "
            f"was not found in the dataset."
        )

    return None


# ============================================================
# RESOLVE MENU ITEM
# ============================================================

def resolve_menu_item(request):
    """
    Convert menu item ID/name into the menu_item_id
    used by the dataset.
    """

    menu_item_id = request.get(
        "menu_item_id"
    )

    menu_item_name = request.get(
        "menu_item_name"
    )

    # --------------------------------------------------------
    # MENU ITEM ID
    # --------------------------------------------------------

    if menu_item_id is not None:

        match = menu_lookup[
            menu_lookup[
                "menu_item_id"
            ]
            .astype(str)
            .str.lower()
            ==
            str(menu_item_id)
            .strip()
            .lower()
        ]

        if not match.empty:

            return match.iloc[0][
                "menu_item_id"
            ]

        raise ValueError(
            f"Menu item ID "
            f"'{menu_item_id}' "
            f"was not found in the dataset."
        )

    # --------------------------------------------------------
    # MENU ITEM NAME
    # --------------------------------------------------------

    if (
        menu_item_name is not None
        and
        "menu_item_name"
        in menu_lookup.columns
    ):

        # Exact match

        match = menu_lookup[
            menu_lookup[
                "menu_item_name"
            ]
            .astype(str)
            .str.lower()
            ==
            str(menu_item_name)
            .strip()
            .lower()
        ]

        if not match.empty:

            return match.iloc[0][
                "menu_item_id"
            ]

        # Partial match

        match = menu_lookup[
            menu_lookup[
                "menu_item_name"
            ]
            .astype(str)
            .str.lower()
            .str.contains(
                str(menu_item_name)
                .strip()
                .lower(),
                regex=False,
                na=False
            )
        ]

        if len(match) == 1:

            return match.iloc[0][
                "menu_item_id"
            ]

        if len(match) > 1:

            names = (
                match[
                    "menu_item_name"
                ]
                .astype(str)
                .tolist()
            )

            raise ValueError(
                "Multiple menu items matched: "
                + ", ".join(names)
            )

        raise ValueError(
            f"Menu item "
            f"'{menu_item_name}' "
            f"was not found in the dataset."
        )

    return None


# ============================================================
# FILTER DATA FOR REQUEST
# ============================================================

def filter_data_for_request(
    data,
    request
):
    """
    Filter Data Analyst output according to the restaurant
    and menu item selected by the user.

    The original Data Analyst output remains unchanged.
    """

    filtered = data.copy()

    restaurant_scope = request.get(
        "restaurant_scope"
    )

    # --------------------------------------------------------
    # RESTAURANT FILTER
    # --------------------------------------------------------

    if restaurant_scope != "all":

        restaurant_id = (
            resolve_restaurant(
                request
            )
        )

        if restaurant_id is not None:

            filtered = filtered[
                filtered[
                    "restaurant_id"
                ]
                .astype(str)
                ==
                str(restaurant_id)
            ]

            # Store resolved ID so downstream agents use it

            request[
                "restaurant_id"
            ] = restaurant_id

            request[
                "restaurant_scope"
            ] = "single"

    # --------------------------------------------------------
    # MENU ITEM FILTER
    # --------------------------------------------------------

    if (
        request.get(
            "menu_item_id"
        )
        is not None

        or

        request.get(
            "menu_item_name"
        )
        is not None
    ):

        menu_item_id = (
            resolve_menu_item(
                request
            )
        )

        if menu_item_id is not None:

            filtered = filtered[
                filtered[
                    "menu_item_id"
                ]
                .astype(str)
                ==
                str(menu_item_id)
            ]

            request[
                "menu_item_id"
            ] = menu_item_id

    # --------------------------------------------------------
    # CHECK RESULT
    # --------------------------------------------------------

    if filtered.empty:

        raise ValueError(
            "No data was found for the "
            "selected restaurant/menu item."
        )

    return filtered


# ============================================================
# GET DATA ANALYST OUTPUT
# ============================================================

def get_data_analyst_output(state):
    """
    Retrieve the dataframe produced by Data Analyst.
    """

    results = state.get(
        "agent_results",
        {}
    )
    data_result = results.get(
        "data_analyst"
    )

    if data_result is None:

        raise ValueError(
            "Data Analyst Agent must run "
            "before this agent."
        )

    data = data_result.get(
        "data"
    )

    if data is None:

        raise ValueError(
            "Data Analyst Agent did not "
            "return processed data."
        )

    return data


# ============================================================
# REAL DATA ANALYST AGENT WRAPPER
# ============================================================

def run_data_analyst(state):

    print(
        "[INTEGRATION] "
        "Starting Data Analyst Agent..."
    )

    # --------------------------------------------------------
    # PRINT PARAMETERS RECEIVED FROM ORCHESTRATOR
    # --------------------------------------------------------

    # print("\n[DEBUG] Parameters received by Data Analyst:")
    # print(state["request"])

    # --------------------------------------------------------
    # RUN THE REAL DATA ANALYST AGENT
    # --------------------------------------------------------

    data_agent = DataAnalystAgent(
        demand_path=DATASET_PATH,
        inventory_path=None
    )

        # Make sure data/processed exists
    DATA_ANALYST_OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # Run Data Analyst AND save the complete processed dataset
    result = data_agent.run(
        str(DATA_ANALYST_OUTPUT_PATH)
    )

    # --------------------------------------------------------
    # REGISTER DATA ANALYST OUTPUT IN SHARED CONTEXT
    # --------------------------------------------------------

    state["shared_context"]["data"][
        "unified_demand_path"
    ] = str(DATA_ANALYST_OUTPUT_PATH)

    # print(
    #     "[SHARED CONTEXT] Unified demand:",
    #     state["shared_context"]["data"][
    #         "unified_demand_path"
    #     ]
    # )

    # --------------------------------------------------------
    # GET PROCESSED DATAFRAME
    # --------------------------------------------------------

    data = result.get("data")

    # Get request extracted by orchestrator
    request = state["request"]

    # --------------------------------------------------------
    # APPLY RESTAURANT / MENU ITEM SCOPE
    # --------------------------------------------------------

    if data is not None:

        filtered_data = filter_data_for_request(
            data,
            request
        )

        # Replace full dataframe with requested subset
        result["data"] = filtered_data

        # ----------------------------------------------------
        # UPDATE SUMMARY TO MATCH FILTERED DATA
        # ----------------------------------------------------

        summary = result.get(
            "summary",
            {}
        )

        summary["rows"] = len(
            filtered_data
        )

        summary["restaurants"] = (
            filtered_data[
                "restaurant_id"
            ].nunique()
        )

        summary["menu_items"] = (
            filtered_data[
                "menu_item_id"
            ].nunique()
        )

        if "date" in filtered_data.columns:

            summary["date_min"] = str(
                filtered_data[
                    "date"
                ].min()
            )

            summary["date_max"] = str(
                filtered_data[
                    "date"
                ].max()
            )

        result["summary"] = summary

    # --------------------------------------------------------
    # COMPLETE
    # --------------------------------------------------------

    print(
        "[INTEGRATION] "
        "Data Analyst Agent completed."
    )

    return result


# ============================================================
# REAL CONTEXT / SEASONALITY AGENT WRAPPER
# ============================================================

def run_context_seasonality(state):

    print(
        "[INTEGRATION] "
        "Starting Context / Seasonality Agent..."
    )

    # --------------------------------------------------------
    # GET DATA ANALYST OUTPUT
    # --------------------------------------------------------

    data = get_data_analyst_output(state)

    request = state["request"]

    required_capabilities = state.get(
        "required_capabilities",
        []
    )

    # print(
    #     "[DEBUG] Context request:",
    #     request
    # )

    print(
        "[ORCHESTRATOR] Context capabilities:",
        required_capabilities
    )

    # --------------------------------------------------------
    # FILTER DATA FOR USER REQUEST
    # --------------------------------------------------------

    filtered_data = filter_data_for_request(
        data,
        request
    )

    # print(
    #     "[INTEGRATION] "
    #     f"Context analysis using "
    #     f"{len(filtered_data):,} rows."
    # )

    # --------------------------------------------------------
    # RUN REAL CONTEXT / SEASONALITY AGENT
    # --------------------------------------------------------
    #
    # The agent performs the calculations and generates:
    #
    # promotion_impact.csv
    # holiday_impact.csv
    # event_impact.csv
    # seasonality_weekday.csv
    # seasonality_month.csv
    # seasonality_quarter.csv
    # seasonality_week_of_year.csv
    # sku_context_signals.csv
    #
    # --------------------------------------------------------

    seasonality_agent = ContextSeasonalityAgent()

    raw_result = seasonality_agent.run(
        filtered_data
    )

    # --------------------------------------------------------
    # REGISTER CONTEXT / SEASONALITY OUTPUTS IN SHARED CONTEXT
    # --------------------------------------------------------

    project_root = Path(__file__).resolve().parent

    state["shared_context"]["data"][
        "context_features_path"
    ] = str(
        project_root
        / "data"
        / "processed"
        / "context_features.csv"
    )

    state["shared_context"]["context"][
        "context_signals_path"
    ] = str(
        CONTEXT_OUTPUT_DIR
        / "sku_context_signals.csv"
    )

    state["shared_context"]["context"][
        "agent_context_path"
    ] = str(
        CONTEXT_OUTPUT_DIR
        / "agent_context.json"
    )

    # print(
    #     "[SHARED CONTEXT] "
    #     "Context / Seasonality outputs registered."
    # )

    # --------------------------------------------------------
    # HELPER
    # Convert numpy/pandas values to normal Python values
    # --------------------------------------------------------

    def clean_value(value):

        if pd.isna(value):
            return None

        if hasattr(value, "item"):
            try:
                return value.item()
            except Exception:
                pass

        return value

    # --------------------------------------------------------
    # PROMOTION ANALYSIS
    # --------------------------------------------------------

    if "promotion_analysis" in required_capabilities:

        promotion_file = (
            CONTEXT_OUTPUT_DIR
            / "promotion_impact.csv"
        )

        if not promotion_file.exists():

            raise FileNotFoundError(
                f"Promotion output not found: "
                f"{promotion_file}"
            )

        promotion_df = pd.read_csv(
            promotion_file
        )

        # Get promotion-active row
        active = promotion_df[
            promotion_df["is_promotion"] == 1
        ]

        if active.empty:

            return {
                "analysis_type": "promotion_analysis",
                "restaurant_id": request.get(
                    "restaurant_id"
                ),
                "menu_item_id": request.get(
                    "menu_item_id"
                ),
                "message": (
                    "No promotion observations "
                    "were available for this request."
                )
            }

        row = active.iloc[0]

        return {
            "analysis_type": "promotion_analysis",

            "restaurant_id": request.get(
                "restaurant_id"
            ),

            "menu_item_id": request.get(
                "menu_item_id"
            ),

            "promotion": {
                "mean_demand": clean_value(
                    row.get("mean")
                ),

                "median_demand": clean_value(
                    row.get("median")
                ),

                "active_count": clean_value(
                    row.get("count")
                ),

                "baseline_demand": clean_value(
                    row.get("baseline_mean")
                ),

                "baseline_count": clean_value(
                    row.get("baseline_count")
                ),

                "uplift_pct": clean_value(
                    row.get("uplift_pct")
                ),

                "interpretation": clean_value(
                    row.get("interpretation")
                )
            }
        }

    # --------------------------------------------------------
    # HOLIDAY ANALYSIS
    # --------------------------------------------------------

    if "holiday_analysis" in required_capabilities:

        holiday_file = (
            CONTEXT_OUTPUT_DIR
            / "holiday_impact.csv"
        )

        if not holiday_file.exists():

            raise FileNotFoundError(
                f"Holiday output not found: "
                f"{holiday_file}"
            )

        holiday_df = pd.read_csv(
            holiday_file
        )

        holiday_name = request.get(
            "holiday_name"
        )

        # ----------------------------------------------------
        # SPECIFIC HOLIDAY
        # Example: Christmas
        # ----------------------------------------------------

        if holiday_name:

            matches = holiday_df[
                holiday_df[
                    "holiday_name"
                ]
                .astype(str)
                .str.contains(
                    str(holiday_name),
                    case=False,
                    na=False,
                    regex=False
                )
            ]

            if matches.empty:

                return {
                    "analysis_type": "holiday_analysis",
                    "restaurant_id": request.get(
                        "restaurant_id"
                    ),
                    "menu_item_id": request.get(
                        "menu_item_id"
                    ),
                    "holiday_name": holiday_name,
                    "message": (
                        "No historical observations "
                        "were found for this holiday."
                    )
                }

            row = matches.iloc[0]

            return {
                "analysis_type": "holiday_analysis",

                "restaurant_id": request.get(
                    "restaurant_id"
                ),

                "menu_item_id": request.get(
                    "menu_item_id"
                ),

                "holiday_name": clean_value(
                    row.get("holiday_name")
                ),

                "holiday": {
                    "mean_demand": clean_value(
                        row.get("mean")
                    ),

                    "median_demand": clean_value(
                        row.get("median")
                    ),

                    "active_count": clean_value(
                        row.get("count")
                    ),

                    "baseline_demand": clean_value(
                        row.get("baseline_mean")
                    ),

                    "baseline_count": clean_value(
                        row.get("baseline_count")
                    ),

                    "uplift_pct": clean_value(
                        row.get("uplift_pct")
                    ),

                    "interpretation": clean_value(
                        row.get("interpretation")
                    )
                }
            }

        # ----------------------------------------------------
        # GENERAL HOLIDAY ANALYSIS
        # ----------------------------------------------------

        holiday_records = []

        for _, row in holiday_df.iterrows():

            if clean_value(
                row.get("is_holiday")
            ) != 1:
                continue

            holiday_records.append(
                {
                    "holiday_name": clean_value(
                        row.get("holiday_name")
                    ),

                    "mean_demand": clean_value(
                        row.get("mean")
                    ),

                    "median_demand": clean_value(
                        row.get("median")
                    ),

                    "active_count": clean_value(
                        row.get("count")
                    ),

                    "baseline_demand": clean_value(
                        row.get("baseline_mean")
                    ),

                    "baseline_count": clean_value(
                        row.get("baseline_count")
                    ),

                    "uplift_pct": clean_value(
                        row.get("uplift_pct")
                    ),

                    "interpretation": clean_value(
                        row.get("interpretation")
                    )
                }
            )

        return {
            "analysis_type": "holiday_analysis",

            "restaurant_id": request.get(
                "restaurant_id"
            ),

            "menu_item_id": request.get(
                "menu_item_id"
            ),

            "holidays": holiday_records
        }

    # --------------------------------------------------------
    # EVENT ANALYSIS
    # --------------------------------------------------------

    if "event_analysis" in required_capabilities:

        event_file = (
            CONTEXT_OUTPUT_DIR
            / "event_impact.csv"
        )

        if not event_file.exists():

            raise FileNotFoundError(
                f"Event output not found: "
                f"{event_file}"
            )

        event_df = pd.read_csv(
            event_file
        )

        event_name = request.get(
            "special_event_name"
        )

        # ----------------------------------------------------
        # SPECIFIC EVENT
        # ----------------------------------------------------

        if event_name:

            matches = event_df[
                event_df[
                    "special_event_name"
                ]
                .astype(str)
                .str.contains(
                    str(event_name),
                    case=False,
                    na=False,
                    regex=False
                )
            ]

            if matches.empty:

                return {
                    "analysis_type": "event_analysis",
                    "restaurant_id": request.get(
                        "restaurant_id"
                    ),
                    "menu_item_id": request.get(
                        "menu_item_id"
                    ),
                    "special_event_name": event_name,
                    "message": (
                        "No historical observations "
                        "were found for this event."
                    )
                }

            row = matches.iloc[0]

            return {
                "analysis_type": "event_analysis",

                "restaurant_id": request.get(
                    "restaurant_id"
                ),

                "menu_item_id": request.get(
                    "menu_item_id"
                ),

                "special_event_name": clean_value(
                    row.get(
                        "special_event_name"
                    )
                ),

                "event": {
                    "mean_demand": clean_value(
                        row.get("mean")
                    ),

                    "median_demand": clean_value(
                        row.get("median")
                    ),

                    "active_count": clean_value(
                        row.get("count")
                    ),

                    "baseline_demand": clean_value(
                        row.get("baseline_mean")
                    ),

                    "baseline_count": clean_value(
                        row.get("baseline_count")
                    ),

                    "uplift_pct": clean_value(
                        row.get("uplift_pct")
                    ),

                    "interpretation": clean_value(
                        row.get("interpretation")
                    )
                }
            }

        # ----------------------------------------------------
        # GENERAL EVENT ANALYSIS
        # ----------------------------------------------------

        event_records = []

        for _, row in event_df.iterrows():

            if clean_value(
                row.get("is_special_event")
            ) != 1:
                continue

            event_records.append(
                {
                    "special_event_name": clean_value(
                        row.get(
                            "special_event_name"
                        )
                    ),

                    "mean_demand": clean_value(
                        row.get("mean")
                    ),

                    "median_demand": clean_value(
                        row.get("median")
                    ),

                    "active_count": clean_value(
                        row.get("count")
                    ),

                    "baseline_demand": clean_value(
                        row.get("baseline_mean")
                    ),

                    "baseline_count": clean_value(
                        row.get("baseline_count")
                    ),

                    "uplift_pct": clean_value(
                        row.get("uplift_pct")
                    ),

                    "interpretation": clean_value(
                        row.get("interpretation")
                    )
                }
            )

        return {
            "analysis_type": "event_analysis",

            "restaurant_id": request.get(
                "restaurant_id"
            ),

            "menu_item_id": request.get(
                "menu_item_id"
            ),

            "events": event_records
        }

    # --------------------------------------------------------
    # SEASONALITY ANALYSIS
    # --------------------------------------------------------

    if "seasonality_analysis" in required_capabilities:

        seasonality_files = {
            "weekday":
                CONTEXT_OUTPUT_DIR
                / "seasonality_weekday.csv",

            "month":
                CONTEXT_OUTPUT_DIR
                / "seasonality_month.csv",

            "quarter":
                CONTEXT_OUTPUT_DIR
                / "seasonality_quarter.csv",

            "week_of_year":
                CONTEXT_OUTPUT_DIR
                / "seasonality_week_of_year.csv"
        }

        seasonality_result = {}

        for dimension, file_path in (
            seasonality_files.items()
        ):

            if not file_path.exists():
                continue

            seasonal_df = pd.read_csv(
                file_path
            )

            records = []

            for _, row in seasonal_df.iterrows():

                record = {}

                for column in seasonal_df.columns:

                    record[column] = (
                        clean_value(
                            row[column]
                        )
                    )

                records.append(
                    record
                )

            seasonality_result[
                dimension
            ] = records

        return {
            "analysis_type":
                "seasonality_analysis",

            "restaurant_id":
                request.get(
                    "restaurant_id"
                ),

            "menu_item_id":
                request.get(
                    "menu_item_id"
                ),

            "seasonality":
                seasonality_result
        }

    # --------------------------------------------------------
    # FALLBACK
    # --------------------------------------------------------
    #
    # If no specific Context capability was requested,
    # return the Context Agent's original metadata.
    #
    # --------------------------------------------------------

    return raw_result
# ============================================================
# REAL CUSTOMER PATTERN AGENT WRAPPER
# ============================================================

def run_customer_pattern(state):

    print(
        "[INTEGRATION] "
        "Starting Customer Pattern Agent..."
    )

    # --------------------------------------------------------
    # GET DATA FROM DATA ANALYST
    # --------------------------------------------------------

    data = (
        get_data_analyst_output(
            state
        )
    )

    # --------------------------------------------------------
    # GET REQUEST
    # --------------------------------------------------------

    request = state[
        "request"
    ]

    # --------------------------------------------------------
    # RESOLVE RESTAURANT
    # --------------------------------------------------------

    if (
        request.get(
            "restaurant_scope"
        )
        != "all"
    ):

        restaurant_id = (
            resolve_restaurant(
                request
            )
        )

        if restaurant_id is not None:

            request[
                "restaurant_id"
            ] = restaurant_id

            request[
                "restaurant_scope"
            ] = "single"

    # --------------------------------------------------------
    # RESOLVE MENU ITEM
    # --------------------------------------------------------

    if (
        request.get(
            "menu_item_id"
        )
        is not None

        or

        request.get(
            "menu_item_name"
        )
        is not None
    ):

        menu_item_id = (
            resolve_menu_item(
                request
            )
        )

        request[
            "menu_item_id"
        ] = menu_item_id

    # --------------------------------------------------------
    # RUN REAL AGENT
    # --------------------------------------------------------

    result = (
        customer_pattern_agent(
            state,
            data
        )
    )

    # --------------------------------------------------------
    # REGISTER CUSTOMER PATTERN OUTPUT IN SHARED CONTEXT
    # --------------------------------------------------------

    state["shared_context"][
        "customer_pattern"
    ] = result

    print(
        "[SHARED CONTEXT] "
        "Customer pattern registered."
    )

    print(
        "[INTEGRATION] "
        "Customer Pattern Agent completed."
    )

    return result


# ============================================================
# REAL DEMAND FORECASTING AGENT WRAPPER
# ============================================================

def run_demand_forecasting(state):

    print(
        "[INTEGRATION] "
        "Starting Demand Forecasting Agent..."
    )

    # --------------------------------------------------------
    # GET SHARED CONTEXT
    # --------------------------------------------------------

    shared_context = state.get(
        "shared_context",
        {}
    )

    # --------------------------------------------------------
    # CHECK DATA ANALYST OUTPUT
    # --------------------------------------------------------

    shared_data = shared_context.get(
        "data",
        {}
    )

    unified_path = shared_data.get(
        "unified_demand_path"
    )

    if not unified_path:

        raise ValueError(
            "Demand Forecasting Agent requires "
            "unified_demand_path from Data Analyst."
        )

    # print(
    #     "[DEMAND FORECAST] Unified demand:",
    #     unified_path
    # )

    # --------------------------------------------------------
    # CHECK CONTEXT / SEASONALITY OUTPUT
    # --------------------------------------------------------

    context_data = shared_context.get(
        "context",
        {}
    )

    if context_data:

        print(
            "[DEMAND FORECAST] "
            "Context / Seasonality output available."
        )

    # --------------------------------------------------------
    # CHECK CUSTOMER PATTERN OUTPUT
    # --------------------------------------------------------

    customer_pattern = shared_context.get(
        "customer_pattern",
        {}
    )

    if customer_pattern:

        print(
            "[DEMAND FORECAST] "
            "Customer Pattern output available."
        )

    # --------------------------------------------------------
    # FORECAST HORIZON
    # --------------------------------------------------------

    request = state.get(
        "request",
        {}
    )

    forecast_horizon = int(
        request.get(
            "forecast_horizon"
        )
        or 7
    )

    request[
        "forecast_horizon"
    ] = forecast_horizon

    print(
        "[DEMAND FORECAST] "
        "Forecast horizon:",
        forecast_horizon,
        "days"
    )

    # --------------------------------------------------------
    # RUN REAL DEMAND FORECASTING AGENT
    # --------------------------------------------------------

    forecasting_agent = (
        DemandForecastingAgent()
    )

    result = forecasting_agent.run(
        state
    )

    # --------------------------------------------------------
    # REGISTER FORECAST IN SHARED CONTEXT
    # --------------------------------------------------------

    state[
        "shared_context"
    ].setdefault(
        "forecast",
        {}
    )

    state[
        "shared_context"
    ][
        "forecast"
    ].update(
        {
            "output_path":
                result.get(
                    "output_path"
                ),

            "forecast_horizon":
                result.get(
                    "forecast_horizon"
                ),

            "forecast_start":
                result.get(
                    "forecast_start"
                ),

            "forecast_end":
                result.get(
                    "forecast_end"
                ),

            "evaluation":
                result.get(
                    "evaluation",
                    {}
                )
        }
    )

    print(
        "[SHARED CONTEXT] "
        "Demand forecast registered."
    )

    print(
        "[INTEGRATION] "
        "Demand Forecasting Agent completed."
    )

    return result


# ============================================================
# CREATE ORCHESTRATOR
# ============================================================

orchestrator = (
    OrchestratorAgent()
)


# ============================================================
# REGISTER DATA ANALYST
# ============================================================

orchestrator.register_agent(

    agent_name=
        "data_analyst",

    agent_function=
        run_data_analyst,

    capabilities=[

        "data_retrieval",

        "historical_sales_analysis",

        "recent_demand_analysis"
    ],

    dependencies=[],

    required_user_inputs=[
        "restaurant"
    ]
)


# ============================================================
# REGISTER CONTEXT / SEASONALITY
# ============================================================

orchestrator.register_agent(

    agent_name=
        "context_seasonality",

    agent_function=
        run_context_seasonality,

    capabilities=[

        "seasonality_analysis",

        "promotion_analysis",

        "holiday_analysis",

        "event_analysis"
    ],

    # Context agent needs processed data first.

    dependencies=[
        "data_retrieval"
    ],

    required_user_inputs=[
        "restaurant"
    ]
)


# ============================================================
# REGISTER CUSTOMER PATTERN
# ============================================================

orchestrator.register_agent(

    agent_name=
        "customer_pattern",

    agent_function=
        run_customer_pattern,

    capabilities=[
        "customer_pattern_analysis"
    ],

    # Customer Pattern also needs Data Analyst output.

    dependencies=[
        "data_retrieval"
    ],

    required_user_inputs=[
        "restaurant"
    ]
)

# ============================================================
# REGISTER DEMAND FORECASTING
# ============================================================

orchestrator.register_agent(

    agent_name=
        "demand_forecasting",

    agent_function=
        run_demand_forecasting,

    capabilities=[

        "demand_forecasting",

        "future_demand_forecasting",

        "demand_prediction"
    ],

    dependencies=[

        "data_retrieval",

        "seasonality_analysis",

        "customer_pattern_analysis"
    ],

    required_user_inputs=[
        "restaurant"
    ]
)

# ============================================================
# PRINT RESULT
# ============================================================

def print_result(response):
    """
    Display orchestrator results without printing
    the entire DataFrame.
    """

    print(
        "\n"
        + "=" * 70
    )

    print(
        "ORCHESTRATOR RESULT"
    )

    print(
        "=" * 70
    )

    # --------------------------------------------------------
    # STATUS
    # --------------------------------------------------------

    print(
        "\nStatus:",
        response.get(
            "status"
        )
    )

    # --------------------------------------------------------
    # GOAL
    # --------------------------------------------------------

    if response.get(
        "goal"
    ):

        print(
            "Goal:",
            response.get(
                "goal"
            )
        )

    # --------------------------------------------------------
    # WORKFLOW
    # --------------------------------------------------------

    workflow = response.get(
        "workflow",
        []
    )

    if workflow:

        print(
            "\nWorkflow:"
        )

        for index, agent_name in enumerate(
            workflow,
            start=1
        ):

            status = (
                response
                .get(
                    "workflow_status",
                    {}
                )
                .get(
                    agent_name,
                    "unknown"
                )
            )

            print(
                f"  {index}. "
                f"{agent_name} "
                f"[{status}]"
            )

    # --------------------------------------------------------
    # RESULTS
    # --------------------------------------------------------

    results = response.get(
        "results",
        {}
    )

    # DATA ANALYST

    if "data_analyst" in results:

        result = results[
            "data_analyst"
        ]

        print(
            "\n--- DATA ANALYST ---"
        )

        summary = result.get(
            "summary",
            {}
        )

        # for key, value in summary.items():

        #     # Avoid huge output

        #     if key == "columns":

        #         print(
        #             "columns:",
        #             ", ".join(
        #                 map(
        #                     str,
        #                     value
        #                 )
        #             )
        #         )

        #     else:

        #         print(
        #             f"{key}: {value}"
        #         )

    # CONTEXT / SEASONALITY
    # ========================================================
    # CONTEXT / SEASONALITY
    # ========================================================

    if "context_seasonality" in results:

        result = results[
            "context_seasonality"
        ]

        print(
            "\n--- CONTEXT / SEASONALITY ---"
        )

        analysis_type = result.get(
            "analysis_type"
        )

        # ====================================================
        # PROMOTION ANALYSIS
        # ====================================================

        if analysis_type == "promotion_analysis":

            promotion = result.get(
                "promotion",
                {}
            )

            mean_demand = promotion.get(
                "mean_demand"
            )

            baseline = promotion.get(
                "baseline_demand"
            )

            uplift = promotion.get(
                "uplift_pct"
            )

            if mean_demand is not None:

                print(
                    "Promotion Mean Demand:",
                    round(
                        float(mean_demand),
                        2
                    )
                )

            if baseline is not None:

                print(
                    "Baseline Mean Demand:",
                    round(
                        float(baseline),
                        2
                    )
                )

            if uplift is not None:

                print(
                    "Demand Uplift:",
                    f"{float(uplift):.2f}%"
                )

        # ====================================================
        # SPECIFIC HOLIDAY
        # ====================================================

        elif (
            analysis_type == "holiday_analysis"
            and result.get("holiday")
        ):

            holiday = result.get(
                "holiday",
                {}
            )

            holiday_name = result.get(
                "holiday_name"
            )

            if holiday_name:

                print(
                    "Holiday:",
                    holiday_name
                )

            mean_demand = holiday.get(
                "mean_demand"
            )

            baseline = holiday.get(
                "baseline_demand"
            )

            uplift = holiday.get(
                "uplift_pct"
            )

            if mean_demand is not None:

                print(
                    "Mean Demand:",
                    round(
                        float(mean_demand),
                        2
                    )
                )

            if baseline is not None:

                print(
                    "Baseline Mean Demand:",
                    round(
                        float(baseline),
                        2
                    )
                )

            if uplift is not None:

                print(
                    "Demand Uplift:",
                    f"{float(uplift):.2f}%"
                )

        # ====================================================
        # GENERAL HOLIDAY ANALYSIS
        # ====================================================

        elif (
            analysis_type == "holiday_analysis"
            and result.get("holidays")
        ):

            holidays = result.get(
                "holidays",
                []
            )

            for holiday in holidays:

                name = holiday.get(
                    "holiday_name"
                )

                mean_demand = holiday.get(
                    "mean_demand"
                )

                if (
                    name is not None
                    and mean_demand is not None
                ):

                    print(
                        f"{name}: "
                        f"{float(mean_demand):.2f}"
                    )

        # ====================================================
        # SPECIFIC EVENT
        # ====================================================

        elif (
            analysis_type == "event_analysis"
            and result.get("event")
        ):

            event = result.get(
                "event",
                {}
            )

            event_name = result.get(
                "special_event_name"
            )

            if event_name:

                print(
                    "Event:",
                    event_name
                )

            mean_demand = event.get(
                "mean_demand"
            )

            baseline = event.get(
                "baseline_demand"
            )

            uplift = event.get(
                "uplift_pct"
            )

            if mean_demand is not None:

                print(
                    "Mean Demand:",
                    round(
                        float(mean_demand),
                        2
                    )
                )

            if baseline is not None:

                print(
                    "Baseline Mean Demand:",
                    round(
                        float(baseline),
                        2
                    )
                )

            if uplift is not None:

                print(
                    "Demand Uplift:",
                    f"{float(uplift):.2f}%"
                )

        # ====================================================
        # GENERAL EVENT ANALYSIS
        # ====================================================

        elif (
            analysis_type == "event_analysis"
            and result.get("events")
        ):

            events = result.get(
                "events",
                []
            )

            for event in events:

                name = event.get(
                    "special_event_name"
                )

                mean_demand = event.get(
                    "mean_demand"
                )

                if (
                    name is not None
                    and mean_demand is not None
                ):

                    print(
                        f"{name}: "
                        f"{float(mean_demand):.2f}"
                    )

        # ====================================================
        # SEASONALITY
        # ====================================================

        elif analysis_type == "seasonality_analysis":

            seasonality = result.get(
                "seasonality",
                {}
            )

            # ----------------------------------------------
            # Weekday
            # ----------------------------------------------

            weekday_data = seasonality.get(
                "weekday",
                []
            )

            if weekday_data:

                print(
                    "\nWeekday Mean Demand:"
                )

                for row in weekday_data:

                    # Find demand column
                    mean_demand = (
                        row.get("mean")
                        or row.get("mean_demand")
                    )

                    # Find weekday column
                    weekday = (
                        row.get("day_of_week")
                        or row.get(
                            "day_of_week_num"
                        )
                    )

                    if mean_demand is not None:

                        print(
                            f"  {weekday}: "
                            f"{float(mean_demand):.2f}"
                        )

            # ----------------------------------------------
            # Month
            # ----------------------------------------------

            month_data = seasonality.get(
                "month",
                []
            )

            if month_data:

                print(
                    "\nMonthly Mean Demand:"
                )

                for row in month_data:

                    month = row.get(
                        "month"
                    )

                    mean_demand = (
                        row.get("mean")
                        or row.get("mean_demand")
                    )

                    if mean_demand is not None:

                        print(
                            f"  Month {month}: "
                            f"{float(mean_demand):.2f}"
                        )

            # ----------------------------------------------
            # Quarter
            # ----------------------------------------------

            quarter_data = seasonality.get(
                "quarter",
                []
            )

            if quarter_data:

                print(
                    "\nQuarterly Mean Demand:"
                )

                for row in quarter_data:

                    quarter = row.get(
                        "quarter"
                    )

                    mean_demand = (
                        row.get("mean")
                        or row.get("mean_demand")
                    )

                    if mean_demand is not None:

                        print(
                            f"  Q{quarter}: "
                            f"{float(mean_demand):.2f}"
                        )

            # ----------------------------------------------
            # Week of year
            # ----------------------------------------------

            week_data = seasonality.get(
                "week_of_year",
                []
            )

            if week_data:

                print(
                    "\nWeekly Mean Demand:"
                )

                for row in week_data:

                    week = row.get(
                        "week_of_year"
                    )

                    mean_demand = (
                        row.get("mean")
                        or row.get("mean_demand")
                    )

                    if mean_demand is not None:

                        print(
                            f"  Week {week}: "
                            f"{float(mean_demand):.2f}"
                        )

        # ====================================================
        # FALLBACK
        # ====================================================

        else:

            print(
                "Context / seasonality analysis completed."
            )

    # CUSTOMER PATTERN

    if "customer_pattern" in results:

        result = results[
            "customer_pattern"
        ]

        print(
            "\n--- CUSTOMER PATTERN ---"
        )

        if isinstance(
            result,
            dict
        ):

            for key, value in result.items():

                print(
                    f"{key}: {value}"
                )

        else:

            print(
                result
            )

        # ========================================================
    # DEMAND FORECASTING
    # ========================================================

    if "demand_forecasting" in results:

        result = results[
            "demand_forecasting"
        ]

        forecast = result.get(
            "result",
            {}
        )

        print(
            "\n--- DEMAND FORECASTING ---"
        )

        # ----------------------------------------------------
        # Restaurant
        # ----------------------------------------------------

        if forecast.get(
            "restaurant_id"
        ) is not None:

            print(
                "Restaurant:",
                forecast[
                    "restaurant_id"
                ]
            )

        # ----------------------------------------------------
        # Menu Item
        # ----------------------------------------------------

        if forecast.get(
            "menu_item_id"
        ) is not None:

            print(
                "Menu Item:",
                forecast[
                    "menu_item_id"
                ]
            )

        # ----------------------------------------------------
        # Specific Date
        # ----------------------------------------------------

        if forecast.get(
            "date"
        ) is not None:

            print(
                "Date:",
                forecast[
                    "date"
                ]
            )

        # ----------------------------------------------------
        # Total predicted demand
        # ----------------------------------------------------

        total_demand = forecast.get(
            "total_predicted_demand"
        )

        if total_demand is not None:

            print(
                "\nTotal Predicted Demand:",
                round(
                    total_demand,
                    2
                )
            )

        # ----------------------------------------------------
        # Display values read from demand_forecast.csv
        # ----------------------------------------------------

        daily_forecast = forecast.get(
            "daily_forecast",
            []
        )

        if daily_forecast:

            print(
                "\nForecast:"
            )

            for row in daily_forecast:

                print(
                    f"  {row['date']} : "
                    f"{row['predicted_quantity']:.2f}"
                )

        # --------------------------------------------
        # Evaluation metrics from historical backtest
        # --------------------------------------------

        evaluation = result.get(
            "evaluation",
            {}
        )

        if evaluation:

            print(
                "\nBacktest Evaluation:"
            )

            print(
                "wMAPE:",
                evaluation.get(
                    "wmape_percent"
                ),
                "%"
            )

            print(
                "MAE:",
                evaluation.get(
                    "mae"
                )
            )

            print(
                "RMSE:",
                evaluation.get(
                    "rmse"
                )
            )
    # --------------------------------------------------------
    # ERRORS
    # --------------------------------------------------------

    errors = response.get(
        "errors",
        []
    )

    if errors:

        print(
            "\n--- ERRORS ---"
        )

        for error in errors:

            print(
                error
            )

    print()


# ============================================================
# CHATBOT
# ============================================================

def start_chatbot():

    print(
        "\n"
        + "=" * 70
    )

    print(
        "        MULTI-AGENT DEMAND FORECASTING SYSTEM"
    )

    print(
        "=" * 70
    )

    print(
        "\nAvailable real agents:"
    )

    print(
        "  1. Data Analyst Agent"
    )

    print(
        "  2. Context / Seasonality Agent"
    )

    print(
        "  3. Customer Pattern Agent"
    )

    print(
        "  4. Demand Forecasting Agent"
    )

    print(
        "\nThe Orchestrator will automatically "
        "select the required agents."
    )

    print(
        "\nType 'exit' to stop.\n"
    )

    # ========================================================
    # CONVERSATION LOOP
    # ========================================================

    while True:

        user_query = input(
            "You: "
        ).strip()

        # ----------------------------------------------------
        # EMPTY QUERY
        # ----------------------------------------------------

        if not user_query:

            continue

        # ----------------------------------------------------
        # EXIT
        # ----------------------------------------------------

        if user_query.lower() in [
            "exit",
            "quit",
            "bye"
        ]:

            print(
                "\nBot: Goodbye!"
            )

            break

        # ----------------------------------------------------
        # SEND QUERY TO ORCHESTRATOR
        # ----------------------------------------------------

        try:

            response = (
                orchestrator
                .process_request(
                    user_query
                )
            )

        except Exception as error:

            print(
                "\nBot: Unexpected error:"
            )

            print(
                error
            )

            continue

        # ----------------------------------------------------
        # CLARIFICATION
        # ----------------------------------------------------

        status = response.get(
            "status"
        )

        if status in [
            "clarification_required",
            "missing_parameters",
            "unsupported_request"
        ]:

            print(
                "\nBot:",
                response.get(
                    "message"
                )
            )

            print()

            continue

        # ----------------------------------------------------
        # COMPLETED
        # ----------------------------------------------------

        print_result(
            response
        )


# ============================================================
# RUN APPLICATION
# ============================================================

if __name__ == "__main__":

    start_chatbot()