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
from agents.inventory_decision.agent import InventoryDecisionAgent

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
    Filter data for single, list, or explicitly paired restaurant/menu requests.

    If restaurant_item_pairs is present, pairing is preserved:
    (R01,M02) + (R02,M03) means exactly those two combinations.
    """

    filtered = data.copy()

    pairs = request.get("restaurant_item_pairs") or []

    if pairs:
        masks = []

        for pair in pairs:
            restaurant_id = pair.get("restaurant_id")
            menu_item_id = pair.get("menu_item_id")

            mask = pd.Series(True, index=filtered.index)

            if restaurant_id is not None:
                mask &= (
                    filtered["restaurant_id"].astype(str)
                    == str(restaurant_id)
                )

            if menu_item_id is not None:
                mask &= (
                    filtered["menu_item_id"].astype(str)
                    == str(menu_item_id)
                )

            masks.append(mask)

        if masks:
            combined = masks[0].copy()
            for mask in masks[1:]:
                combined |= mask
            filtered = filtered[combined]

    else:
        restaurant_ids = request.get("restaurant_ids") or []
        menu_item_ids = request.get("menu_item_ids") or []

        if restaurant_ids:
            filtered = filtered[
                filtered["restaurant_id"].astype(str).isin(
                    [str(value) for value in restaurant_ids]
                )
            ]
        elif request.get("restaurant_scope") != "all":
            restaurant_id = resolve_restaurant(request)
            if restaurant_id is not None:
                filtered = filtered[
                    filtered["restaurant_id"].astype(str)
                    == str(restaurant_id)
                ]
                request["restaurant_id"] = restaurant_id
                request["restaurant_scope"] = "single"

        if menu_item_ids:
            filtered = filtered[
                filtered["menu_item_id"].astype(str).isin(
                    [str(value) for value in menu_item_ids]
                )
            ]
        elif (
            request.get("menu_item_id") is not None
            or request.get("menu_item_name") is not None
        ):
            menu_item_id = resolve_menu_item(request)
            if menu_item_id is not None:
                filtered = filtered[
                    filtered["menu_item_id"].astype(str)
                    == str(menu_item_id)
                ]
                request["menu_item_id"] = menu_item_id

    if filtered.empty:
        raise ValueError(
            "No data was found for the selected restaurant/menu item combination(s)."
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
    # GET COMPLETE DATASET FROM SHARED CONTEXT
    # --------------------------------------------------------
    # Customer-pattern features are model-training features.
    # Therefore, whenever this agent is part of the forecasting
    # workflow, patterns must be generated for EVERY
    # restaurant/menu-item combination, not only the combination
    # requested by the user.

    unified_path = (
        state.get("shared_context", {})
        .get("data", {})
        .get("unified_demand_path")
    )

    if not unified_path:
        raise ValueError(
            "Customer Pattern Agent requires unified_demand_path "
            "from Data Analyst."
        )

    unified_path = Path(unified_path)

    if not unified_path.exists():
        raise FileNotFoundError(
            f"Unified demand file not found: {unified_path}"
        )

    data = pd.read_csv(
        unified_path,
        low_memory=False
    )

    # --------------------------------------------------------
    # PRESERVE THE ORIGINAL USER REQUEST
    # --------------------------------------------------------
    # Example:
    # restaurant 1 + menu item 2 + next 2 days
    #
    # This request must remain unchanged for Demand Forecasting.
    # We create a separate state only for dataset-level customer
    # pattern generation.

    original_request = state.get(
        "request",
        {}
    )

    pattern_state = dict(state)

    pattern_request = dict(
        original_request
    )

    # --------------------------------------------------------
    # FORCE DATASET-LEVEL CUSTOMER PATTERN ANALYSIS
    # --------------------------------------------------------
    # The Customer Pattern Agent switches to single-item mode when
    # restaurant_scope == "single" AND restaurant_id/menu_item_id
    # are present. Clear those fields only in pattern_state so that
    # analyze_all_items() runs over the complete dataset.

    pattern_request[
        "restaurant_scope"
    ] = "all"

    pattern_request[
        "restaurant_id"
    ] = None

    pattern_request[
        "restaurant_name"
    ] = None

    pattern_request[
        "menu_item_id"
    ] = None

    pattern_request[
        "menu_item_name"
    ] = None

    pattern_state[
        "request"
    ] = pattern_request

    # Give the Customer Pattern Agent the COMPLETE processed
    # dataset directly. Its load_sales_data() checks sales_data
    # before other sources.
    pattern_state[
        "sales_data"
    ] = data

    # --------------------------------------------------------
    # RUN CUSTOMER PATTERN AGENT
    # --------------------------------------------------------

    result = customer_pattern_agent(
        pattern_state
    )

    if not isinstance(
        result,
        dict
    ):
        raise ValueError(
            "Customer Pattern Agent returned an invalid result."
        )

    # --------------------------------------------------------
    # GET GENERATED FEATURE FILE
    # --------------------------------------------------------

    customer_pattern_path = (
        result.get("output_path")
        or result.get("patterns_file")
    )

    if not customer_pattern_path:
        raise ValueError(
            "Customer Pattern Agent did not generate the complete "
            "customer-pattern feature file."
        )

    customer_pattern_path = Path(
        customer_pattern_path
    )

    if not customer_pattern_path.is_absolute():

        customer_pattern_path = (
            Path(__file__).resolve().parent
            / customer_pattern_path
        )

    customer_pattern_path = (
        customer_pattern_path.resolve()
    )

    if not customer_pattern_path.exists():

        raise FileNotFoundError(
            "Customer pattern feature file was not created: "
            f"{customer_pattern_path}"
        )

    # --------------------------------------------------------
    # VALIDATE FEATURE FILE
    # --------------------------------------------------------

    pattern_features = pd.read_csv(
        customer_pattern_path,
        low_memory=False
    )

    required_columns = {
        "restaurant_id",
        "menu_item_id",
        "customer_demand_trend",
        "pattern_strength",
    }

    missing_columns = (
        required_columns
        - set(pattern_features.columns)
    )

    if missing_columns:

        raise ValueError(
            "Customer pattern feature file is missing columns: "
            + ", ".join(
                sorted(
                    missing_columns
                )
            )
        )

    if pattern_features.empty:

        raise ValueError(
            "Customer pattern feature file was created but contains "
            "no restaurant/menu-item patterns."
        )

    combination_count = (
        pattern_features[
            [
                "restaurant_id",
                "menu_item_id",
            ]
        ]
        .drop_duplicates()
        .shape[0]
    )

    # --------------------------------------------------------
    # REGISTER FULL FEATURE PATH IN SHARED CONTEXT
    # --------------------------------------------------------

    state.setdefault(
        "shared_context",
        {}
    )

    state[
        "shared_context"
    ].setdefault(
        "data",
        {}
    )

    state[
        "shared_context"
    ][
        "data"
    ][
        "customer_pattern_features_path"
    ] = str(
        customer_pattern_path
    )

    # Keep lightweight metadata under customer_pattern.
    # Do NOT replace the user's original request or store the full
    # DataFrame in shared context.

    state[
        "shared_context"
    ][
        "customer_pattern"
    ] = {
        "output_path":
            str(
                customer_pattern_path
            ),

        "pattern_count":
            int(
                combination_count
            ),
    }

    print(
        "[CUSTOMER PATTERN] "
        f"Generated patterns for "
        f"{combination_count:,} "
        "restaurant/menu-item combinations."
    )

    print(
        "[SHARED CONTEXT] "
        "customer_pattern_features_path:",
        str(
            customer_pattern_path
        )
    )

    print(
        "[INTEGRATION] "
        "Customer Pattern Agent completed."
    )

    # Return dataset-level metadata to the orchestrator.
    # The original state['request'] is untouched, so the Demand
    # Forecasting Agent can later retrieve exactly what the user
    # requested from the complete 7-day forecast.

    requested_pairs = original_request.get("restaurant_item_pairs") or []

    if requested_pairs:
        selected_patterns = []

        for pair in requested_pairs:
            match = pattern_features.copy()

            if pair.get("restaurant_id") is not None:
                match = match[
                    match["restaurant_id"].astype(str)
                    == str(pair.get("restaurant_id"))
                ]

            if pair.get("menu_item_id") is not None:
                match = match[
                    match["menu_item_id"].astype(str)
                    == str(pair.get("menu_item_id"))
                ]

            for _, row in match.iterrows():
                selected_patterns.append({
                    "restaurant_id": row.get("restaurant_id"),
                    "menu_item_id": row.get("menu_item_id"),
                    "customer_demand_trend": row.get("customer_demand_trend"),
                    "pattern_strength": row.get("pattern_strength"),
                })

        return {
            "analysis_type": "customer_pattern_analysis",
            "scope": "requested_restaurant_item_pairs",
            "patterns": selected_patterns,
            "pattern_count": len(selected_patterns),
            "output_path": str(customer_pattern_path),
        }

    return {
        "analysis_type": "customer_pattern_analysis",
        "scope": "all_restaurant_menu_combinations",
        "pattern_count": int(combination_count),
        "output_path": str(customer_pattern_path),
    }


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

    requested_forecast_horizon = int(
        request.get("forecast_horizon") or 7
    )
    request["requested_forecast_horizon"] = requested_forecast_horizon

    # The production forecast is deliberately limited to the next 7 days.
    # Longer-horizon values are not exposed because their accuracy degrades.
    forecast_horizon = 7
    request["forecast_horizon"] = forecast_horizon

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
# REAL INVENTORY DECISION AGENT WRAPPER
# ============================================================

def run_inventory_decision(state):
    """Run inventory decisions from the forecast produced in this workflow."""

    print("[INTEGRATION] Starting Inventory Decision Agent...")

    shared_context = state.get("shared_context", {})
    forecast_context = shared_context.get("forecast", {})
    forecast_path = forecast_context.get("output_path")

    if not forecast_path:
        raise ValueError(
            "Inventory Decision Agent requires the demand forecast output "
            "from shared_context['forecast']['output_path']."
        )

    shared_data = shared_context.get("data", {})
    unified_path = shared_data.get("unified_demand_path")

    if not unified_path:
        raise ValueError(
            "Inventory Decision Agent requires unified_demand_path from Data Analyst."
        )

    project_root = Path(__file__).resolve().parent
    output_path = project_root / "data" / "outputs" / "inventory_decisions.csv"

    inventory_agent = InventoryDecisionAgent(
        inventory_path=project_root / "data" / "inventory_dataset.csv",
        recipe_path=project_root / "data" / "ingredients.csv",
        unified_demand_path=unified_path,
    )

    result = inventory_agent.run(
        forecast_path=forecast_path,
        output_path=output_path,
    )

    shared_context.setdefault("inventory", {}).update({
        "output_path": result.get("output_path"),
        "forecast_horizon_days": result.get("forecast_horizon_days"),
        "orders_recommended": result.get("orders_recommended"),
        "stockout_risk_counts": result.get("stockout_risk_counts", {}),
    })

    print("[SHARED CONTEXT] Inventory decision registered.")
    print("[INTEGRATION] Inventory Decision Agent completed.")

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
# REGISTER INVENTORY DECISION
# ============================================================

orchestrator.register_agent(
    agent_name="inventory_decision",
    agent_function=run_inventory_decision,
    capabilities=[
        "inventory_decision",
        "order_recommendation",
        "stockout_analysis",
    ],
    dependencies=[
        "demand_forecasting"
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
        # Warning / unavailable forecast
        # ----------------------------------------------------

        if forecast.get("status") == "warning":

            print(
                forecast.get(
                    "message",
                    "Forecast is not available."
                )
            )

            if forecast.get("available_forecast_start"):

                print(
                    "Available Forecast Start:",
                    forecast[
                        "available_forecast_start"
                    ]
                )

            if forecast.get("available_forecast_end"):

                print(
                    "Available Forecast End:",
                    forecast[
                        "available_forecast_end"
                    ]
                )

        # ----------------------------------------------------
        # Multiple explicitly paired restaurant/item results
        # ----------------------------------------------------

        elif forecast.get("combination_results"):

            combination_results = forecast.get(
                "combination_results",
                []
            )

            print(
                "Number of combinations:",
                len(combination_results)
            )

            for combination in combination_results:

                restaurant_id = combination.get(
                    "restaurant_id"
                )

                menu_item_id = combination.get(
                    "menu_item_id"
                )

                total_demand = combination.get(
                    "total_predicted_demand"
                )

                daily_forecast = combination.get(
                    "daily_forecast",
                    []
                )

                print(
                    "\n"
                    + "-" * 45
                )

                print(
                    "Restaurant:",
                    restaurant_id
                )

                print(
                    "Menu Item:",
                    menu_item_id
                )

                if total_demand is not None:

                    print(
                        "Total Predicted Demand:",
                        round(
                            float(total_demand),
                            2
                        )
                    )

                if daily_forecast:

                    print(
                        "\nForecast:"
                    )

                    for row in daily_forecast:

                        date = row.get(
                            "date"
                        )

                        predicted_quantity = row.get(
                            "predicted_quantity"
                        )

                        if predicted_quantity is not None:

                            print(
                                f"  {date} : "
                                f"{float(predicted_quantity):.2f}"
                            )

            print(
                "\n"
                + "-" * 45
            )

        # ----------------------------------------------------
        # Original single restaurant/item result
        # ----------------------------------------------------

        else:

            restaurant_id = forecast.get(
                "restaurant_id"
            )

            menu_item_id = forecast.get(
                "menu_item_id"
            )

            requested_date = forecast.get(
                "date"
            )

            total_demand = forecast.get(
                "total_predicted_demand"
            )

            daily_forecast = forecast.get(
                "daily_forecast",
                []
            )

            if restaurant_id is not None:

                print(
                    "Restaurant:",
                    restaurant_id
                )

            if menu_item_id is not None:

                print(
                    "Menu Item:",
                    menu_item_id
                )

            if requested_date is not None:

                print(
                    "Date:",
                    requested_date
                )

            if total_demand is not None:

                print(
                    "\nTotal Predicted Demand:",
                    round(
                        float(total_demand),
                        2
                    )
                )

            if daily_forecast:

                print(
                    "\nForecast:"
                )

                for row in daily_forecast:

                    date = row.get(
                        "date"
                    )

                    predicted_quantity = row.get(
                        "predicted_quantity"
                    )

                    if predicted_quantity is not None:

                        print(
                            f"  {date} : "
                            f"{float(predicted_quantity):.2f}"
                        )

        # ----------------------------------------------------
        # Backtest evaluation
        # ----------------------------------------------------

        evaluation = result.get(
            "evaluation",
            {}
        )

        if evaluation:

            print(
                "\nBacktest Evaluation:"
            )

            wmape = evaluation.get(
                "wmape_percent"
            )

            if wmape is not None:

                print(
                    "wMAPE:",
                    wmape,
                    "%"
                )

            mae = evaluation.get(
                "mae"
            )

            if mae is not None:

                print(
                    "MAE:",
                    mae
                )

            rmse = evaluation.get(
                "rmse"
            )

            if rmse is not None:

                print(
                    "RMSE:",
                    rmse
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