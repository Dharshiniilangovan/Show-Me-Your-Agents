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
from agents.anomaly_validation.agent import anomaly_validation_agent
from agents.waste_reduction.agent import agent as waste_reduction_agent

# ============================================================
# DATASET PATH
# ============================================================
#
# Change ONLY this path if your CSV is somewhere else.
#
# ============================================================

DATASET_PATH = (
   r"C:\Users\saaral\Desktop\hackathon\code\Show-Me-Your-Agents\data\qsr_demand_dataset.csv"
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
    # PRESERVE RESTAURANT + MENU-ITEM DETAIL FOR ALL-RESTAURANT
    # MULTI-ITEM REQUESTS
    # --------------------------------------------------------
    # The forecasting agent stores a complete 7-day forecast. For a request
    # such as "item 1 and item 2 in all restaurants for the next 2 days",
    # the default all-scope response aggregates all restaurants/items by date.
    # Rebuild the user-facing result from the detailed forecast CSV so that
    # each restaurant/menu-item combination is returned separately.

    requested_menu_items = request.get("menu_item_ids") or []
    restaurant_scope = request.get("restaurant_scope")

    if restaurant_scope == "all" and requested_menu_items:
        forecast_path = result.get("output_path")

        if forecast_path and Path(forecast_path).exists():
            detailed_forecast = pd.read_csv(forecast_path, low_memory=False)
            detailed_forecast["date"] = pd.to_datetime(
                detailed_forecast["date"], errors="coerce"
            )
            detailed_forecast["predicted_quantity"] = pd.to_numeric(
                detailed_forecast["predicted_quantity"], errors="coerce"
            )
            detailed_forecast = detailed_forecast.dropna(
                subset=["date", "predicted_quantity"]
            )

            detailed_forecast = detailed_forecast[
                detailed_forecast["menu_item_id"].astype(str).isin(
                    [str(value) for value in requested_menu_items]
                )
            ].copy()

            requested_days = int(
                request.get("requested_forecast_horizon")
                or request.get("forecast_horizon")
                or 7
            )
            requested_days = max(1, min(requested_days, 7))

            requested_dates = (
                detailed_forecast["date"]
                .drop_duplicates()
                .sort_values()
                .head(requested_days)
            )
            detailed_forecast = detailed_forecast[
                detailed_forecast["date"].isin(requested_dates)
            ].copy()

            restaurant_names = {}
            if "restaurant_name" in restaurant_lookup.columns:
                restaurant_names = dict(
                    zip(
                        restaurant_lookup["restaurant_id"].astype(str),
                        restaurant_lookup["restaurant_name"].astype(str),
                    )
                )

            menu_names = {}
            if "menu_item_name" in menu_lookup.columns:
                menu_names = dict(
                    zip(
                        menu_lookup["menu_item_id"].astype(str),
                        menu_lookup["menu_item_name"].astype(str),
                    )
                )

            combination_results = []

            grouped = detailed_forecast.groupby(
                ["restaurant_id", "menu_item_id"],
                sort=True,
                dropna=False,
            )

            for (restaurant_id, menu_item_id), group in grouped:
                group = group.sort_values("date")
                daily_forecast = []

                for _, row in group.iterrows():
                    daily_forecast.append({
                        "date": row["date"].strftime("%Y-%m-%d"),
                        "predicted_quantity": float(row["predicted_quantity"]),
                    })

                combination_results.append({
                    "restaurant_id": restaurant_id,
                    "restaurant_name": restaurant_names.get(str(restaurant_id)),
                    "menu_item_id": menu_item_id,
                    "menu_item_name": menu_names.get(str(menu_item_id)),
                    "total_predicted_demand": float(
                        group["predicted_quantity"].sum()
                    ),
                    "daily_forecast": daily_forecast,
                })

            if combination_results:
                result["result"] = {
                    "scope": "all_restaurants_selected_items",
                    "requested_forecast_horizon": requested_days,
                    "combination_results": combination_results,
                    "combination_count": len(combination_results),
                }

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
# REAL ANOMALY / VALIDATION AGENT WRAPPER
# ============================================================

def run_anomaly_validation(state):
    """
    Validate the demand forecast for each requested restaurant/menu-item pair.

    The current Demand Forecasting Agent returns combination_results, while
    the existing anomaly-validation agent expects one restaurant/item and a
    predicted_demand value. This wrapper adapts between those formats.
    """

    print(
        "[INTEGRATION] Starting Anomaly / Validation Agent..."
    )

    request = state.get(
        "request",
        {}
    )

    agent_results = state.get(
        "agent_results",
        {}
    )

    demand_result = agent_results.get(
        "demand_forecasting",
        {}
    )

    forecast_result = demand_result.get(
        "result",
        {}
    )

    combinations = forecast_result.get(
        "combination_results",
        []
    )

    # Backward compatibility for a single-result forecast.
    if not combinations:

        total_predicted_demand = forecast_result.get(
            "total_predicted_demand"
        )

        if total_predicted_demand is not None:

            combinations = [
                {
                    "restaurant_id":
                        forecast_result.get(
                            "restaurant_id"
                        )
                        or request.get(
                            "restaurant_id"
                        ),

                    "menu_item_id":
                        forecast_result.get(
                            "menu_item_id"
                        )
                        or request.get(
                            "menu_item_id"
                        ),

                    "total_predicted_demand":
                        total_predicted_demand,
                }
            ]

    if not combinations:

        raise ValueError(
            "Anomaly Validation Agent requires a demand forecast result."
        )

    dataset_path = (
        state.get(
            "shared_context",
            {}
        )
        .get(
            "data",
            {}
        )
        .get(
            "unified_demand_path"
        )
    )

    if not dataset_path:

        dataset_path = str(
            DATA_ANALYST_OUTPUT_PATH
        )

    validation_results = []

    for combination in combinations:

        restaurant_id = combination.get(
            "restaurant_id"
        )

        menu_item_id = combination.get(
            "menu_item_id"
        )

        predicted_demand = combination.get(
            "total_predicted_demand"
        )

        if predicted_demand is None:
            continue

        validation_state = dict(
            state
        )

        validation_state[
            "request"
        ] = dict(
            request
        )

        validation_state[
            "request"
        ][
            "restaurant_scope"
        ] = "single"

        validation_state[
            "request"
        ][
            "restaurant_id"
        ] = restaurant_id

        validation_state[
            "request"
        ][
            "menu_item_id"
        ] = menu_item_id

        validation_state[
            "agent_results"
        ] = dict(
            agent_results
        )

        # Format expected by the existing anomaly-validation agent.
        validation_state[
            "agent_results"
        ][
            "demand_forecasting"
        ] = {
            "restaurant_id":
                restaurant_id,

            "menu_item_id":
                menu_item_id,

            "predicted_demand":
                float(
                    predicted_demand
                ),

            "forecast_horizon":
                demand_result.get(
                    "forecast_horizon",
                    7
                ),

            "confidence":
                None,
        }

        validation = anomaly_validation_agent(
            validation_state,
            dataset_path=dataset_path,
        )

        validation_results.append(
            validation
        )

    if not validation_results:

        raise ValueError(
            "No forecast combinations were available "
            "for anomaly validation."
        )

    overall_status = "pass"

    if any(
        result.get(
            "validation_status"
        ) == "fail"
        for result in validation_results
    ):

        overall_status = "fail"

    elif any(
        result.get(
            "validation_status"
        ) == "warning"
        for result in validation_results
    ):

        overall_status = "warning"

    result = {
        "analysis_type":
            "forecast_validation",

        "validation_status":
            overall_status,

        "combination_results":
            validation_results,

        "combination_count":
            len(
                validation_results
            ),

        "reforecast_recommended":
            any(
                bool(
                    validation.get(
                        "reforecast_recommended"
                    )
                )
                for validation in validation_results
            ),
    }

    state.setdefault(
        "shared_context",
        {}
    ).setdefault(
        "validation",
        {}
    ).update(
        {
            "validation_status":
                overall_status,

            "combination_count":
                len(
                    validation_results
                ),

            "reforecast_recommended":
                result[
                    "reforecast_recommended"
                ],
        }
    )

    print(
        "[SHARED CONTEXT] Anomaly validation registered."
    )

    print(
        "[INTEGRATION] Anomaly / Validation Agent completed."
    )

    return result


# ============================================================
# REAL INVENTORY DECISION AGENT WRAPPER
# ============================================================

def run_inventory_decision(state):
    """
    Run the Inventory Decision Agent, then retrieve only the ingredient-level
    inventory insights relevant to the restaurant/menu-item pairs in the
    user's original request.

    The complete inventory_decisions.csv is still generated and stored.
    Retrieval does not recalculate the inventory decisions.
    """

    print(
        "[INTEGRATION] Starting Inventory Decision Agent..."
    )

    shared_context = state.get(
        "shared_context",
        {}
    )

    request = state.get(
        "request",
        {}
    )

    # --------------------------------------------------------
    # REQUIRE DEMAND FORECAST OUTPUT
    # --------------------------------------------------------

    forecast_context = shared_context.get(
        "forecast",
        {}
    )

    forecast_path = forecast_context.get(
        "output_path"
    )

    if not forecast_path:

        raise ValueError(
            "Inventory Decision Agent requires the demand forecast output "
            "from shared_context['forecast']['output_path']."
        )

    forecast_path = Path(
        forecast_path
    )

    if not forecast_path.exists():

        raise FileNotFoundError(
            f"Demand forecast file not found: {forecast_path}"
        )

    # --------------------------------------------------------
    # REQUIRE UNIFIED DEMAND DATA
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
            "Inventory Decision Agent requires unified_demand_path "
            "from Data Analyst."
        )

    # --------------------------------------------------------
    # PROJECT FILES
    # --------------------------------------------------------

    project_root = (
        Path(__file__)
        .resolve()
        .parent
    )

    inventory_path = (
        project_root
        / "data"
        / "inventory_dataset.csv"
    )

    recipe_path = (
        project_root
        / "data"
        / "ingredients.csv"
    )

    output_path = (
        project_root
        / "data"
        / "outputs"
        / "inventory_decisions.csv"
    )

    if not recipe_path.exists():

        raise FileNotFoundError(
            f"Ingredient mapping file not found: {recipe_path}"
        )

    # --------------------------------------------------------
    # RUN INVENTORY DECISION AGENT
    # --------------------------------------------------------
    # This continues to generate the COMPLETE inventory decision
    # file. We filter it only after generation.

    inventory_agent = InventoryDecisionAgent(
        inventory_path=inventory_path,
        recipe_path=recipe_path,
        unified_demand_path=unified_path,
    )

    result = inventory_agent.run(
        forecast_path=forecast_path,
        output_path=output_path,
    )

    if not isinstance(
        result,
        dict
    ):

        raise ValueError(
            "Inventory Decision Agent returned an invalid result."
        )

    # Prefer the path returned by the agent.
    decision_path = Path(
        result.get(
            "output_path",
            output_path
        )
    )

    if not decision_path.is_absolute():

        decision_path = (
            project_root
            / decision_path
        )

    decision_path = decision_path.resolve()

    if not decision_path.exists():

        raise FileNotFoundError(
            "Inventory decision output was not created: "
            f"{decision_path}"
        )

    # --------------------------------------------------------
    # LOAD STORED RESULTS + MENU-ITEM/INGREDIENT MAPPING
    # --------------------------------------------------------

    decisions_df = pd.read_csv(
        decision_path,
        low_memory=False
    )

    ingredients_df = pd.read_csv(
        recipe_path,
        low_memory=False
    )

    required_mapping_columns = {
        "menu_item_id",
        "ingredient_id",
    }

    missing_mapping_columns = (
        required_mapping_columns
        - set(ingredients_df.columns)
    )

    if missing_mapping_columns:

        raise ValueError(
            "ingredients.csv is missing required columns: "
            + ", ".join(
                sorted(
                    missing_mapping_columns
                )
            )
        )

    if "ingredient_id" not in decisions_df.columns:

        raise ValueError(
            "inventory_decisions.csv must contain ingredient_id "
            "for menu-item-level retrieval."
        )

    # --------------------------------------------------------
    # GET EXACT RESTAURANT / MENU-ITEM PAIRS
    # --------------------------------------------------------

    requested_pairs = (
        request.get(
            "restaurant_item_pairs"
        )
        or []
    )

    # Backward compatibility for a normal single request.
    if not requested_pairs:

        restaurant_id = request.get(
            "restaurant_id"
        )

        menu_item_id = request.get(
            "menu_item_id"
        )

        if menu_item_id is not None:

            requested_pairs = [
                {
                    "restaurant_id":
                        restaurant_id,

                    "menu_item_id":
                        menu_item_id,
                }
            ]

    # --------------------------------------------------------
    # LOAD FORECAST TO GET PAIR-SPECIFIC FORECAST DEMAND
    # --------------------------------------------------------

    forecast_df = pd.read_csv(
        forecast_path,
        low_memory=False
    )

    # --------------------------------------------------------
    # BUILD DETAILED USER-REQUESTED INVENTORY INSIGHTS
    # --------------------------------------------------------

    combination_results = []

    for pair in requested_pairs:

        restaurant_id = pair.get(
            "restaurant_id"
        )

        menu_item_id = pair.get(
            "menu_item_id"
        )

        if menu_item_id is None:
            continue

        # -----------------------------------------------
        # Find the requested menu item's ingredients.
        # -----------------------------------------------

        item_ingredients = ingredients_df[
            ingredients_df[
                "menu_item_id"
            ].astype(str)
            == str(
                menu_item_id
            )
        ].copy()

        if item_ingredients.empty:
            continue

        ingredient_ids = (
            item_ingredients[
                "ingredient_id"
            ]
            .astype(str)
            .unique()
            .tolist()
        )

        # -----------------------------------------------
        # Retrieve stored inventory decisions.
        # -----------------------------------------------

        item_decisions = decisions_df[
            decisions_df[
                "ingredient_id"
            ].astype(str)
            .isin(
                ingredient_ids
            )
        ].copy()

        # Merge recipe information so count_per_item/unit/menu
        # names remain available in the final response.
        merge_columns = [
            column
            for column in [
                "menu_item_id",
                "menu_item_name",
                "ingredient_id",
                "ingredient_name",
                "count_per_item",
                "unit",
            ]
            if column in item_ingredients.columns
        ]

        mapping = (
            item_ingredients[
                merge_columns
            ]
            .drop_duplicates()
        )

        merged = mapping.merge(
            item_decisions,
            on="ingredient_id",
            how="left",
            suffixes=(
                "_recipe",
                "_inventory"
            )
        )

        # -----------------------------------------------
        # Get forecast demand for THIS restaurant/item.
        # -----------------------------------------------

        pair_forecast = forecast_df.copy()

        if (
            restaurant_id is not None
            and "restaurant_id"
            in pair_forecast.columns
        ):

            pair_forecast = pair_forecast[
                pair_forecast[
                    "restaurant_id"
                ].astype(str)
                == str(
                    restaurant_id
                )
            ]

        if "menu_item_id" in pair_forecast.columns:

            pair_forecast = pair_forecast[
                pair_forecast[
                    "menu_item_id"
                ].astype(str)
                == str(
                    menu_item_id
                )
            ]

        forecast_demand = None

        if (
            not pair_forecast.empty
            and "predicted_quantity"
            in pair_forecast.columns
        ):

            forecast_demand = float(
                pair_forecast[
                    "predicted_quantity"
                ].sum()
            )

        # -----------------------------------------------
        # Convert rows into clean dictionaries.
        # -----------------------------------------------

        ingredient_results = []

        for _, row in merged.iterrows():

            record = {
                "ingredient_id":
                    row.get(
                        "ingredient_id"
                    ),

                "ingredient_name":
                    (
                        row.get(
                            "ingredient_name_recipe"
                        )
                        if pd.notna(
                            row.get(
                                "ingredient_name_recipe"
                            )
                        )
                        else row.get(
                            "ingredient_name_inventory"
                        )
                    ),

                "count_per_item":
                    row.get(
                        "count_per_item"
                    ),

                "unit":
                    (
                        row.get(
                            "unit_recipe"
                        )
                        if pd.notna(
                            row.get(
                                "unit_recipe"
                            )
                        )
                        else row.get(
                            "unit_inventory"
                        )
                    ),
            }

            # Copy useful inventory-decision columns when present.
            for column in [
                "forecast_period_demand",
                "current_stock",
                "safety_stock",
                "lead_time_days",
                "lead_time_demand",
                "reorder_point",
                "recommended_order_quantity",
                "stockout_risk",
                "order_required",
                "decision_reason",
            ]:

                if column in merged.columns:

                    value = row.get(
                        column
                    )

                    if pd.isna(value):
                        value = None

                    elif hasattr(
                        value,
                        "item"
                    ):

                        try:
                            value = value.item()
                        except Exception:
                            pass

                    record[
                        column
                    ] = value

            ingredient_results.append(
                record
            )

        menu_item_name = None

        if (
            "menu_item_name"
            in item_ingredients.columns
            and not item_ingredients.empty
        ):

            menu_item_name = (
                item_ingredients.iloc[0]
                .get(
                    "menu_item_name"
                )
            )

        combination_results.append(
            {
                "restaurant_id":
                    restaurant_id,

                "menu_item_id":
                    menu_item_id,

                "menu_item_name":
                    menu_item_name,

                "forecast_demand":
                    forecast_demand,

                "ingredients":
                    ingredient_results,
            }
        )

    # --------------------------------------------------------
    # REGISTER COMPLETE OUTPUT IN SHARED CONTEXT
    # --------------------------------------------------------

    shared_context.setdefault(
        "inventory",
        {}
    ).update(
        {
            "output_path":
                str(
                    decision_path
                ),

            "forecast_horizon_days":
                result.get(
                    "forecast_horizon_days"
                ),

            "orders_recommended":
                result.get(
                    "orders_recommended"
                ),

            "stockout_risk_counts":
                result.get(
                    "stockout_risk_counts",
                    {}
                ),
        }
    )

    print(
        "[SHARED CONTEXT] Inventory decision registered."
    )

    print(
        "[INTEGRATION] Inventory Decision Agent completed."
    )

    # --------------------------------------------------------
    # RETURN REQUEST-SPECIFIC INSIGHTS
    # --------------------------------------------------------

    return {
        "status":
            "success",

        "output_path":
            str(
                decision_path
            ),

        "forecast_horizon_days":
            result.get(
                "forecast_horizon_days"
            ),

        "orders_recommended":
            result.get(
                "orders_recommended"
            ),

        "stockout_risk_counts":
            result.get(
                "stockout_risk_counts",
                {}
            ),

        "combination_results":
            combination_results,
    }


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
# REGISTER ANOMALY / VALIDATION
# ============================================================

orchestrator.register_agent(

    agent_name=
        "anomaly_validation",

    agent_function=
        run_anomaly_validation,

    capabilities=[
        "forecast_validation",
        "anomaly_detection",
        "data_quality_validation",
        "stockout_risk_validation",
        "overstock_risk_validation",
    ],

    dependencies=[
        "demand_forecasting"
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
# REGISTER WASTE REDUCTION
# ============================================================

orchestrator.register_agent(
    agent_name="waste_reduction",
    agent_function=waste_reduction_agent,
    capabilities=[
        "waste_reduction",
        "expiry_risk_analysis",
        "waste_risk_analysis",
        "promotion_recommendation",
        "stock_reallocation_recommendation",
        "priority_selling_recommendation",
        "future_order_reduction_recommendation",
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

                restaurant_name = combination.get(
                    "restaurant_name"
                )

                if restaurant_name is not None:
                    print(
                        "Restaurant Name:",
                        restaurant_name
                    )

                print(
                    "Menu Item:",
                    menu_item_id
                )

                menu_item_name = combination.get(
                    "menu_item_name"
                )

                if menu_item_name is not None:
                    print(
                        "Menu Item Name:",
                        menu_item_name
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
    # ========================================================
    # ANOMALY / VALIDATION
    # ========================================================

    if "anomaly_validation" in results:

        validation_result = results[
            "anomaly_validation"
        ]

        print(
            "\n--- ANOMALY / VALIDATION ---"
        )

        print(
            "Overall Validation Status:",
            validation_result.get(
                "validation_status"
            )
        )

        print(
            "Reforecast Recommended:",
            validation_result.get(
                "reforecast_recommended"
            )
        )

        validations = validation_result.get(
            "combination_results",
            []
        )

        for validation in validations:

            print(
                "\n"
                + "-" * 45
            )

            print(
                "Restaurant:",
                validation.get(
                    "restaurant_id"
                )
            )

            print(
                "Menu Item:",
                validation.get(
                    "menu_item_id"
                )
            )

            print(
                "Validation Status:",
                validation.get(
                    "validation_status"
                )
            )

            print(
                "Forecast Valid:",
                validation.get(
                    "forecast_valid"
                )
            )

            metrics = validation.get(
                "forecast_metrics",
                {}
            )

            if metrics:

                for label, key in [
                    ("Predicted Demand", "predicted_demand"),
                    ("Historical Baseline", "historical_baseline"),
                    ("Deviation %", "deviation_pct"),
                ]:

                    value = metrics.get(
                        key
                    )

                    if value is not None:

                        print(
                            f"{label}:",
                            value
                        )

            historical = validation.get(
                "historical_anomalies",
                {}
            )

            if historical:

                for label, key in [
                    ("Historical Anomaly Count", "count"),
                    ("Historical Anomaly Rate", "rate"),
                    ("Recent Demand Anomaly", "recent_anomaly"),
                ]:

                    value = historical.get(
                        key
                    )

                    if value is not None:

                        print(
                            f"{label}:",
                            value
                        )

            issues = validation.get(
                "issues",
                []
            )

            if issues:

                print(
                    "Issues:"
                )

                for issue in issues:

                    print(
                        "  -",
                        issue.get(
                            "severity"
                        ),
                        issue.get(
                            "code"
                        ),
                        ":",
                        issue.get(
                            "message"
                        )
                    )

            else:

                print(
                    "Issues: None"
                )

        if validations:

            print(
                "\n"
                + "-" * 45
            )


    # ========================================================
    # INVENTORY DECISION
    # ========================================================

    if "inventory_decision" in results:

        inventory_result = results[
            "inventory_decision"
        ]

        print(
            "\n--- INVENTORY DECISION ---"
        )

        combination_results = (
            inventory_result.get(
                "combination_results",
                []
            )
        )

        if combination_results:

            for combination in combination_results:

                print(
                    "\n"
                    + "=" * 45
                )

                print(
                    "Restaurant:",
                    combination.get(
                        "restaurant_id"
                    )
                )

                print(
                    "Menu Item:",
                    combination.get(
                        "menu_item_id"
                    )
                )

                if combination.get(
                    "menu_item_name"
                ) is not None:

                    print(
                        "Menu Item Name:",
                        combination.get(
                            "menu_item_name"
                        )
                    )

                if combination.get(
                    "forecast_demand"
                ) is not None:

                    print(
                        "Forecast Demand:",
                        round(
                            float(
                                combination.get(
                                    "forecast_demand"
                                )
                            ),
                            2
                        )
                    )

                ingredients = combination.get(
                    "ingredients",
                    []
                )

                if not ingredients:

                    print(
                        "No ingredient inventory decisions were found."
                    )

                for ingredient in ingredients:

                    print(
                        "\n"
                        + "-" * 45
                    )

                    display_fields = [
                        ("Ingredient ID", "ingredient_id"),
                        ("Ingredient", "ingredient_name"),
                        ("Quantity Per Item", "count_per_item"),
                        ("Unit", "unit"),
                        ("Forecast Period Demand", "forecast_period_demand"),
                        ("Current Stock", "current_stock"),
                        ("Safety Stock", "safety_stock"),
                        ("Lead Time Days", "lead_time_days"),
                        ("Lead Time Demand", "lead_time_demand"),
                        ("Reorder Point", "reorder_point"),
                        ("Recommended Order Quantity", "recommended_order_quantity"),
                        ("Stockout Risk", "stockout_risk"),
                        ("Order Required", "order_required"),
                        ("Decision Reason", "decision_reason"),
                    ]

                    for label, key in display_fields:

                        value = ingredient.get(
                            key
                        )

                        if value is None:
                            continue

                        if isinstance(
                            value,
                            (int, float)
                        ):

                            value = round(
                                float(value),
                                2
                            )

                        print(
                            f"{label}:",
                            value
                        )

            print(
                "\n"
                + "=" * 45
            )

        else:

            print(
                "No request-specific ingredient inventory "
                "insights were found."
            )

            if inventory_result.get(
                "output_path"
            ):

                print(
                    "Complete inventory decisions stored at:",
                    inventory_result.get(
                        "output_path"
                    )
                )

    # ========================================================
    # WASTE REDUCTION
    # ========================================================

    if "waste_reduction" in results:

        waste_result = results["waste_reduction"]

        print(
            "\n--- WASTE REDUCTION ---"
        )

        print(
            "Analysis Date:",
            waste_result.get("analysis_date")
        )

        if waste_result.get("analysis_type") == "single_item_waste_reduction":
            display_fields = [
                ("Ingredient ID", "ingredient_id"),
                ("Ingredient", "ingredient_name"),
                ("Waste Risk", "waste_risk"),
                ("Days to Expiry", "days_to_expiry"),
                ("Expected Waste Quantity", "expected_waste_quantity"),
                ("Promotion Recommended", "promotion_recommended"),
                ("Recommendations", "recommendations"),
            ]

            for label, key in display_fields:
                value = waste_result.get(key)
                if value is not None:
                    print(f"{label}:", value)

        else:
            print(
                "Inventory Items Analyzed:",
                waste_result.get("total_inventory_items_analyzed", 0)
            )
            print(
                "Products At Risk:",
                waste_result.get("products_at_risk", 0)
            )
            print(
                "Critical / High Risk Items:",
                waste_result.get("critical_or_high_risk_items", 0)
            )
            print(
                "Promotion Recommended Items:",
                waste_result.get("promotion_recommended_items", 0)
            )
            # Print the actual waste-risk items instead of only showing the
            # generated CSV path. The CSV is still created in the background.
            items = waste_result.get("items", [])
            risk_items = [
                item for item in items
                if str(item.get("waste_risk", "")).lower()
                in {"critical", "high", "medium"}
            ]

            print("\nProducts At Risk of Waste:")

            if not risk_items:
                print(
                    "No products currently identified at significant "
                    "waste risk."
                )
            else:
                for item in risk_items:
                    print("\n---------------------------------------------")
                    print("Ingredient:", item.get("ingredient_name"))
                    print("Ingredient ID:", item.get("ingredient_id"))
                    print(
                        "Current Stock:",
                        item.get("current_stock"),
                        item.get("unit") or ""
                    )
                    print("Expiry Date:", item.get("expiry_date"))
                    print("Days to Expiry:", max(item.get("days_to_expiry"),0))
                    print(
                        "Expected Usage Before Expiry:",
                        item.get("expected_demand_until_expiry")
                    )
                    print(
                        "Surplus Quantity:",
                        item.get("surplus_quantity")
                    )
                    print(
                        "Waste Risk:",
                        str(item.get("waste_risk", "")).upper()
                    )
                    print(
                        "Demand Source:",
                        item.get("demand_source")
                    )

                    recommendations = item.get("recommendations", [])
                    print("Recommendations:")
                    if isinstance(recommendations, list):
                        for recommendation in recommendations:
                            print(
                                "  -",
                                str(recommendation)
                                .replace("_", " ")
                                .title()
                            )
                    elif recommendations:
                        print("  -", recommendations)
                    else:
                        print("  - No waste-reduction action required")

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
        "        MULTI-AGENT DEMAND & WASTE REDUCTION SYSTEM"
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
    "  5. Inventory Decision Agent"
    )

    print(
        "  6. Anomaly / Validation Agent"
    )

    print(
        "  7. Waste Reduction Agent"
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