"""
Context & Seasonality Agent
===========================

This module exposes two interfaces:

1. ContextSeasonalityAgent.run(df)
   --------------------------------
   Offline / batch analysis interface.

   Used for:
   - generating context_features.csv
   - generating seasonality summaries
   - generating contextual impact reports
   - generating sku_context_signals.csv
   - generating agent_context.json

2. context_seasonality_agent(state)
   ---------------------------------
   Runtime / Orchestrator interface.

   Used by the multi-agent system.

   The runtime interface:
   - receives the common shared state
   - obtains the Data Analyst output path from
     state["shared_data"]["unified_demand_path"]
   - DOES NOT perform generic data preprocessing
   - performs context/seasonality feature engineering
   - generates contextual signals
   - returns a Python dictionary

The Data Analyst Agent owns generic preprocessing.
"""

import json

import pandas as pd

from .config import (
    PROCESSED,
    OUTPUTS,
)

from .data_loader import (
    load_unified_data,
)

from .feature_engineering import (
    add_context_features,
)

from .seasonality import (
    seasonality_summary,
)

from .context_analysis import (
    binary_impact,
    weather_impact,
)

from .signals import (
    build_sku_signals,
)

from .schemas import (
    AgentRunSummary,
)


# ============================================================
# CONSTANTS
# ============================================================

AGENT_NAME = "context_seasonality"
AGENT_VERSION = "1.2"


SUPPORTED_CAPABILITIES = {
    "seasonality_analysis",
    "promotion_analysis",
    "holiday_analysis",
    "event_analysis",
    "weather_analysis",
}


# ============================================================
# OFFLINE / BATCH ANALYSIS AGENT
# ============================================================

class ContextSeasonalityAgent:
    """
    Offline/batch implementation of the Context &
    Seasonality Agent.

    This class receives an already prepared DataFrame.

    It does NOT own generic preprocessing.

    The class is useful for:
    - development
    - EDA
    - diagnostics
    - generating forecasting features
    - generating cached context knowledge
    """

    VERSION = AGENT_VERSION

    def run(self, df: pd.DataFrame) -> dict:

        # ----------------------------------------------------
        # Create output directories
        # ----------------------------------------------------

        PROCESSED.mkdir(
            parents=True,
            exist_ok=True,
        )

        OUTPUTS.mkdir(
            parents=True,
            exist_ok=True,
        )

        # ----------------------------------------------------
        # Context-specific feature engineering
        # ----------------------------------------------------

        x = add_context_features(df)

        files = []

        # ====================================================
        # MODEL FEATURE DATASET
        # ====================================================

        feature_path = (
            PROCESSED
            / "context_features.csv"
        )

        x.to_csv(
            feature_path,
            index=False,
        )

        files.append(
            str(feature_path)
        )

        sample_path = (
            PROCESSED
            / "context_features_sample.csv"
        )

        x.head(10000).to_csv(
            sample_path,
            index=False,
        )

        # ====================================================
        # SEASONALITY ANALYSIS
        # ====================================================

        for frame in seasonality_summary(x):

            if frame.empty:
                continue

            dimension = (
                frame["dimension"]
                .iloc[0]
            )

            path = (
                OUTPUTS
                / f"seasonality_{dimension}.csv"
            )

            frame.to_csv(
                path,
                index=False,
            )

            files.append(
                str(path)
            )

        # ====================================================
        # CONTEXT IMPACT ANALYSIS
        # ====================================================

        analyses = [
            (
                "holiday_impact.csv",
                binary_impact(
                    x,
                    "is_holiday",
                    "holiday_name",
                ),
            ),

            (
                "promotion_impact.csv",
                binary_impact(
                    x,
                    "is_promotion",
                ),
            ),

            (
                "event_impact.csv",
                binary_impact(
                    x,
                    "is_special_event",
                    "special_event_name",
                ),
            ),
        ]

        for name, frame in analyses:

            path = (
                OUTPUTS
                / name
            )

            frame.to_csv(
                path,
                index=False,
            )

            files.append(
                str(path)
            )

        # ====================================================
        # WEATHER ANALYSIS
        # ====================================================

        (
            temperature,
            precipitation,
            precipitation_type,
        ) = weather_impact(x)

        weather_files = [
            (
                "weather_temperature.csv",
                temperature,
            ),

            (
                "weather_precipitation.csv",
                precipitation,
            ),

            (
                "weather_precip_type.csv",
                precipitation_type,
            ),
        ]

        for name, frame in weather_files:

            path = (
                OUTPUTS
                / name
            )

            frame.to_csv(
                path,
                index=False,
            )

            files.append(
                str(path)
            )

        # ====================================================
        # RESTAURANT / SKU CONTEXT KNOWLEDGE
        # ====================================================

        signals = build_sku_signals(x)

        signal_path = (
            OUTPUTS
            / "sku_context_signals.csv"
        )

        signals.to_csv(
            signal_path,
            index=False,
        )

        files.append(
            str(signal_path)
        )

        # ====================================================
        # AGENT SUMMARY
        # ====================================================

        summary = AgentRunSummary(
            len(x),

            str(
                x["date"]
                .min()
                .date()
            ),

            str(
                x["date"]
                .max()
                .date()
            ),
            
            x["restaurant_id"].nunique(),
            x["menu_item_id"].nunique(),
            int(
                x["quantity"]
                .isna()
                .sum()
            ),
            files,
        )

        # ====================================================
        # FEATURE CONTRACT
        # ====================================================

        payload = {
            "agent":
                "ContextSeasonalityAgent",

            "version":
                self.VERSION,

            "summary":
                summary.to_dict(),

            "feature_contract": {
                "forecast_keys": [
                    "date",
                    "restaurant_id",
                    "menu_item_id",
                ],

                "target":
                    "quantity",

                "categorical_context_features": [
                    "holiday_name",
                    "special_event_name",
                    "precip_type",
                ],

                "numeric_context_features": [
                    "dow_sin",
                    "dow_cos",
                    "month_sin",
                    "month_cos",
                    "doy_sin",
                    "doy_cos",
                    "week_of_year",
                    "quarter",
                    "day_of_month",
                    "is_month_start",
                    "is_month_end",
                    "is_weekend",
                    "is_holiday",
                    "is_special_event",
                    "is_promotion",
                    "avg_temp_f",
                    "temp_c",
                    "precip_inches",
                    "has_precipitation",
                    "context_event_count",
                    "is_context_day",
                    "weekend_promotion",
                    "holiday_promotion",
                    "event_promotion",
                    "precipitation_weekend",
                ],

                "historical_signal_file":
                    "sku_context_signals.csv",

                "signal_note": (
                    "Historical uplifts are "
                    "descriptive associations, "
                    "not causal effects."
                ),
            },
        }

        # ====================================================
        # OPTIONAL OFFLINE METADATA FILE
        # ====================================================

        context_path = (
            OUTPUTS
            / "agent_context.json"
        )

        with open(
            context_path,
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                payload,
                file,
                indent=2,
            )

        return payload


# ============================================================
# SERIALIZATION HELPERS
# ============================================================

def _safe_float(value):
    """
    Convert pandas/numpy numeric values into normal
    Python floats.

    Returns None for missing values.

    This makes the returned dictionary safe for:
    - shared state
    - JSON serialization
    - API responses
    """

    if value is None:
        return None

    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass

    return float(value)


def _safe_int(value):
    """
    Convert pandas/numpy numeric values into normal
    Python integers.

    Returns None for missing values.
    """

    if value is None:
        return None

    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass

    return int(value)


# ============================================================
# SIGNAL SERIALIZATION
# ============================================================

def _build_signal(
    row: pd.Series,
    prefix: str,
) -> dict:
    """
    Convert one historical context signal into the
    common agent-output representation.

    Example prefix:
        promotion

    Expected source columns:
        promotion_uplift_pct
        promotion_active_count
        promotion_baseline_count
        promotion_signal_level
        promotion_confidence
    """

    return {
        "uplift_pct":
            _safe_float(
                row[
                    f"{prefix}_uplift_pct"
                ]
            ),

        "active_count":
            _safe_int(
                row[
                    f"{prefix}_active_count"
                ]
            ),

        "baseline_count":
            _safe_int(
                row[
                    f"{prefix}_baseline_count"
                ]
            ),

        "signal_level":
            str(
                row[
                    f"{prefix}_signal_level"
                ]
            ),

        "confidence":
            str(
                row[
                    f"{prefix}_confidence"
                ]
            ),
    }


# ============================================================
# RESPONSE BUILDER
# ============================================================

def _build_context_response(
    row: pd.Series,
    required_capabilities: list,
) -> dict:
    """
    Build the Context & Seasonality Agent response for
    one restaurant/SKU combination.

    Only requested context signals are returned when
    context-specific capabilities are supplied.

    If no Context & Seasonality capability is supplied,
    all signals are returned.
    """

    all_signals = {
        "weekend":
            _build_signal(
                row,
                "weekend",
            ),

        "holiday":
            _build_signal(
                row,
                "holiday",
            ),

        "event":
            _build_signal(
                row,
                "event",
            ),

        "promotion":
            _build_signal(
                row,
                "promotion",
            ),

        "precipitation":
            _build_signal(
                row,
                "precipitation",
            ),
    }

    # --------------------------------------------------------
    # Capability → signal mapping
    # --------------------------------------------------------

    capability_map = {
        "seasonality_analysis": {
            "weekend",
        },

        "holiday_analysis": {
            "holiday",
        },

        "event_analysis": {
            "event",
        },

        "promotion_analysis": {
            "promotion",
        },

        "weather_analysis": {
            "precipitation",
        },
    }

    # --------------------------------------------------------
    # Identify Context Agent capabilities requested
    # --------------------------------------------------------

    context_capabilities = [
        capability
        for capability
        in required_capabilities
        if capability
        in SUPPORTED_CAPABILITIES
    ]

    # --------------------------------------------------------
    # No Context-specific filtering requested
    # --------------------------------------------------------

    if not context_capabilities:

        selected_signals = (
            all_signals
        )

    else:

        requested_signals = set()

        for capability in context_capabilities:

            requested_signals.update(
                capability_map[
                    capability
                ]
            )

        selected_signals = {
            signal_name:
                signal_value

            for (
                signal_name,
                signal_value
            )
            in all_signals.items()

            if signal_name
            in requested_signals
        }

    # --------------------------------------------------------
    # Final dictionary
    # --------------------------------------------------------

    return {
        "restaurant_scope":
            "single",

        "restaurant_id":
            str(
                row[
                    "restaurant_id"
                ]
            ),

        "menu_item_id":
            str(
                row[
                    "menu_item_id"
                ]
            ),

        "category":
            str(
                row[
                    "category"
                ]
            ),

        "baseline_demand":
            _safe_float(
                row[
                    "baseline_demand"
                ]
            ),

        "baseline_observation_count":
            _safe_int(
                row[
                    "baseline_observation_count"
                ]
            ),

        "signals":
            selected_signals,

        "signal_note": (
            "Historical uplifts are descriptive "
            "associations, not causal effects."
        ),
    }


# ============================================================
# ORCHESTRATOR ENTRY POINT
# ============================================================

def context_seasonality_agent(
    state: dict,
) -> dict:
    """
    Runtime entry point used by the Orchestrator.

    ----------------------------------------------------------
    EXPECTED STATE
    ----------------------------------------------------------

    state = {
        "request": {
            "restaurant_scope": "single",
            "restaurant_id": "R01",
            "menu_item_id": "M01",
            ...
        },

        "required_capabilities": [
            "promotion_analysis",
            "holiday_analysis",
            ...
        ],

        "agent_results": {
            ...
        },

        "shared_data": {
            "unified_demand_path":
                ".../unified_demand.csv"
        }
    }

    ----------------------------------------------------------
    IMPORTANT
    ----------------------------------------------------------

    The Data Analyst Agent owns generic preprocessing.

    This function does NOT:
    - load the original raw QSR dataset;
    - hard-code an input dataset path;
    - clean raw data;
    - remove duplicates;
    - perform outlier correction;
    - repair generic missing data.

    It receives the processed dataset produced by the
    Data Analyst Agent.

    ----------------------------------------------------------
    RETURNS
    ----------------------------------------------------------

    A normal Python dictionary suitable for:

        state["agent_results"][
            "context_seasonality"
        ]

    The Orchestrator is responsible for storing the
    returned dictionary in shared state.
    """

    # ========================================================
    # 1. VALIDATE STATE
    # ========================================================

    if not isinstance(
        state,
        dict,
    ):
        raise TypeError(
            "Context & Seasonality Agent expected "
            "state to be a dictionary."
        )

    # ========================================================
    # 2. READ COMMON STATE
    # ========================================================

    request = state.get(
        "request",
        {},
    )

    required_capabilities = state.get(
        "required_capabilities",
        [],
    )

    # Read for compatibility with the common contract.
    # The current Context Agent does not require previous
    # agent results because its processed dataset path is
    # supplied through shared_data.
    previous_results = state.get(
        "agent_results",
        {},
    )

    shared_data = state.get(
        "shared_data",
        {},
    )

    # Avoid unused-variable warnings while documenting
    # compatibility with the common contract.
    _ = previous_results

    # ========================================================
    # 3. GET DATA ANALYST OUTPUT PATH
    # ========================================================

    unified_path = shared_data.get(
        "unified_demand_path"
    )

    if not unified_path:

        raise ValueError(
            "Context & Seasonality Agent requires "
            "state['shared_data']"
            "['unified_demand_path']. "
            "The Data Analyst Agent must run first."
        )

    # ========================================================
    # 4. LOAD PROCESSED DATA ANALYST OUTPUT
    # ========================================================

    df = load_unified_data(
        unified_path
    )

    # ========================================================
    # 5. CONTEXT-SPECIFIC FEATURE ENGINEERING
    # ========================================================

    context_df = (
        add_context_features(
            df
        )
    )

    # ========================================================
    # 6. BUILD CONTEXT SIGNALS
    #
    # IMPORTANT:
    # Use the complete unified dataset here.
    #
    # Do NOT filter to one restaurant/SKU before this step,
    # because hierarchical fallback may need:
    #
    # restaurant+SKU
    #       ↓
    # SKU
    #       ↓
    # category
    #       ↓
    # global
    # ========================================================

    signals = build_sku_signals(
        context_df
    )

    # ========================================================
    # 7. READ REQUEST SCOPE
    # ========================================================

    restaurant_scope = request.get(
        "restaurant_scope",
        "single",
    )

    restaurant_id = request.get(
        "restaurant_id"
    )

    menu_item_id = request.get(
        "menu_item_id"
    )

    # ========================================================
    # 8. VALIDATE RESTAURANT SCOPE
    # ========================================================

    valid_scopes = {
        "single",
        "all",
    }

    if restaurant_scope not in valid_scopes:

        raise ValueError(
            "restaurant_scope must be either "
            "'single' or 'all'."
        )

    # ========================================================
    # 9. VALIDATE MENU ITEM
    # ========================================================

    if not menu_item_id:

        raise ValueError(
            "menu_item_id is required for "
            "Context & Seasonality analysis."
        )

    # ========================================================
    # 10. FILTER SIGNAL RESULT BY MENU ITEM
    # ========================================================

    selected = signals[
        signals[
            "menu_item_id"
        ].astype(str)
        == str(menu_item_id)
    ]

    # ========================================================
    # 11. SINGLE RESTAURANT FILTER
    # ========================================================

    if restaurant_scope == "single":

        if not restaurant_id:

            raise ValueError(
                "restaurant_id is required "
                "when restaurant_scope='single'."
            )

        selected = selected[
            selected[
                "restaurant_id"
            ].astype(str)
            == str(restaurant_id)
        ]

    # ========================================================
    # 12. VALIDATE RESULT EXISTS
    # ========================================================

    if selected.empty:

        if (
            restaurant_scope
            == "single"
        ):

            raise ValueError(
                "No Context & Seasonality "
                "result found for "
                f"restaurant_id={restaurant_id}, "
                f"menu_item_id={menu_item_id}."
            )

        raise ValueError(
            "No Context & Seasonality "
            "result found for "
            f"menu_item_id={menu_item_id}."
        )

    # ========================================================
    # 13. SINGLE RESTAURANT RESPONSE
    # ========================================================

    if restaurant_scope == "single":

        row = selected.iloc[0]

        result = (
            _build_context_response(
                row,
                required_capabilities,
            )
        )

        result["agent"] = (
            AGENT_NAME
        )

        result["version"] = (
            AGENT_VERSION
        )

        return result

    # ========================================================
    # 14. ALL RESTAURANTS RESPONSE
    # ========================================================

    restaurant_breakdown = []

    selected = selected.sort_values(
        "restaurant_id"
    )

    for _, row in selected.iterrows():

        response = (
            _build_context_response(
                row,
                required_capabilities,
            )
        )

        restaurant_breakdown.append(
            response
        )

    return {
        "agent":
            AGENT_NAME,

        "version":
            AGENT_VERSION,

        "restaurant_scope":
            "all",

        "menu_item_id":
            str(menu_item_id),

        "restaurant_count":
            len(
                restaurant_breakdown
            ),

        "restaurant_breakdown":
            restaurant_breakdown,

        "signal_note": (
            "Historical uplifts are descriptive "
            "associations, not causal effects."
        ),
    }