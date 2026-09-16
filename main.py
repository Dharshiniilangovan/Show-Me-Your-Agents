import pandas as pd

from agents.orchestrator.agent import (
    OrchestratorAgent
)


# ============================================================
# DATASET
# ============================================================

DATASET_PATH = (
    r"C:\Users\saaral\Desktop\hackathon\data"
    r"\2026-havi-niu-hackathon"
    r"\qsr_demand_dataset.csv"
)


print(
    "\nLoading QSR demand dataset..."
)

df = pd.read_csv(
    DATASET_PATH
)

print(
    f"Dataset loaded: "
    f"{len(df):,} rows"
)


# ============================================================
# DATASET LOOKUPS
# ============================================================

restaurant_lookup = (
    df[
        [
            "restaurant_id",
            "restaurant_name"
        ]
    ]
    .drop_duplicates()
)


menu_lookup = (
    df[
        [
            "menu_item_id",
            "menu_item_name"
        ]
    ]
    .drop_duplicates()
)


# ============================================================
# RESOLVE RESTAURANT
# ============================================================

def resolve_restaurant(
    request
):

    restaurant_id = (
        request.get(
            "restaurant_id"
        )
    )

    restaurant_name = (
        request.get(
            "restaurant_name"
        )
    )

    # --------------------------------------------------------
    # ID PROVIDED
    # --------------------------------------------------------

    if restaurant_id:

        match = restaurant_lookup[
            restaurant_lookup[
                "restaurant_id"
            ]
            .astype(str)
            .str.lower()
            ==
            str(
                restaurant_id
            ).lower()
        ]

        if not match.empty:

            row = match.iloc[0]

            return (
                row[
                    "restaurant_id"
                ],

                row[
                    "restaurant_name"
                ]
            )

        raise ValueError(
            f'I couldn\'t find restaurant '
            f'ID "{restaurant_id}". '
            "Please provide another restaurant "
            "ID, name, or number."
        )

    # --------------------------------------------------------
    # NAME PROVIDED
    # --------------------------------------------------------

    if restaurant_name:

        match = restaurant_lookup[
            restaurant_lookup[
                "restaurant_name"
            ]
            .astype(str)
            .str.lower()
            ==
            str(
                restaurant_name
            ).lower()
        ]

        if not match.empty:

            row = match.iloc[0]

            return (
                row[
                    "restaurant_id"
                ],

                row[
                    "restaurant_name"
                ]
            )

        raise ValueError(
            f'I couldn\'t find '
            f'"{restaurant_name}". '
            "Please provide another restaurant "
            "ID, name, or number."
        )

    raise ValueError(
        "Please provide a restaurant ID, "
        "restaurant name, or restaurant number."
    )


# ============================================================
# RESOLVE MENU ITEM
# ============================================================

def resolve_menu_item(
    request
):

    menu_item_id = (
        request.get(
            "menu_item_id"
        )
    )

    menu_item_name = (
        request.get(
            "menu_item_name"
        )
    )

    # --------------------------------------------------------
    # ID PROVIDED
    # --------------------------------------------------------

    if menu_item_id:

        match = menu_lookup[
            menu_lookup[
                "menu_item_id"
            ]
            .astype(str)
            .str.lower()
            ==
            str(
                menu_item_id
            ).lower()
        ]

        if not match.empty:

            row = match.iloc[0]

            return (
                row[
                    "menu_item_id"
                ],

                row[
                    "menu_item_name"
                ]
            )

        raise ValueError(
            f'I couldn\'t find menu item '
            f'ID "{menu_item_id}". '
            "Please provide another menu item "
            "ID, name, or number."
        )

    # --------------------------------------------------------
    # NAME PROVIDED
    # --------------------------------------------------------

    if menu_item_name:

        match = menu_lookup[
            menu_lookup[
                "menu_item_name"
            ]
            .astype(str)
            .str.lower()
            ==
            str(
                menu_item_name
            ).lower()
        ]

        if not match.empty:

            row = match.iloc[0]

            return (
                row[
                    "menu_item_id"
                ],

                row[
                    "menu_item_name"
                ]
            )

        raise ValueError(
            f'I couldn\'t find menu item '
            f'"{menu_item_name}". '
            "Please provide another menu item "
            "ID, name, or number."
        )

    raise ValueError(
        "Please provide a menu item ID, "
        "menu item name, or item number."
    )


# ============================================================
# HELPERS
# ============================================================

def has_capability(
    state,
    capability
):

    return (
        capability
        in state.get(
            "required_capabilities",
            []
        )
    )


def has_menu_item(
    request
):

    return (
        request.get(
            "menu_item_id"
        )
        is not None

        or

        request.get(
            "menu_item_name"
        )
        is not None
    )


# ============================================================
# DUMMY DATA ANALYST
# ============================================================

def dummy_data_analyst(
    state
):

    request = (
        state[
            "request"
        ]
    )

    # ========================================================
    # GET RESTAURANT SCOPE
    # ========================================================

    restaurant_scope = (
        request.get(
            "restaurant_scope"
        )
    )

    # ========================================================
    # ALL RESTAURANTS
    # ========================================================

    if restaurant_scope == "all":

        # ----------------------------------------------------
        # STOCKOUT WITHOUT MENU ITEM
        #
        # This case will eventually require an all-restaurant
        # inventory strategy. For now we keep it separate.
        # ----------------------------------------------------

        if (
            has_capability(
                state,
                "stockout_analysis"
            )

            and

            not has_menu_item(
                request
            )
        ):

            raise ValueError(
                "All-restaurant stockout analysis "
                "is not implemented in the dummy "
                "agent yet."
            )

        # ----------------------------------------------------
        # RESOLVE MENU ITEM
        # ----------------------------------------------------

        menu_item_id, menu_item_name = (
            resolve_menu_item(
                request
            )
        )

        # ----------------------------------------------------
        # GET THIS ITEM ACROSS ALL RESTAURANTS
        # ----------------------------------------------------

        item_data = (
            df[
                df[
                    "menu_item_id"
                ]
                == menu_item_id
            ]
        )

        if item_data.empty:

            raise ValueError(
                f"I couldn't find records for "
                f"{menu_item_name} across "
                f"all restaurants."
            )

        # ----------------------------------------------------
        # OVERALL STATISTICS
        # ----------------------------------------------------

        average_quantity = (
            item_data[
                "quantity"
            ].mean()
        )

        total_quantity = (
            item_data[
                "quantity"
            ].sum()
        )

        # ----------------------------------------------------
        # RESTAURANT-BY-RESTAURANT BREAKDOWN
        # ----------------------------------------------------

        restaurant_breakdown = (
            item_data
            .groupby(
                [
                    "restaurant_id",
                    "restaurant_name"
                ],
                as_index=False
            )
            .agg(
                avg_daily_sales=(
                    "quantity",
                    "mean"
                ),

                total_historical_quantity=(
                    "quantity",
                    "sum"
                )
            )
        )

        breakdown = []

        for _, row in (
            restaurant_breakdown
            .iterrows()
        ):

            breakdown.append({

                "restaurant_id":
                    row[
                        "restaurant_id"
                    ],

                "restaurant_name":
                    row[
                        "restaurant_name"
                    ],

                "avg_daily_sales":
                    round(
                        float(
                            row[
                                "avg_daily_sales"
                            ]
                        ),
                        2
                    ),

                "total_historical_quantity":
                    int(
                        row[
                            "total_historical_quantity"
                        ]
                    )
            })

        # ----------------------------------------------------
        # RETURN ALL-RESTAURANT RESULT
        # ----------------------------------------------------

        return {

            "restaurant_scope":
                "all",

            "menu_item_id":
                menu_item_id,

            "menu_item_name":
                menu_item_name,

            "avg_daily_sales":
                round(
                    float(
                        average_quantity
                    ),
                    2
                ),

            "total_historical_quantity":
                int(
                    total_quantity
                ),

            "restaurant_breakdown":
                breakdown,

            # There is no single inventory value
            # for all restaurants.
            "current_stock":
                None
        }

    # ========================================================
    # SINGLE RESTAURANT
    # ========================================================

    restaurant_id, restaurant_name = (
        resolve_restaurant(
            request
        )
    )

    # ========================================================
    # RESTAURANT-WIDE STOCKOUT ANALYSIS
    # ========================================================

    if (
        has_capability(
            state,
            "stockout_analysis"
        )

        and

        not has_menu_item(
            request
        )
    ):

        restaurant_data = (
            df[
                df[
                    "restaurant_id"
                ]
                == restaurant_id
            ]
        )

        items = (
            restaurant_data[
                [
                    "menu_item_id",
                    "menu_item_name"
                ]
            ]
            .drop_duplicates()
            .head(10)
            .reset_index(
                drop=True
            )
        )
        holiday_name = (
    request.get(
        "holiday_name"
    )
)

    if holiday_name:

        if "holiday_name" not in item_data.columns:

            raise ValueError(
                "Holiday information is not "
                "available in the dataset."
            )

        item_data = (
            item_data[
                item_data[
                    "holiday_name"
                ]
                .astype(str)
                .str.lower()
                ==
                str(
                    holiday_name
                ).lower()
            ]
        )

        # ----------------------------------------------------
        # Temporary dummy inventory
        # ----------------------------------------------------

        dummy_stock = [
            500,
            150,
            100,
            600,
            80,
            220,
            350,
            90,
            410,
            175
        ]

        output_items = []

        for (
            index,
            row
        ) in items.iterrows():

            output_items.append({

                "menu_item_id":
                    row[
                        "menu_item_id"
                    ],

                "menu_item_name":
                    row[
                        "menu_item_name"
                    ],

                "current_stock":
                    dummy_stock[
                        index
                        % len(
                            dummy_stock
                        )
                    ]
            })

        return {

            "restaurant_scope":
                "single",

            "restaurant_id":
                restaurant_id,

            "restaurant_name":
                restaurant_name,

            "items":
                output_items
        }

    # ========================================================
    # SINGLE MENU ITEM + SINGLE RESTAURANT
    # ========================================================

    menu_item_id, menu_item_name = (
        resolve_menu_item(
            request
        )
    )

    item_data = (
        df[
            (
                df[
                    "restaurant_id"
                ]
                == restaurant_id
            )
            &
            (
                df[
                    "menu_item_id"
                ]
                == menu_item_id
            )
        ]
    )

    if item_data.empty:

        raise ValueError(
            f"I couldn't find records for "
            f"{menu_item_name} at "
            f"{restaurant_name}."
        )

    average_quantity = (
        item_data[
            "quantity"
        ].mean()
    )

    total_quantity = (
        item_data[
            "quantity"
        ].sum()
    )

    return {

        "restaurant_scope":
            "single",

        "restaurant_id":
            restaurant_id,

        "restaurant_name":
            restaurant_name,

        "menu_item_id":
            menu_item_id,

        "menu_item_name":
            menu_item_name,

        "avg_daily_sales":
            round(
                float(
                    average_quantity
                ),
                2
            ),

        "total_historical_quantity":
            int(
                total_quantity
            ),

        # Temporary simulated inventory
        "current_stock":
            500
    }
# ============================================================
# DUMMY CONTEXT / SEASONALITY AGENT
# ============================================================

def dummy_seasonality(state):

    request = state["request"]

    required_capabilities = (
        state.get(
            "required_capabilities",
            []
        )
    )

    # ========================================================
    # COMPLETE DUMMY CONTEXT RESULT
    # ========================================================

    all_signals = {

        "weekend": {
            "uplift_pct": 20.86,
            "active_count": 520,
            "baseline_count": 1302,
            "signal_level": "restaurant_sku",
            "confidence": "high"
        },

        "holiday": {
            "uplift_pct": 18.76,
            "active_count": 61,
            "baseline_count": 1761,
            "signal_level": "restaurant_sku",
            "confidence": "medium"
        },

        "event": {
            "uplift_pct": 78.76,
            "active_count": 55,
            "baseline_count": 1767,
            "signal_level": "restaurant_sku",
            "confidence": "medium"
        },

        "promotion": {
            "uplift_pct": 27.13,
            "active_count": 283,
            "baseline_count": 26997,
            "signal_level": "sku",
            "confidence": "high"
        },

        "precipitation": {
            "uplift_pct": -5.66,
            "active_count": 628,
            "baseline_count": 1194,
            "signal_level": "restaurant_sku",
            "confidence": "high"
        }
    }

    # ========================================================
    # MAP CAPABILITY -> RELEVANT SIGNAL
    # ========================================================

    capability_signal_map = {

        "promotion_analysis":
            "promotion",

        "holiday_analysis":
            "holiday",

        "event_analysis":
            "event"
    }

    # ========================================================
    # FILTER SIGNALS
    # ========================================================

    filtered_signals = {}

    for capability in required_capabilities:

        signal_name = (
            capability_signal_map.get(
                capability
            )
        )

        if (
            signal_name is not None
            and
            signal_name in all_signals
        ):

            filtered_signals[
                signal_name
            ] = all_signals[
                signal_name
            ]

    # ========================================================
    # GENERAL SEASONALITY REQUEST
    #
    # If the user asks for general seasonality rather than
    # one specific context, return the complete context.
    # ========================================================

    if (
        "seasonality_analysis"
        in required_capabilities
    ):

        filtered_signals = (
            all_signals
        )

    # ========================================================
    # FALLBACK
    #
    # Useful if the Context Agent is executed as a dependency
    # of another agent such as Demand Forecasting.
    #
    # In that situation the user's final capability may be
    # demand_forecasting, so there may be no direct
    # context-specific capability in required_capabilities.
    #
    # The forecasting agent can receive the complete context.
    # ========================================================

    if not filtered_signals:

        filtered_signals = (
            all_signals
        )

    # ========================================================
    # RETURN
    # ========================================================

    return {

        "restaurant_id":
            request.get(
                "restaurant_id"
            ),

        "menu_item_id":
            request.get(
                "menu_item_id"
            ),

        "category":
            "Burgers",

        "baseline_demand":
            19.84,

        "signals":
            filtered_signals
    }

# ============================================================
# DUMMY CUSTOMER PATTERN AGENT
# ============================================================

def dummy_customer_pattern(
    state
):

    return {

        "customer_demand_trend":
            "increasing",

        "pattern_strength":
            0.82
    }


# ============================================================
# DUMMY DEMAND FORECASTING AGENT
# ============================================================

def dummy_forecasting(
    state
):

    request = (
        state[
            "request"
        ]
    )

    data_result = (
        state[
            "agent_results"
        ].get(
            "data_analyst",
            {}
        )
    )

    # ========================================================
    # MULTI-ITEM FORECAST FOR STOCKOUT ANALYSIS
    # ========================================================

    if (
        has_capability(
            state,
            "stockout_analysis"
        )

        and

        not has_menu_item(
            request
        )
    ):

        forecasts = []

        restaurant_id = (
            data_result.get(
                "restaurant_id"
            )
        )

        for item in (
            data_result.get(
                "items",
                []
            )
        ):

            menu_item_id = (
                item[
                    "menu_item_id"
                ]
            )

            historical = df[
                (
                    df[
                        "restaurant_id"
                    ]
                    == restaurant_id
                )
                &
                (
                    df[
                        "menu_item_id"
                    ]
                    == menu_item_id
                )
            ]

            average = (
                historical[
                    "quantity"
                ].mean()
            )

            # Temporary 7-day forecast.
            predicted = (
                average
                * 7
            )

            forecasts.append({

                "menu_item_id":
                    menu_item_id,

                "predicted_demand":
                    round(
                        float(
                            predicted
                        )
                    )
            })

        return {

            "item_forecasts":
                forecasts,

            "forecast_horizon":
                7
        }

    # ========================================================
    # SINGLE ITEM FORECAST
    # ========================================================

    horizon = (
        request.get(
            "forecast_horizon"
        )
        or 7
    )

    average = (
        data_result.get(
            "avg_daily_sales",
            0
        )
    )

    predicted = (
        average
        * horizon
    )

    return {

        "restaurant_id":
            data_result.get(
                "restaurant_id"
            ),

        "restaurant_name":
            data_result.get(
                "restaurant_name"
            ),

        "menu_item_id":
            data_result.get(
                "menu_item_id"
            ),

        "menu_item_name":
            data_result.get(
                "menu_item_name"
            ),

        "predicted_demand":
            round(
                float(
                    predicted
                )
            ),

        "forecast_horizon":
            horizon,

        "confidence":
            0.87
    }


# ============================================================
# DUMMY INVENTORY DECISION AGENT
# ============================================================

def dummy_inventory(
    state
):

    request = (
        state[
            "request"
        ]
    )

    data_result = (
        state[
            "agent_results"
        ].get(
            "data_analyst",
            {}
        )
    )

    forecast_result = (
        state[
            "agent_results"
        ].get(
            "demand_forecasting",
            {}
        )
    )

    # ========================================================
    # MULTI-ITEM STOCKOUT ANALYSIS
    # ========================================================

    if (
        has_capability(
            state,
            "stockout_analysis"
        )

        and

        not has_menu_item(
            request
        )
    ):

        forecast_lookup = {

            item[
                "menu_item_id"
            ]:
                item[
                    "predicted_demand"
                ]

            for item in (
                forecast_result.get(
                    "item_forecasts",
                    []
                )
            )
        }

        risky_items = []

        for item in (
            data_result.get(
                "items",
                []
            )
        ):

            item_id = (
                item[
                    "menu_item_id"
                ]
            )

            stock = (
                item[
                    "current_stock"
                ]
            )

            demand = (
                forecast_lookup.get(
                    item_id,
                    0
                )
            )

            shortage = (
                demand
                - stock
            )

            if shortage <= 0:

                continue

            ratio = (
                shortage
                / demand
                if demand > 0
                else 0
            )

            if ratio >= 0.5:

                risk = "high"

            elif ratio >= 0.25:

                risk = "medium"

            else:

                risk = "low"

            risky_items.append({

                "menu_item_id":
                    item_id,

                "menu_item_name":
                    item[
                        "menu_item_name"
                    ],

                "current_stock":
                    stock,

                "predicted_demand":
                    demand,

                "shortage_quantity":
                    shortage,

                "stockout_risk":
                    risk
            })

        return {

            "analysis_type":
                "multi_item_stockout",

            "restaurant_id":
                data_result.get(
                    "restaurant_id"
                ),

            "restaurant_name":
                data_result.get(
                    "restaurant_name"
                ),

            "number_of_items_at_risk":
                len(
                    risky_items
                ),

            "stockout_items":
                risky_items
        }

    # ========================================================
    # SINGLE ITEM INVENTORY
    # ========================================================

    predicted = (
        forecast_result.get(
            "predicted_demand",
            0
        )
    )

    current_stock = (
        data_result.get(
            "current_stock",
            500
        )
    )

    shortage = max(
        predicted
        - current_stock,
        0
    )

    if predicted <= current_stock:

        risk = "low"

    else:

        ratio = (
            shortage
            / predicted
            if predicted > 0
            else 0
        )

        if ratio >= 0.5:

            risk = "high"

        elif ratio >= 0.25:

            risk = "medium"

        else:

            risk = "low"

    return {

        "analysis_type":
            "single_item_inventory",

        "restaurant_id":
            forecast_result.get(
                "restaurant_id"
            ),

        "restaurant_name":
            forecast_result.get(
                "restaurant_name"
            ),

        "menu_item_id":
            forecast_result.get(
                "menu_item_id"
            ),

        "menu_item_name":
            forecast_result.get(
                "menu_item_name"
            ),

        "current_stock":
            current_stock,

        "predicted_demand":
            predicted,

        "shortage_quantity":
            shortage,

        "recommended_order_quantity":
            shortage,

        "stockout_risk":
            risk
    }


# ============================================================
# CREATE ORCHESTRATOR
# ============================================================

orchestrator = (
    OrchestratorAgent()
)


# ============================================================
# REGISTER AGENTS
# ============================================================

orchestrator.register_agent(

    "data_analyst",

    dummy_data_analyst,

    capabilities=[
        "historical_sales_analysis",
        "recent_demand_analysis",
        "data_retrieval"
    ],

    dependencies=[],

    required_user_inputs=[
        "restaurant"
    ]
)


orchestrator.register_agent(

    "context_seasonality",

    dummy_seasonality,

    capabilities=[
        "seasonality_analysis",
        "promotion_analysis",
        "holiday_analysis",
        "event_analysis"
    ],

    dependencies=[],

    required_user_inputs=[
        "restaurant"
    ]
)


orchestrator.register_agent(

    "customer_pattern",

    dummy_customer_pattern,

    capabilities=[
        "customer_pattern_analysis"
    ],

    dependencies=[],

    required_user_inputs=[
        "restaurant"
    ]
)


orchestrator.register_agent(

    "demand_forecasting",

    dummy_forecasting,

    capabilities=[
        "demand_forecasting"
    ],

    dependencies=[
        "historical_sales_analysis",
        "seasonality_analysis",
        "customer_pattern_analysis"
    ],

    required_user_inputs=[
        "restaurant",
        "menu_item",
        "forecast_horizon"
    ]
)


orchestrator.register_agent(

    "inventory_decision",

    dummy_inventory,

    capabilities=[
        "inventory_analysis",
        "stockout_analysis",
        "order_recommendation"
    ],

    dependencies=[
        "demand_forecasting"
    ],

    required_user_inputs=[
        "restaurant"
    ]
)


# ============================================================
# GENERIC RESULT DISPLAY
# ============================================================

def format_label(key):
    """
    Convert:
        seasonal_factor
    into:
        Seasonal factor
    """

    return (
        str(key)
        .replace("_", " ")
        .strip()
        .capitalize()
    )


def display_dictionary(
    data,
    indent=0
):
    """
    Generic dictionary display.

    This allows results from ANY agent to be shown
    without adding agent-specific print logic.
    """

    spacing = " " * indent

    for key, value in data.items():

        label = format_label(key)

        # ----------------------------------------------------
        # NESTED DICTIONARY
        # ----------------------------------------------------

        if isinstance(
            value,
            dict
        ):

            print(
                f"{spacing}{label}:"
            )

            display_dictionary(
                value,
                indent + 4
            )

        # ----------------------------------------------------
        # LIST
        # ----------------------------------------------------

        elif isinstance(
            value,
            list
        ):

            print(
                f"{spacing}{label}:"
            )

            if not value:

                print(
                    f"{spacing}    None"
                )

                continue

            for item in value:

                if isinstance(
                    item,
                    dict
                ):

                    print(
                        f"{spacing}    -"
                    )

                    display_dictionary(
                        item,
                        indent + 8
                    )

                else:

                    print(
                        f"{spacing}    - {item}"
                    )

        # ----------------------------------------------------
        # NORMAL VALUE
        # ----------------------------------------------------

        else:

            print(
                f"{spacing}{label}: {value}"
            )


def display_agent_results(
    response
):
    """
    Display results from every agent that actually
    participated in the workflow.

    This prevents results from an agent being silently
    ignored by main.py.
    """

    results = response.get(
        "results",
        {}
    )

    workflow = response.get(
        "workflow",
        []
    )

    if not results:

        print(
            "\nBot: The analysis completed, "
            "but no agent returned a result."
        )

        return

    print(
        "\nBot: Analysis completed."
    )

    goal = response.get(
        "goal"
    )

    if goal:

        print(
            f"\nGoal: {goal}"
        )

    # --------------------------------------------------------
    # DISPLAY RESULTS IN WORKFLOW ORDER
    # --------------------------------------------------------

    displayed_agents = set()

    for agent_name in workflow:

        result = results.get(
            agent_name
        )

        if result is None:

            continue

        displayed_agents.add(
            agent_name
        )

        title = (
            agent_name
            .replace("_", " ")
            .title()
        )

        print(
            f"\n{title}:"
        )

        print(
            "-" * len(
                title
            )
        )

        if isinstance(
            result,
            dict
        ):

            display_dictionary(
                result,
                indent=0
            )

        else:

            print(
                result
            )

    # --------------------------------------------------------
    # SAFETY CHECK
    #
    # If an agent returned something but for some reason
    # wasn't in workflow, still show it.
    # --------------------------------------------------------

    for (
        agent_name,
        result
    ) in results.items():

        if (
            agent_name
            in displayed_agents
        ):

            continue

        title = (
            agent_name
            .replace("_", " ")
            .title()
        )

        print(
            f"\n{title}:"
        )

        print(
            "-" * len(
                title
            )
        )

        if isinstance(
            result,
            dict
        ):

            display_dictionary(
                result
            )

        else:

            print(
                result
            )


# ============================================================
# CHATBOT
# ============================================================

print()

print(
    "=" * 65
)

print(
    "             DEMAND FORECASTING ASSISTANT"
)

print(
    "=" * 65
)

print(
    "\nYou can use restaurant/menu item IDs, "
    "names, or simple numbers."
)

print(
    "\nType 'exit' to stop.\n"
)


# ============================================================
# CHAT LOOP
# ============================================================

while True:

    user_input = input(
        "You: "
    ).strip()

    # --------------------------------------------------------
    # EXIT
    # --------------------------------------------------------

    if user_input.lower() in [
        "exit",
        "quit",
        "bye"
    ]:

        print(
            "\nBot: Goodbye!"
        )

        break

    if not user_input:

        continue

    # --------------------------------------------------------
    # SEND QUERY TO ORCHESTRATOR
    # --------------------------------------------------------

    response = (
        orchestrator
        .process_request(
            user_input
        )
    )

    status = response.get(
        "status"
    )

    # --------------------------------------------------------
    # CONVERSATIONAL FOLLOW-UP
    # --------------------------------------------------------

    if status in [
        "clarification_required",
        "missing_parameters",
        "needs_user_input",
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

    # --------------------------------------------------------
    # INTERNAL FAILURE
    # --------------------------------------------------------

    if status == "completed_with_errors":

        print(
            "\nBot: I couldn't complete the "
            "analysis because one of the required "
            "agents encountered a problem."
        )

        continue

    # --------------------------------------------------------
    # DISPLAY ALL AGENT RESULTS
    # --------------------------------------------------------

    display_agent_results(
        response
    )

    print()