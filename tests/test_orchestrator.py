import pytest

from agents.orchestrator.agent import OrchestratorAgent


# ============================================================
# MOCK AGENTS
# ============================================================

def mock_data_analyst(state):

    return {
        "restaurant_id": "R01",
        "menu_item_id": "M12",
        "avg_daily_sales": 120,
        "current_stock": 500
    }


def mock_seasonality(state):

    return {
        "weekly_pattern": "weekend_peak",
        "seasonal_factor": 1.15,
        "holiday_effect": 0.05
    }


def mock_customer_pattern(state):

    return {
        "customer_demand_trend": "increasing",
        "pattern_strength": 0.82
    }


def mock_forecasting(state):

    return {
        "restaurant_id": "R01",
        "menu_item_id": "M12",
        "predicted_demand": 850,
        "forecast_horizon": 7,
        "confidence": 0.87
    }


def mock_inventory(state):

    forecast = (
        state["agent_results"]
        ["demand_forecasting"]
    )

    predicted_demand = (
        forecast["predicted_demand"]
    )

    current_stock = 500

    recommended_quantity = max(
        predicted_demand - current_stock,
        0
    )

    return {
        "current_stock":
            current_stock,

        "predicted_demand":
            predicted_demand,

        "recommended_order_quantity":
            recommended_quantity,

        "stockout_risk":
            "medium"
    }


# ============================================================
# CREATE TEST ORCHESTRATOR
# ============================================================

def create_test_orchestrator():

    orchestrator = OrchestratorAgent()

    # --------------------------------------------------------
    # DATA ANALYST
    # --------------------------------------------------------

    orchestrator.register_agent(
        "data_analyst",
        mock_data_analyst,

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

    # --------------------------------------------------------
    # CONTEXT / SEASONALITY
    # --------------------------------------------------------

    orchestrator.register_agent(
        "context_seasonality",
        mock_seasonality,

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

    # --------------------------------------------------------
    # CUSTOMER PATTERN
    # --------------------------------------------------------

    orchestrator.register_agent(
        "customer_pattern",
        mock_customer_pattern,

        capabilities=[
            "customer_pattern_analysis"
        ],

        dependencies=[],

        required_user_inputs=[
            "restaurant"
        ]
    )

    # --------------------------------------------------------
    # DEMAND FORECASTING
    # --------------------------------------------------------

    orchestrator.register_agent(
        "demand_forecasting",
        mock_forecasting,

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

    # --------------------------------------------------------
    # INVENTORY DECISION
    # --------------------------------------------------------

    orchestrator.register_agent(
        "inventory_decision",
        mock_inventory,

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

    return orchestrator


# ============================================================
# TEST 1
# NATURAL LANGUAGE CAPABILITY SELECTION
# ============================================================

@pytest.mark.parametrize(
    "query, expected_capability",
    [

        # ----------------------------------------------------
        # DEMAND FORECASTING
        # ----------------------------------------------------

        (
            "Forecast demand for M12 at R01 next week",
            "demand_forecasting"
        ),

        (
            "What will the demand for M12 "
            "at R01 be next week?",
            "demand_forecasting"
        ),

        (
            "Predict how much M05 will be needed "
            "at R02 tomorrow",
            "demand_forecasting"
        ),

        (
            "Estimate next month's demand "
            "for M10 at R03",
            "demand_forecasting"
        ),

        (
            "Predict demand for M12 at R01 "
            "for the coming week",
            "demand_forecasting"
        ),

        (
            "What will sales of M12 look like "
            "at R01 next week?",
            "demand_forecasting"
        ),

        (
            "Can you estimate how much M12 "
            "restaurant 1 will need next week?",
            "demand_forecasting"
        ),

        # ----------------------------------------------------
        # ORDER RECOMMENDATION
        # ----------------------------------------------------

        (
            "How much M12 should R01 "
            "order next week?",
            "order_recommendation"
        ),

        (
            "How many units of M05 should "
            "restaurant 2 prepare next week?",
            "order_recommendation"
        ),

        (
            "What quantity of M10 should R03 "
            "replenish for the coming week?",
            "order_recommendation"
        ),

        # ----------------------------------------------------
        # STOCKOUT
        # ----------------------------------------------------

        (
            "Will M12 run out of stock "
            "at R01 next week?",
            "stockout_analysis"
        ),

        (
            "Is there enough stock of M05 "
            "at R02 next week?",
            "stockout_analysis"
        ),

        # ----------------------------------------------------
        # HISTORICAL / RECENT SALES
        #
        # Both map to historical_sales_analysis.
        # ----------------------------------------------------

        (
            "Show recent demand for M12 at R01",
            "historical_sales_analysis"
        ),

        (
            "What are the recent sales for M05 "
            "in restaurant 2?",
            "historical_sales_analysis"
        ),

        (
            "Give me the historical sales "
            "of M10 at R03",
            "historical_sales_analysis"
        ),

        # ----------------------------------------------------
        # PROMOTION
        # ----------------------------------------------------

        (
            "How are promotions affecting "
            "sales at restaurant 1?",
            "promotion_analysis"
        ),

        (
            "Analyse the effect of promotions "
            "on R02",
            "promotion_analysis"
        ),

        # ----------------------------------------------------
        # HOLIDAY
        # ----------------------------------------------------

        (
            "How do holidays affect sales "
            "at R01?",
            "holiday_analysis"
        ),

        # ----------------------------------------------------
        # EVENTS
        # ----------------------------------------------------

        (
            "Are special events affecting "
            "demand at restaurant 3?",
            "event_analysis"
        ),

        # ----------------------------------------------------
        # CUSTOMER PATTERN
        # ----------------------------------------------------

        (
            "What are the customer purchasing "
            "patterns at R01?",
            "customer_pattern_analysis"
        ),

        (
            "How is customer demand behaving "
            "at restaurant 2?",
            "customer_pattern_analysis"
        ),
    ]
)
def test_complete_queries(
    query,
    expected_capability
):

    orchestrator = (
        create_test_orchestrator()
    )

    response = (
        orchestrator.process_request(
            query
        )
    )

    assert (
        response["status"]
        == "completed"
    )

    assert (
        expected_capability
        in response[
            "required_capabilities"
        ]
    )


# ============================================================
# TEST 2
# FORECAST WORKFLOW
# ============================================================

def test_demand_forecast_workflow():

    orchestrator = (
        create_test_orchestrator()
    )

    response = (
        orchestrator.process_request(
            "Forecast demand for "
            "M12 at R01 next week"
        )
    )

    assert (
        response["status"]
        == "completed"
    )

    assert (
        "demand_forecasting"
        in response[
            "required_capabilities"
        ]
    )

    assert (
        response["workflow"]
        == [
            "data_analyst",
            "context_seasonality",
            "customer_pattern",
            "demand_forecasting"
        ]
    )


# ============================================================
# TEST 3
# FORECAST RESULT
# ============================================================

def test_forecast_result():

    orchestrator = (
        create_test_orchestrator()
    )

    response = (
        orchestrator.process_request(
            "Forecast demand for "
            "M12 at R01 next week"
        )
    )

    forecast = (
        response[
            "results"
        ][
            "demand_forecasting"
        ]
    )

    assert (
        forecast[
            "predicted_demand"
        ]
        == 850
    )

    assert (
        forecast[
            "confidence"
        ]
        == 0.87
    )


# ============================================================
# TEST 4
# ORDER RECOMMENDATION WORKFLOW
# ============================================================

def test_order_recommendation():

    orchestrator = (
        create_test_orchestrator()
    )

    response = (
        orchestrator.process_request(
            "How much should R01 order "
            "for M12 next week?"
        )
    )

    assert (
        response["status"]
        == "completed"
    )

    assert (
        "order_recommendation"
        in response[
            "required_capabilities"
        ]
    )

    assert (
        response["workflow"]
        == [
            "data_analyst",
            "context_seasonality",
            "customer_pattern",
            "demand_forecasting",
            "inventory_decision"
        ]
    )


# ============================================================
# TEST 5
# ORDER QUANTITY
# ============================================================

def test_order_quantity():

    orchestrator = (
        create_test_orchestrator()
    )

    response = (
        orchestrator.process_request(
            "How much M12 should "
            "R01 order next week?"
        )
    )

    inventory = (
        response[
            "results"
        ][
            "inventory_decision"
        ]
    )

    # Forecast = 850
    # Stock = 500
    # Recommended order = 350

    assert (
        inventory[
            "recommended_order_quantity"
        ]
        == 350
    )


# ============================================================
# TEST 6
# PROMOTION WORKFLOW
# ============================================================

def test_promotion_analysis():

    orchestrator = (
        create_test_orchestrator()
    )

    response = (
        orchestrator.process_request(
            "How are promotions affecting "
            "sales at R01?"
        )
    )

    assert (
        response["status"]
        == "completed"
    )

    assert (
        "promotion_analysis"
        in response[
            "required_capabilities"
        ]
    )

    assert (
        response["workflow"]
        == [
            "context_seasonality"
        ]
    )


# ============================================================
# TEST 7
# SEASONALITY OUTPUT
# ============================================================

def test_seasonality_output():

    orchestrator = (
        create_test_orchestrator()
    )

    response = (
        orchestrator.process_request(
            "How are promotions affecting "
            "sales at R01?"
        )
    )

    result = (
        response[
            "results"
        ][
            "context_seasonality"
        ]
    )

    assert (
        result[
            "seasonal_factor"
        ]
        == 1.15
    )

    assert (
        result[
            "weekly_pattern"
        ]
        == "weekend_peak"
    )


# ============================================================
# TEST 8
# CUSTOMER PATTERN
# ============================================================

def test_customer_pattern():

    orchestrator = (
        create_test_orchestrator()
    )

    response = (
        orchestrator.process_request(
            "What are the customer purchasing "
            "patterns at R01?"
        )
    )

    assert (
        response["status"]
        == "completed"
    )

    assert (
        "customer_pattern_analysis"
        in response[
            "required_capabilities"
        ]
    )

    assert (
        response["workflow"]
        == [
            "customer_pattern"
        ]
    )

    result = (
        response[
            "results"
        ][
            "customer_pattern"
        ]
    )

    assert (
        result[
            "customer_demand_trend"
        ]
        == "increasing"
    )

    assert (
        result[
            "pattern_strength"
        ]
        == 0.82
    )


# ============================================================
# TEST 9
# RECENT DEMAND
# ============================================================

def test_recent_demand():

    orchestrator = (
        create_test_orchestrator()
    )

    response = (
        orchestrator.process_request(
            "Show recent demand for "
            "M12 at R01"
        )
    )

    assert (
        response["status"]
        == "completed"
    )

    # Recent demand maps to historical_sales_analysis
    # in the current orchestrator prompt.

    assert (
        "historical_sales_analysis"
        in response[
            "required_capabilities"
        ]
    )

    assert (
        response["workflow"]
        == [
            "data_analyst"
        ]
    )

    assert (
        response[
            "results"
        ][
            "data_analyst"
        ][
            "avg_daily_sales"
        ]
        == 120
    )


# ============================================================
# TEST 10
# MISSING RESTAURANT
# ============================================================

def test_missing_restaurant():

    orchestrator = (
        create_test_orchestrator()
    )

    response = (
        orchestrator.process_request(
            "Forecast demand for "
            "M12 next week"
        )
    )

    assert (
        response["status"]
        == "missing_parameters"
    )

    missing = (
        response[
            "missing_parameters"
        ]
    )

    assert (
        "restaurant"
        in missing
    )

    assert (
        "menu_item"
        not in missing
    )

    assert (
        "forecast_horizon"
        not in missing
    )


# ============================================================
# TEST 11
# MISSING MENU ITEM
# ============================================================

def test_missing_menu_item():

    orchestrator = (
        create_test_orchestrator()
    )

    response = (
        orchestrator.process_request(
            "Forecast demand at "
            "R01 next week"
        )
    )

    assert (
        response["status"]
        == "missing_parameters"
    )

    missing = (
        response[
            "missing_parameters"
        ]
    )

    assert (
        "menu_item"
        in missing
    )

    assert (
        "restaurant"
        not in missing
    )

    assert (
        "forecast_horizon"
        not in missing
    )


# ============================================================
# TEST 12
# MULTIPLE MISSING PARAMETERS
# ============================================================

def test_multiple_missing_parameters():

    orchestrator = (
        create_test_orchestrator()
    )

    response = (
        orchestrator.process_request(
            "Forecast demand next week"
        )
    )

    assert (
        response["status"]
        == "missing_parameters"
    )

    missing = (
        response[
            "missing_parameters"
        ]
    )

    assert (
        "restaurant"
        in missing
    )

    assert (
        "menu_item"
        in missing
    )

    assert (
        "forecast_horizon"
        not in missing
    )


# ============================================================
# TEST 13
# MISSING FORECAST PERIOD
# ============================================================

def test_missing_forecast_period():

    orchestrator = (
        create_test_orchestrator()
    )

    response = (
        orchestrator.process_request(
            "Forecast demand for "
            "M12 at R01"
        )
    )

    assert (
        response["status"]
        == "missing_parameters"
    )

    missing = (
        response[
            "missing_parameters"
        ]
    )

    assert (
        "forecast_horizon"
        in missing
    )

    assert (
        "restaurant"
        not in missing
    )

    assert (
        "menu_item"
        not in missing
    )


# ============================================================
# TEST 14
# EXPLICIT MULTI-TURN FOLLOW-UP
# ============================================================

def test_followup_parameters():

    orchestrator = (
        create_test_orchestrator()
    )

    # --------------------------------------------------------
    # Initial request
    # --------------------------------------------------------

    response = (
        orchestrator.process_request(
            "Forecast demand next week"
        )
    )

    assert (
        response["status"]
        == "missing_parameters"
    )

    assert (
        "restaurant"
        in response[
            "missing_parameters"
        ]
    )

    assert (
        "menu_item"
        in response[
            "missing_parameters"
        ]
    )

    # --------------------------------------------------------
    # User provides restaurant
    # --------------------------------------------------------

    response = (
        orchestrator.process_request(
            "R01"
        )
    )

    assert (
        response["status"]
        == "missing_parameters"
    )

    assert (
        "restaurant"
        not in response[
            "missing_parameters"
        ]
    )

    assert (
        "menu_item"
        in response[
            "missing_parameters"
        ]
    )

    # --------------------------------------------------------
    # User provides menu item
    # --------------------------------------------------------

    response = (
        orchestrator.process_request(
            "M12"
        )
    )

    assert (
        response["status"]
        == "completed"
    )

    assert (
        "demand_forecasting"
        in response[
            "required_capabilities"
        ]
    )


# ============================================================
# TEST 15
# FORECAST HORIZONS
# ============================================================

@pytest.mark.parametrize(
    "query, expected_horizon",
    [

        (
            "Forecast M12 at R01 tomorrow",
            1
        ),

        (
            "Forecast M12 at R01 next week",
            7
        ),

        (
            "Forecast M12 at R01 "
            "for the next 14 days",
            14
        ),

        (
            "Forecast M12 at R01 next month",
            30
        ),
    ]
)
def test_forecast_horizon(
    query,
    expected_horizon
):

    orchestrator = (
        create_test_orchestrator()
    )

    parsed = (
        orchestrator.parse_query(
            query,
            {}
        )
    )

    assert (
        parsed[
            "forecast_horizon"
        ]
        == expected_horizon
    )


# ============================================================
# TEST 16
# NEXT WEEK NORMALIZATION
# ============================================================

def test_next_week_normalization():

    orchestrator = (
        create_test_orchestrator()
    )

    parsed = (
        orchestrator.parse_query(
            "Forecast demand for "
            "M12 at R01 next week",
            {}
        )
    )

    assert (
        parsed[
            "time_period"
        ]
        == "next_week"
    )

    assert (
        parsed[
            "forecast_horizon"
        ]
        == 7
    )


# ============================================================
# TEST 17
# ALL FORECAST AGENTS EXECUTED
# ============================================================

def test_all_forecast_agents_return_results():

    orchestrator = (
        create_test_orchestrator()
    )

    response = (
        orchestrator.process_request(
            "Forecast M12 at R01 "
            "next week"
        )
    )

    assert (
        response["status"]
        == "completed"
    )

    expected_agents = [
        "data_analyst",
        "context_seasonality",
        "customer_pattern",
        "demand_forecasting"
    ]

    for agent_name in expected_agents:

        assert (
            agent_name
            in response[
                "results"
            ]
        )

        assert (
            response[
                "results"
            ][agent_name]
            is not None
        )


# ============================================================
# TEST 18
# WORKFLOW STATUS
# ============================================================

def test_workflow_status():

    orchestrator = (
        create_test_orchestrator()
    )

    response = (
        orchestrator.process_request(
            "Forecast M12 at R01 "
            "next week"
        )
    )

    assert (
        response["status"]
        == "completed"
    )

    for agent_name in (
        response[
            "workflow"
        ]
    ):

        assert (
            response[
                "workflow_status"
            ][agent_name]
            == "completed"
        )


# ============================================================
# TEST 19
# DEPENDENCY ORDER
# ============================================================

def test_forecast_dependency_order():

    orchestrator = (
        create_test_orchestrator()
    )

    response = (
        orchestrator.process_request(
            "Forecast M12 at R01 "
            "next week"
        )
    )

    workflow = (
        response[
            "workflow"
        ]
    )

    # All prerequisite agents must run before
    # demand forecasting.

    forecasting_position = (
        workflow.index(
            "demand_forecasting"
        )
    )

    assert (
        workflow.index(
            "data_analyst"
        )
        < forecasting_position
    )

    assert (
        workflow.index(
            "context_seasonality"
        )
        < forecasting_position
    )

    assert (
        workflow.index(
            "customer_pattern"
        )
        < forecasting_position
    )


# ============================================================
# TEST 20
# ORDER DEPENDENCY ORDER
# ============================================================

def test_inventory_runs_after_forecasting():

    orchestrator = (
        create_test_orchestrator()
    )

    response = (
        orchestrator.process_request(
            "How much M12 should "
            "R01 order next week?"
        )
    )

    workflow = (
        response[
            "workflow"
        ]
    )

    assert (
        workflow.index(
            "demand_forecasting"
        )
        <
        workflow.index(
            "inventory_decision"
        )
    )
    def test_all_restaurants_followup():

        orchestrator = (
            create_test_orchestrator()
        )

        response = (
            orchestrator.process_request(
                "Forecast M12 next week"
            )
        )

        assert (
            response["status"]
            == "missing_parameters"
        )

        response = (
            orchestrator.process_request(
                "all restaurants"
            )
        )

        assert (
            response["status"]
            == "completed"
        )