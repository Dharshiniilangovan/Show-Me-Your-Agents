"""
Tests for the Phase-2 Anomaly / Validation Agent.

Run from the repository root:

    pytest tests/test_anomaly_validation.py -v

These tests validate the public agent(state) interface rather than
the internal implementation.
"""

import pytest

from agents.anomaly_validation.agent import agent


# ---------------------------------------------------------------------
# Shared state factory
# ---------------------------------------------------------------------

def make_state(
    predicted_demand=850,
    current_stock=500,
    avg_daily_sales=36.58,
    confidence=0.87,
    forecast_horizon=7,
):
    """
    Build a Phase-1-compatible shared state for testing.

    data_analyst and demand_forecasting are mocked because this test
    exercises the Anomaly / Validation Agent independently.
    """

    return {
        "request": {
            "request_id": "TEST-001",
            "query": "Validate demand forecast",
            "goal": "forecast_validation",

            "restaurant_scope": "single",
            "restaurant_id": "R01",
            "restaurant_name": "Downtown Chicago Grill",

            "menu_item_id": "M12",
            "menu_item_name": "Chicken Nuggets 12pc",

            "time_period": "next_week",
            "forecast_horizon": forecast_horizon,

            "holiday_name": None,
            "special_event_name": None,
            "promotion_percentage": None,
        },

        "required_capabilities": [
            "forecast_validation",
            "anomaly_detection",
            "data_quality_validation",
            "stockout_risk_validation",
            "overstock_risk_validation",
        ],

        "agent_results": {

            "data_analyst": {
                "restaurant_scope": "single",
                "restaurant_id": "R01",
                "restaurant_name": "Downtown Chicago Grill",

                "menu_item_id": "M12",
                "menu_item_name": "Chicken Nuggets 12pc",

                "avg_daily_sales": avg_daily_sales,
                "total_historical_quantity": 66504,
                "current_stock": current_stock,
            },

            "demand_forecasting": {
                "restaurant_id": "R01",
                "restaurant_name": "Downtown Chicago Grill",

                "menu_item_id": "M12",
                "menu_item_name": "Chicken Nuggets 12pc",

                "predicted_demand": predicted_demand,
                "forecast_horizon": forecast_horizon,
                "confidence": confidence,
            },
        },
    }


# ---------------------------------------------------------------------
# 1. Basic execution
# ---------------------------------------------------------------------

def test_agent_runs_successfully():

    state = make_state()

    result = agent(state)

    assert result is not None
    assert isinstance(result, dict)


# ---------------------------------------------------------------------
# 2. Agent identity
# ---------------------------------------------------------------------

def test_agent_identity():

    result = agent(make_state())

    assert result["agent_name"] == "anomaly_validation"
    assert result["analysis_type"] == "forecast_validation"


# ---------------------------------------------------------------------
# 3. Required output contract
# ---------------------------------------------------------------------

def test_required_output_fields_exist():

    result = agent(make_state())

    required_fields = [
        "analysis_type",
        "restaurant_id",
        "restaurant_name",
        "menu_item_id",
        "menu_item_name",
        "validation_status",
        "forecast_valid",
        "reforecast_recommended",
        "forecast_metrics",
        "historical_anomalies",
        "inventory_risk",
        "issues",
        "agent_name",
        "capabilities",
    ]

    for field in required_fields:
        assert field in result, f"Missing output field: {field}"


# ---------------------------------------------------------------------
# 4. Correct entity propagation
# ---------------------------------------------------------------------

def test_restaurant_and_item_are_preserved():

    result = agent(make_state())

    assert result["restaurant_id"] == "R01"
    assert result["restaurant_name"] == "Downtown Chicago Grill"

    assert result["menu_item_id"] == "M12"
    assert result["menu_item_name"] == "Chicken Nuggets 12pc"


# ---------------------------------------------------------------------
# 5. Forecast value propagation
# ---------------------------------------------------------------------

def test_forecast_value_is_preserved():

    result = agent(
        make_state(predicted_demand=850)
    )

    assert result["forecast_metrics"]["predicted_demand"] == pytest.approx(850)


# ---------------------------------------------------------------------
# 6. Extreme forecast detection
# ---------------------------------------------------------------------

def test_extreme_forecast_is_rejected():

    result = agent(
        make_state(
            predicted_demand=5000,
            current_stock=500
        )
    )

    assert result["forecast_valid"] is False
    assert result["reforecast_recommended"] is True

    issue_codes = {
        issue["code"]
        for issue in result["issues"]
    }

    assert "FORECAST_DEVIATION" in issue_codes


# ---------------------------------------------------------------------
# 7. Extreme forecast should have large deviation
# ---------------------------------------------------------------------

def test_extreme_forecast_has_large_deviation():

    result = agent(
        make_state(predicted_demand=5000)
    )

    deviation = result["forecast_metrics"]["deviation_pct"]

    assert deviation > 100


# ---------------------------------------------------------------------
# 8. Stockout detection
# ---------------------------------------------------------------------

def test_stockout_risk_detected():

    result = agent(
        make_state(
            predicted_demand=5000,
            current_stock=500
        )
    )

    inventory = result["inventory_risk"]

    assert inventory["stockout_risk"] == "high"

    issue_codes = {
        issue["code"]
        for issue in result["issues"]
    }

    assert "STOCKOUT_RISK" in issue_codes


# ---------------------------------------------------------------------
# 9. Coverage ratio
# ---------------------------------------------------------------------

def test_inventory_coverage_ratio():

    result = agent(
        make_state(
            predicted_demand=1000,
            current_stock=500
        )
    )

    coverage = result["inventory_risk"]["coverage_ratio"]

    assert coverage == pytest.approx(0.5, abs=0.01)


# ---------------------------------------------------------------------
# 10. Overstock scenario
# ---------------------------------------------------------------------

def test_overstock_detection():

    result = agent(
        make_state(
            predicted_demand=100,
            current_stock=1000
        )
    )

    inventory = result["inventory_risk"]

    assert inventory["overstock_risk"] in {
        "medium",
        "high"
    }


# ---------------------------------------------------------------------
# 11. Historical anomaly structure
# ---------------------------------------------------------------------

def test_historical_anomaly_output():

    result = agent(make_state())

    anomaly = result["historical_anomalies"]

    assert "method" in anomaly
    assert "threshold" in anomaly
    assert "count" in anomaly
    assert "rate" in anomaly
    assert "recent_anomaly" in anomaly
    assert "status" in anomaly

    assert anomaly["method"] == "robust_zscore"

    assert anomaly["count"] >= 0
    assert 0 <= anomaly["rate"] <= 1


# ---------------------------------------------------------------------
# 12. Issues must be structured
# ---------------------------------------------------------------------

def test_issues_are_structured():

    result = agent(make_state())

    assert isinstance(result["issues"], list)

    for issue in result["issues"]:

        assert isinstance(issue, dict)

        assert "code" in issue
        assert "severity" in issue
        assert "message" in issue


# ---------------------------------------------------------------------
# 13. Capabilities
# ---------------------------------------------------------------------

def test_agent_capabilities():

    result = agent(make_state())

    capabilities = result["capabilities"]

    assert "forecast_validation" in capabilities
    assert "anomaly_detection" in capabilities
    assert "data_quality_validation" in capabilities
    assert "stockout_risk_validation" in capabilities
    assert "overstock_risk_validation" in capabilities


# ---------------------------------------------------------------------
# 14. Deterministic behaviour
# ---------------------------------------------------------------------

def test_same_input_produces_same_validation_result():

    state1 = make_state(
        predicted_demand=850,
        current_stock=500
    )

    state2 = make_state(
        predicted_demand=850,
        current_stock=500
    )

    result1 = agent(state1)
    result2 = agent(state2)

    assert (
        result1["validation_status"]
        == result2["validation_status"]
    )

    assert (
        result1["forecast_valid"]
        == result2["forecast_valid"]
    )

    assert (
        result1["reforecast_recommended"]
        == result2["reforecast_recommended"]
    )


# ---------------------------------------------------------------------
# 15. Agent must return dictionary for orchestrator
# ---------------------------------------------------------------------

def test_orchestrator_compatible_return_type():

    state = make_state()

    result = agent(state)

    assert isinstance(result, dict)

    # Simulate what the orchestrator will eventually do.
    state["agent_results"]["anomaly_validation"] = result

    assert "anomaly_validation" in state["agent_results"]

    assert (
        state["agent_results"]["anomaly_validation"]["agent_name"]
        == "anomaly_validation"
    )