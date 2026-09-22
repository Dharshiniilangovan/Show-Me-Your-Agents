from __future__ import annotations

import pandas as pd

from .data import load_dataset, select_history
from .models import ValidationConfig
from .validators import (
    data_quality_checks,
    historical_anomaly_summary,
    forecast_checks,
    inventory_risk,
)


AGENT_NAME = "anomaly_validation"

CAPABILITIES = [
    "forecast_validation",
    "anomaly_detection",
    "data_quality_validation",
    "stockout_risk_validation",
    "overstock_risk_validation",
]


def _severity(issues):
    """Determine overall validation status from issue severities."""

    levels = {x.get("severity") for x in issues}

    if "high" in levels:
        return "fail"

    if "medium" in levels:
        return "warning"

    return "pass"


def anomaly_validation_agent(
    state: dict,
    dataset_path=None,
    config: ValidationConfig | None = None,
) -> dict:
    """
    Phase-2 Anomaly / Validation Agent.

    Reads:
        state["request"]
        state["agent_results"]
        state["shared_context"]["data"]["unified_demand_path"]

    The agent never invokes other agents directly.
    """

    cfg = config or ValidationConfig()

    if not isinstance(state, dict):
        raise TypeError("state must be a dictionary")

    # ========================================================
    # 1. READ COMMON STATE
    # ========================================================

    request = state.get("request") or {}
    results = state.get("agent_results") or {}

    forecast = results.get("demand_forecasting") or {}
    data_result = results.get("data_analyst") or {}

    # Resolve identifiers from request first, then fall back
    # to previous agent outputs.

    restaurant_id = (
        request.get("restaurant_id")
        or forecast.get("restaurant_id")
        or data_result.get("restaurant_id")
    )

    menu_item_id = (
        request.get("menu_item_id")
        or forecast.get("menu_item_id")
        or data_result.get("menu_item_id")
    )

    # ========================================================
    # 2. LOAD HISTORICAL DATA
    # ========================================================

    try:

        # ----------------------------------------------------
        # Optional explicit path for standalone testing
        # ----------------------------------------------------

        if dataset_path is not None:

            temporary_state = dict(state)

            temporary_shared_context = dict(
                temporary_state.get(
                    "shared_context",
                    {},
                )
            )

            temporary_data = dict(
                temporary_shared_context.get(
                    "data",
                    {},
                )
            )

            temporary_data["unified_demand_path"] = str(
                dataset_path
            )

            temporary_shared_context["data"] = temporary_data

            temporary_state["shared_context"] = (
                temporary_shared_context
            )

            df = load_dataset(temporary_state)

        else:

            # Integrated execution:
            #
            # state
            #   -> shared_context
            #       -> data
            #           -> unified_demand_path

            df = load_dataset(state)

        # ----------------------------------------------------
        # Select history for requested restaurant + SKU
        # ----------------------------------------------------

        history = select_history(
            df,
            restaurant_id=restaurant_id,
            menu_item_id=menu_item_id,
        )

        if history.empty:
            raise ValueError(
                "No historical demand records found for "
                f"restaurant_id={restaurant_id}, "
                f"menu_item_id={menu_item_id}."
            )

        # ----------------------------------------------------
        # Historical validation
        # ----------------------------------------------------

        dq = data_quality_checks(
            history,
            cfg,
        )

        hist = historical_anomaly_summary(
            history,
            cfg,
        )

    except Exception as exc:

        history = None

        dq = [
            {
                "code": "DATASET_ERROR",
                "severity": "high",
                "message": str(exc),
            }
        ]

        # Keep a stable output schema even when historical
        # data cannot be accessed.
        hist = {
            "method": "robust_zscore",
            "threshold": getattr(
                cfg,
                "robust_z_threshold",
                4.0,
            ),
            "count": 0,
            "rate": 0.0,
            "recent_anomaly": False,
            "median": None,
            "mad": None,
            "status": "unavailable",
        }

    # ========================================================
    # 3. FORECAST VALIDATION
    # ========================================================

    if history is not None:

        forecast_history = history

    else:

        # forecast_checks expects a DataFrame containing
        # a quantity column.
        forecast_history = pd.DataFrame(
            {
                "quantity": [],
            }
        )

    fissues, fmetrics = forecast_checks(
        forecast,
        forecast_history,
        cfg,
    )

    # ========================================================
    # 4. INVENTORY RISK
    # ========================================================

    risk = inventory_risk(
        data_result,
        forecast,
        cfg,
    )

    # ========================================================
    # 5. COMBINE ISSUES
    # ========================================================

    issues = dq + fissues

    if hist.get("recent_anomaly"):

        issues.append(
            {
                "code": "RECENT_DEMAND_ANOMALY",
                "severity": "medium",
                "message": (
                    "Most recent historical demand is "
                    "anomalous versus the SKU history."
                ),
            }
        )

    if risk.get("stockout_risk") == "high":

        issues.append(
            {
                "code": "STOCKOUT_RISK",
                "severity": "high",
                "message": (
                    "Current stock is materially below "
                    "forecast demand."
                ),
            }
        )

    if risk.get("overstock_risk") == "high":

        issues.append(
            {
                "code": "OVERSTOCK_RISK",
                "severity": "medium",
                "message": (
                    "Current stock is materially above "
                    "forecast demand."
                ),
            }
        )

    # ========================================================
    # 6. FINAL VALIDATION DECISION
    # ========================================================

    status = _severity(issues)

    reforecast = any(
        issue.get("code")
        in {
            "INVALID_FORECAST",
            "NEGATIVE_FORECAST",
            "FORECAST_DEVIATION",
            "DATASET_ERROR",
        }
        and issue.get("severity") == "high"
        for issue in issues
    )

    # ========================================================
    # 7. RETURN COMMON AGENT OUTPUT
    # ========================================================

    return {
        "analysis_type": "forecast_validation",

        "restaurant_id": restaurant_id,

        "restaurant_name": (
            request.get("restaurant_name")
            or forecast.get("restaurant_name")
            or data_result.get("restaurant_name")
        ),

        "menu_item_id": menu_item_id,

        "menu_item_name": (
            request.get("menu_item_name")
            or forecast.get("menu_item_name")
            or data_result.get("menu_item_name")
        ),

        "validation_status": status,

        "forecast_valid": status != "fail",

        "reforecast_recommended": reforecast,

        "forecast_metrics": fmetrics,

        "historical_anomalies": hist,

        "inventory_risk": risk,

        "issues": issues,

        "agent_name": AGENT_NAME,

        "capabilities": CAPABILITIES,
    }


def agent(state: dict) -> dict:
    """
    Integration alias matching the Phase-1
    preferred def agent(state) contract.
    """

    return anomaly_validation_agent(state)