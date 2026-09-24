"""
Waste Reduction Agent

Purpose
-------
Identifies inventory items approaching expiry, estimates waste risk using
available demand information, and recommends actions such as priority
selling, markdown/promotion, stock reallocation, and reducing future orders.

Common project interface
-------------------------
def agent(state):
    request = state["request"]
    previous_results = state["agent_results"]
    return {...}

Important integration note
--------------------------
The current inventory dataset is ingredient-level:

    ingredient_id / ingredient_name / current_stock / expiry_date / ...

The Demand Forecasting Agent is menu-item-level:

    restaurant_id / menu_item_id / predicted_quantity

Therefore this agent NEVER assumes that a menu item is an ingredient.
A forecast is used directly only when the forecast output contains a
matching ingredient identifier/name (or an explicit inventory mapping is
provided in state/request). Otherwise the agent uses avg_daily_usage from
inventory as the operational depletion baseline and reports that no direct
forecast-to-inventory match was available.

Default inventory file:
    data/inventory_data_dec2025_updated.csv

Dataset-level output:
    data/waste_reduction.csv
"""

from pathlib import Path
import math
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

DEFAULT_INVENTORY_FILE = "data/inventory_dataset.csv"
OUTPUT_FILE = "data/waste_reduction.csv"

# Expiry thresholds are deliberately explicit so they can be changed
# without changing the rest of the agent.
CRITICAL_DAYS = 1
HIGH_DAYS = 4
MEDIUM_DAYS = 7


# ============================================================
# GENERAL HELPERS
# ============================================================

def _project_root():
    """Return the project root when this file is inside agents/.../."""
    return Path(__file__).resolve().parents[2]


def _resolve_path(path_value, default_value):
    """Resolve a user/project path without requiring absolute paths."""
    value = path_value or default_value
    path = Path(value)

    if path.is_absolute():
        return path

    # First try relative to the current working directory.
    if path.exists():
        return path

    # Then try relative to the project root.
    return _project_root() / path


def _safe_float(value, default=0.0):
    try:
        number = float(value)
        if math.isfinite(number):
            return number
    except (TypeError, ValueError):
        pass
    return float(default)


def _normalize_id(value):
    if value is None:
        return None
    return str(value).strip().upper()


def _normalize_name(value):
    if value is None:
        return None
    return " ".join(str(value).strip().lower().split())


# ============================================================
# LOAD INVENTORY
# ============================================================

def load_inventory(state):
    """
    Priority:
    1. state["inventory_data"]
    2. request["inventory_data_path"]
    3. default project inventory CSV
    """
    request = state.get("request", {})

    inventory_data = state.get("inventory_data")

    if inventory_data is not None:
        if isinstance(inventory_data, pd.DataFrame):
            return inventory_data.copy()
        return pd.DataFrame(inventory_data)

    inventory_path = request.get("inventory_data_path")
    path = _resolve_path(inventory_path, DEFAULT_INVENTORY_FILE)

    if not path.exists():
        raise FileNotFoundError(
            f"Inventory dataset not found: {path}"
        )

    return pd.read_csv(path)


def validate_inventory(df):
    required = [
        "ingredient_id",
        "ingredient_name",
        "current_stock",
        "avg_daily_usage",
        "expiry_date",
    ]

    missing = [column for column in required if column not in df.columns]

    if missing:
        raise ValueError(
            "Inventory dataset is missing: " + ", ".join(missing)
        )

    result = df.copy()

    result["current_stock"] = pd.to_numeric(
        result["current_stock"], errors="coerce"
    ).fillna(0)

    result["avg_daily_usage"] = pd.to_numeric(
        result["avg_daily_usage"], errors="coerce"
    ).fillna(0)

    result["expiry_date"] = pd.to_datetime(
        result["expiry_date"], errors="coerce", dayfirst=True
    )

    result = result.dropna(subset=["expiry_date"]).copy()

    result["current_stock"] = result["current_stock"].clip(lower=0)
    result["avg_daily_usage"] = result["avg_daily_usage"].clip(lower=0)

    return result


# ============================================================
# APPLICATION / INVENTORY DATE
# ============================================================

def get_analysis_date(request, inventory_df):
    """
    Prefer an explicit request/application date.

    For this project, the inventory data was constructed around the
    Dec-2025 inventory snapshot. If no date is supplied, use the latest
    received_date when available, otherwise the latest inventory date.
    """
    explicit = (
        request.get("analysis_date")
        or request.get("current_date")
        or request.get("forecast_start")
    )

    if explicit is not None:
        parsed = pd.to_datetime(explicit, errors="coerce")
        if not pd.isna(parsed):
            return parsed.normalize()

    if "received_date" in inventory_df.columns:
        received = pd.to_datetime(
            inventory_df["received_date"],
            errors="coerce",
            dayfirst=True
        ).dropna()

        if not received.empty:
            return received.max().normalize()

    return pd.Timestamp("2025-12-31")


# ============================================================
# FORECAST EXTRACTION
# ============================================================

def _get_forecast_result(previous_results):
    """
    Demand Forecasting Agent returns a wrapper containing:
        output_path
        forecast_horizon
        result

    The result can contain:
        combination_results
        daily_forecast
        total_predicted_demand
    """
    return previous_results.get("demand_forecasting", {})


def _read_forecast_rows(forecast_result):
    """
    Return a normalized forecast dataframe.

    Preferred source:
        demand_forecasting["output_path"]

    Fallback:
        demand_forecasting["result"] / nested daily_forecast.
    """
    if not isinstance(forecast_result, dict):
        return pd.DataFrame()

    output_path = forecast_result.get("output_path")

    if output_path:
        path = Path(output_path)

        if not path.is_absolute():
            if not path.exists():
                path = _project_root() / path

        if path.exists():
            try:
                df = pd.read_csv(path)

                required = {
                    "date",
                    "restaurant_id",
                    "menu_item_id",
                    "predicted_quantity",
                }

                if required.issubset(df.columns):
                    df["date"] = pd.to_datetime(
                        df["date"], errors="coerce"
                    )
                    df["predicted_quantity"] = pd.to_numeric(
                        df["predicted_quantity"], errors="coerce"
                    ).fillna(0)
                    return df.dropna(subset=["date"]).copy()
            except Exception:
                pass

    result = forecast_result.get("result", {})

    if not isinstance(result, dict):
        return pd.DataFrame()

    rows = []

    combinations = result.get("combination_results") or []

    for combination in combinations:
        restaurant_id = combination.get("restaurant_id")
        menu_item_id = combination.get("menu_item_id")

        for row in combination.get("daily_forecast", []) or []:
            rows.append(
                {
                    "date": row.get("date"),
                    "restaurant_id": restaurant_id,
                    "menu_item_id": menu_item_id,
                    "predicted_quantity": row.get("predicted_quantity", 0),
                }
            )

    if not rows:
        daily = result.get("daily_forecast") or []

        # The no-pair response does not necessarily identify a menu item
        # at every row, so preserve what is actually available.
        for row in daily:
            rows.append(
                {
                    "date": row.get("date"),
                    "restaurant_id": result.get("restaurant_id"),
                    "menu_item_id": result.get("menu_item_id"),
                    "predicted_quantity": row.get(
                        "predicted_quantity",
                        row.get("quantity", 0),
                    ),
                }
            )

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["predicted_quantity"] = pd.to_numeric(
        df["predicted_quantity"], errors="coerce"
    ).fillna(0)

    return df.dropna(subset=["date"]).copy()


def _extract_inventory_mapping(state):
    """
    Optional explicit mapping.

    Supported forms:

    request["inventory_mapping"] = {
        "M12": ["ING003", "ING004"]
    }

    or

    state["inventory_mapping"] = {
        "M12": ["ING003", "ING004"]
    }

    This is intentionally optional because the current inventory dataset
    does not contain menu_item_id.
    """
    request = state.get("request", {})

    mapping = (
        request.get("inventory_mapping")
        or state.get("inventory_mapping")
        or {}
    )

    if not isinstance(mapping, dict):
        return {}

    normalized = {}

    for menu_item_id, ingredients in mapping.items():
        if not isinstance(ingredients, list):
            ingredients = [ingredients]

        normalized[_normalize_id(menu_item_id)] = {
            _normalize_id(value)
            for value in ingredients
            if value is not None
        }

    return normalized


# ============================================================
# FORECAST-TO-INVENTORY MATCHING
# ============================================================

def _forecast_for_inventory_item(
    inventory_row,
    forecast_df,
    mapping,
    request,
):
    """
    Try to obtain a direct forecast for an inventory item.

    The order is:
    1. Explicit menu-item -> ingredient mapping.
    2. Direct ingredient_id in forecast data, if available.
    3. Direct ingredient_name in forecast data, if available.
    4. Otherwise no direct forecast match.
    """
    ingredient_id = _normalize_id(
        inventory_row.get("ingredient_id")
    )
    ingredient_name = _normalize_name(
        inventory_row.get("ingredient_name")
    )

    mapped_menu_items = []

    for menu_item_id, ingredients in mapping.items():
        if ingredient_id in ingredients:
            mapped_menu_items.append(menu_item_id)

    # A request-level menu item can also be used with an explicit mapping.
    requested_menu_item = _normalize_id(
        request.get("menu_item_id")
    )

    if requested_menu_item and requested_menu_item in mapping:
        if ingredient_id in mapping[requested_menu_item]:
            mapped_menu_items.append(requested_menu_item)

    mapped_menu_items = list(dict.fromkeys(mapped_menu_items))

    if forecast_df.empty:
        return None, "no_forecast_data"

    candidate = forecast_df.copy()

    if "ingredient_id" in candidate.columns:
        candidate = candidate[
            candidate["ingredient_id"].map(_normalize_id)
            == ingredient_id
        ]

        if not candidate.empty:
            return candidate, "direct_ingredient_id"

    if "ingredient_name" in candidate.columns:
        candidate = candidate[
            candidate["ingredient_name"].map(_normalize_name)
            == ingredient_name
        ]

        if not candidate.empty:
            return candidate, "direct_ingredient_name"

    if mapped_menu_items:
        candidate = forecast_df[
            forecast_df["menu_item_id"].map(_normalize_id).isin(
                mapped_menu_items
            )
        ]

        if not candidate.empty:
            return candidate, "explicit_menu_item_mapping"

    return None, "no_inventory_forecast_mapping"


# ============================================================
# WASTE RISK
# ============================================================

def calculate_expiry_risk(days_to_expiry):
    if days_to_expiry <= CRITICAL_DAYS:
        return "critical"

    if days_to_expiry <= HIGH_DAYS:
        return "high"

    if days_to_expiry <= MEDIUM_DAYS:
        return "medium"

    return "low"


def calculate_waste_risk(
    days_to_expiry,
    current_stock,
    expected_demand_until_expiry,
):
    """
    Waste risk combines:
    - expiry urgency
    - stock remaining after expected consumption before expiry
    """
    surplus = max(
        current_stock - expected_demand_until_expiry,
        0.0
    )

    surplus_ratio = (
        surplus / current_stock
        if current_stock > 0
        else 0.0
    )

    expiry_risk = calculate_expiry_risk(days_to_expiry)

    if current_stock <= 0:
        return "none", surplus, surplus_ratio

    if days_to_expiry <= CRITICAL_DAYS and surplus > 0:
        return "critical", surplus, surplus_ratio

    if (
        days_to_expiry <= HIGH_DAYS
        and surplus_ratio >= 0.25
    ):
        return "high", surplus, surplus_ratio

    if (
        days_to_expiry <= MEDIUM_DAYS
        and surplus_ratio >= 0.40
    ):
        return "high", surplus, surplus_ratio

    if days_to_expiry <= HIGH_DAYS:
        return "medium", surplus, surplus_ratio

    if surplus_ratio >= 0.60 and days_to_expiry <= MEDIUM_DAYS:
        return "medium", surplus, surplus_ratio

    return expiry_risk if expiry_risk != "low" else "low", surplus, surplus_ratio


# ============================================================
# RECOMMENDATIONS
# ============================================================

def build_recommendations(
    days_to_expiry,
    waste_risk,
    surplus_quantity,
    surplus_ratio,
    current_stock,
):
    recommendations = []

    if current_stock <= 0:
        return ["No stock available; no waste-reduction action required."]

    if surplus_quantity <= 0:
        if days_to_expiry <= HIGH_DAYS:
            recommendations.append(
                "Prioritize normal selling/use before expiry."
            )
        else:
            recommendations.append(
                "Continue normal stock rotation and monitor expiry."
            )
        return recommendations

    # Expiry-first action.
    if waste_risk in {"critical", "high"}:
        recommendations.append(
            "priority_selling"
        )

    # Markdown/promotion is appropriate when a material surplus is
    # likely to remain before expiry.
    if (
        days_to_expiry <= MEDIUM_DAYS
        and surplus_ratio >= 0.25
    ):
        recommendations.append(
            "promotion_or_markdown"
        )

    # Reallocation is recommended as an action category, but the current
    # inventory file has no restaurant/location destination field.
    if waste_risk in {"critical", "high"}:
        recommendations.append(
            "consider_stock_reallocation"
        )

    # Avoid additional procurement while surplus inventory remains.
    if surplus_quantity > 0:
        recommendations.append(
            "reduce_or_delay_future_ordering"
        )

    if not recommendations:
        recommendations.append(
            "monitor_inventory_and_use_fifo"
        )

    return recommendations


# ============================================================
# ANALYZE ONE INVENTORY ROW
# ============================================================

def analyze_inventory_row(
    row,
    analysis_date,
    forecast_df,
    mapping,
    request,
):
    expiry_date = pd.Timestamp(row["expiry_date"]).normalize()
    days_to_expiry = int(
        (expiry_date - analysis_date).days
    )

    current_stock = _safe_float(
        row.get("current_stock"),
        0
    )

    avg_daily_usage = _safe_float(
        row.get("avg_daily_usage"),
        0
    )

    # Inventory Decision output may already contain a forecast-adjusted
    # ingredient usage rate. Prefer it when it is present and positive.
    forecast_avg_daily_usage = _safe_float(
        row.get("forecast_avg_daily_usage"),
        0
    )

    # Do not let "days until expiry" become negative for demand
    # consumption calculations.
    usable_days = max(days_to_expiry, 0)

    matched_forecast, forecast_match_status = (
        _forecast_for_inventory_item(
            row,
            forecast_df,
            mapping,
            request,
        )
    )

    forecasted_demand = None
    forecast_horizon = None

    if matched_forecast is not None and not matched_forecast.empty:
        matched_forecast = matched_forecast.copy()

        # Use only forecast rows on or after the analysis date.
        matched_forecast = matched_forecast[
            matched_forecast["date"].dt.normalize()
            >= analysis_date
        ]

        if not matched_forecast.empty:
            forecast_horizon = int(
                matched_forecast["date"].dt.normalize().nunique()
            )

            forecasted_demand = _safe_float(
                matched_forecast["predicted_quantity"].sum(),
                0
            )

    # If a direct forecast-to-inventory relationship exists, estimate
    # demand before expiry from the forecast's daily rate.
    if forecasted_demand is not None:
        if forecast_horizon and forecast_horizon > 0:
            forecast_daily_demand = (
                forecasted_demand / forecast_horizon
            )
        else:
            forecast_daily_demand = 0.0

        expected_demand_until_expiry = (
            forecast_daily_demand * usable_days
        )

        demand_source = "demand_forecasting_agent"

    else:
        # If the Inventory Decision Agent has already produced an
        # ingredient-level forecast usage rate, use it directly. This avoids
        # treating menu-item quantities as ingredient quantities.
        if forecast_avg_daily_usage > 0:
            forecast_daily_demand = forecast_avg_daily_usage
            expected_demand_until_expiry = (
                forecast_avg_daily_usage * usable_days
            )
            forecast_horizon = int(_safe_float(
                row.get("forecast_horizon_days"), 0
            )) or None
            forecasted_demand = (
                _safe_float(row.get("forecast_period_demand"), 0)
                or None
            )
            demand_source = "inventory_decision_forecast"
            forecast_match_status = "ingredient_level_inventory_decision"
        else:
            # Operational fallback when no ingredient-level forecast exists.
            forecast_daily_demand = avg_daily_usage
            expected_demand_until_expiry = (
                avg_daily_usage * usable_days
            )
            demand_source = "inventory_avg_daily_usage"

    (
        waste_risk,
        surplus_quantity,
        surplus_ratio,
    ) = calculate_waste_risk(
        days_to_expiry=days_to_expiry,
        current_stock=current_stock,
        expected_demand_until_expiry=expected_demand_until_expiry,
    )

    recommendations = build_recommendations(
        days_to_expiry=days_to_expiry,
        waste_risk=waste_risk,
        surplus_quantity=surplus_quantity,
        surplus_ratio=surplus_ratio,
        current_stock=current_stock,
    )

    promotion_recommended = (
        "promotion_or_markdown" in recommendations
    )

    return {
        "ingredient_id": row.get("ingredient_id"),
        "ingredient_name": row.get("ingredient_name"),
        "category": row.get("category"),
        "current_stock": round(current_stock, 2),
        "unit": row.get("unit"),
        "expiry_date": expiry_date.strftime("%Y-%m-%d"),
        "days_to_expiry": days_to_expiry,
        "expiry_risk": calculate_expiry_risk(days_to_expiry),
        "forecasted_demand": (
            round(forecasted_demand, 2)
            if forecasted_demand is not None
            else None
        ),
        "forecast_horizon_days": forecast_horizon,
        "forecast_daily_demand": round(
            forecast_daily_demand,
            2
        ),
        "expected_demand_until_expiry": round(
            expected_demand_until_expiry,
            2
        ),
        "surplus_quantity": round(
            surplus_quantity,
            2
        ),
        "surplus_ratio": round(
            surplus_ratio,
            3
        ),
        "waste_risk": waste_risk,
        "recommendations": recommendations,
        "promotion_recommended": promotion_recommended,
        "demand_source": demand_source,
        "forecast_match_status": forecast_match_status,
        "supplier_id": row.get("supplier_id"),
        "supplier_name": row.get("supplier_name"),
        "storage_type": row.get("storage_type"),
        "batch_id": row.get("batch_id"),
        "projected_ending_stock": row.get("projected_ending_stock"),
        "stockout_risk": row.get("stockout_risk"),
        "order_required": row.get("order_required"),
    }


# ============================================================
# FILTER REQUEST SCOPE
# ============================================================

def filter_inventory_scope(df, request):
    result = df.copy()

    ingredient_id = (
        request.get("ingredient_id")
        or request.get("inventory_item_id")
    )

    ingredient_name = (
        request.get("ingredient_name")
        or request.get("inventory_item_name")
    )

    if ingredient_id is not None:
        result = result[
            result["ingredient_id"].map(_normalize_id)
            == _normalize_id(ingredient_id)
        ]

    if ingredient_name is not None:
        result = result[
            result["ingredient_name"].map(_normalize_name)
            == _normalize_name(ingredient_name)
        ]

    return result


# ============================================================
# MAIN AGENT
# ============================================================

def agent(state):
    """
    Standard project entry point.

    Returns a dictionary and does not orchestrate other agents.
    """
    request = state.get("request", {})
    previous_results = state.get("agent_results", {})

    inventory_df = validate_inventory(
        load_inventory(state)
    )

    if inventory_df.empty:
        raise ValueError("Inventory dataset contains no valid rows.")

    analysis_date = get_analysis_date(
        request,
        inventory_df
    )

    scoped_inventory = filter_inventory_scope(
        inventory_df,
        request
    )

    if scoped_inventory.empty:
        raise ValueError(
            "No inventory item matches the requested scope."
        )

    forecast_result = _get_forecast_result(
        previous_results
    )

    forecast_df = _read_forecast_rows(
        forecast_result
    )

    mapping = _extract_inventory_mapping(
        state
    )

    # --------------------------------------------------------
    # Analyze every selected inventory item
    # --------------------------------------------------------

    records = []

    # Waste-reduction scope: only inventory expiring in the next 1 to 3 days.
    # Already-expired items (negative days), items expiring today (0 days),
    # and items more than 3 days from expiry are excluded from this analysis.
    for _, row in scoped_inventory.iterrows():
        expiry_date = pd.Timestamp(row["expiry_date"]).normalize()
        days_to_expiry = int((expiry_date - analysis_date).days)

        if not (1 <= days_to_expiry <= 3):
            continue

        records.append(
            analyze_inventory_row(
                row=row,
                analysis_date=analysis_date,
                forecast_df=forecast_df,
                mapping=mapping,
                request=request,
            )
        )

    result_df = pd.DataFrame(records)

    # Keep dataset-level handling safe when no inventory item falls inside
    # the 1-to-3-day expiry window.
    if result_df.empty:
        output_path = _resolve_path(
            request.get("waste_reduction_output_path"),
            OUTPUT_FILE,
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame().to_csv(output_path, index=False)

        return {
            "analysis_type": "dataset_waste_reduction",
            "analysis_date": analysis_date.strftime("%Y-%m-%d"),
            "waste_risk_file": str(output_path),
            "total_inventory_items_analyzed": 0,
            "products_at_risk": 0,
            "critical_or_high_risk_items": 0,
            "promotion_recommended_items": 0,
            "items": [],
            "message": "No inventory items expire within the next 1 to 3 days.",
        }

    # --------------------------------------------------------
    # Determine whether this is a single-item request
    # --------------------------------------------------------

    is_single_item = (
        request.get("ingredient_id") is not None
        or request.get("inventory_item_id") is not None
        or request.get("ingredient_name") is not None
        or request.get("inventory_item_name") is not None
    )

    if is_single_item and len(result_df) == 1:
        record = result_df.iloc[0].to_dict()

        return {
            "analysis_type": "single_item_waste_reduction",
            "analysis_date": analysis_date.strftime("%Y-%m-%d"),
            **record,
        }

    # --------------------------------------------------------
    # Dataset-level output
    # --------------------------------------------------------

    output_path = _resolve_path(
        request.get("waste_reduction_output_path"),
        OUTPUT_FILE,
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # Store recommendations as a readable string in CSV.
    csv_df = result_df.copy()

    csv_df["recommendations"] = csv_df[
        "recommendations"
    ].apply(
        lambda values: " | ".join(values)
        if isinstance(values, list)
        else str(values)
    )

    csv_df.to_csv(
        output_path,
        index=False
    )

    at_risk = result_df[
        result_df["waste_risk"].isin(
            ["critical", "high", "medium"]
        )
    ]

    critical_or_high = result_df[
        result_df["waste_risk"].isin(
            ["critical", "high"]
        )
    ]

    promotions = result_df[
        result_df["promotion_recommended"] == True
    ]

    return {
        "analysis_type": "dataset_waste_reduction",
        "analysis_date": analysis_date.strftime("%Y-%m-%d"),
        "waste_risk_file": str(output_path),
        "total_inventory_items_analyzed": int(
            len(result_df)
        ),
        "products_at_risk": int(
            len(at_risk)
        ),
        "critical_or_high_risk_items": int(
            len(critical_or_high)
        ),
        "promotion_recommended_items": int(
            len(promotions)
        ),
        # Return the detailed analysis to the orchestrator so the user gets
        # the requested answer directly instead of only a CSV path.
        "items": result_df.to_dict(orient="records"),
    }


# ============================================================
# OPTIONAL CLASS WRAPPER
# ============================================================

class WasteReductionAgent:
    """Compatibility wrapper for registries that expect an agent class."""

    def run(self, state):
        return agent(state)


if __name__ == "__main__":
    # Minimal local smoke test.
    test_state = {
        "request": {
            "inventory_data_path": DEFAULT_INVENTORY_FILE,
        },
        "agent_results": {},
    }

    print(agent(test_state))
