"""Inventory Decision Agent for Phase 2.

Converts menu-item demand forecasts into ingredient-level inventory decisions.
Core decisions are deterministic (no LLM required).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
import pandas as pd


class InventoryDecisionAgent:
    REQUIRED_FORECAST = {"date", "restaurant_id", "menu_item_id", "predicted_quantity"}
    REQUIRED_RECIPE = {"menu_item_name", "ingredient_name", "count_per_item"}
    REQUIRED_INVENTORY = {
        "ingredient_id", "ingredient_name", "current_stock", "safety_stock",
        "lead_time_days", "avg_daily_usage"
    }

    def __init__(
        self,
        inventory_path: str | Path = "data/inventory_dataset.csv",
        recipe_path: str | Path = "data/ingredients.csv",
        unified_demand_path: str | Path = "data/processed/unified_demand.csv",
    ):
        self.inventory_path = Path(inventory_path)
        self.recipe_path = Path(recipe_path)
        self.unified_demand_path = Path(unified_demand_path)

    @staticmethod
    def _load_csv(path: Path, label: str) -> pd.DataFrame:
        if not path.exists():
            raise FileNotFoundError(f"{label} file not found: {path}")
        return pd.read_csv(path, low_memory=False)

    @staticmethod
    def _require(df: pd.DataFrame, required: set[str], label: str) -> None:
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"{label} is missing required columns: {sorted(missing)}")

    def _menu_lookup(self) -> pd.DataFrame:
        demand = self._load_csv(self.unified_demand_path, "Unified demand")
        self._require(demand, {"menu_item_id", "menu_item_name"}, "Unified demand")
        lookup = demand[["menu_item_id", "menu_item_name"]].dropna().drop_duplicates()
        conflicts = lookup.groupby("menu_item_id")["menu_item_name"].nunique()
        bad = conflicts[conflicts > 1]
        if not bad.empty:
            raise ValueError("Some menu_item_id values map to multiple menu_item_name values.")
        return lookup.drop_duplicates("menu_item_id")

    @staticmethod
    def _risk(row: pd.Series) -> str:
        current = float(row["scenario_current_stock"])
        lead_demand = float(row["lead_time_demand"])
        reorder_point = float(row["reorder_point"])
        projected = float(row["projected_ending_stock"])
        if projected < 0 or current < lead_demand:
            return "Critical"
        if current < reorder_point:
            return "High"
        if current < reorder_point * 1.25:
            return "Medium"
        return "Low"

    def run(
        self,
        forecast_path: str | Path = "data/outputs/demand_forecast.csv",
        output_path: str | Path = "data/outputs/inventory_decisions.csv",
        scenario: dict[str, float] | None = None,
    ) -> dict[str, Any]:
        """Create ingredient-level inventory decisions.

        scenario supports optional multipliers/deltas:
        - demand_multiplier (default 1.0)
        - stock_multiplier (default 1.0)
        - safety_stock_multiplier (default 1.0)
        - lead_time_extra_days (default 0)
        """
        scenario = scenario or {}
        demand_multiplier = float(scenario.get("demand_multiplier", 1.0))
        stock_multiplier = float(scenario.get("stock_multiplier", 1.0))
        safety_multiplier = float(scenario.get("safety_stock_multiplier", 1.0))
        lead_extra = float(scenario.get("lead_time_extra_days", 0.0))
        if min(demand_multiplier, stock_multiplier, safety_multiplier) < 0:
            raise ValueError("Scenario multipliers cannot be negative.")

        forecast = self._load_csv(Path(forecast_path), "Forecast")
        recipes = self._load_csv(self.recipe_path, "Recipe mapping")
        inventory = self._load_csv(self.inventory_path, "Inventory")
        self._require(forecast, self.REQUIRED_FORECAST, "Forecast")
        self._require(recipes, self.REQUIRED_RECIPE, "Recipe mapping")
        self._require(inventory, self.REQUIRED_INVENTORY, "Inventory")

        forecast = forecast.copy()
        forecast["date"] = pd.to_datetime(forecast["date"], errors="coerce")
        forecast["predicted_quantity"] = pd.to_numeric(forecast["predicted_quantity"], errors="coerce")
        forecast = forecast.dropna(subset=["date", "predicted_quantity"])
        forecast["predicted_quantity"] = forecast["predicted_quantity"].clip(lower=0) * demand_multiplier
        if forecast.empty:
            raise ValueError("Forecast contains no usable rows.")

        # Forecast output has IDs; recipes use names. Resolve via Data Analyst output.
        forecast = forecast.merge(self._menu_lookup(), on="menu_item_id", how="left", validate="many_to_one")
        missing_menu = sorted(forecast.loc[forecast["menu_item_name"].isna(), "menu_item_id"].astype(str).unique())
        if missing_menu:
            raise ValueError(f"No menu name mapping for forecast menu_item_id(s): {missing_menu[:10]}")

        recipes = recipes.copy()
        recipes["count_per_item"] = pd.to_numeric(recipes["count_per_item"], errors="coerce")
        recipes = recipes.dropna(subset=["menu_item_name", "ingredient_name", "count_per_item"])
        recipes = recipes[recipes["count_per_item"] > 0]

        expanded = forecast.merge(recipes, on="menu_item_name", how="left", validate="many_to_many")
        missing_recipe = sorted(expanded.loc[expanded["ingredient_name"].isna(), "menu_item_name"].dropna().unique())
        if missing_recipe:
            raise ValueError(f"No ingredient recipe found for menu item(s): {missing_recipe[:10]}")

        expanded["ingredient_demand"] = expanded["predicted_quantity"] * expanded["count_per_item"]
        horizon_days = int(forecast["date"].nunique())
        if horizon_days <= 0:
            raise ValueError("Could not determine forecast horizon.")

        ingredient_demand = (
            expanded.groupby("ingredient_name", as_index=False)["ingredient_demand"].sum()
            .rename(columns={"ingredient_demand": "forecast_period_demand"})
        )

        inv = inventory.copy()
        numeric = ["current_stock", "safety_stock", "lead_time_days", "avg_daily_usage"]
        for col in numeric:
            inv[col] = pd.to_numeric(inv[col], errors="coerce")
        inv = inv.dropna(subset=["ingredient_name", *numeric])
        inv = inv.drop_duplicates("ingredient_name", keep="last")

        decisions = ingredient_demand.merge(inv, on="ingredient_name", how="left", validate="one_to_one")
        missing_inventory = sorted(decisions.loc[decisions["ingredient_id"].isna(), "ingredient_name"].unique())
        if missing_inventory:
            raise ValueError(f"No inventory record for ingredient(s): {missing_inventory[:15]}")

        decisions["forecast_horizon_days"] = horizon_days
        decisions["forecast_avg_daily_usage"] = decisions["forecast_period_demand"] / horizon_days
        # Prefer forecast-driven usage; historical avg_daily_usage remains visible for comparison.
        decisions["scenario_current_stock"] = decisions["current_stock"] * stock_multiplier
        decisions["scenario_safety_stock"] = decisions["safety_stock"] * safety_multiplier
        decisions["scenario_lead_time_days"] = (decisions["lead_time_days"] + lead_extra).clip(lower=0)
        decisions["lead_time_demand"] = decisions["forecast_avg_daily_usage"] * decisions["scenario_lead_time_days"]
        decisions["reorder_point"] = decisions["lead_time_demand"] + decisions["scenario_safety_stock"]
        decisions["projected_ending_stock"] = decisions["scenario_current_stock"] - decisions["forecast_period_demand"]
        # Target covers the forecast horizon plus safety stock. This is a recommendation, not a purchase order.
        decisions["recommended_inventory_level"] = decisions["forecast_period_demand"] + decisions["scenario_safety_stock"]
        decisions["recommended_order_quantity"] = (
            decisions["recommended_inventory_level"] - decisions["scenario_current_stock"]
        ).clip(lower=0)
        decisions["stockout_risk"] = decisions.apply(self._risk, axis=1)
        decisions["order_required"] = decisions["recommended_order_quantity"] > 0
        decisions["days_of_cover_after_order"] = (
            decisions["recommended_inventory_level"] /
            decisions["forecast_avg_daily_usage"].replace(0, pd.NA)
        )

        def reason(row: pd.Series) -> str:
            if row["stockout_risk"] == "Critical":
                return "Stock may run out before replenishment or before forecast demand is met."
            if row["order_required"]:
                return "Stock is below the recommended forecast-horizon level plus safety stock."
            return "Current stock is sufficient for forecast demand and safety stock."

        decisions["decision_reason"] = decisions.apply(reason, axis=1)

        round_cols = [
            "forecast_period_demand", "forecast_avg_daily_usage", "scenario_current_stock",
            "scenario_safety_stock", "scenario_lead_time_days", "lead_time_demand",
            "reorder_point", "projected_ending_stock", "recommended_inventory_level",
            "recommended_order_quantity", "days_of_cover_after_order"
        ]
        decisions[round_cols] = decisions[round_cols].astype(float).round(2)

        priority = pd.Categorical(decisions["stockout_risk"], ["Critical", "High", "Medium", "Low"], ordered=True)
        decisions = decisions.assign(_priority=priority).sort_values(
            ["_priority", "recommended_order_quantity"], ascending=[True, False]
        ).drop(columns="_priority")

        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        decisions.to_csv(out, index=False)

        risk_counts = decisions["stockout_risk"].value_counts().to_dict()
        return {
            "status": "success",
            "forecast_horizon_days": horizon_days,
            "ingredients_evaluated": int(len(decisions)),
            "orders_recommended": int(decisions["order_required"].sum()),
            "stockout_risk_counts": {str(k): int(v) for k, v in risk_counts.items()},
            "total_recommended_order_quantity": round(float(decisions["recommended_order_quantity"].sum()), 2),
            "scenario": {
                "demand_multiplier": demand_multiplier,
                "stock_multiplier": stock_multiplier,
                "safety_stock_multiplier": safety_multiplier,
                "lead_time_extra_days": lead_extra,
            },
            "output_path": str(out),
        }
