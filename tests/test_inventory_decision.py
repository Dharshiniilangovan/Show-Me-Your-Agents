import pandas as pd
from agents.inventory_decision import InventoryDecisionAgent


def test_inventory_decision_basic(tmp_path):
    forecast = pd.DataFrame({
        "date": ["2026-10-01", "2026-10-02"],
        "restaurant_id": ["R01", "R01"],
        "menu_item_id": ["M01", "M01"],
        "predicted_quantity": [10, 12],
    })
    unified = pd.DataFrame({"menu_item_id": ["M01"], "menu_item_name": ["Burger"]})
    recipes = pd.DataFrame({"menu_item_name": ["Burger"], "ingredient_name": ["Bun"], "count_per_item": [1]})
    inventory = pd.DataFrame({
        "ingredient_id": ["I01"], "ingredient_name": ["Bun"], "current_stock": [5],
        "safety_stock": [4], "lead_time_days": [2], "avg_daily_usage": [8]
    })
    paths = {}
    for name, df in [("forecast", forecast), ("unified", unified), ("recipes", recipes), ("inventory", inventory)]:
        p = tmp_path / f"{name}.csv"; df.to_csv(p, index=False); paths[name] = p
    out = tmp_path / "decisions.csv"
    agent = InventoryDecisionAgent(paths["inventory"], paths["recipes"], paths["unified"])
    summary = agent.run(paths["forecast"], out)
    result = pd.read_csv(out)
    assert summary["status"] == "success"
    assert result.loc[0, "forecast_period_demand"] == 22
    assert result.loc[0, "recommended_order_quantity"] == 21
    assert result.loc[0, "stockout_risk"] == "Critical"


def test_what_if_increases_order(tmp_path):
    forecast = pd.DataFrame({"date": ["2026-10-01"], "restaurant_id": ["R01"], "menu_item_id": ["M01"], "predicted_quantity": [10]})
    unified = pd.DataFrame({"menu_item_id": ["M01"], "menu_item_name": ["Burger"]})
    recipes = pd.DataFrame({"menu_item_name": ["Burger"], "ingredient_name": ["Bun"], "count_per_item": [1]})
    inventory = pd.DataFrame({"ingredient_id": ["I01"], "ingredient_name": ["Bun"], "current_stock": [8], "safety_stock": [2], "lead_time_days": [1], "avg_daily_usage": [5]})
    for name, df in [("forecast", forecast), ("unified", unified), ("recipes", recipes), ("inventory", inventory)]:
        df.to_csv(tmp_path / f"{name}.csv", index=False)
    agent = InventoryDecisionAgent(tmp_path/"inventory.csv", tmp_path/"recipes.csv", tmp_path/"unified.csv")
    agent.run(tmp_path/"forecast.csv", tmp_path/"base.csv")
    agent.run(tmp_path/"forecast.csv", tmp_path/"stress.csv", {"demand_multiplier": 1.2})
    base = pd.read_csv(tmp_path/"base.csv").loc[0, "recommended_order_quantity"]
    stress = pd.read_csv(tmp_path/"stress.csv").loc[0, "recommended_order_quantity"]
    assert stress > base
