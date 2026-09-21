import argparse
import json
from agents.inventory_decision import InventoryDecisionAgent


def main():
    parser = argparse.ArgumentParser(description="Run the Inventory Decision Agent")
    parser.add_argument("--forecast", default="data/outputs/demand_forecast.csv")
    parser.add_argument("--inventory", default="data/inventory_dataset.csv")
    parser.add_argument("--recipes", default="data/ingredients.csv")
    parser.add_argument("--unified-demand", default="data/processed/unified_demand.csv")
    parser.add_argument("--output", default="data/outputs/inventory_decisions.csv")
    parser.add_argument("--demand-multiplier", type=float, default=1.0)
    parser.add_argument("--stock-multiplier", type=float, default=1.0)
    parser.add_argument("--safety-stock-multiplier", type=float, default=1.0)
    parser.add_argument("--lead-time-extra-days", type=float, default=0.0)
    args = parser.parse_args()

    agent = InventoryDecisionAgent(args.inventory, args.recipes, args.unified_demand)
    result = agent.run(
        forecast_path=args.forecast,
        output_path=args.output,
        scenario={
            "demand_multiplier": args.demand_multiplier,
            "stock_multiplier": args.stock_multiplier,
            "safety_stock_multiplier": args.safety_stock_multiplier,
            "lead_time_extra_days": args.lead_time_extra_days,
        },
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
