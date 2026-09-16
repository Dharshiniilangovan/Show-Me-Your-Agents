import argparse
from agents.data_analyst import DataAnalystAgent

def main():
    parser = argparse.ArgumentParser(description="Run the Data Analyst Agent")
    parser.add_argument("--demand", required=True, help="Path to qsr_demand_dataset.csv")
    parser.add_argument("--inventory", default=None, help="Optional matching inventory CSV")
    parser.add_argument("--output", default="data/processed/unified_demand.csv")
    args = parser.parse_args()

    result = DataAnalystAgent(args.demand, args.inventory).run(args.output)
    summary = result["summary"]
    print("Data Analyst Agent finished")
    print(f"Rows: {summary['rows']:,}")
    print(f"Dates: {summary['date_min']} -> {summary['date_max']}")
    print(f"Restaurants: {summary['restaurants']}")
    print(f"Menu items: {summary['menu_items']}")
    print(f"Inventory attached: {summary['inventory']['inventory_available']}")
    print(f"Output: {summary.get('output_path')}")


if __name__ == "__main__":
    main()
