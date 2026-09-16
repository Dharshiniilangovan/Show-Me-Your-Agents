from pathlib import Path
import json
import pandas as pd

from .data_loader import load_demand_data, load_inventory_data
from .data_cleaning import clean_demand_data
from .feature_engineering import add_features
from .inventory_handler import integrate_inventory, inventory_snapshot


class DataAnalystAgent:
    """Central data-processing agent for the demand-forecasting system."""

    def __init__(self, demand_path: str, inventory_path: str | None = None):
        self.demand_path = demand_path
        self.inventory_path = inventory_path

    def run(self, output_path: str | None = None) -> dict:
        raw = load_demand_data(self.demand_path)
        clean, quality = clean_demand_data(raw)
        featured = add_features(clean)

        inventory = load_inventory_data(self.inventory_path) if self.inventory_path else None
        unified = integrate_inventory(featured, inventory)

        summary = {
            "status": "success",
            "rows": int(len(unified)),
            "date_min": str(unified["date"].min().date()),
            "date_max": str(unified["date"].max().date()),
            "restaurants": int(unified["restaurant_id"].nunique()),
            "menu_items": int(unified["menu_item_id"].nunique()),
            "quality": quality,
            "inventory": inventory_snapshot(unified),
            "columns": list(unified.columns),
        }

        if output_path:
            output = Path(output_path)
            output.parent.mkdir(parents=True, exist_ok=True)
            if output.suffix.lower() == ".parquet":
                unified.to_parquet(output, index=False)
            else:
                unified.to_csv(output, index=False)
            summary["output_path"] = str(output)

            report_path = output.with_suffix(".report.json")
            report_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
            summary["report_path"] = str(report_path)

        return {"data": unified, "summary": summary}
