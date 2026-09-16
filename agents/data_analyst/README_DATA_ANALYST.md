# Data Analyst Agent

## Overview

The Data Analyst Agent prepares the raw demand data before it is used by the other agents in our demand forecasting system.

Its main job is to clean, validate and transform the dataset into a reliable format for forecasting and analysis.

## Flow

Raw Data → Cleaning → Validation → Feature Engineering → Unified Dataset → Other Agents

## Files

- `agent.py` — Runs the complete data pipeline.
- `data_loader.py` — Loads and checks the dataset.
- `data_cleaning.py` — Handles missing values, duplicates, invalid data and outliers.
- `feature_engineering.py` — Creates useful historical demand and rolling features.
- `inventory_handler.py` — Supports inventory data when a matching dataset is available.

## Input

Current dataset(assumed):

`data/qsr_demand_dataset.csv`

The main demand variable is `quantity`, representing the quantity sold/demanded for a menu item.

## Output

The processed dataset is saved as:

`data/processed/unified_demand.csv`

This dataset can then be used by the forecasting, seasonality and customer-pattern agents.

## Run

From the project root:

```bash
python3 run_data_analyst.py \
  --demand data/qsr_demand_dataset.csv \
  --output data/processed/unified_demand.csv