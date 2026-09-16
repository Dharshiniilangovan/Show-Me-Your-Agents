from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / 'data' / 'raw' / 'qsr_demand_dataset.csv'
PROCESSED = ROOT / 'data' / 'processed'
OUTPUTS = ROOT / 'data' / 'outputs'
ID_COLS = ['restaurant_id','menu_item_id']
TARGET = 'quantity'
