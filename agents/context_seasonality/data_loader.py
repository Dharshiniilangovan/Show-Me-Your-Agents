import pandas as pd
REQUIRED = {'date','restaurant_id','menu_item_id','quantity','day_of_week_num','month','year','avg_temp_f','precip_inches','is_holiday','holiday_name','is_special_event','special_event_name','is_promotion'}
def load_data(path):
    df = pd.read_csv(path, parse_dates=['date'])
    missing = REQUIRED - set(df.columns)
    if missing: raise ValueError(f'Missing required columns: {sorted(missing)}')
    df['quantity'] = pd.to_numeric(df['quantity'], errors='coerce')
    df['avg_temp_f'] = pd.to_numeric(df['avg_temp_f'], errors='coerce')
    df['precip_inches'] = pd.to_numeric(df['precip_inches'], errors='coerce').fillna(0)
    for c in ['is_holiday','is_special_event','is_promotion','is_weekend']:
        if c in df: df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0).astype('int8')
    return df
