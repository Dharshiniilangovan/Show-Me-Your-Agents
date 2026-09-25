# app.py
# QSR Demand Forecast & Inventory Dashboard
# Uses the project's real qsr_demand_dataset.csv, inventory_dataset.csv,
# and ingredients.csv files. No synthetic fallback data is generated.

from __future__ import annotations

from datetime import timedelta
from pathlib import Path
import math

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="DemandRadar: Forecast & Inventory Insights",
    page_icon="🍴",
    layout="wide",
    initial_sidebar_state="collapsed",
)

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"

DEMAND_FILE = DATA_DIR / "qsr_demand_dataset.csv"
INVENTORY_FILE = DATA_DIR / "inventory_dataset.csv"
INGREDIENT_FILE = DATA_DIR / "ingredients.csv"

RANDOM_SEED = 42


# ============================================================
# SINGLE CONSOLIDATED DASHBOARD THEME
# ============================================================
st.markdown(
    """
    <style>
    :root {
        --navy:#0A4D9C; --text:#111827; --muted:#486581;
        --surface:#F7FBFF; --border:#AFCFEF; --border-hover:#86B8E8;
        --control:#CFE6F5; --control-hover:#BDDDF1; --control-border:#AFCFE5;
        --red-bg:#FEE2E2; --red:#DC2626; --red-border:#EF4444;
        --amber-bg:#FFF7E6; --amber-border:#F2C66D;
        --green-bg:#EFFAF4; --green-border:#B9DFC9;
    }

    .stApp {background:#F8FBFF;color:var(--text);}
    [data-testid="stHeader"] {background:transparent;}
    .block-container {max-width:1540px;padding:.45rem 4.8rem 1.2rem 2.4rem;}
    html,body,[class*="css"],.stMarkdown,.stText,p,span {color:var(--text);}
    div[data-testid="stVerticalBlock"] {gap:.48rem;}
    div[data-testid="stHorizontalBlock"] {gap:.72rem;}
    hr {margin:.55rem 0 !important;}

    .hero {
        background:linear-gradient(110deg,#0A4D9C 0%,#2B6CB0 62%,#0A4D9C 100%);
        color:white;border-radius:16px;padding:14px 22px;margin:0 0 20px;
        box-shadow:0 7px 20px rgba(17,49,88,.14);
    }
    .hero * {color:white !important;}
    .hero-title {font-size:2.12rem;font-weight:850;line-height:1.05;}
    .hero-subtitle {font-size:.91rem;margin-top:4px;line-height:1.25;opacity:.92;}

    .section-heading {
        font-size:1.04rem;font-weight:800;color:var(--navy) !important;
        margin:15px 0 !important;line-height:1.2;
    }

        /* ==========================================
       CARD SYSTEM - ONE-SIDED ACCENT BORDERS
       ========================================== */

    /* Default informational cards */
    .card,
    .info-card,
    .kpi-blue {
        background:#F7FBFF !important;
        border-top:1px solid #AFCFEF !important;
        border-right:1px solid #AFCFEF !important;
        border-bottom:1px solid #AFCFEF !important;
        border-left:7px solid #9DCCF3 !important;
        border-radius:13px !important;
        padding:12px 14px !important;
        box-shadow:0 4px 12px rgba(59,130,246,.08) !important;
    }

    /* Hover keeps the left accent intact */
    .card:hover,
    .info-card:hover,
    .kpi-blue:hover {
        border-top-color:#86B8E8 !important;
        border-right-color:#86B8E8 !important;
        border-bottom-color:#86B8E8 !important;
        border-left-color:#9DCCF3 !important;
    }

    .kpi-label {
        font-size:.88rem;
        font-weight:800;
        color:var(--navy) !important;
        margin-bottom:6px;
    }
    .kpi-value {
        color:var(--text) !important;
        font-size:1.82rem;
        font-weight:700;
        line-height:1.02;
        margin-bottom:5px;
    }
    .kpi-note {
        color:var(--muted) !important;
        font-size:.76rem;
        line-height:1.25;
    }
    .id-label {
        color:var(--text) !important;
        font-size:.72rem;
        font-weight:800;
        text-transform:uppercase;
    }
    .id-value {
        color:var(--text) !important;
        font-size:.92rem;
        font-weight:500;
        margin-bottom:6px;
    }

    /* Critical / expired / at-risk cards */
    .card.kpi-red,
    .kpi-red,
    .alert-card-red,
    .health-fail {
        background:#FEE2E2 !important;
        border-top:1px solid #EF4444 !important;
        border-right:1px solid #EF4444 !important;
        border-bottom:1px solid #EF4444 !important;
        border-left:7px solid #DC2626 !important;
        box-shadow:0 4px 12px rgba(220,38,38,.12) !important;
    }

    .card.kpi-red:hover,
    .kpi-red:hover,
    .alert-card-red:hover,
    .health-fail:hover {
        border-top-color:#EF4444 !important;
        border-right-color:#EF4444 !important;
        border-bottom-color:#EF4444 !important;
        border-left-color:#DC2626 !important;
    }

    .kpi-red .kpi-label,
    .alert-card-red b,
    .health-fail b {
        color:#991B1B !important;
    }

    /* Amber warning cards */
    .card.kpi-orange,
    .kpi-orange {
        background:#FFF7E6 !important;
        border-top:1px solid #F2C66D !important;
        border-right:1px solid #F2C66D !important;
        border-bottom:1px solid #F2C66D !important;
        border-left:7px solid #E5A623 !important;
        box-shadow:0 4px 12px rgba(217,145,17,.09) !important;
    }

    .card.kpi-orange:hover,
    .kpi-orange:hover {
        border-top-color:#F2C66D !important;
        border-right-color:#F2C66D !important;
        border-bottom-color:#F2C66D !important;
        border-left-color:#E5A623 !important;
    }

    .kpi-orange .kpi-label {
        color:#8A5A00 !important;
    }

    /* Positive status only */
    .card.kpi-green,
    .kpi-green,
    .health-pass,
    .alert-green,
    .validation-pass {
        background:#EFFAF4 !important;
        border-top:1px solid #B9DFC9 !important;
        border-right:1px solid #B9DFC9 !important;
        border-bottom:1px solid #B9DFC9 !important;
        border-left:7px solid #49A66B !important;
    }

    .card.kpi-green:hover,
    .kpi-green:hover {
        border-top-color:#B9DFC9 !important;
        border-right-color:#B9DFC9 !important;
        border-bottom-color:#B9DFC9 !important;
        border-left-color:#49A66B !important;
    }

    .kpi-green .kpi-label,
    .health-pass,
    .alert-green {
        color:#176B39 !important;
    }

    .alert-card {
        padding:8px 12px !important;min-height:64px !important;border-radius:10px;
        margin:2px 0 14px !important;font-size:.82rem;line-height:1.28;
    }
    .alert-card b {display:block;font-size:.84rem;margin-bottom:4px;}
    .alert-green {padding:10px 14px;margin:6px 0;border-radius:10px;}
    .health-pass,.health-fail {
        padding:8px 12px !important;border-radius:10px;line-height:1.25;
        min-height:0 !important;margin:2px 0 8px !important;
    }
    .health-pass div,.health-fail div {font-size:1.35rem !important;line-height:1.05 !important;margin-bottom:4px;}

    label,[data-testid="stWidgetLabel"],[data-testid="stWidgetLabel"] p {
        color:var(--text) !important;font-weight:750 !important;font-size:.84rem !important;
    }
    [data-testid="stWidgetLabel"] {margin-bottom:6px !important;}

    /* ==========================================
       FILTER CONTROLS
       Streamlit theme supplies the persistent widget border.
       CSS here controls background, radius, and hover only.
       ========================================== */

    div[data-testid="stSelectbox"] [data-baseweb="select"] > div,
    div[data-testid="stDateInput"] > div > div,
    div[data-testid="stTextInput"] input,
    div[data-testid="stMultiSelect"] [data-baseweb="select"] > div {
        background:#F7FBFF !important;
        border-radius:9px !important;
        color:#111827 !important;
        box-shadow:0 2px 7px rgba(59,130,246,.05) !important;
    }

    div[data-testid="stSelectbox"] [data-baseweb="select"] > div:hover,
    div[data-testid="stDateInput"] > div > div:hover,
    div[data-testid="stTextInput"] input:hover,
    div[data-testid="stMultiSelect"] [data-baseweb="select"] > div:hover {
        border-color:#86B8E8 !important;
    }

    [data-testid="stMultiSelect"] [data-baseweb="tag"],
    [data-testid="stMultiSelect"] span[data-baseweb="tag"],
    [data-testid="stMultiSelect"] [data-baseweb="select"] [role="button"] {
        background:var(--control) !important;
        border:1px solid var(--control-border) !important;
        color:var(--navy) !important;
        border-radius:7px !important;
        box-shadow:0 2px 6px rgba(59,130,246,.06) !important;
    }

    [data-testid="stMultiSelect"] [data-baseweb="tag"]:hover,
    [data-testid="stMultiSelect"] span[data-baseweb="tag"]:hover,
    [data-testid="stMultiSelect"] [data-baseweb="select"] [role="button"]:hover {
        background:var(--control-hover) !important;
        border-color:#94BFDC !important;
    }
    [data-testid="stMultiSelect"] [data-baseweb="tag"] *,
    [data-testid="stMultiSelect"] span[data-baseweb="tag"] *,
    [data-testid="stMultiSelect"] [data-baseweb="select"] [role="button"] * {
        color:var(--navy) !important;-webkit-text-fill-color:var(--navy) !important;
    }

    div[data-testid="stButton"] > button,
    div[data-testid="stButton"] > button[kind="primary"] {
        background:var(--control) !important;border:1px solid var(--control-border) !important;
        color:var(--navy) !important;border-radius:10px !important;font-weight:700 !important;
        min-height:2.25rem !important;box-shadow:0 3px 9px rgba(59,130,246,.08) !important;
    }
    div[data-testid="stButton"] > button * {
        color:var(--navy) !important;-webkit-text-fill-color:var(--navy) !important;
    }
    div[data-testid="stButton"] > button:hover {
        background:var(--control-hover) !important;border-color:#94BFDC !important;
    }

    /* View Forecast alignment with the input controls */
    .forecast-button-anchor {
        height:25px;
        margin:0;
        padding:0;
    }

    /* Keep the primary forecast button the same height as the filter controls */
    div[data-testid="stButton"] > button[kind="primary"] {
        height:46px !important;
        min-height:46px !important;
        margin-top:0 !important;
    }

    div[data-testid="stDownloadButton"] > button {
        background:var(--navy) !important;color:white !important;border:1px solid var(--navy) !important;
        border-radius:8px !important;font-weight:700 !important;min-height:2rem !important;height:2rem !important;
        padding:.25rem .8rem !important;font-size:.78rem !important;box-shadow:none !important;
    }
    div[data-testid="stDownloadButton"] > button * {color:white !important;-webkit-text-fill-color:white !important;}
    div[data-testid="stDownloadButton"] > button:hover {background:#073B78 !important;border-color:#073B78 !important;}

    [data-testid="stDataFrame"] {
        background:#FFF !important;border:1.5px solid var(--border) !important;border-radius:12px !important;
        overflow:hidden !important;box-shadow:0 3px 12px rgba(59,130,246,.06) !important;
        font-size:.79rem !important;margin-top:4px !important;
    }
    [data-testid="stDataFrame"] [role="columnheader"],
    [data-testid="stDataFrame"] [role="columnheader"] *,
    [data-testid="stDataFrame"] th,[data-testid="stDataFrame"] th *,
    thead th,thead th * {
        background:#F3F6FA !important;color:#111111 !important;
        -webkit-text-fill-color:#111111 !important;font-weight:900 !important;opacity:1 !important;
    }
    [data-testid="stDataFrame"] [role="gridcell"] {color:var(--text) !important;font-weight:400 !important;}
    table {background:#FFF !important;color:var(--text) !important;border-color:var(--border) !important;}
    tbody td {color:var(--text) !important;padding:4px 7px !important;font-weight:400 !important;}
    tbody tr:nth-child(even) {background:#F8FAFC !important;}

    [data-testid="stPlotlyChart"] {margin-top:4px !important;margin-bottom:8px !important;}
    [data-testid="stDateInput"],[data-testid="stSelectbox"],
    [data-testid="stTextInput"],[data-testid="stMultiSelect"] {margin-bottom:4px !important;}

    .insights-list,.insight-list {
        font-size:.86rem !important;line-height:1.4 !important;
        margin:.2rem 0 .3rem 1.05rem !important;padding:0 !important;
    }
    .insights-list li,.insight-list li {margin:.28rem 0 !important;padding-left:.15rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# CUSTOM CSS
# ============================================================




# Executive compact UI overrides



# v5 visual hierarchy: richer cards + breathing room between sections



# ============================================================
# DATA LOADING
# ============================================================

def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    result.columns = [
        str(c).replace("\ufeff", "").strip().lower()
        for c in result.columns
    ]
    return result


@st.cache_data(show_spinner=False)
def load_demand_data() -> pd.DataFrame:
    if not DEMAND_FILE.exists():
        raise FileNotFoundError(
            f"Demand dataset not found: {DEMAND_FILE}\n"
            "Place qsr_demand_dataset.csv inside the data folder."
        )

    df = _normalise_columns(pd.read_csv(DEMAND_FILE, low_memory=False))

    required = {
        "date", "restaurant_id", "restaurant_name",
        "menu_item_id", "menu_item_name", "quantity"
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(
            "Demand dataset is missing required columns: "
            + ", ".join(missing)
        )

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce")
    df = df.dropna(
        subset=["date", "restaurant_id", "menu_item_id", "quantity"]
    ).copy()

    df["restaurant_id"] = df["restaurant_id"].astype(str).str.strip()
    df["menu_item_id"] = df["menu_item_id"].astype(str).str.strip()

    if "is_promotion" not in df.columns:
        df["is_promotion"] = 0
    df["is_promotion"] = pd.to_numeric(
        df["is_promotion"], errors="coerce"
    ).fillna(0).astype(int)

    return df


@st.cache_data(show_spinner=False)
def load_inventory_data() -> pd.DataFrame:
    if not INVENTORY_FILE.exists():
        raise FileNotFoundError(
            f"Inventory dataset not found: {INVENTORY_FILE}"
        )

    df = _normalise_columns(pd.read_csv(INVENTORY_FILE, low_memory=False))

    required = {
        "restaurant_id", "ingredient_id", "ingredient_name",
        "current_stock", "expiry_date"
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(
            "Inventory dataset is missing required columns: "
            + ", ".join(missing)
        )

    df["restaurant_id"] = df["restaurant_id"].astype(str).str.strip()
    df["ingredient_id"] = df["ingredient_id"].astype(str).str.strip()
    df["current_stock"] = pd.to_numeric(
        df["current_stock"], errors="coerce"
    ).fillna(0)

    # Your uploaded inventory uses day/month/year dates such as 15/01/2026.
    df["expiry_date"] = pd.to_datetime(
        df["expiry_date"], errors="coerce", dayfirst=True
    )
    if "received_date" in df.columns:
        df["received_date"] = pd.to_datetime(
            df["received_date"], errors="coerce", dayfirst=True
        )

    return df.dropna(subset=["expiry_date"]).copy()


@st.cache_data(show_spinner=False)
def load_ingredient_mapping() -> pd.DataFrame:
    if not INGREDIENT_FILE.exists():
        raise FileNotFoundError(
            f"Ingredient mapping not found: {INGREDIENT_FILE}"
        )

    df = _normalise_columns(pd.read_csv(INGREDIENT_FILE, low_memory=False))

    required = {
        "menu_item_id", "ingredient_id",
        "ingredient_name", "count_per_item"
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(
            "Ingredient mapping is missing required columns: "
            + ", ".join(missing)
        )

    df["menu_item_id"] = df["menu_item_id"].astype(str).str.strip()
    df["ingredient_id"] = df["ingredient_id"].astype(str).str.strip()
    return df


# ============================================================
# FEATURE ENGINEERING / FORECASTING
# ============================================================

FEATURES = [
    "day_of_week", "month", "quarter", "is_weekend",
    "day_of_year", "is_promotion",
    "lag_1", "lag_7", "lag_14",
    "rolling_mean_7", "rolling_std_7", "rolling_mean_14",
]


def prepare_features(history: pd.DataFrame) -> pd.DataFrame:
    x = history.sort_values("date").copy()
    x["day_of_week"] = x["date"].dt.dayofweek
    x["month"] = x["date"].dt.month
    x["quarter"] = x["date"].dt.quarter
    x["is_weekend"] = x["day_of_week"].isin([5, 6]).astype(int)
    x["day_of_year"] = x["date"].dt.dayofyear

    x["lag_1"] = x["quantity"].shift(1)
    x["lag_7"] = x["quantity"].shift(7)
    x["lag_14"] = x["quantity"].shift(14)

    shifted = x["quantity"].shift(1)
    x["rolling_mean_7"] = shifted.rolling(7, min_periods=3).mean()
    x["rolling_std_7"] = shifted.rolling(7, min_periods=3).std()
    x["rolling_mean_14"] = shifted.rolling(14, min_periods=5).mean()

    return x


def validation_metrics(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mae = mean_absolute_error(y_true, y_pred)
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))

    mask = np.abs(y_true) > 1e-8
    mape = (
        float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)
        if mask.any()
        else 0.0
    )

    return {
        "MAE": float(mae),
        "RMSE": float(rmse),
        "MAPE": mape,
        "R2": float(r2_score(y_true, y_pred)),
    }


@st.cache_resource(show_spinner=False)
def train_model(restaurant_id: str, menu_item_id: str, signature: tuple):
    df = load_demand_data()
    history = df[
        (df["restaurant_id"] == restaurant_id)
        & (df["menu_item_id"] == menu_item_id)
    ].copy()

    feature_data = prepare_features(history).dropna(
        subset=FEATURES + ["quantity"]
    )

    if len(feature_data) < 60:
        raise ValueError("Not enough history to train the selected forecast.")

    test_size = min(max(30, int(len(feature_data) * 0.2)), len(feature_data) - 30)
    train = feature_data.iloc[:-test_size]
    test = feature_data.iloc[-test_size:]

    validation_model = RandomForestRegressor(
        n_estimators=250,
        max_depth=14,
        min_samples_leaf=2,
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    validation_model.fit(train[FEATURES], train["quantity"])
    test_pred = validation_model.predict(test[FEATURES])
    metrics = validation_metrics(test["quantity"], test_pred)

    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=14,
        min_samples_leaf=2,
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    model.fit(feature_data[FEATURES], feature_data["quantity"])

    return model, metrics


def future_feature_row(future_date, values):
    def lag(n):
        return float(values[-n]) if len(values) >= n else float(values[0])

    last7 = values[-7:]
    last14 = values[-14:]

    return {
        "day_of_week": future_date.dayofweek,
        "month": future_date.month,
        "quarter": future_date.quarter,
        "is_weekend": int(future_date.dayofweek >= 5),
        "day_of_year": future_date.dayofyear,
        "is_promotion": 0,
        "lag_1": lag(1),
        "lag_7": lag(7),
        "lag_14": lag(14),
        "rolling_mean_7": float(np.mean(last7)),
        "rolling_std_7": float(np.std(last7, ddof=1)) if len(last7) > 1 else 0.0,
        "rolling_mean_14": float(np.mean(last14)),
    }


def make_forecast(model, history, start_date, end_date):
    dates = pd.date_range(start_date, end_date, freq="D")
    values = history.sort_values("date")["quantity"].astype(float).tolist()
    records = []

    for future_date in dates:
        row = future_feature_row(future_date, values)
        pred = max(0.0, float(model.predict(pd.DataFrame([row])[FEATURES])[0]))
        records.append(
            {
                "Date": future_date,
                "Day": future_date.day_name()[:3],
                "Predicted Demand": pred,
            }
        )
        values.append(pred)

    return pd.DataFrame(records)


# ============================================================
# INVENTORY / EXPIRY ANALYSIS
# ============================================================

def inventory_analysis_date(inventory: pd.DataFrame) -> pd.Timestamp:
    # This project inventory is a December-2025 snapshot. Use the latest
    # received date as the operational snapshot date, matching the agent logic.
    if "received_date" in inventory.columns:
        received = inventory["received_date"].dropna()
        if not received.empty:
            return received.max().normalize()
    return pd.Timestamp("2025-12-31")


def selected_item_inventory(
    inventory: pd.DataFrame,
    mapping: pd.DataFrame,
    restaurant_id: str,
    menu_item_id: str,
) -> pd.DataFrame:
    if menu_item_id == "ALL":
        mapped = mapping.copy()
    else:
        mapped = mapping[
            mapping["menu_item_id"].astype(str) == str(menu_item_id)
        ].copy()

    if mapped.empty:
        return pd.DataFrame()

    ingredient_ids = mapped["ingredient_id"].astype(str).unique().tolist()
    inv = inventory[
        inventory["ingredient_id"].astype(str).isin(ingredient_ids)
    ].copy()

    if restaurant_id != "ALL":
        inv = inv[
            inv["restaurant_id"].astype(str) == str(restaurant_id)
        ].copy()

    if inv.empty:
        return inv

    map_cols = [
        c for c in [
            "menu_item_id", "ingredient_id", "ingredient_name",
            "count_per_item", "unit"
        ]
        if c in mapped.columns
    ]
    mapped_small = mapped[map_cols].drop_duplicates()

    inv = inv.merge(
        mapped_small,
        on="ingredient_id",
        how="left",
        suffixes=("", "_recipe"),
    )
    return inv


def add_expiry_status(inv: pd.DataFrame, analysis_date: pd.Timestamp):
    result = inv.copy()
    result["actual_days_left"] = (
        result["expiry_date"].dt.normalize() - analysis_date
    ).dt.days

    # Requirement: never display negative days for expired items.
    result["days_left"] = result["actual_days_left"].clip(lower=0)

    def classify(days):
        if days <= 0:
            return "Expired"
        if days <= 3:
            return "Critical"
        if days <= 7:
            return "Nearing Expiry"
        return "Fresh"

    result["expiry_status"] = result["actual_days_left"].apply(classify)
    return result


def stock_status_from_inventory(inv: pd.DataFrame):
    if inv.empty:
        return "Not Available", "orange"

    if "stock_status" in inv.columns:
        statuses = inv["stock_status"].astype(str).str.lower()
        if statuses.str.contains("critical|out|low").any():
            return "At Risk", "red"
        if statuses.str.contains("watch|medium").any():
            return "Low Stock", "orange"

    return "Sufficient", "green"


def expiry_counts(inv: pd.DataFrame):
    counts = inv["expiry_status"].value_counts().to_dict() if not inv.empty else {}
    return {
        "Expired": int(counts.get("Expired", 0)),
        "Critical": int(counts.get("Critical", 0)),
        "Nearing Expiry": int(counts.get("Nearing Expiry", 0)),
        "Fresh": int(counts.get("Fresh", 0)),
    }



# ============================================================
# OPERATIONAL RISK / RECOMMENDATION LOGIC
# ============================================================

def add_inventory_monitoring_fields(inv: pd.DataFrame) -> pd.DataFrame:
    result = inv.copy()

    # Use the real reorder_level as the operational minimum threshold.
    # If unavailable, fall back to safety_stock, then zero.
    if "reorder_level" in result.columns:
        result["minimum_threshold"] = pd.to_numeric(
            result["reorder_level"], errors="coerce"
        ).fillna(0)
    elif "safety_stock" in result.columns:
        result["minimum_threshold"] = pd.to_numeric(
            result["safety_stock"], errors="coerce"
        ).fillna(0)
    else:
        result["minimum_threshold"] = 0.0

    quantity = pd.to_numeric(
        result["current_stock"], errors="coerce"
    ).fillna(0)

    threshold = result["minimum_threshold"]

    def inventory_status(row):
        qty = float(row["current_stock"])
        minimum = float(row["minimum_threshold"])

        if minimum <= 0:
            return "Sufficient"
        if qty <= minimum * 0.5:
            return "Critical"
        if qty <= minimum:
            return "Low Stock"
        return "Sufficient"

    result["inventory_status"] = result.apply(inventory_status, axis=1)

    def recommendation(row):
        inv_status = row["inventory_status"]
        expiry_status = row["expiry_status"]

        if expiry_status == "Expired":
            return "Discard Immediately"
        if inv_status == "Critical":
            return "Immediate Procurement"
        if inv_status == "Low Stock" and expiry_status in {
            "Critical", "Nearing Expiry"
        }:
            return "Urgent Action Required"
        if inv_status == "Low Stock":
            return "Reorder"
        if expiry_status in {"Critical", "Nearing Expiry"}:
            return "Use First (FIFO)"
        return "No Immediate Action"

    result["recommended_action"] = result.apply(recommendation, axis=1)
    return result


def detect_abnormal_demand(
    history: pd.DataFrame,
    forecast: pd.DataFrame,
) -> tuple[bool, float, float, float]:
    recent = (
        history.sort_values("date")["quantity"]
        .astype(float)
        .tail(30)
    )

    historical_average = float(recent.mean()) if not recent.empty else 0.0
    historical_std = float(recent.std(ddof=1)) if len(recent) > 1 else 0.0
    forecast_average = float(forecast["Predicted Demand"].mean())

    if historical_std <= 1e-8:
        z_score = 0.0
        anomaly = False
    else:
        z_score = abs(forecast_average - historical_average) / historical_std
        anomaly = z_score > 2.0

    return anomaly, historical_average, historical_std, z_score


def row_highlight(row):
    inventory_status = row.get("Inventory Status", "")
    expiry_status = row.get("Expiry Status", "")

    if expiry_status == "Expired" or inventory_status == "Critical":
        return ["background-color:#FDECEC;color:#172B4D;"] * len(row)
    if inventory_status == "Low Stock":
        return ["background-color:#FFF1E6;color:#172B4D;"] * len(row)
    if expiry_status in {"Critical", "Nearing Expiry"}:
        return ["background-color:#FFF8D8;color:#172B4D;"] * len(row)
    if expiry_status == "Fresh":
        return ["background-color:#EFFAF4;color:#172B4D;"] * len(row)
    return ["background-color:#FFFFFF;color:#172B4D;"] * len(row)


# ============================================================
# UI HELPERS
# ============================================================

def kpi(title, value, note, theme):
    st.markdown(
        f"""
        <div class="card kpi-{theme}">
            <div class="kpi-label">{title}</div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def date_pair(value):
    if isinstance(value, (tuple, list)):
        if len(value) == 2:
            return value[0], value[1]
        if len(value) == 1:
            return value[0], value[0]
    return value, value


# v6 card color strategy: neutral by default; warning colors only for real issues



# ============================================================
# LOAD REAL PROJECT DATA
# ============================================================

try:
    demand = load_demand_data()
    inventory = load_inventory_data()
    ingredient_map = load_ingredient_mapping()
except Exception as exc:
    st.error(f"Unable to load dashboard data: {exc}")
    st.stop()


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div class="hero">
        <div class="hero-title">🍴 QSR Demand Forecast &amp; Inventory Dashboard</div>
        <div class="hero-subtitle">
            Restaurant-wise and Menu Item-wise Forecast, Validation,
            Inventory and Ingredient Expiry Insights
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)




# ============================================================
# DYNAMIC FILTER OPTIONS FROM REAL DATASET
# ============================================================

restaurants = (
    demand[["restaurant_id", "restaurant_name"]]
    .drop_duplicates()
    .sort_values("restaurant_id")
)

menus = (
    demand[["menu_item_id", "menu_item_name"]]
    .drop_duplicates()
    .sort_values("menu_item_id")
)

restaurant_options = {
    "All Restaurants": "ALL",
    **{
        f"{row.restaurant_id} - {row.restaurant_name}": row.restaurant_id
        for row in restaurants.itertuples()
    },
}

menu_options = {
    "All Menu Items": "ALL",
    **{
        f"{row.menu_item_id} - {row.menu_item_name}": row.menu_item_id
        for row in menus.itertuples()
    },
}

# Forecasts should start after the last historical date.
forecast_start_default = demand["date"].max().date() + timedelta(days=1)
forecast_end_default = forecast_start_default + timedelta(days=6)

c1, c2, c3, c4 = st.columns([1.2, 1.35, 1.35, .75])

with c1:
    chosen_dates = st.date_input(
        "Select Date Range",
        value=(forecast_start_default, forecast_end_default),
    )

with c2:
    restaurant_labels = list(restaurant_options.keys())

    default_restaurant_index = next(
        (
            i for i, label in enumerate(restaurant_labels)
            if restaurant_options[label] == "R01"
        ),
        0
    )

    restaurant_label = st.selectbox(
        "Select Restaurant",
        restaurant_labels,
        index=default_restaurant_index,
    )


with c3:
    menu_labels = list(menu_options.keys())

    default_menu_index = next(
        (
            i for i, label in enumerate(menu_labels)
            if menu_options[label] == "M01"
        ),
        0
    )

    menu_label = st.selectbox(
        "Select Menu Item",
        menu_labels,
        index=default_menu_index,
    )

with c4:
    st.markdown(
        '<div style="font-size:.84rem;font-weight:750;'
        'margin-bottom:6px;visibility:hidden;">Action</div>',
        unsafe_allow_html=True,
    )

    st.button(
        "🔎 View Forecast",
        type="primary",
        use_container_width=True
    )

start_date, end_date = date_pair(chosen_dates)
restaurant_id = restaurant_options[restaurant_label]
menu_item_id = menu_options[menu_label]

if restaurant_id == "ALL":
    restaurant_name = "All Restaurants"
else:
    restaurant_name = restaurants.loc[
        restaurants["restaurant_id"] == restaurant_id, "restaurant_name"
    ].iloc[0]

if menu_item_id == "ALL":
    menu_item_name = "All Menu Items"
else:
    menu_item_name = menus.loc[
        menus["menu_item_id"] == menu_item_id, "menu_item_name"
    ].iloc[0]


# ============================================================
# FORECAST
# ============================================================

selected_demand = demand.copy()
if restaurant_id != "ALL":
    selected_demand = selected_demand[
        selected_demand["restaurant_id"] == restaurant_id
    ]
if menu_item_id != "ALL":
    selected_demand = selected_demand[
        selected_demand["menu_item_id"] == menu_item_id
    ]

if selected_demand.empty:
    st.error("No demand history exists for the selected filters.")
    st.stop()

selected_pairs = (
    selected_demand[["restaurant_id", "menu_item_id"]]
    .drop_duplicates()
    .sort_values(["restaurant_id", "menu_item_id"])
)

signature = (
    len(demand),
    str(demand["date"].min()),
    str(demand["date"].max()),
)

forecast_parts = []
metric_rows = []

try:
    with st.spinner("Preparing forecast..."):
        for pair in selected_pairs.itertuples(index=False):
            rid = str(pair.restaurant_id)
            mid = str(pair.menu_item_id)
            pair_history = demand[
                (demand["restaurant_id"] == rid)
                & (demand["menu_item_id"] == mid)
            ].copy()

            model, pair_metrics = train_model(rid, mid, signature)
            pair_forecast = make_forecast(
                model, pair_history, start_date, end_date
            )
            pair_forecast["restaurant_id"] = rid
            pair_forecast["menu_item_id"] = mid
            forecast_parts.append(pair_forecast)
            metric_rows.append(pair_metrics)
except Exception as exc:
    st.error(f"Forecasting failed: {exc}")
    st.stop()

forecast_detail = pd.concat(forecast_parts, ignore_index=True)
forecast = (
    forecast_detail.groupby(["Date", "Day"], as_index=False)["Predicted Demand"]
    .sum()
    .sort_values("Date")
)

history = selected_demand.copy()
total_forecast = float(forecast["Predicted Demand"].sum())
metrics = {
    key: float(np.mean([m[key] for m in metric_rows]))
    for key in ["MAE", "RMSE", "MAPE", "R2"]
}
validation_pass = metrics["MAPE"] < 20


# ============================================================
# SELECTED MENU ITEM INVENTORY
# ============================================================

item_inventory = selected_item_inventory(
    inventory, ingredient_map, restaurant_id, menu_item_id
)

analysis_date = inventory_analysis_date(inventory)
item_inventory = add_expiry_status(item_inventory, analysis_date)
item_inventory = add_inventory_monitoring_fields(item_inventory)

# Overall stock status is based on the new operational threshold logic.
if not item_inventory.empty and (
    item_inventory["inventory_status"] == "Critical"
).any():
    stock_status, stock_theme = "At Risk", "red"
elif not item_inventory.empty and (
    item_inventory["inventory_status"] == "Low Stock"
).any():
    stock_status, stock_theme = "Low Stock", "orange"
else:
    stock_status, stock_theme = "Sufficient", "green"

counts = expiry_counts(item_inventory)

anomaly_detected, historical_average, historical_std, demand_z_score = (
    detect_abnormal_demand(history, forecast)
)

# Current Stock is ingredient inventory, so show ingredient count + summed
# stock separately instead of pretending it is directly the same unit as
# menu-item demand.
total_ingredient_stock = (
    float(item_inventory["current_stock"].sum())
    if not item_inventory.empty
    else 0.0
)



# ============================================================
# TOP CARDS
# ============================================================

info_col, demand_col, stock_col, status_col = st.columns([2.1, 1.15, 1.05, 1.15])

with info_col:
    st.markdown(
        f"""
        <div class="card info-card">
            <div class="section-heading">🍔 Restaurant &amp; Menu Item</div>
            <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px 24px;">
                <div><div class="id-label">Restaurant ID</div>
                <div class="id-value">{restaurant_id}</div></div>
                <div><div class="id-label">Restaurant Name</div>
                <div class="id-value">{restaurant_name}</div></div>
                <div><div class="id-label">Menu Item ID</div>
                <div class="id-value">{menu_item_id}</div></div>
                <div><div class="id-label">Menu Item Name</div>
                <div class="id-value">{menu_item_name}</div></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with demand_col:
    kpi(
        "📊 Total Predicted Demand",
        f"{total_forecast:.2f}",
        f"menu-item units for {len(forecast)} day(s)",
        "blue",
    )

with stock_col:
    kpi(
        "📦 Current Stock",
        f"{total_ingredient_stock:.1f}",
        f"across {len(item_inventory)} mapped ingredient(s)",
        "blue",
    )

with status_col:
    kpi(
        "⚠ Stock Status",
        stock_status,
        "based on ingredient inventory status",
        stock_theme,
    )


# ============================================================
# EXPIRY KPI SUMMARY
# ============================================================

st.markdown(
    '<div class="section-heading">⏳ Ingredient Expiry Alert Summary</div>',
    unsafe_allow_html=True,
)

e1, e2, e3, e4 = st.columns(4)
with e1:
    expired_theme = "red" if counts["Expired"] > 0 else "blue"
    kpi("Expired Ingredients", counts["Expired"], "Displayed as 0 days left", expired_theme)
with e2:
    critical_theme = "red" if counts["Critical"] > 0 else "blue"
    kpi("Critical Ingredients", counts["Critical"], "1 to 3 days remaining", critical_theme)
with e3:
    nearing_theme = "orange" if counts["Nearing Expiry"] > 0 else "blue"
    kpi("Nearing Expiry", counts["Nearing Expiry"], "4 to 7 days remaining", nearing_theme)
with e4:
    kpi("Fresh Ingredients", counts["Fresh"], "More than 7 days remaining", "blue")


# ============================================================
# FORECAST TREND + EXPIRY DISTRIBUTION
# ============================================================

trend_col, expiry_col = st.columns([1.35, 1])

with trend_col:
    st.markdown(
        '<div class="section-heading">📈 Forecast Trend</div>',
        unsafe_allow_html=True,
    )

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=forecast["Date"],
            y=forecast["Predicted Demand"],
            mode="lines+markers+text",
            text=[f"{v:.1f}" for v in forecast["Predicted Demand"]],
            textposition="top center",
            line=dict(color="#0A4D9C", width=3, shape="spline"),
            marker=dict(size=8, color="#0A4D9C"),
            fill="tozeroy",
            fillcolor="rgba(10,77,156,0.06)",
            hovertemplate="%{x|%Y-%m-%d}<br>%{y:.2f} units<extra></extra>",
        )
    )
    fig.update_layout(
        height=235,
        margin=dict(l=6, r=6, t=4, b=4),
        plot_bgcolor="white",
        paper_bgcolor="white",
        xaxis_title="Date",
        yaxis_title="Predicted Demand",
        showlegend=False,
        font=dict(color="#172B4D"),
    )
    # Give edge points enough space for their labels
    x_min = forecast["Date"].min() - pd.Timedelta(days=0.5)
    x_max = forecast["Date"].max() + pd.Timedelta(days=0.5)

    y_max = forecast["Predicted Demand"].max()

    fig.update_xaxes(
        range=[x_min, x_max],
        gridcolor="#EAF4FF",
        linecolor="#D6E4F0"
    )

    fig.update_yaxes(
        range=[0, y_max * 1.15],
        gridcolor="#EAF4FF",
        zeroline=False
    )
    st.plotly_chart(
        fig,
        use_container_width=True,
        config={"displayModeBar": False},
    )

with expiry_col:
    st.markdown(
        '<div class="section-heading">🍩 Ingredient Expiry Distribution</div>',
        unsafe_allow_html=True,
    )

    distribution = pd.DataFrame({
        "Status": ["Fresh", "Nearing Expiry", "Critical", "Expired"],
        "Count": [
            counts["Fresh"],
            counts["Nearing Expiry"],
            counts["Critical"],
            counts["Expired"],
        ],
    })

    donut = px.pie(
        distribution,
        names="Status",
        values="Count",
        hole=.58,
        color="Status",
        color_discrete_map={
            "Fresh": "#2FA866",
            "Nearing Expiry": "#F4D35E",
            "Critical": "#E66B2E",
            "Expired": "#A82032",
        },
    )
    donut.update_traces(textposition="inside", textinfo="percent+label")
    donut.update_layout(
        height=235,
        margin=dict(l=2, r=2, t=2, b=2),
        font=dict(color="#172B4D"),
        legend_title_text="",
    )
    st.plotly_chart(
        donut,
        use_container_width=True,
        config={"displayModeBar": False},
    )



# ============================================================
# DAILY FORECAST + DEMAND HEALTH + RECOMMENDATIONS
# ============================================================

daily_col, health_recommendation_col = st.columns([1.05, 1.35])

with daily_col:
    st.markdown(
        '<div class="section-heading">📊 Daily Forecast</div>',
        unsafe_allow_html=True,
    )

    display = forecast.copy()
    display["Date"] = display["Date"].dt.strftime("%Y-%m-%d")
    display["Predicted Demand"] = display["Predicted Demand"].round(2)
    total_row = pd.DataFrame([{
        "Date": "Total",
        "Day": "-",
        "Predicted Demand": round(total_forecast, 2),
    }])

    st.dataframe(
        pd.concat([display, total_row], ignore_index=True),
        use_container_width=True,
        hide_index=True,
    )



with health_recommendation_col:

    st.markdown(
        '<div class="section-heading">❤️ Demand Health Status</div>',
        unsafe_allow_html=True,
    )

    if anomaly_detected:
        st.markdown(
            """
            <div class="health-fail">
                <div style="font-size:1.7rem;font-weight:850;">FAIL</div>
                <b>Abnormal Demand Pattern Detected</b><br>
                Recent forecast shows unusual demand behaviour.<br>
                <b>Recommended Action:</b> Review historical demand,
                promotional campaigns, and external influencing factors.
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <div class="health-pass">
                <div style="font-size:1.7rem;font-weight:850;">PASS</div>
                <b>Demand Pattern Normal</b><br>
                Historical demand trend is consistent.
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        '<div class="section-heading" style="margin-top:10px;margin-bottom:9px;">💡 Insights &amp; Recommendations</div>',
        unsafe_allow_html=True,
    )

    critical_stock_count = int(
        (item_inventory["inventory_status"] == "Critical").sum()
    ) if not item_inventory.empty else 0

    low_stock_count = int(
        (item_inventory["inventory_status"] == "Low Stock").sum()
    ) if not item_inventory.empty else 0

    actions = []

    if critical_stock_count:
        actions.append(
            f"{critical_stock_count} ingredient(s) require immediate procurement."
        )

    if low_stock_count:
        actions.append(
            f"{low_stock_count} ingredient(s) are below the minimum threshold and should be reordered."
        )

    if counts["Expired"]:
        actions.append(
            f"Discard {counts['Expired']} expired ingredient(s) immediately."
        )

    expiring_count = counts["Critical"] + counts["Nearing Expiry"]
    if expiring_count:
        actions.append(
            f"Use {expiring_count} ingredient(s) expiring within 7 days first "
            "using First In, First Out rotation."
        )

    if anomaly_detected:
        actions.append(
            "Review the abnormal demand forecast before placing inventory orders."
        )
    else:
        actions.append(
            "Demand pattern is normal relative to recent historical sales."
        )

    actions.append(
        "Consider transferring excess usable ingredients to nearby restaurants "
        "when operationally appropriate."
    )

    st.markdown(
        '<ul class="insight-list">' + ''.join(f'<li>{action}</li>' for action in actions) + '</ul>' +
        '<style>.insight-list{font-size:.86rem;margin:.2rem 0 .3rem 1.05rem;padding:0;line-height:1.35}.insight-list li{margin:.28rem 0;padding-left:.15rem}</style>',
        unsafe_allow_html=True,
    )


# ============================================================
# INGREDIENT EXPIRY MONITORING TABLE
# ============================================================

st.markdown("---")
st.markdown(
    '<div class="section-heading">🧾 Inventory &amp; Expiry Monitoring</div>',
    unsafe_allow_html=True,
)

search_col, filter_col = st.columns([1.4, 1])

with search_col:
    search = st.text_input(
        "Search Inventory",
        placeholder="Search ingredient, identifier or supplier...",
    )

with filter_col:
    selected_statuses = st.multiselect(
        "Filter Expiry Status",
        ["Expired", "Critical", "Nearing Expiry", "Fresh"],
        default=["Expired", "Critical", "Nearing Expiry", "Fresh"],
    )

filtered = item_inventory.copy()

if search and not filtered.empty:
    q = search.lower().strip()
    supplier = (
        filtered["supplier_name"].astype(str)
        if "supplier_name" in filtered.columns
        else pd.Series("", index=filtered.index)
    )
    filtered = filtered[
        filtered["ingredient_name"].astype(str).str.lower().str.contains(q, regex=False, na=False)
        | filtered["ingredient_id"].astype(str).str.lower().str.contains(q, regex=False, na=False)
        | supplier.str.lower().str.contains(q, regex=False, na=False)
    ]

if selected_statuses and not filtered.empty:
    filtered = filtered[filtered["expiry_status"].isin(selected_statuses)]

if not filtered.empty:
    filtered = filtered.sort_values(
        ["actual_days_left", "inventory_status", "ingredient_name"],
        ascending=[True, True, True],
    )

    inventory_table = filtered[
        [
            "restaurant_id",
            "menu_item_id",
            "ingredient_id",
            "ingredient_name",
            "current_stock",
            "inventory_status",
            "expiry_date",
            "days_left",
            "expiry_status",
            "recommended_action",
        ]
    ].copy()

    inventory_table["expiry_date"] = (
        inventory_table["expiry_date"].dt.strftime("%Y-%m-%d")
    )
    inventory_table["days_left"] = inventory_table["days_left"].astype(int)

    inventory_table = inventory_table.rename(
        columns={
            "restaurant_id": "Restaurant ID",
            "menu_item_id": "Menu Item ID",
            "ingredient_id": "Ingredient ID",
            "ingredient_name": "Ingredient Name",
            "current_stock": "Current Quantity",
            "inventory_status": "Inventory Status",
            "expiry_date": "Expiry Date",
            "days_left": "Days Remaining",
            "expiry_status": "Expiry Status",
            "recommended_action": "Recommended Action",
        }
    )

    styled_inventory = (
        inventory_table.style
        .set_properties(
            **{
                "background-color": "#FFFFFF",
                "color": "#172B4D",
                "border-color": "#C9DDF2",
            }
        )
        .apply(row_highlight, axis=1)
        .set_table_styles(
            [
                {
                    "selector": "table",
                    "props": [
                        ("background-color", "#FFFFFF"),
                        ("border-color", "#AFCFEF"),
                    ],
                },
                {
                    "selector": "thead th",
                    "props": [
                        ("background-color", "#F3F6FA"),
                        ("color", "#111827"),
                        ("font-weight", "700"),
                        ("border-color", "#AFCFEF"),
                    ],
                },
                {
                    "selector": "tbody td",
                    "props": [
                        ("color", "#172B4D"),
                        ("border-color", "#C9DDF2"),
                    ],
                },
            ]
        )
    )

    st.dataframe(
        styled_inventory,
        use_container_width=True,
        hide_index=True,
        height=315,
    )
else:
    st.info("No ingredients match the current filters.")




# ============================================================
# DOWNLOADS
# ============================================================

st.markdown("---")
d1, d2, _ = st.columns([0.62, 0.78, 2.6])

forecast_export = forecast.copy()
forecast_export["Date"] = forecast_export["Date"].dt.strftime("%Y-%m-%d")

with d1:
    st.download_button(
        "Download Forecast CSV",
        forecast_export.to_csv(index=False).encode("utf-8"),
        file_name=f"forecast_{restaurant_id}_{menu_item_id}.csv",
        mime="text/csv",
        use_container_width=True,
    )

with d2:
    export_cols = [
        c for c in [
            "restaurant_id", "menu_item_id", "ingredient_id", "ingredient_name",
            "current_stock", "unit", "expiry_date",
            "days_left", "expiry_status", "minimum_threshold",
            "inventory_status", "recommended_action", "stock_status",
            "supplier_name"
        ]
        if c in item_inventory.columns
    ]
    inventory_export = item_inventory[export_cols].copy()
    if "expiry_date" in inventory_export.columns:
        inventory_export["expiry_date"] = (
            inventory_export["expiry_date"].dt.strftime("%Y-%m-%d")
        )

    st.download_button(
        "Download Ingredient Report CSV",
        inventory_export.to_csv(index=False).encode("utf-8"),
        file_name=f"ingredient_report_{restaurant_id}_{menu_item_id}.csv",
        mime="text/csv",
        use_container_width=True,
    )



