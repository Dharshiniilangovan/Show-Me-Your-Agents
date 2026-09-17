"""
Tests for the Context & Seasonality Agent.

Run from the repository root:

    python -m pytest tests/test_context_seasonality.py -v

Repository structure expected:

Show-Me-Your-Agents/
├── agents/
│   └── context_seasonality/
│       ├── __init__.py
│       ├── data_loader.py
│       ├── feature_engineering.py
│       ├── signals.py
│       └── ...
└── tests/
    └── test_context_seasonality.py
"""

import pandas as pd

from agents.context_seasonality.data_loader import REQUIRED

from agents.context_seasonality.feature_engineering import (
    add_context_features,
)

from agents.context_seasonality.signals import (
    build_sku_signals,
)

from agents.context_seasonality.agent import (
    context_seasonality_agent,
)

from agents.context_seasonality.data_loader import (
    load_unified_data,
)

# ============================================================
# 1. INPUT CONTRACT
# ============================================================

def test_required_contract():
    """
    Verify that important columns required by the
    Context & Seasonality Agent are present in the
    data-loader contract.
    """

    assert "quantity" in REQUIRED
    assert "is_promotion" in REQUIRED
    assert "avg_temp_f" in REQUIRED


# ============================================================
# 2. FEATURE ENGINEERING
# ============================================================

def test_features():
    """
    Verify that calendar, weather, context and
    interaction features are generated correctly.

    Optional descriptive context columns are deliberately
    omitted to verify that the agent creates safe defaults.
    """

    df = pd.DataFrame(
        {
            "date": pd.to_datetime(
                ["2026-01-01"]
            ),
            "day_of_week_num": [3],
            "month": [1],
            "avg_temp_f": [50.0],
            "precip_inches": [0.0],
            "is_holiday": [1],
            "is_special_event": [0],
            "is_promotion": [0],
        }
    )

    x = add_context_features(df)

    # --------------------------------------------------------
    # Calendar / cyclical features
    # --------------------------------------------------------

    assert "dow_sin" in x.columns
    assert "dow_cos" in x.columns

    assert "month_sin" in x.columns
    assert "month_cos" in x.columns

    assert "doy_sin" in x.columns
    assert "doy_cos" in x.columns

    assert "week_of_year" in x.columns
    assert "quarter" in x.columns
    assert "day_of_month" in x.columns

    assert "is_month_start" in x.columns
    assert "is_month_end" in x.columns

    # --------------------------------------------------------
    # Context features
    # --------------------------------------------------------

    assert "is_weekend" in x.columns
    assert "context_event_count" in x.columns
    assert "is_context_day" in x.columns

    # --------------------------------------------------------
    # Optional columns should automatically be created
    # --------------------------------------------------------

    assert "holiday_name" in x.columns
    assert "special_event_name" in x.columns
    assert "precip_type" in x.columns

    assert x.loc[0, "holiday_name"] == "None"
    assert x.loc[0, "special_event_name"] == "None"
    assert x.loc[0, "precip_type"] == "None"

    # --------------------------------------------------------
    # Weather
    # --------------------------------------------------------

    assert "has_precipitation" in x.columns
    assert "temp_band" in x.columns
    assert "temp_c" in x.columns

    assert x.loc[0, "has_precipitation"] == 0

    # --------------------------------------------------------
    # Interactions
    # --------------------------------------------------------

    assert "weekend_promotion" in x.columns
    assert "holiday_promotion" in x.columns
    assert "event_promotion" in x.columns
    assert "precipitation_weekend" in x.columns

    # Jan 1 2026 is Thursday.
    assert x.loc[0, "is_weekend"] == 0

    # Only the holiday context is active.
    assert x.loc[0, "context_event_count"] == 1
    assert x.loc[0, "is_context_day"] == 1


# ============================================================
# 3. NAMED CONTEXT PRESERVATION
# ============================================================

def test_named_context_is_preserved():
    """
    Verify that real holiday, event and precipitation
    descriptions are preserved rather than replaced
    with the default 'None' value.
    """

    df = pd.DataFrame(
        {
            "date": pd.to_datetime(
                ["2026-07-04"]
            ),

            "day_of_week_num": [5],
            "month": [7],

            "avg_temp_f": [85.0],
            "precip_inches": [0.2],

            "is_holiday": [1],
            "holiday_name": [
                "Independence Day"
            ],

            "is_special_event": [1],
            "special_event_name": [
                "City Festival"
            ],

            "is_promotion": [1],

            "precip_type": [
                "rain"
            ],
        }
    )

    x = add_context_features(df)

    # --------------------------------------------------------
    # Named context should survive feature engineering
    # --------------------------------------------------------

    assert (
        x.loc[0, "holiday_name"]
        == "Independence Day"
    )

    assert (
        x.loc[0, "special_event_name"]
        == "City Festival"
    )

    assert (
        x.loc[0, "precip_type"]
        == "rain"
    )

    # --------------------------------------------------------
    # Three context conditions are active:
    # holiday + event + promotion
    # --------------------------------------------------------

    assert (
        x.loc[0, "context_event_count"]
        == 3
    )

    assert (
        x.loc[0, "is_context_day"]
        == 1
    )

    # --------------------------------------------------------
    # Interaction features
    # --------------------------------------------------------

    assert (
        x.loc[0, "weekend_promotion"]
        == 1
    )

    assert (
        x.loc[0, "holiday_promotion"]
        == 1
    )

    assert (
        x.loc[0, "event_promotion"]
        == 1
    )

    assert (
        x.loc[
            0,
            "precipitation_weekend"
        ]
        == 1
    )


# ============================================================
# 4. HIERARCHICAL CONTEXT SIGNALS
# ============================================================

def test_hierarchical_signal_fallback_and_metadata():
    """
    Verify that restaurant/SKU context signals contain
    uplift, support counts, signal source and confidence
    metadata.

    Synthetic data:
        baseline quantity = 10
        context quantity  = 12

    Therefore expected historical uplift is 20%.
    """

    rows = []

    for restaurant in [
        "R1",
        "R2",
    ]:

        # ----------------------------------------------------
        # Baseline observations
        # ----------------------------------------------------

        for _ in range(60):

            rows.append(
                {
                    "restaurant_id":
                        restaurant,

                    "menu_item_id":
                        "M1",

                    "category":
                        "Burger",

                    "quantity":
                        10.0,

                    "is_weekend":
                        0,

                    "is_holiday":
                        0,

                    "is_special_event":
                        0,

                    "is_promotion":
                        0,

                    "has_precipitation":
                        0,
                }
            )

        # ----------------------------------------------------
        # Context-active observations
        # ----------------------------------------------------

        for _ in range(15):

            rows.append(
                {
                    "restaurant_id":
                        restaurant,

                    "menu_item_id":
                        "M1",

                    "category":
                        "Burger",

                    "quantity":
                        12.0,

                    "is_weekend":
                        1,

                    "is_holiday":
                        1,

                    "is_special_event":
                        1,

                    "is_promotion":
                        1,

                    "has_precipitation":
                        1,
                }
            )

    df = pd.DataFrame(rows)

    result = build_sku_signals(df)

    # --------------------------------------------------------
    # Basic output
    # --------------------------------------------------------

    assert not result.empty

    assert len(result) == 2

    assert "restaurant_id" in result.columns
    assert "menu_item_id" in result.columns
    assert "category" in result.columns

    assert "baseline_demand" in result.columns
    assert (
        "baseline_observation_count"
        in result.columns
    )

    # --------------------------------------------------------
    # Promotion metadata
    # --------------------------------------------------------

    assert (
        "promotion_uplift_pct"
        in result.columns
    )

    assert (
        "promotion_active_count"
        in result.columns
    )

    assert (
        "promotion_baseline_count"
        in result.columns
    )

    assert (
        "promotion_signal_level"
        in result.columns
    )

    assert (
        "promotion_confidence"
        in result.columns
    )

    # --------------------------------------------------------
    # No missing promotion signal should remain
    # --------------------------------------------------------

    assert (
        result[
            "promotion_uplift_pct"
        ]
        .notna()
        .all()
    )

    # --------------------------------------------------------
    # Synthetic uplift should be approximately +20%
    #
    # (12 / 10 - 1) * 100 = 20
    # --------------------------------------------------------

    assert (
        result[
            "promotion_uplift_pct"
        ]
        .round(6)
        .eq(20.0)
        .all()
    )

    # --------------------------------------------------------
    # Restaurant/SKU has sufficient evidence:
    #
    # active   = 15 >= MIN_ACTIVE (10)
    # baseline = 60 >= MIN_BASELINE (30)
    # --------------------------------------------------------

    assert (
        result[
            "promotion_signal_level"
        ]
        .eq("restaurant_sku")
        .all()
    )

    # 15 active observations -> LOW confidence
    assert (
        result[
            "promotion_confidence"
        ]
        .eq("low")
        .all()
    )


# ============================================================
# 5. ALL CONTEXT SIGNAL FAMILIES
# ============================================================

def test_all_context_signal_metadata_exists():
    """
    Verify that every context family exposes the same
    standard metadata contract expected by downstream
    agents.
    """

    rows = []

    for _ in range(40):

        rows.append(
            {
                "restaurant_id": "R1",
                "menu_item_id": "M1",
                "category": "Burger",
                "quantity": 10.0,
                "is_weekend": 0,
                "is_holiday": 0,
                "is_special_event": 0,
                "is_promotion": 0,
                "has_precipitation": 0,
            }
        )

    for _ in range(10):

        rows.append(
            {
                "restaurant_id": "R1",
                "menu_item_id": "M1",
                "category": "Burger",
                "quantity": 12.0,
                "is_weekend": 1,
                "is_holiday": 1,
                "is_special_event": 1,
                "is_promotion": 1,
                "has_precipitation": 1,
            }
        )

    df = pd.DataFrame(rows)

    result = build_sku_signals(df)

    contexts = [
        "weekend",
        "holiday",
        "event",
        "promotion",
        "precipitation",
    ]

    for context in contexts:

        assert (
            f"{context}_uplift_pct"
            in result.columns
        )

        assert (
            f"{context}_active_count"
            in result.columns
        )

        assert (
            f"{context}_baseline_count"
            in result.columns
        )

        assert (
            f"{context}_signal_level"
            in result.columns
        )

        assert (
            f"{context}_confidence"
            in result.columns
        )


# ============================================================
# 6. INVALID INPUT CONTRACT
# ============================================================

def test_missing_required_feature_column_raises_error():
    """
    Verify that feature engineering fails clearly when
    a genuinely required column is missing.
    """

    df = pd.DataFrame(
        {
            "date": pd.to_datetime(
                ["2026-01-01"]
            ),
            "day_of_week_num": [3],
            "month": [1],

            # avg_temp_f intentionally missing

            "precip_inches": [0.0],
            "is_holiday": [0],
            "is_special_event": [0],
            "is_promotion": [0],
        }
    )

    try:

        add_context_features(df)

        raised = False

    except ValueError as exc:

        raised = True

        assert (
            "avg_temp_f"
            in str(exc)
        )

    assert raised

def test_context_agent_requires_data_analyst_path():

    state = {
        "request": {
            "restaurant_scope": "single",
            "restaurant_id": "R01",
            "menu_item_id": "M01",
        },

        "required_capabilities": [
            "promotion_analysis"
        ],

        "agent_results": {},

        "shared_data": {},
    }

    try:

        context_seasonality_agent(
            state
        )

        raised = False

    except ValueError as exc:

        raised = True

        assert (
            "unified_demand_path"
            in str(exc)
        )

    assert raised

def test_load_unified_data_from_shared_path(
    tmp_path
):

    path = (
        tmp_path
        / "unified_demand.csv"
    )

    df = pd.DataFrame(
        {
            "date": ["2026-01-01"],
            "restaurant_id": ["R01"],
            "menu_item_id": ["M01"],
            "category": ["Burgers"],
            "quantity": [20],
            "day_of_week_num": [3],
            "month": [1],
            "avg_temp_f": [50.0],
            "precip_inches": [0.0],
            "is_holiday": [0],
            "is_special_event": [0],
            "is_promotion": [0],
        }
    )

    df.to_csv(
        path,
        index=False
    )

    loaded = load_unified_data(
        str(path)
    )

    assert len(loaded) == 1

    assert (
        loaded.loc[
            0,
            "restaurant_id"
        ]
        == "R01"
    )