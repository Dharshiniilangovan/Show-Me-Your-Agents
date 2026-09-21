"""
Demand Forecasting Agent

Uses outputs from Phase-1 agents:

1. Data Analyst
   state["shared_context"]["data"]["unified_demand_path"]

2. Context / Seasonality
   state["shared_context"]["data"]["context_features_path"]

3. Customer Pattern
   state["shared_context"]["customer_pattern"]

Workflow:
1. Load unified demand data.
2. Merge Context / Seasonality features.
3. Add usable Customer Pattern features.
4. Filter requested restaurant/item.
5. Create calendar, lag and rolling features.
6. Backtest on the last N historical days.
7. Print validation wMAPE.
8. Train final LightGBM model on all historical data.
9. Generate recursive future demand forecasts.
10. Save detailed demand_forecast.csv.
11. Read demand_forecast.csv.
12. Filter/aggregate according to the user's request.
13. Return requested demand values.
"""

from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd


class DemandForecastingAgent:

    # ============================================================
    # 1. INITIALIZE MODEL
    # ============================================================

    def __init__(self, model_params=None):

        self.model_params = model_params or {
            "n_estimators": 1500,
            "learning_rate": 0.02,
            "num_leaves": 80,
            "max_depth": 12,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42,
            "verbosity": -1,
        }

    # ============================================================
    # 2. GET DATA ANALYST OUTPUT PATH
    # ============================================================

    @staticmethod
    def _get_unified_path(state):

        shared_context = state.get(
            "shared_context",
            {}
        )

        shared_data = shared_context.get(
            "data",
            {}
        )

        unified_path = shared_data.get(
            "unified_demand_path"
        )

        if not unified_path:

            raise ValueError(
                "unified_demand_path is missing from "
                "state['shared_context']['data']. "
                "Run the Data Analyst Agent first."
            )

        unified_path = Path(
            unified_path
        )

        if not unified_path.exists():

            raise FileNotFoundError(
                f"Unified demand file not found: "
                f"{unified_path}"
            )

        return unified_path

    # ============================================================
    # 3. GET CONTEXT / SEASONALITY OUTPUT PATH
    # ============================================================

    @staticmethod
    def _get_context_features_path(state):

        shared_context = state.get(
            "shared_context",
            {}
        )

        shared_data = shared_context.get(
            "data",
            {}
        )

        context_path = shared_data.get(
            "context_features_path"
        )

        if not context_path:
            return None

        context_path = Path(
            context_path
        )

        if not context_path.exists():
            return None

        return context_path

    # ============================================================
    # 4. GET CUSTOMER PATTERN OUTPUT
    # ============================================================

    @staticmethod
    def _get_customer_pattern(state):

        shared_context = state.get(
            "shared_context",
            {}
        )

        return shared_context.get(
            "customer_pattern",
            {}
        )

    # ============================================================
    # 5. MERGE CONTEXT / SEASONALITY FEATURES
    # ============================================================

    @staticmethod
    def _merge_context_features(
        demand_df,
        context_path
    ):

        if context_path is None:
            return demand_df

        context_df = pd.read_csv(
            context_path
        )

        if context_df.empty:
            return demand_df

        demand_df = demand_df.copy()

        # --------------------------------------------------------
        # Date conversion
        # --------------------------------------------------------

        if "date" in demand_df.columns:

            demand_df["date"] = pd.to_datetime(
                demand_df["date"],
                errors="coerce"
            )

        if "date" in context_df.columns:

            context_df["date"] = pd.to_datetime(
                context_df["date"],
                errors="coerce"
            )

        # --------------------------------------------------------
        # Determine common merge keys
        # --------------------------------------------------------

        possible_keys = [
            "date",
            "restaurant_id",
            "menu_item_id"
        ]

        merge_keys = [
            key
            for key in possible_keys
            if (
                key in demand_df.columns
                and key in context_df.columns
            )
        ]

        if not merge_keys:
            return demand_df

        # --------------------------------------------------------
        # Add only NEW context features
        # --------------------------------------------------------

        new_context_columns = [
            column
            for column in context_df.columns
            if (
                column not in demand_df.columns
                and column not in merge_keys
            )
        ]

        if not new_context_columns:
            return demand_df

        context_subset = context_df[
            merge_keys
            + new_context_columns
        ].copy()

        # Avoid many-to-many merge

        context_subset = (
            context_subset
            .drop_duplicates(
                subset=merge_keys
            )
        )

        merged = demand_df.merge(
            context_subset,
            on=merge_keys,
            how="left"
        )

        return merged

    # ============================================================
    # 6. ADD CUSTOMER PATTERN FEATURES
    # ============================================================

    @staticmethod
    def _add_customer_pattern_features(
        data,
        customer_pattern
    ):

        result = data.copy()

        if not isinstance(
            customer_pattern,
            dict
        ):
            return result

        # --------------------------------------------------------
        # Helper
        # --------------------------------------------------------

        def add_numeric_feature(
            output_column,
            value
        ):

            if value is None:
                return

            try:

                result[
                    output_column
                ] = float(value)

            except (
                TypeError,
                ValueError
            ):
                pass

        # --------------------------------------------------------
        # Direct numeric customer-pattern features
        # --------------------------------------------------------

        add_numeric_feature(
            "customer_demand_shift_pct",
            customer_pattern.get(
                "demand_shift_pct"
            )
        )

        add_numeric_feature(
            "customer_purchase_frequency",
            customer_pattern.get(
                "purchase_frequency"
            )
        )

        add_numeric_feature(
            "customer_popularity_score",
            customer_pattern.get(
                "popularity_score"
            )
        )

        add_numeric_feature(
            "customer_relationship_score",
            customer_pattern.get(
                "relationship_score"
            )
        )

        # --------------------------------------------------------
        # Unusual-pattern flag
        # --------------------------------------------------------

        unusual_pattern = (
            customer_pattern.get(
                "unusual_pattern"
            )
        )

        if unusual_pattern is not None:

            result[
                "customer_unusual_pattern"
            ] = int(
                bool(
                    unusual_pattern
                )
            )

        # --------------------------------------------------------
        # Nested numeric metrics
        # --------------------------------------------------------

        metrics = customer_pattern.get(
            "metrics",
            {}
        )

        if isinstance(
            metrics,
            dict
        ):

            for key, value in metrics.items():

                if isinstance(
                    value,
                    (
                        int,
                        float,
                        np.integer,
                        np.floating
                    )
                ):

                    safe_key = (
                        str(key)
                        .strip()
                        .lower()
                        .replace(
                            " ",
                            "_"
                        )
                    )

                    result[
                        f"customer_{safe_key}"
                    ] = float(value)

        return result

    # ============================================================
    # 7. FILTER REQUESTED SCOPE
    # ============================================================

    @staticmethod
    def _filter_scope(
        df,
        request
    ):

        result = df.copy()

        restaurant_scope = (
            request.get(
                "restaurant_scope"
            )
        )

        restaurant_id = (
            request.get(
                "restaurant_id"
            )
        )

        menu_item_id = (
            request.get(
                "menu_item_id"
            )
        )

        # Restaurant

        if (
            restaurant_scope == "single"
            and restaurant_id is not None
        ):

            result = result[
                result[
                    "restaurant_id"
                ].astype(str)
                == str(
                    restaurant_id
                )
            ]

        # Menu item

        if menu_item_id is not None:

            result = result[
                result[
                    "menu_item_id"
                ].astype(str)
                == str(
                    menu_item_id
                )
            ]

        if result.empty:

            raise ValueError(
                "No historical demand data matches "
                "the requested restaurant/menu item."
            )

        return result

    # ============================================================
    # 8. PREPARE BASE DATA
    # ============================================================

    @staticmethod
    def _prepare_base_data(
        df
    ):

        data = df.copy()

        required_columns = [
            "date",
            "restaurant_id",
            "menu_item_id",
            "quantity"
        ]

        missing_columns = [
            column
            for column in required_columns
            if column not in data.columns
        ]

        if missing_columns:

            raise ValueError(
                "Unified demand dataset is missing: "
                + ", ".join(
                    missing_columns
                )
            )

        data["date"] = pd.to_datetime(
            data["date"],
            errors="coerce"
        )

        data["quantity"] = pd.to_numeric(
            data["quantity"],
            errors="coerce"
        )

        data = data.dropna(
            subset=[
                "date",
                "quantity"
            ]
        )

        data["quantity"] = (
            data["quantity"]
            .clip(
                lower=0
            )
        )

        data = (
            data.sort_values(
                [
                    "restaurant_id",
                    "menu_item_id",
                    "date"
                ]
            )
            .reset_index(
                drop=True
            )
        )

        return data

    # ============================================================
    # 9. CREATE CALENDAR FEATURES
    # ============================================================

    @staticmethod
    def _create_calendar_features(
        data
    ):

        data = data.copy()

        data["day"] = (
            data["date"].dt.day
        )

        data["month"] = (
            data["date"].dt.month
        )

        data["year"] = (
            data["date"].dt.year
        )

        data["day_of_week_num"] = (
            data["date"].dt.dayofweek
        )

        data["week_of_year"] = (
            data["date"]
            .dt
            .isocalendar()
            .week
            .astype(int)
        )

        data["quarter"] = (
            data["date"].dt.quarter
        )

        data["is_weekend"] = (
            data[
                "day_of_week_num"
            ]
            .isin(
                [5, 6]
            )
            .astype(int)
        )

        # --------------------------------------------------------
        # Cyclic day-of-week
        # --------------------------------------------------------

        data["dow_sin"] = np.sin(
            2
            * np.pi
            * data[
                "day_of_week_num"
            ]
            / 7
        )

        data["dow_cos"] = np.cos(
            2
            * np.pi
            * data[
                "day_of_week_num"
            ]
            / 7
        )

        # --------------------------------------------------------
        # Cyclic month
        # --------------------------------------------------------

        data["month_sin"] = np.sin(
            2
            * np.pi
            * data["month"]
            / 12
        )

        data["month_cos"] = np.cos(
            2
            * np.pi
            * data["month"]
            / 12
        )

        return data

    # ============================================================
    # 10. CREATE LAG / ROLLING FEATURES
    # ============================================================

    @staticmethod
    def _create_lag_features(
        data
    ):

        data = data.copy()

        group_columns = [
            "restaurant_id",
            "menu_item_id"
        ]

        grouped = (
            data.groupby(
                group_columns,
                sort=False
            )[
                "quantity"
            ]
        )

        # --------------------------------------------------------
        # Lag features
        # --------------------------------------------------------

        for lag in [
            1,
            2,
            3,
            7,
            14,
            21,
            28
        ]:

            data[
                f"model_lag_{lag}"
            ] = grouped.shift(
                lag
            )

        # --------------------------------------------------------
        # Rolling means
        # --------------------------------------------------------

        shifted = grouped.shift(1)

        for window in [
            7,
            14,
            28
        ]:

            data[
                f"model_rolling_mean_{window}"
            ] = (
                shifted.groupby(
                    [
                        data[
                            "restaurant_id"
                        ],
                        data[
                            "menu_item_id"
                        ]
                    ]
                )
                .transform(
                    lambda x:
                        x.rolling(
                            window,
                            min_periods=1
                        ).mean()
                )
            )

        return data

    # ============================================================
    # 11. PREPARE MODEL FEATURES
    # ============================================================

    @staticmethod
    def _prepare_model_features(
        data
    ):

        model_data = data.copy()

        excluded = {

            "date",

            "quantity",

            # Direct target leakage
            "revenue",

            # Descriptive text
            "restaurant_name",
            "menu_item_name",
            "holiday_name",
            "special_event_name",
            "day_of_week",

            # Existing quantity-derived features
            "quantity_lag_1",
            "quantity_lag_7",
            "quantity_lag_14",
            "quantity_lag_28",

            "rolling_mean_7",
            "rolling_mean_14",
            "rolling_mean_28",

            "rolling_std_7",
            "rolling_std_14",
            "rolling_std_28",
        }

        features = [
            column
            for column in model_data.columns
            if column not in excluded
        ]

        encoders = {}

        # --------------------------------------------------------
        # Encode categorical columns
        # --------------------------------------------------------

        for column in features:

            if (
                model_data[
                    column
                ].dtype == "object"

                or

                str(
                    model_data[
                        column
                    ].dtype
                ).startswith(
                    "string"
                )

                or

                str(
                    model_data[
                        column
                    ].dtype
                ) == "category"
            ):

                values = (
                    model_data[
                        column
                    ]
                    .fillna(
                        "unknown"
                    )
                    .astype(str)
                )

                codes, uniques = (
                    pd.factorize(
                        values,
                        sort=True
                    )
                )

                model_data[
                    column
                ] = codes

                encoders[
                    column
                ] = [
                    str(value)
                    for value in uniques
                ]

            else:

                model_data[
                    column
                ] = pd.to_numeric(
                    model_data[
                        column
                    ],
                    errors="coerce"
                )

        model_data[
            features
        ] = (
            model_data[
                features
            ]
            .replace(
                [
                    np.inf,
                    -np.inf
                ],
                np.nan
            )
            .fillna(0)
        )

        return (
            model_data,
            features,
            encoders
        )

    # ============================================================
    # 12. CREATE SAMPLE WEIGHTS
    # ============================================================

    @staticmethod
    def _create_sample_weights(
        train
    ):

        train_y = (
            train[
                "quantity"
            ]
        )

        recency = (
            train["date"]
            - train["date"].min()
        ).dt.days.astype(
            float
        )

        max_recency = max(
            float(
                recency.max()
            ),
            1.0
        )

        recency_weight = (
            recency
            / max_recency
        )

        mean_demand = max(
            float(
                train_y.mean()
            ),
            1e-8
        )

        demand_weight = (
            train_y
            / mean_demand
        )

        weights = (
            demand_weight
            * (
                1.0
                + recency_weight.values
            )
        )

        return weights

    # ============================================================
    # 13. TRAIN MODEL
    # ============================================================

    def _train_model(
        self,
        train,
        features
    ):

        train_X = (
            train[
                features
            ]
        )

        train_y = (
            train[
                "quantity"
            ]
        )

        train_y_log = (
            np.log1p(
                train_y
            )
        )

        weights = (
            self._create_sample_weights(
                train
            )
        )

        model = (
            lgb.LGBMRegressor(
                **self.model_params
            )
        )

        model.fit(
            train_X,
            train_y_log,
            sample_weight=weights
        )

        return model

    # ============================================================
    # 14. CALCULATE wMAPE
    # ============================================================

    @staticmethod
    def _wmape(
        y_true,
        y_pred
    ):

        y_true = np.asarray(
            y_true,
            dtype=float
        )

        y_pred = np.asarray(
            y_pred,
            dtype=float
        )

        denominator = (
            np.abs(
                y_true
            ).sum()
        )

        if denominator == 0:

            return 0.0

        wmape = (
            np.abs(
                y_true - y_pred
            ).sum()
            / denominator
        )

        return float(
            wmape
        )

    # ============================================================
    # 15. HISTORICAL BACKTEST
    # ============================================================

    def _evaluate_model(
        self,
        model_data,
        features,
        forecast_horizon
    ):

        unique_dates = np.sort(
            model_data[
                "date"
            ]
            .dropna()
            .unique()
        )

        # --------------------------------------------------------
        # Need enough history
        # --------------------------------------------------------

        if (
            len(unique_dates)
            <= forecast_horizon
        ):

            return None

        # --------------------------------------------------------
        # Last N historical dates become validation
        # --------------------------------------------------------

        cutoff = pd.Timestamp(
            unique_dates[
                -forecast_horizon
            ]
        )

        train = model_data[
            model_data[
                "date"
            ] < cutoff
        ].copy()

        validation = model_data[
            model_data[
                "date"
            ] >= cutoff
        ].copy()

        if (
            train.empty
            or validation.empty
        ):

            return None

        # --------------------------------------------------------
        # Train temporary validation model
        # --------------------------------------------------------

        validation_model = (
            self._train_model(
                train,
                features
            )
        )

        # --------------------------------------------------------
        # Predict historical validation period
        # --------------------------------------------------------

        validation_predictions = (
            np.expm1(
                validation_model.predict(
                    validation[
                        features
                    ]
                )
            )
        )

        validation_predictions = (
            np.clip(
                validation_predictions,
                0,
                None
            )
        )

        # --------------------------------------------------------
        # Calculate wMAPE
        # --------------------------------------------------------

        wmape = (
            self._wmape(
                validation[
                    "quantity"
                ],
                validation_predictions
            )
        )

        return wmape

    # ============================================================
    # 16. GET LATEST FEATURE VALUE
    # ============================================================

    @staticmethod
    def _latest_value(
        history,
        column,
        default=0
    ):

        if column not in history.columns:

            return default

        values = (
            history[
                column
            ]
            .dropna()
        )

        if values.empty:

            return default

        return (
            values.iloc[-1]
        )

    # ============================================================
    # 17. CREATE ONE FUTURE ROW
    # ============================================================

    def _create_future_row(
        self,
        history,
        restaurant_id,
        menu_item_id,
        future_date,
        features
    ):

        item_history = history[
            (
                history[
                    "restaurant_id"
                ].astype(str)
                == str(
                    restaurant_id
                )
            )
            &
            (
                history[
                    "menu_item_id"
                ].astype(str)
                == str(
                    menu_item_id
                )
            )
        ].sort_values(
            "date"
        )

        if item_history.empty:

            raise ValueError(
                "No historical data available for "
                f"{restaurant_id} / {menu_item_id}"
            )

        latest = (
            item_history.iloc[-1]
        )

        row = {}

        # --------------------------------------------------------
        # Copy latest known/static feature values
        # --------------------------------------------------------

        for column in history.columns:

            if column in [
                "date",
                "quantity"
            ]:

                continue

            row[
                column
            ] = latest.get(
                column,
                0
            )

        # --------------------------------------------------------
        # Future identity
        # --------------------------------------------------------

        row[
            "date"
        ] = pd.Timestamp(
            future_date
        )

        row[
            "restaurant_id"
        ] = restaurant_id

        row[
            "menu_item_id"
        ] = menu_item_id

        # --------------------------------------------------------
        # Future calendar features
        # --------------------------------------------------------

        row[
            "day"
        ] = future_date.day

        row[
            "month"
        ] = future_date.month

        row[
            "year"
        ] = future_date.year

        row[
            "day_of_week_num"
        ] = future_date.dayofweek

        row[
            "week_of_year"
        ] = int(
            future_date
            .isocalendar()
            .week
        )

        row[
            "quarter"
        ] = future_date.quarter

        row[
            "is_weekend"
        ] = int(
            future_date.dayofweek
            in [5, 6]
        )

        row[
            "dow_sin"
        ] = np.sin(
            2
            * np.pi
            * future_date.dayofweek
            / 7
        )

        row[
            "dow_cos"
        ] = np.cos(
            2
            * np.pi
            * future_date.dayofweek
            / 7
        )

        row[
            "month_sin"
        ] = np.sin(
            2
            * np.pi
            * future_date.month
            / 12
        )

        row[
            "month_cos"
        ] = np.cos(
            2
            * np.pi
            * future_date.month
            / 12
        )

        # --------------------------------------------------------
        # Context Agent calendar features
        # --------------------------------------------------------

        day_of_year = (
            future_date.dayofyear
        )

        if "doy_sin" in features:

            row[
                "doy_sin"
            ] = np.sin(
                2
                * np.pi
                * day_of_year
                / 365.25
            )

        if "doy_cos" in features:

            row[
                "doy_cos"
            ] = np.cos(
                2
                * np.pi
                * day_of_year
                / 365.25
            )

        if "day_of_month" in features:

            row[
                "day_of_month"
            ] = future_date.day

        if "is_month_start" in features:

            row[
                "is_month_start"
            ] = int(
                future_date.is_month_start
            )

        if "is_month_end" in features:

            row[
                "is_month_end"
            ] = int(
                future_date.is_month_end
            )

        # --------------------------------------------------------
        # Historical quantity
        # --------------------------------------------------------

        quantity_history = (
            item_history[
                "quantity"
            ]
            .astype(float)
            .tolist()
        )

        # --------------------------------------------------------
        # Lag helper
        # --------------------------------------------------------

        def get_lag(
            lag
        ):

            if (
                len(
                    quantity_history
                )
                >= lag
            ):

                return float(
                    quantity_history[
                        -lag
                    ]
                )

            if quantity_history:

                return float(
                    quantity_history[
                        0
                    ]
                )

            return 0.0

        # --------------------------------------------------------
        # Lag features
        # --------------------------------------------------------

        for lag in [
            1,
            2,
            3,
            7,
            14,
            21,
            28
        ]:

            row[
                f"model_lag_{lag}"
            ] = get_lag(
                lag
            )

        # --------------------------------------------------------
        # Rolling mean helper
        # --------------------------------------------------------

        def rolling_mean(
            window
        ):

            values = (
                quantity_history[
                    -window:
                ]
            )

            if not values:

                return 0.0

            return float(
                np.mean(
                    values
                )
            )

        # --------------------------------------------------------
        # Rolling means
        # --------------------------------------------------------

        for window in [
            7,
            14,
            28
        ]:

            row[
                f"model_rolling_mean_{window}"
            ] = rolling_mean(
                window
            )

        # --------------------------------------------------------
        # Ensure every model feature exists
        # --------------------------------------------------------

        for feature in features:

            if feature not in row:

                row[
                    feature
                ] = self._latest_value(
                    item_history,
                    feature,
                    0
                )

        return row

    # ============================================================
    # 18. ENCODE FUTURE FEATURES
    # ============================================================

    @staticmethod
    def _encode_future_features(
        future_df,
        features,
        encoders
    ):

        encoded = (
            future_df.copy()
        )

        for column in features:

            # ----------------------------------------------------
            # Categorical
            # ----------------------------------------------------

            if column in encoders:

                mapping = {
                    value: index
                    for index, value
                    in enumerate(
                        encoders[
                            column
                        ]
                    )
                }

                encoded[
                    column
                ] = (
                    encoded[
                        column
                    ]
                    .fillna(
                        "unknown"
                    )
                    .astype(str)
                    .map(
                        mapping
                    )
                    .fillna(-1)
                    .astype(int)
                )

            # ----------------------------------------------------
            # Numeric
            # ----------------------------------------------------

            else:

                encoded[
                    column
                ] = (
                    pd.to_numeric(
                        encoded[
                            column
                        ],
                        errors="coerce"
                    )
                    .fillna(0)
                )

        encoded[
            features
        ] = (
            encoded[
                features
            ]
            .replace(
                [
                    np.inf,
                    -np.inf
                ],
                np.nan
            )
            .fillna(0)
        )

        return encoded

    # ============================================================
    # 19. GENERATE FUTURE FORECAST
    # ============================================================

    def _forecast_future(
        self,
        history,
        model,
        features,
        encoders,
        forecast_horizon
    ):

        history = (
            history.copy()
        )

        last_date = (
            history[
                "date"
            ].max()
        )

        combinations = (
            history[
                [
                    "restaurant_id",
                    "menu_item_id"
                ]
            ]
            .drop_duplicates()
            .reset_index(
                drop=True
            )
        )

        forecast_records = []

        # --------------------------------------------------------
        # Recursive forecast
        # --------------------------------------------------------

        for step in range(
            1,
            forecast_horizon + 1
        ):

            future_date = (
                last_date
                + pd.Timedelta(
                    days=step
                )
            )

            future_rows = []

            for _, combination in (
                combinations.iterrows()
            ):

                restaurant_id = (
                    combination[
                        "restaurant_id"
                    ]
                )

                menu_item_id = (
                    combination[
                        "menu_item_id"
                    ]
                )

                row = (
                    self._create_future_row(
                        history=
                            history,

                        restaurant_id=
                            restaurant_id,

                        menu_item_id=
                            menu_item_id,

                        future_date=
                            future_date,

                        features=
                            features
                    )
                )

                future_rows.append(
                    row
                )

            future_raw = (
                pd.DataFrame(
                    future_rows
                )
            )

            future_encoded = (
                self._encode_future_features(
                    future_raw,
                    features,
                    encoders
                )
            )

            predictions = (
                np.expm1(
                    model.predict(
                        future_encoded[
                            features
                        ]
                    )
                )
            )

            predictions = (
                np.clip(
                    predictions,
                    0,
                    None
                )
            )

            future_raw[
                "quantity"
            ] = predictions

            # ----------------------------------------------------
            # Store detailed forecast
            # ----------------------------------------------------

            for index in range(
                len(
                    future_raw
                )
            ):

                forecast_records.append(
                    {
                        "date":
                            future_raw
                            .iloc[index][
                                "date"
                            ],

                        "restaurant_id":
                            future_raw
                            .iloc[index][
                                "restaurant_id"
                            ],

                        "menu_item_id":
                            future_raw
                            .iloc[index][
                                "menu_item_id"
                            ],

                        "predicted_quantity":
                            float(
                                predictions[
                                    index
                                ]
                            )
                    }
                )

            # ----------------------------------------------------
            # Recursive update
            # ----------------------------------------------------

            history = pd.concat(
                [
                    history,
                    future_raw
                ],
                ignore_index=True,
                sort=False
            )

        return pd.DataFrame(
            forecast_records
        )

    # ============================================================
    # 20. READ FORECAST CSV
    # ============================================================

    @staticmethod
    def _read_forecast_csv(
        output_path
    ):

        if not output_path.exists():

            raise FileNotFoundError(
                f"Forecast output not found: "
                f"{output_path}"
            )

        forecast_data = (
            pd.read_csv(
                output_path
            )
        )

        required_columns = [
            "date",
            "restaurant_id",
            "menu_item_id",
            "predicted_quantity"
        ]

        missing = [
            column
            for column in required_columns
            if column not in forecast_data.columns
        ]

        if missing:

            raise ValueError(
                "Forecast CSV is missing: "
                + ", ".join(
                    missing
                )
            )

        forecast_data[
            "date"
        ] = pd.to_datetime(
            forecast_data[
                "date"
            ],
            errors="coerce"
        )

        forecast_data[
            "predicted_quantity"
        ] = pd.to_numeric(
            forecast_data[
                "predicted_quantity"
            ],
            errors="coerce"
        )

        forecast_data = (
            forecast_data.dropna(
                subset=[
                    "date",
                    "predicted_quantity"
                ]
            )
        )

        return forecast_data

    # ============================================================
    # 21. GET REQUESTED FORECAST
    # ============================================================

    @staticmethod
    def _get_requested_forecast(
        forecast_data,
        request
    ):

        result = (
            forecast_data.copy()
        )

        restaurant_scope = (
            request.get(
                "restaurant_scope"
            )
        )

        restaurant_id = (
            request.get(
                "restaurant_id"
            )
        )

        menu_item_id = (
            request.get(
                "menu_item_id"
            )
        )

        requested_date = (
            request.get(
                "forecast_date"
            )
            or
            request.get(
                "date"
            )
        )

        # --------------------------------------------------------
        # Restaurant
        # --------------------------------------------------------

        if (
            restaurant_scope == "single"
            and restaurant_id is not None
        ):

            result = result[
                result[
                    "restaurant_id"
                ].astype(str)
                == str(
                    restaurant_id
                )
            ]

        # --------------------------------------------------------
        # Menu item
        # --------------------------------------------------------

        if menu_item_id is not None:

            result = result[
                result[
                    "menu_item_id"
                ].astype(str)
                == str(
                    menu_item_id
                )
            ]

        # --------------------------------------------------------
        # Date
        # --------------------------------------------------------

        if requested_date is not None:

            requested_date = (
                pd.to_datetime(
                    requested_date,
                    errors="coerce"
                )
            )

            if pd.isna(
                requested_date
            ):

                raise ValueError(
                    "The requested forecast date "
                    "could not be understood."
                )

            result = result[
                result[
                    "date"
                ].dt.date
                ==
                requested_date.date()
            ]

        # --------------------------------------------------------
        # No match
        # --------------------------------------------------------

        if result.empty:

            available_start = (
                forecast_data[
                    "date"
                ]
                .min()
                .date()
            )

            available_end = (
                forecast_data[
                    "date"
                ]
                .max()
                .date()
            )

            raise ValueError(
                "No forecast was found for the "
                "requested restaurant/item/date. "
                f"Available forecast dates are "
                f"{available_start} to "
                f"{available_end}."
            )

        # --------------------------------------------------------
        # Total predicted demand
        # --------------------------------------------------------

        total_predicted_demand = float(
            result[
                "predicted_quantity"
            ].sum()
        )

        # --------------------------------------------------------
        # Daily predicted demand
        # --------------------------------------------------------

        daily_forecast = (
            result.groupby(
                "date",
                as_index=False
            )[
                "predicted_quantity"
            ]
            .sum()
        )

        daily_forecast[
            "date"
        ] = (
            daily_forecast[
                "date"
            ]
            .dt.strftime(
                "%Y-%m-%d"
            )
        )

        daily_records = (
            daily_forecast.to_dict(
                orient="records"
            )
        )

        # --------------------------------------------------------
        # Determine scope
        # --------------------------------------------------------

        if (
            restaurant_id is not None
            and menu_item_id is not None
        ):

            scope = (
                "restaurant_item"
            )

        elif menu_item_id is not None:

            scope = (
                "item"
            )

        elif (
            restaurant_scope == "single"
            and restaurant_id is not None
        ):

            scope = (
                "restaurant"
            )

        else:

            scope = "all"

        response = {
            "scope":
                scope,

            "total_predicted_demand":
                total_predicted_demand,

            "daily_forecast":
                daily_records
        }

        if restaurant_id is not None:

            response[
                "restaurant_id"
            ] = restaurant_id

        if menu_item_id is not None:

            response[
                "menu_item_id"
            ] = menu_item_id

        if requested_date is not None:

            response[
                "date"
            ] = (
                requested_date.strftime(
                    "%Y-%m-%d"
                )
            )

        return response

    # ============================================================
    # 22. RUN AGENT
    # ============================================================

    def run(
        self,
        state
    ):

        print(
            "\n===== DEMAND FORECASTING AGENT ====="
        )

        # ========================================================
        # REQUEST
        # ========================================================

        request = state.get(
            "request",
            {}
        )

        forecast_horizon = int(
            request.get(
                "forecast_horizon"
            )
            or 7
        )

        if forecast_horizon <= 0:

            raise ValueError(
                "forecast_horizon must "
                "be greater than 0."
            )

        # ========================================================
        # DATA ANALYST OUTPUT
        # ========================================================

        unified_path = (
            self._get_unified_path(
                state
            )
        )

        demand_df = pd.read_csv(
            unified_path
        )

        # ========================================================
        # CONTEXT / SEASONALITY OUTPUT
        # ========================================================

        context_path = (
            self._get_context_features_path(
                state
            )
        )

        if context_path is not None:

            demand_df = (
                self._merge_context_features(
                    demand_df,
                    context_path
                )
            )

        # ========================================================
        # CUSTOMER PATTERN OUTPUT
        # ========================================================

        customer_pattern = (
            self._get_customer_pattern(
                state
            )
        )

        demand_df = (
            self._add_customer_pattern_features(
                demand_df,
                customer_pattern
            )
        )

        # ========================================================
        # FILTER REQUEST
        # ========================================================

        demand_df = (
            self._filter_scope(
                demand_df,
                request
            )
        )

        # ========================================================
        # PREPARE MODEL DATA
        # ========================================================

        base_data = (
            self._prepare_base_data(
                demand_df
            )
        )

        base_data = (
            self._create_calendar_features(
                base_data
            )
        )

        feature_data = (
            self._create_lag_features(
                base_data
            )
        )

        (
            model_data,
            features,
            encoders
        ) = (
            self._prepare_model_features(
                feature_data
            )
        )

        if model_data.empty:

            raise ValueError(
                "No data available "
                "for model training."
            )

        # ========================================================
        # HISTORICAL VALIDATION
        # ========================================================

        print(
            "\nEvaluating model..."
        )

        wmape = (
            self._evaluate_model(
                model_data,
                features,
                forecast_horizon
            )
        )

        if wmape is not None:

            print(
                f"Validation wMAPE: "
                f"{wmape * 100:.2f}%"
            )

        else:

            print(
                "Validation wMAPE: "
                "Not enough historical data."
            )

        # ========================================================
        # TRAIN FINAL MODEL ON ALL HISTORICAL DATA
        # ========================================================

        print(
            "Training final demand forecasting model..."
        )

        model = (
            self._train_model(
                model_data,
                features
            )
        )

        print(
            "Model training completed."
        )

        # ========================================================
        # FUTURE FORECAST
        # ========================================================

        future_forecast = (
            self._forecast_future(
                history=
                    feature_data,

                model=
                    model,

                features=
                    features,

                encoders=
                    encoders,

                forecast_horizon=
                    forecast_horizon
            )
        )

        if future_forecast.empty:

            raise ValueError(
                "No future forecast was generated."
            )

        # ========================================================
        # SAVE DETAILED FORECAST
        # ========================================================

        output_dir = (
            Path(__file__)
            .resolve()
            .parents[2]
            / "data"
            / "outputs"
        )

        output_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        output_path = (
            output_dir
            / "demand_forecast.csv"
        )

        future_forecast.to_csv(
            output_path,
            index=False
        )

        # ========================================================
        # READ GENERATED CSV
        # ========================================================

        forecast_data = (
            self._read_forecast_csv(
                output_path
            )
        )

        # ========================================================
        # FORECAST PERIOD
        # ========================================================

        forecast_start = (
            forecast_data[
                "date"
            ]
            .min()
            .date()
        )

        forecast_end = (
            forecast_data[
                "date"
            ]
            .max()
            .date()
        )

        # ========================================================
        # GET USER-REQUESTED FORECAST
        # ========================================================

        requested_forecast = (
            self._get_requested_forecast(
                forecast_data,
                request
            )
        )

        # ========================================================
        # RETURN
        # ========================================================

        return {

            "status":
                "success",

            "forecast_horizon":
                forecast_horizon,

            "forecast_start":
                str(
                    forecast_start
                ),

            "forecast_end":
                str(
                    forecast_end
                ),

            "result":
                requested_forecast
        }