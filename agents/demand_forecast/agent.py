"""
Demand Forecasting Agent

Uses outputs from Phase-1 agents:

1. Data Analyst
   state["shared_context"]["data"]["unified_demand_path"]

2. Context / Seasonality
   state["shared_context"]["data"]["context_features_path"]

3. Customer Pattern
   state["shared_context"]["data"]["customer_pattern_features_path"]

Workflow:
1. Load the COMPLETE unified demand dataset.
2. Merge Context / Seasonality features.
3. Merge Customer Pattern features for every restaurant/menu-item combination.
4. Create calendar, lag and rolling features on the complete dataset.
5. Backtest on the last 7 historical days.
6. Train the final LightGBM model on the complete historical dataset.
7. Forecast every restaurant/menu-item combination for the next 7 days.
8. Save the complete detailed forecast to demand_forecast.csv.
9. Read demand_forecast.csv.
10. Apply the user's restaurant/menu-item/date request only to the generated forecast.
11. Return the requested values when they are inside the 7-day window.
12. Return an accuracy warning for requests outside the 7-day window.
"""

from pathlib import Path
import pickle

import lightgbm as lgb
import numpy as np
import pandas as pd


class DemandForecastingAgent:

    # ============================================================
    # APPLICATION CLOCK / WEEKLY FORECAST POLICY
    # ============================================================
    #
    # For this project/demo, 2025-12-01 is treated as the current date.
    # Forecasts are generated in non-overlapping 7-day blocks:
    #
    #   2025-12-01 -> 2025-12-07
    #   2025-12-08 -> 2025-12-14
    #   2025-12-15 -> 2025-12-21
    #   ...
    #
    # The model is persisted to disk and reused. It is NOT retrained for
    # every chatbot request. A new weekly forecast is generated only when
    # the requested date falls outside the currently stored forecast file.
    #
    CURRENT_DATE = pd.Timestamp("2026-01-01")
    FORECAST_HORIZON = 7

    @classmethod
    def _get_week_start(cls, request):
        """Return the 7-day block start for the user's requested date."""

        requested_date = (
            request.get("forecast_date")
            or request.get("date")
        )

        if requested_date is None:
            return cls.CURRENT_DATE.normalize()

        requested_date = pd.to_datetime(
            requested_date,
            errors="coerce"
        )

        if pd.isna(requested_date):
            raise ValueError(
                "The requested forecast date could not be understood."
            )

        requested_date = requested_date.normalize()
        base_date = cls.CURRENT_DATE.normalize()

        if requested_date < base_date:
            raise ValueError(
                f"Forecasts are available from {base_date.date()} onward."
            )

        days_from_base = (
            requested_date - base_date
        ).days

        block_offset = (
            days_from_base
            // cls.FORECAST_HORIZON
        ) * cls.FORECAST_HORIZON

        return (
            base_date
            + pd.Timedelta(days=block_offset)
        )

    @staticmethod
    def _forecast_covers_week(output_path, week_start, forecast_horizon):
        """Check whether the saved forecast already covers this weekly block."""

        output_path = Path(output_path)

        if not output_path.exists():
            return False

        try:
            existing = pd.read_csv(output_path)
        except Exception:
            return False

        if existing.empty or "date" not in existing.columns:
            return False

        dates = pd.to_datetime(
            existing["date"],
            errors="coerce"
        ).dropna()

        if dates.empty:
            return False

        expected_end = (
            pd.Timestamp(week_start)
            + pd.Timedelta(days=forecast_horizon - 1)
        )

        return (
            dates.min().normalize()
            == pd.Timestamp(week_start).normalize()
            and
            dates.max().normalize()
            == expected_end.normalize()
        )

    @staticmethod
    def _save_model_bundle(
        model_path,
        model,
        features,
        encoders,
        evaluation_wmape,
    ):
        """Persist the trained model and feature metadata."""

        model_path = Path(model_path)
        model_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        bundle = {
            "model": model,
            "features": features,
            "encoders": encoders,
            "evaluation_wmape": evaluation_wmape,
        }

        with open(model_path, "wb") as file:
            pickle.dump(bundle, file)

    @staticmethod
    def _load_model_bundle(model_path):
        """Load a previously trained model bundle."""

        model_path = Path(model_path)

        if not model_path.exists():
            return None

        try:
            with open(model_path, "rb") as file:
                return pickle.load(file)
        except Exception:
            return None

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
    def _get_customer_pattern_features_path(state):
        shared_context = state.get("shared_context", {})
        shared_data = shared_context.get("data", {})
        pattern_path = shared_data.get("customer_pattern_features_path")

        if not pattern_path:
            raise ValueError(
                "customer_pattern_features_path is missing from "
                "state['shared_context']['data']. Run the Customer Pattern "
                "Agent before Demand Forecasting."
            )

        pattern_path = Path(pattern_path)
        if not pattern_path.exists():
            raise FileNotFoundError(
                f"Customer pattern feature file not found: {pattern_path}"
            )
        return pattern_path

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
    def _add_customer_pattern_features(data, customer_pattern_path):
        """Merge row-level customer-pattern features into model training data."""
        result = data.copy()
        pattern_df = pd.read_csv(customer_pattern_path)

        if pattern_df.empty:
            return result

        merge_keys = [
            key for key in ["restaurant_id", "menu_item_id"]
            if key in result.columns and key in pattern_df.columns
        ]
        if len(merge_keys) != 2:
            raise ValueError(
                "Customer pattern features must contain restaurant_id and "
                "menu_item_id for model training."
            )

        # Normalize identifiers before merging to avoid int/string mismatches.
        for key in merge_keys:
            result[key] = result[key].astype(str)
            pattern_df[key] = pattern_df[key].astype(str)

        feature_columns = [
            c for c in pattern_df.columns
            if c not in merge_keys and c not in result.columns
        ]
        pattern_subset = pattern_df[merge_keys + feature_columns].drop_duplicates(
            subset=merge_keys
        )
        return result.merge(pattern_subset, on=merge_keys, how="left")

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
        forecast_horizon,
        forecast_start_date
    ):

        history = (
            history.copy()
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

        forecast_start_date = pd.Timestamp(
            forecast_start_date
        ).normalize()

        for step in range(
            forecast_horizon
        ):

            future_date = (
                forecast_start_date
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
    @staticmethod
    def _get_requested_forecast(
        forecast_data,
        request
    ):
        """
        Retrieve requested forecast rows from the already-generated 7-day CSV.

        Explicit restaurant_item_pairs preserve pairing and never create a
        Cartesian product.
        """

        result = forecast_data.copy()

        requested_date = (
            request.get("forecast_date")
            or request.get("date")
        )

        requested_forecast_horizon = int(
            request.get("requested_forecast_horizon")
            or request.get("forecast_horizon")
            or 7
        )

        requested_forecast_horizon = max(
            1,
            min(requested_forecast_horizon, 7)
        )

        if requested_date is not None:
            parsed_requested_date = pd.to_datetime(
                requested_date,
                errors="coerce"
            )
            if pd.isna(parsed_requested_date):
                raise ValueError(
                    "The requested forecast date could not be understood."
                )

            available_start = result["date"].min().normalize()
            available_end = result["date"].max().normalize()

            if not (
                available_start
                <= parsed_requested_date.normalize()
                <= available_end
            ):
                return {
                    "status": "warning",
                    "message": (
                        "Demand forecasts are only considered reliable for "
                        "the available 7-day window."
                    ),
                    "available_forecast_start": str(available_start.date()),
                    "available_forecast_end": str(available_end.date()),
                }

        pairs = request.get("restaurant_item_pairs") or []

        # Backward-compatible conversion for singular/list requests.
        if not pairs:
            restaurant_ids = request.get("restaurant_ids") or []
            menu_item_ids = request.get("menu_item_ids") or []

            if not restaurant_ids and request.get("restaurant_id") is not None:
                restaurant_ids = [request.get("restaurant_id")]

            if not menu_item_ids and request.get("menu_item_id") is not None:
                menu_item_ids = [request.get("menu_item_id")]

            if restaurant_ids and menu_item_ids:
                pairs = [
                    {
                        "restaurant_id": restaurant_id,
                        "menu_item_id": menu_item_id,
                    }
                    for restaurant_id in restaurant_ids
                    for menu_item_id in menu_item_ids
                ]

        if pairs:
            combination_results = []

            for pair in pairs:
                pair_result = result.copy()

                restaurant_id = pair.get("restaurant_id")
                menu_item_id = pair.get("menu_item_id")

                if restaurant_id is not None:
                    pair_result = pair_result[
                        pair_result["restaurant_id"].astype(str)
                        == str(restaurant_id)
                    ]

                if menu_item_id is not None:
                    pair_result = pair_result[
                        pair_result["menu_item_id"].astype(str)
                        == str(menu_item_id)
                    ]

                if requested_date is not None:
                    pair_result = pair_result[
                        pair_result["date"].dt.date
                        == parsed_requested_date.date()
                    ]
                else:
                    dates = (
                        pair_result["date"]
                        .drop_duplicates()
                        .sort_values()
                        .head(requested_forecast_horizon)
                    )
                    pair_result = pair_result[
                        pair_result["date"].isin(dates)
                    ]

                if pair_result.empty:
                    continue

                daily = (
                    pair_result.groupby(
                        "date",
                        as_index=False
                    )["predicted_quantity"]
                    .sum()
                )

                daily["date"] = daily["date"].dt.strftime("%Y-%m-%d")

                combination_results.append({
                    "restaurant_id": restaurant_id,
                    "menu_item_id": menu_item_id,
                    "total_predicted_demand": float(
                        pair_result["predicted_quantity"].sum()
                    ),
                    "daily_forecast": daily.to_dict(orient="records"),
                })

            if not combination_results:
                raise ValueError(
                    "No forecast was found for the requested "
                    "restaurant/item combination(s)."
                )

            return {
                "scope": "restaurant_item_pairs",
                "combination_results": combination_results,
                "combination_count": len(combination_results),
            }

        # No explicit pairs: preserve all-restaurants/all-items behavior.
        restaurant_scope = request.get("restaurant_scope")
        restaurant_id = request.get("restaurant_id")
        menu_item_id = request.get("menu_item_id")

        if (
            restaurant_scope == "single"
            and restaurant_id is not None
        ):
            result = result[
                result["restaurant_id"].astype(str)
                == str(restaurant_id)
            ]

        if menu_item_id is not None:
            result = result[
                result["menu_item_id"].astype(str)
                == str(menu_item_id)
            ]

        if requested_date is not None:
            result = result[
                result["date"].dt.date
                == parsed_requested_date.date()
            ]
        else:
            dates = (
                result["date"]
                .drop_duplicates()
                .sort_values()
                .head(requested_forecast_horizon)
            )
            result = result[
                result["date"].isin(dates)
            ]

        if result.empty:
            raise ValueError(
                "No forecast was found for the requested scope."
            )

        daily = (
            result.groupby(
                "date",
                as_index=False
            )["predicted_quantity"]
            .sum()
        )
        daily["date"] = daily["date"].dt.strftime("%Y-%m-%d")

        response = {
            "scope": "all" if restaurant_scope == "all" else "filtered",
            "total_predicted_demand": float(
                result["predicted_quantity"].sum()
            ),
            "daily_forecast": daily.to_dict(orient="records"),
        }

        if restaurant_id is not None:
            response["restaurant_id"] = restaurant_id
        if menu_item_id is not None:
            response["menu_item_id"] = menu_item_id
        if requested_date is not None:
            response["date"] = parsed_requested_date.strftime("%Y-%m-%d")

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

        request = state.get(
            "request",
            {}
        )

        requested_forecast_horizon = int(
            request.get("requested_forecast_horizon")
            or request.get("forecast_horizon")
            or self.FORECAST_HORIZON
        )

        if requested_forecast_horizon <= 0:
            raise ValueError(
                "forecast_horizon must be greater than 0."
            )

        forecast_horizon = self.FORECAST_HORIZON

        # --------------------------------------------------------
        # FIXED DEMO CLOCK + WEEKLY FORECAST BLOCK
        # --------------------------------------------------------

        forecast_start_date = (
            self._get_week_start(
                request
            )
        )

        forecast_end_date = (
            forecast_start_date
            + pd.Timedelta(
                days=forecast_horizon - 1
            )
        )

        print(
            "[DEMAND FORECAST] Forecast block:",
            str(forecast_start_date.date()),
            "to",
            str(forecast_end_date.date()),
        )

        # --------------------------------------------------------
        # OUTPUT / MODEL PATHS
        # --------------------------------------------------------

        project_root = (
            Path(__file__)
            .resolve()
            .parents[2]
        )

        output_dir = (
            project_root
            / "data"
            / "outputs"
        )

        model_dir = (
            project_root
            / "data"
            / "models"
        )

        output_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        model_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        output_path = (
            output_dir
            / "demand_forecast.csv"
        )

        model_path = (
            model_dir
            / "demand_forecasting_model.pkl"
        )

        # --------------------------------------------------------
        # FAST PATH:
        # USE THE ALREADY GENERATED WEEKLY FORECAST
        # --------------------------------------------------------
        #
        # This is what prevents model training on every chatbot query.
        # If the stored CSV already covers the requested 7-day block,
        # simply read it and return the requested restaurant/item/date.

        if self._forecast_covers_week(
            output_path,
            forecast_start_date,
            forecast_horizon,
        ):

            print(
                "[DEMAND FORECAST] Using cached weekly forecast. "
                "No model training required."
            )

            forecast_data = (
                self._read_forecast_csv(
                    output_path
                )
            )

            if requested_forecast_horizon > forecast_horizon:
                requested_forecast = {
                    "status": "warning",
                    "message": (
                        "Demand forecasts are generated in 7-day blocks. "
                        "A request longer than 7 days is not returned as one "
                        "forecast because accuracy may decrease."
                    ),
                    "available_forecast_start":
                        str(forecast_start_date.date()),
                    "available_forecast_end":
                        str(forecast_end_date.date()),
                }
            else:
                requested_forecast = (
                    self._get_requested_forecast(
                        forecast_data,
                        request
                    )
                )

            return {
                "status": "success",
                "forecast_horizon": forecast_horizon,
                "forecast_start":
                    str(forecast_start_date.date()),
                "forecast_end":
                    str(forecast_end_date.date()),
                "output_path":
                    str(output_path),
                "model_path":
                    str(model_path),
                "model_trained":
                    False,
                "forecast_generated":
                    False,
                "result":
                    requested_forecast,
            }

        # --------------------------------------------------------
        # LOAD COMPLETE TRAINING DATA
        # --------------------------------------------------------

        unified_path = (
            self._get_unified_path(
                state
            )
        )

        demand_df = pd.read_csv(
            unified_path,
            low_memory=False
        )

        # --------------------------------------------------------
        # MERGE CONTEXT / SEASONALITY FEATURES
        # --------------------------------------------------------

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

        # --------------------------------------------------------
        # MERGE CUSTOMER PATTERN FEATURES
        # --------------------------------------------------------

        customer_pattern_path = (
            self._get_customer_pattern_features_path(
                state
            )
        )

        demand_df = (
            self._add_customer_pattern_features(
                demand_df,
                customer_pattern_path
            )
        )

        required_pattern_features = {
            "customer_demand_trend",
            "pattern_strength",
        }

        missing_pattern_features = (
            required_pattern_features
            - set(demand_df.columns)
        )

        if missing_pattern_features:
            raise ValueError(
                "Customer pattern features were not merged into the "
                "forecast training data: "
                + ", ".join(
                    sorted(
                        missing_pattern_features
                    )
                )
            )

        print(
            "[DEMAND FORECAST] Customer pattern features loaded: "
            "customer_demand_trend, pattern_strength"
        )

        # --------------------------------------------------------
        # PREPARE COMPLETE MODEL DATA
        # --------------------------------------------------------

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
            prepared_model_data,
            prepared_features,
            prepared_encoders
        ) = (
            self._prepare_model_features(
                feature_data
            )
        )

        if prepared_model_data.empty:
            raise ValueError(
                "No data available for model training."
            )

        combination_count = (
            prepared_model_data[
                [
                    "restaurant_id",
                    "menu_item_id",
                ]
            ]
            .drop_duplicates()
            .shape[0]
        )

        print(
            "[DEMAND FORECAST] Complete dataset:",
            f"{len(prepared_model_data):,} rows,",
            f"{combination_count:,} restaurant/menu-item combinations."
        )

        # --------------------------------------------------------
        # LOAD PERSISTED MODEL IF IT ALREADY EXISTS
        # --------------------------------------------------------

        bundle = (
            self._load_model_bundle(
                model_path
            )
        )

        model_trained = False
        wmape = None

        if bundle is not None:

            model = bundle.get(
                "model"
            )

            features = bundle.get(
                "features",
                prepared_features
            )

            encoders = bundle.get(
                "encoders",
                prepared_encoders
            )

            wmape = bundle.get(
                "evaluation_wmape"
            )

            missing_features = [
                feature
                for feature in features
                if feature
                not in feature_data.columns
            ]

            if (
                model is None
                or missing_features
            ):
                bundle = None

        # --------------------------------------------------------
        # TRAIN ONLY WHEN NO VALID SAVED MODEL EXISTS
        # --------------------------------------------------------

        if bundle is None:

            model_data = prepared_model_data
            features = prepared_features
            encoders = prepared_encoders

            print(
                "\n[DEMAND FORECAST] No valid saved model found."
            )

            print(
                "[DEMAND FORECAST] Evaluating model..."
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

            print(
                "[DEMAND FORECAST] Training final model..."
            )

            model = (
                self._train_model(
                    model_data,
                    features
                )
            )

            self._save_model_bundle(
                model_path=model_path,
                model=model,
                features=features,
                encoders=encoders,
                evaluation_wmape=wmape,
            )

            model_trained = True

            print(
                "[DEMAND FORECAST] Model trained and saved:",
                str(model_path)
            )

        else:

            print(
                "[DEMAND FORECAST] Reusing saved model:",
                str(model_path)
            )

            print(
                "[DEMAND FORECAST] Model training skipped."
            )

            if wmape is not None:
                print(
                    f"[DEMAND FORECAST] Saved validation wMAPE: "
                    f"{float(wmape) * 100:.2f}%"
                )

        # --------------------------------------------------------
        # GENERATE THIS 7-DAY FORECAST BLOCK
        # --------------------------------------------------------

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
                    forecast_horizon,

                forecast_start_date=
                    forecast_start_date,
            )
        )

        if future_forecast.empty:
            raise ValueError(
                "No future forecast was generated."
            )

        future_forecast.to_csv(
            output_path,
            index=False
        )

        print(
            "[DEMAND FORECAST] Weekly forecast saved:",
            str(output_path)
        )

        # --------------------------------------------------------
        # READ STORED FORECAST AND ANSWER USER
        # --------------------------------------------------------

        forecast_data = (
            self._read_forecast_csv(
                output_path
            )
        )

        if requested_forecast_horizon > forecast_horizon:

            requested_forecast = {
                "status": "warning",
                "message": (
                    "Demand forecasts are generated in 7-day blocks. "
                    "A request longer than 7 days is not returned as one "
                    "forecast because accuracy may decrease."
                ),
                "available_forecast_start":
                    str(forecast_start_date.date()),
                "available_forecast_end":
                    str(forecast_end_date.date()),
            }

        else:

            requested_forecast = (
                self._get_requested_forecast(
                    forecast_data,
                    request
                )
            )

        return {
            "status":
                "success",

            "forecast_horizon":
                forecast_horizon,

            "forecast_start":
                str(
                    forecast_start_date.date()
                ),

            "forecast_end":
                str(
                    forecast_end_date.date()
                ),

            "output_path":
                str(
                    output_path
                ),

            "model_path":
                str(
                    model_path
                ),

            "model_trained":
                model_trained,

            "forecast_generated":
                True,

            "evaluation": (
                {
                    "wmape_percent":
                        round(
                            float(wmape) * 100,
                            2
                        )
                }
                if wmape is not None
                else {}
            ),

            "result":
                requested_forecast,
        }

