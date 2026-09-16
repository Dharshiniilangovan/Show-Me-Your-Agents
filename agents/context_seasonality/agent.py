import json

from .config import (
    PROCESSED,
    OUTPUTS
)

from .feature_engineering import (
    add_context_features
)

from .seasonality import (
    seasonality_summary
)

from .context_analysis import (
    binary_impact,
    weather_impact
)

from .signals import (
    build_sku_signals
)

from .schemas import (
    AgentRunSummary
)


class ContextSeasonalityAgent:

    VERSION = "1.1"

    def run(self, df):

        PROCESSED.mkdir(
            parents=True,
            exist_ok=True
        )

        OUTPUTS.mkdir(
            parents=True,
            exist_ok=True
        )

        x = add_context_features(df)

        files = []

        # ========================
        # MODEL FEATURE DATASET
        # ========================

        feature_path = (
            PROCESSED
            / "context_features.csv"
        )

        x.to_csv(
            feature_path,
            index=False
        )

        files.append(
            str(feature_path)
        )

        x.head(10000).to_csv(
            PROCESSED
            / "context_features_sample.csv",
            index=False
        )

        # ========================
        # SEASONALITY
        # ========================

        for frame in seasonality_summary(x):

            dimension = (
                frame["dimension"]
                .iloc[0]
            )

            path = (
                OUTPUTS
                / f"seasonality_{dimension}.csv"
            )

            frame.to_csv(
                path,
                index=False
            )

            files.append(str(path))

        # ========================
        # CONTEXT IMPACT
        # ========================

        analyses = [

            (
                "holiday_impact.csv",
                binary_impact(
                    x,
                    "is_holiday",
                    "holiday_name"
                )
            ),

            (
                "promotion_impact.csv",
                binary_impact(
                    x,
                    "is_promotion"
                )
            ),

            (
                "event_impact.csv",
                binary_impact(
                    x,
                    "is_special_event",
                    "special_event_name"
                )
            )
        ]

        for name, frame in analyses:

            path = OUTPUTS / name

            frame.to_csv(
                path,
                index=False
            )

            files.append(
                str(path)
            )

        # ========================
        # WEATHER
        # ========================

        (
            temp,
            precip,
            precip_type
        ) = weather_impact(x)

        weather_files = [

            (
                "weather_temperature.csv",
                temp
            ),

            (
                "weather_precipitation.csv",
                precip
            ),

            (
                "weather_precip_type.csv",
                precip_type
            )
        ]

        for name, frame in weather_files:

            path = OUTPUTS / name

            frame.to_csv(
                path,
                index=False
            )

            files.append(
                str(path)
            )

        # ========================
        # SKU CONTEXT KNOWLEDGE
        # ========================

        signals = build_sku_signals(x)

        path = (
            OUTPUTS
            / "sku_context_signals.csv"
        )

        signals.to_csv(
            path,
            index=False
        )

        files.append(str(path))

        # ========================
        # AGENT SUMMARY
        # ========================

        summary = AgentRunSummary(

            len(x),

            str(
                x.date.min().date()
            ),

            str(
                x.date.max().date()
            ),

            x.restaurant_id.nunique(),

            x.menu_item_id.nunique(),

            int(
                x.quantity
                .isna()
                .sum()
            ),

            files
        )

        # ========================
        # SHARED CONTEXT CONTRACT
        # ========================

        payload = {

            "agent":
                "ContextSeasonalityAgent",

            "version":
                self.VERSION,

            "summary":
                summary.to_dict(),

            "feature_contract": {

                "forecast_keys": [
                    "date",
                    "restaurant_id",
                    "menu_item_id"
                ],

                "target":
                    "quantity",

                "categorical_context_features": [

                    "holiday_name",
                    "special_event_name",
                    "precip_type"
                ],

                "numeric_context_features": [

                    "dow_sin",
                    "dow_cos",

                    "month_sin",
                    "month_cos",

                    "doy_sin",
                    "doy_cos",

                    "week_of_year",
                    "quarter",
                    "day_of_month",

                    "is_month_start",
                    "is_month_end",

                    "is_weekend",

                    "is_holiday",
                    "is_special_event",
                    "is_promotion",

                    "avg_temp_f",
                    "temp_c",

                    "precip_inches",
                    "has_precipitation",

                    "context_event_count",
                    "is_context_day",

                    "weekend_promotion",
                    "holiday_promotion",
                    "event_promotion",
                    "precipitation_weekend"
                ],

                "historical_signal_file":
                    "sku_context_signals.csv",

                "signal_note":
                    (
                        "Historical uplifts are "
                        "descriptive associations, "
                        "not causal effects."
                    )
            }
        }

        with open(
            OUTPUTS
            / "agent_context.json",
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                payload,
                file,
                indent=2
            )

        return payload