import numpy as np
import pandas as pd

MIN_ACTIVE = 10
MIN_BASELINE = 30


def _confidence(active_count: int, baseline_count: int) -> str:
    support = min(active_count, baseline_count)

    if support >= 100:
        return "high"

    if support >= 30:
        return "medium"

    return "low"


def _stats(
    df: pd.DataFrame,
    group_cols: list[str],
    flag: str
) -> pd.DataFrame:

    g = (
        df.dropna(subset=["quantity"])
        .groupby(
            group_cols + [flag],
            observed=True
        )["quantity"]
        .agg(["mean", "count"])
        .reset_index()
    )

    mean_w = g.pivot(
        index=group_cols,
        columns=flag,
        values="mean"
    )

    count_w = g.pivot(
        index=group_cols,
        columns=flag,
        values="count"
    )

    if 0 not in mean_w.columns or 1 not in mean_w.columns:
        return pd.DataFrame(
            columns=group_cols + [
                "uplift_pct",
                "active_count",
                "baseline_count"
            ]
        )

    out = pd.DataFrame(
        {
            "baseline_mean": mean_w[0],
            "active_mean": mean_w[1],
            "baseline_count": count_w.get(0, 0),
            "active_count": count_w.get(1, 0),
        }
    ).reset_index()

    out["baseline_count"] = (
        out["baseline_count"]
        .fillna(0)
        .astype(int)
    )

    out["active_count"] = (
        out["active_count"]
        .fillna(0)
        .astype(int)
    )

    out["uplift_pct"] = np.where(
        out["baseline_mean"].ne(0),
        (
            out["active_mean"]
            / out["baseline_mean"]
            - 1
        ) * 100,
        np.nan,
    )

    return out


def _global_stats(
    df: pd.DataFrame,
    flag: str
) -> dict:

    valid = df.dropna(subset=["quantity"])

    baseline = valid.loc[
        valid[flag].eq(0),
        "quantity"
    ]

    active = valid.loc[
        valid[flag].eq(1),
        "quantity"
    ]

    uplift = np.nan

    if (
        len(active)
        and len(baseline)
        and baseline.mean() != 0
    ):
        uplift = (
            active.mean()
            / baseline.mean()
            - 1
        ) * 100

    return {
        "uplift_pct": uplift,
        "active_count": int(len(active)),
        "baseline_count": int(len(baseline)),
    }


def _lookup(
    frame: pd.DataFrame,
    cols: list[str]
) -> dict:

    if frame.empty:
        return {}

    return {
        tuple(row[c] for c in cols): row
        for _, row in frame.iterrows()
    }


def _key(
    row: pd.Series,
    cols: list[str]
) -> tuple:

    return tuple(
        row[c]
        for c in cols
    )


def _hierarchical_signal(
    df: pd.DataFrame,
    entities: pd.DataFrame,
    flag: str,
    label: str
) -> pd.DataFrame:

    levels = [
        (
            "restaurant_sku",
            ["restaurant_id", "menu_item_id"]
        ),
        (
            "sku",
            ["menu_item_id"]
        ),
        (
            "category",
            ["category"]
        ),
    ]

    tables = []

    for level_name, cols in levels:

        stats = _stats(
            df,
            cols,
            flag
        )

        tables.append(
            (
                level_name,
                cols,
                _lookup(stats, cols)
            )
        )

    global_signal = _global_stats(
        df,
        flag
    )

    rows = []

    for _, entity in entities.iterrows():

        chosen = None

        for (
            level_name,
            cols,
            lookup
        ) in tables:

            record = lookup.get(
                _key(entity, cols)
            )

            if record is None:
                continue

            active = int(
                record["active_count"]
            )

            baseline = int(
                record["baseline_count"]
            )

            if (
                active >= MIN_ACTIVE
                and baseline >= MIN_BASELINE
                and pd.notna(
                    record["uplift_pct"]
                )
            ):

                chosen = (
                    float(
                        record["uplift_pct"]
                    ),
                    active,
                    baseline,
                    level_name,
                )

                break

        if chosen is None:

            chosen = (
                float(
                    global_signal[
                        "uplift_pct"
                    ]
                )
                if pd.notna(
                    global_signal[
                        "uplift_pct"
                    ]
                )
                else np.nan,

                global_signal[
                    "active_count"
                ],

                global_signal[
                    "baseline_count"
                ],

                "global",
            )

        (
            uplift,
            active,
            baseline,
            level
        ) = chosen

        rows.append(
            {
                "restaurant_id":
                    entity["restaurant_id"],

                "menu_item_id":
                    entity["menu_item_id"],

                f"{label}_uplift_pct":
                    uplift,

                f"{label}_active_count":
                    active,

                f"{label}_baseline_count":
                    baseline,

                f"{label}_signal_level":
                    level,

                f"{label}_confidence":
                    _confidence(
                        active,
                        baseline
                    ),
            }
        )

    return pd.DataFrame(rows)


def build_sku_signals(
    df: pd.DataFrame
) -> pd.DataFrame:

    keys = [
        "restaurant_id",
        "menu_item_id"
    ]

    entities = (
        df[
            keys + ["category"]
        ]
        .drop_duplicates(keys)
        .sort_values(keys)
        .reset_index(drop=True)
    )

    base = (
        df.groupby(
            keys,
            observed=True
        )["quantity"]
        .agg(
            baseline_demand="mean",
            baseline_observation_count="count"
        )
        .reset_index()
    )

    out = entities.merge(
        base,
        on=keys,
        how="left"
    )

    contexts = [
        (
            "is_weekend",
            "weekend"
        ),
        (
            "is_holiday",
            "holiday"
        ),
        (
            "is_special_event",
            "event"
        ),
        (
            "is_promotion",
            "promotion"
        ),
        (
            "has_precipitation",
            "precipitation"
        ),
    ]

    for flag, label in contexts:

        signal = _hierarchical_signal(
            df,
            entities,
            flag,
            label
        )

        out = out.merge(
            signal,
            on=keys,
            how="left"
        )

    return out