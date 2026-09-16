import numpy as np
import pandas as pd


def binary_impact(
    df,
    flag,
    name_col=None
):

    valid = df.dropna(
        subset=["quantity"]
    )

    base = valid.loc[
        valid[flag].eq(0),
        "quantity"
    ].mean()

    base_count = int(
        valid[flag]
        .eq(0)
        .sum()
    )

    keys = [flag]

    if name_col:
        keys.append(name_col)

    event = (
        valid.loc[
            valid[flag].eq(1)
        ]
        .groupby(
            keys,
            dropna=False,
            observed=True
        )["quantity"]
        .agg(
            [
                "mean",
                "median",
                "count"
            ]
        )
        .reset_index()
    )

    event["baseline_mean"] = base

    event["baseline_count"] = (
        base_count
    )

    event["uplift_pct"] = np.where(
        base != 0,
        (
            event["mean"]
            / base
            - 1
        ) * 100,
        np.nan
    )

    event["interpretation"] = (
        "descriptive_association_not_causal"
    )

    return event


def weather_impact(df):

    valid = df.dropna(
        subset=["quantity"]
    )

    temperature = (
        valid
        .groupby(
            "temp_band",
            observed=True
        )["quantity"]
        .agg(
            [
                "mean",
                "median",
                "count"
            ]
        )
        .reset_index()
    )

    precipitation = (
        valid
        .groupby(
            "has_precipitation"
        )["quantity"]
        .agg(
            [
                "mean",
                "median",
                "count"
            ]
        )
        .reset_index()
    )

    precipitation_type = (
        valid
        .groupby(
            "precip_type",
            observed=True
        )["quantity"]
        .agg(
            [
                "mean",
                "median",
                "count"
            ]
        )
        .reset_index()
    )

    return (
        temperature,
        precipitation,
        precipitation_type
    )