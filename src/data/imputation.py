"""Missing-value imputation extracted from the cleaning notebook."""

from __future__ import annotations

import pandas as pd


def impute_rate(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing ratings by restaurant mean, location median, then global median."""

    imputed = df.copy()
    restaurant_mean = imputed.groupby("name")["rate"].transform("mean")
    location_median = imputed.groupby("location")["rate"].transform("median")
    global_median = imputed["rate"].median()

    imputed["rate"] = (
        imputed["rate"]
        .fillna(restaurant_mean)
        .fillna(location_median)
        .fillna(global_median)
    )
    return imputed


def impute_location(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing location from the second-last comma-separated address token."""

    imputed = df.copy()
    imputed["location"] = imputed["location"].fillna(
        imputed["address"].str.split(",").str[-2].str.strip()
    )
    return imputed


def impute_cost(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing cost by restaurant mean, location median, then global median."""

    imputed = df.copy()
    restaurant_cost = imputed.groupby("name")["approx_cost(for two people)"].transform("mean")
    location_cost = imputed.groupby("location")["approx_cost(for two people)"].transform("median")
    global_cost = imputed["approx_cost(for two people)"].median()

    imputed["approx_cost(for two people)"] = (
        imputed["approx_cost(for two people)"]
        .fillna(restaurant_cost)
        .fillna(location_cost)
        .fillna(global_cost)
    )
    return imputed


def impute_cleaned_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all notebook imputation steps."""

    imputed = impute_rate(df)
    imputed = impute_location(imputed)
    return impute_cost(imputed)
