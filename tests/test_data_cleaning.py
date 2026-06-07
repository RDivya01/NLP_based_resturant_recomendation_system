"""Tests for data cleaning and imputation."""

from __future__ import annotations

import pandas as pd

from src.data.cleaning import clean_cost, clean_list_column, clean_rate
from src.data.imputation import impute_cleaned_dataset


def test_clean_rate_cost_and_list_columns() -> None:
    """Cleaning helpers should preserve notebook behavior."""

    assert clean_rate("4.2/5") == 4.2
    assert pd.isna(clean_rate("NEW"))
    assert clean_cost("1,200") == 1200.0
    assert clean_list_column("North Indian, Chinese") == ["North Indian", "Chinese"]


def test_impute_cleaned_dataset_fills_rating_location_and_cost() -> None:
    """Imputation should use restaurant/location/global fallbacks."""

    df = pd.DataFrame(
        {
            "name": ["A", "A", "B"],
            "rate": [4.0, None, None],
            "location": ["Indiranagar", "Indiranagar", None],
            "address": ["x, Indiranagar, Bengaluru", "x, Indiranagar, Bengaluru", "x, Jayanagar, Bengaluru"],
            "approx_cost(for two people)": [500.0, None, None],
        }
    )

    result = impute_cleaned_dataset(df)

    assert result["rate"].isna().sum() == 0
    assert result["location"].isna().sum() == 0
    assert result["approx_cost(for two people)"].isna().sum() == 0
