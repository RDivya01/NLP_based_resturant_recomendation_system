"""Data cleaning functions extracted from the cleaning notebook."""

from __future__ import annotations

import numpy as np
import pandas as pd


DROP_COLUMNS = ["url", "listed_in(city)", "menu_item"]
REQUIRED_RAW_COLUMNS = {
    "address",
    "name",
    "online_order",
    "book_table",
    "rate",
    "votes",
    "location",
    "dish_liked",
    "cuisines",
    "approx_cost(for two people)",
    "reviews_list",
    "listed_in(type)",
}


def clean_rate(value: object) -> float:
    """Clean a Zomato rate value such as '4.1/5' into a float."""

    if pd.isna(value) or value in {"NEW", "-"}:
        return np.nan

    value_text = str(value).split("/")[0].strip()
    try:
        return float(value_text)
    except ValueError:
        return np.nan


def clean_cost(value: object) -> float:
    """Clean approx cost strings by removing commas and coercing to float."""

    series = pd.Series([value])
    cleaned = pd.to_numeric(
        series.astype(str).str.replace(",", "", regex=False).str.strip(),
        errors="coerce",
    )
    return float(cleaned.iloc[0]) if pd.notna(cleaned.iloc[0]) else np.nan


def clean_binary_column(value: object) -> int | float:
    """Map Yes/No values to 1/0 while preserving missing values."""

    return {"Yes": 1, "No": 0}.get(value, np.nan)


def clean_list_column(value: object) -> list[str]:
    """Split comma-separated cuisines or liked dishes into a clean list."""

    if pd.isna(value):
        return []

    items = str(value).split(",")
    cleaned_items = []

    for item in items:
        item = item.strip()
        if item:
            cleaned_items.append(item)

    return cleaned_items


def drop_unused_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop notebook-identified unused columns."""

    return df.drop(columns=DROP_COLUMNS, errors="ignore")


def clean_restaurant_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Apply notebook cleaning steps to rate, cost, booleans, and list columns."""

    cleaned = df.copy()
    cleaned["rate"] = cleaned["rate"].apply(clean_rate)
    cleaned["approx_cost(for two people)"] = cleaned["approx_cost(for two people)"].apply(clean_cost)
    cleaned["online_order"] = cleaned["online_order"].apply(clean_binary_column)
    cleaned["book_table"] = cleaned["book_table"].apply(clean_binary_column)
    cleaned["cuisines"] = cleaned["cuisines"].apply(clean_list_column)
    cleaned["dish_liked"] = cleaned["dish_liked"].apply(clean_list_column)
    return cleaned


def validate_raw_schema(df: pd.DataFrame) -> None:
    """Validate the raw dataset has the columns required by the notebooks."""

    available_columns = set(df.columns) | set(DROP_COLUMNS)
    missing = REQUIRED_RAW_COLUMNS - available_columns
    if missing:
        raise ValueError(f"Missing required raw columns: {sorted(missing)}")


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Run the full non-imputation cleaning flow from the notebook."""

    validate_raw_schema(df)
    cleaned = drop_unused_columns(df)
    return clean_restaurant_columns(cleaned)
