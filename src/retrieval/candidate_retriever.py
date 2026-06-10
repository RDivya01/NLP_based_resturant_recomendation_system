"""Candidate filtering extracted from the NLP notebook."""

from __future__ import annotations
import ast
from typing import Any
import pandas as pd
from src.retrieval.fuzzy_match import fuzzy_match


def _contains_text(value: object, query: str) -> bool:
    """Return whether query is contained in a value, ignoring case."""

    return query.lower() in str(value).lower()


def _cuisines_contain(value: object, cuisine: str) -> bool:
    """Match cuisine against list-like or string-stored cuisine values."""

    if isinstance(value, list):
        return any(cuisine.lower() in str(item).lower() for item in value)

    try:
        parsed = ast.literal_eval(str(value))
        if isinstance(parsed, list):
            return any(cuisine.lower() in str(item).lower() for item in parsed)
    except (ValueError, SyntaxError):
        pass

    return cuisine.lower() in str(value).lower()

def _get_cuisine_choices(restaurant_df: pd.DataFrame) -> list[str]:
    """Extract unique cuisines from the dataset."""

    cuisines = set()

    for value in restaurant_df["cuisines"].dropna():

        if isinstance(value, list):
            cuisines.update(str(item).strip() for item in value)
            continue

        try:
            parsed = ast.literal_eval(str(value))

            if isinstance(parsed, list):
                cuisines.update(str(item).strip() for item in parsed)
                continue

        except (ValueError, SyntaxError):
            pass

        cuisines.update(
            cuisine.strip()
            for cuisine in str(value).split(",")
        )

    return list(cuisines)

def retrieve_candidates(restaurant_df: pd.DataFrame, filters: dict[str, Any]) -> pd.DataFrame:
    """Filter restaurants by location, cuisine, budget, online order, and booking."""

    candidates = restaurant_df.copy()

    location = filters.get("location")

    if location:

        location_choices = (
            restaurant_df["location"]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

        corrected_location = fuzzy_match(
            location,
            location_choices,
        )

        candidates = candidates[
            candidates["location"].apply(
                lambda value:
                _contains_text(
                    value,
                    corrected_location,
                )
            )
        ]

    cuisine = filters.get("cuisine")

    if cuisine:

        cuisine_choices = _get_cuisine_choices(
            restaurant_df,
        )

        corrected_cuisine = fuzzy_match(
            cuisine,
            cuisine_choices,
        )

        candidates = candidates[
            candidates["cuisines"].apply(
                lambda cuisines: _cuisines_contain(
                    cuisines,
                    corrected_cuisine,
                )
            )
        ]

    budget = filters.get("budget")
    if budget:
        candidates = candidates[candidates["approx_cost(for two people)"] <= budget]

    online_order = filters.get("online_order")
    if online_order is not None:
        candidates = candidates[candidates["online_order"] == int(online_order)]

    book_table = filters.get("book_table")
    if book_table is not None:
        candidates = candidates[candidates["book_table"] == int(book_table)]

    return candidates.reset_index(drop=True)
