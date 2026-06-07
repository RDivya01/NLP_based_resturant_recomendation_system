"""Explainable recommendation reason generation."""

from __future__ import annotations

import pandas as pd


REASON_RULES = [
    ("food_score", 0.85, "Excellent food quality"),
    ("ambiance_score", 0.85, "Great ambiance"),
    ("authenticity_score", 0.85, "Highly authentic cuisine"),
    ("service_score", 0.85, "Excellent service"),
    ("value_for_money_score", 0.85, "Great value for money"),
    ("popularity_score", 0.80, "Popular among customers"),
]


def generate_recommendation_reasons(row: pd.Series) -> list[str]:
    """Generate recommendation reasons from score thresholds."""

    reasons = [
        reason
        for column, threshold, reason in REASON_RULES
        if column in row
        and pd.notna(row[column])
        and row[column] >= threshold
    ]

    if reasons:
        return reasons[:3]

    rating = row.get("rate", 0)

    if pd.notna(rating) and rating >= 4.3:
        return ["Highly rated by customers"]

    if (
        "value_for_money_score" in row
        and pd.notna(row["value_for_money_score"])
        and row["value_for_money_score"] >= 0.70
    ):
        return ["Good value for money"]

    return ["Matches your search criteria"]


def add_recommendation_reasons(df: pd.DataFrame) -> pd.DataFrame:
    """Add explainable recommendation reasons to a ranked dataframe."""

    explained = df.copy()
    explained["reasons"] = explained.apply(generate_recommendation_reasons, axis=1)
    return explained
