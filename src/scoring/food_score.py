"""Food quality scoring from the review intelligence notebook."""

from __future__ import annotations

import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer

from src.scoring.aspect_utils import calculate_aspect_columns, calculate_aspect_score, get_keyword_sentences


FOOD_KEYWORDS = {
    "food",
    "taste",
    "tasty",
    "delicious",
    "dish",
    "meal",
    "biryani",
    "pizza",
    "burger",
    "dessert",
    "flavor",
    "flavour",
    "fresh",
    "cuisine",
    "starter",
    "main course",
    "portion",
}


def get_food_sentences(review: str, food_keywords: set[str] = FOOD_KEYWORDS) -> list[str]:
    """Return food-related sentences."""

    return get_keyword_sentences(review, food_keywords)


def calculate_food_score(review: str, analyzer: SentimentIntensityAnalyzer) -> float:
    """Calculate review-level food score."""

    return calculate_aspect_score(review, FOOD_KEYWORDS, analyzer)


def add_food_score(df: pd.DataFrame, analyzer: SentimentIntensityAnalyzer | None = None) -> pd.DataFrame:
    """Add restaurant-level food score and food mention count."""

    return calculate_aspect_columns(
        df,
        FOOD_KEYWORDS,
        "food_scores_review",
        "food_score_raw",
        "food_score",
        "food_mentions",
        analyzer,
    )
