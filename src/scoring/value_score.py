"""Value-for-money scoring from the review intelligence notebook."""

from __future__ import annotations

import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer

from src.scoring.aspect_utils import calculate_aspect_columns, calculate_aspect_score, get_keyword_sentences


VALUE_KEYWORDS = {
    "worth",
    "worth it",
    "worth every penny",
    "affordable",
    "reasonable",
    "budget",
    "economical",
    "cheap",
    "value for money",
    "pocket friendly",
    "good value",
    "expensive",
    "overpriced",
    "costly",
    "pricey",
    "not worth",
    "waste of money",
}


def get_value_sentences(review: str) -> list[str]:
    """Return value-related sentences."""

    return get_keyword_sentences(review, VALUE_KEYWORDS)


def calculate_value_score(review: str, analyzer: SentimentIntensityAnalyzer) -> float:
    """Calculate review-level value-for-money score."""

    return calculate_aspect_score(review, VALUE_KEYWORDS, analyzer)


def add_value_score(df: pd.DataFrame, analyzer: SentimentIntensityAnalyzer | None = None) -> pd.DataFrame:
    """Add restaurant-level value-for-money score and mention count."""

    return calculate_aspect_columns(
        df,
        VALUE_KEYWORDS,
        "value_scores_review",
        "value_for_money_score_raw",
        "value_for_money_score",
        "value_mentions",
        analyzer,
    )
