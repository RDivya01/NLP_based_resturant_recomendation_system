"""Service scoring from the review intelligence notebook."""

from __future__ import annotations

import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer

from src.scoring.aspect_utils import calculate_aspect_columns, calculate_aspect_score, get_keyword_sentences


SERVICE_KEYWORDS = {
    "service",
    "staff",
    "waiter",
    "waiters",
    "waitress",
    "manager",
    "hospitality",
    "courteous",
    "friendly",
    "attentive",
    "helpful",
    "professional",
    "polite",
    "behavior",
    "behaviour",
    "served",
    "serving",
    "server",
    "customer service",
}


def get_service_sentences(review: str) -> list[str]:
    """Return service-related sentences."""

    return get_keyword_sentences(review, SERVICE_KEYWORDS)


def calculate_service_score(review: str, analyzer: SentimentIntensityAnalyzer) -> float:
    """Calculate review-level service score."""

    return calculate_aspect_score(review, SERVICE_KEYWORDS, analyzer)


def add_service_score(df: pd.DataFrame, analyzer: SentimentIntensityAnalyzer | None = None) -> pd.DataFrame:
    """Add restaurant-level service score and mention count."""

    return calculate_aspect_columns(
        df,
        SERVICE_KEYWORDS,
        "service_scores_review",
        "service_score_raw",
        "service_score",
        "service_mentions",
        analyzer,
    )
