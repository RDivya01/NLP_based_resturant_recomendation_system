"""Authenticity scoring from the review intelligence notebook."""

from __future__ import annotations

import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer

from src.scoring.aspect_utils import calculate_aspect_columns, calculate_aspect_score, get_keyword_sentences


AUTHENTICITY_KEYWORDS = {
    "authentic",
    "traditional",
    "original",
    "genuine",
    "heritage",
    "classic",
    "homestyle",
    "home style",
    "homemade",
    "native",
    "local cuisine",
    "real",
    "typical",
    "true taste",
    "signature",
    "andhra style",
    "south indian style",
    "north indian style",
}


def get_authenticity_sentences(review: str) -> list[str]:
    """Return authenticity-related sentences."""

    return get_keyword_sentences(review, AUTHENTICITY_KEYWORDS)


def calculate_authenticity_score(review: str, analyzer: SentimentIntensityAnalyzer) -> float:
    """Calculate review-level authenticity score."""

    return calculate_aspect_score(review, AUTHENTICITY_KEYWORDS, analyzer)


def add_authenticity_score(df: pd.DataFrame, analyzer: SentimentIntensityAnalyzer | None = None) -> pd.DataFrame:
    """Add restaurant-level authenticity score and mention count."""

    return calculate_aspect_columns(
        df,
        AUTHENTICITY_KEYWORDS,
        "authenticity_scores_review",
        "authenticity_score_raw",
        "authenticity_score",
        "authenticity_mentions",
        analyzer,
    )
