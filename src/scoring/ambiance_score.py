"""Ambiance scoring from the review intelligence notebook."""

from __future__ import annotations

import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer

from src.scoring.aspect_utils import calculate_aspect_columns, calculate_aspect_score, get_keyword_sentences


AMBIANCE_KEYWORDS = {
    "ambience",
    "ambiance",
    "atmosphere",
    "decor",
    "interior",
    "interiors",
    "lighting",
    "music",
    "seating",
    "view",
    "environment",
    "vibe",
    "vibes",
    "crowd",
    "rooftop",
    "peaceful",
    "cozy",
    "comfortable",
    "romantic",
    "spacious",
}


def get_ambiance_sentences(review: str) -> list[str]:
    """Return ambiance-related sentences."""

    return get_keyword_sentences(review, AMBIANCE_KEYWORDS)


def calculate_ambiance_score(review: str, analyzer: SentimentIntensityAnalyzer) -> float:
    """Calculate review-level ambiance score."""

    return calculate_aspect_score(review, AMBIANCE_KEYWORDS, analyzer)


def add_ambiance_score(df: pd.DataFrame, analyzer: SentimentIntensityAnalyzer | None = None) -> pd.DataFrame:
    """Add restaurant-level ambiance score and mention count."""

    return calculate_aspect_columns(
        df,
        AMBIANCE_KEYWORDS,
        "ambiance_scores_review",
        "ambiance_score_raw",
        "ambiance_score",
        "ambiance_mentions",
        analyzer,
    )
