"""Sentiment scoring extracted from the review intelligence notebook."""

from __future__ import annotations

import numpy as np
import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer


def get_sentiment_score(review: object, analyzer: SentimentIntensityAnalyzer) -> float:
    """Return VADER compound sentiment for a single review."""

    if not isinstance(review, str):
        return 0
    return analyzer.polarity_scores(review)["compound"]


def calculate_sentiment_scores(
    df: pd.DataFrame,
    analyzer: SentimentIntensityAnalyzer | None = None,
) -> pd.DataFrame:
    """Add review-level and normalized restaurant-level sentiment scores."""

    scored = df.copy()
    sia = analyzer or SentimentIntensityAnalyzer()
    sentiment_cache: dict[str, float] = {}

    def cached_sentiment(review: object) -> float:
        if not isinstance(review, str):
            return 0
        if review not in sentiment_cache:
            sentiment_cache[review] = get_sentiment_score(review, sia)
        return sentiment_cache[review]

    scored["review_sentiments"] = scored["review_texts"].apply(
        lambda reviews: [cached_sentiment(review) for review in reviews]
    )
    scored["sentiment_score"] = scored["review_sentiments"].apply(
        lambda scores: np.mean(scores) if len(scores) > 0 else np.nan
    )
    scored["sentiment_score"] = (scored["sentiment_score"] + 1) / 2
    return scored
