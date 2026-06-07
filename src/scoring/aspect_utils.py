"""Shared helpers for aspect-based VADER scoring."""

from __future__ import annotations

import numpy as np
import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import sent_tokenize


def get_keyword_sentences(review: str, keywords: set[str]) -> list[str]:
    """Return sentences containing one or more aspect keywords."""

    aspect_sentences = []
    try:
        sentences = sent_tokenize(review)
    except LookupError:
        sentences = [sentence.strip() for sentence in review.split(".") if sentence.strip()]

    for sentence in sentences:
        sentence_lower = sentence.lower()
        if any(keyword in sentence_lower for keyword in keywords):
            aspect_sentences.append(sentence)

    return aspect_sentences


def calculate_aspect_score(
    review: str,
    keywords: set[str],
    analyzer: SentimentIntensityAnalyzer,
) -> float:
    """Calculate the mean VADER compound score for matching aspect sentences."""

    aspect_sentences = get_keyword_sentences(review, keywords)
    if len(aspect_sentences) == 0:
        return np.nan

    scores = [analyzer.polarity_scores(sentence)["compound"] for sentence in aspect_sentences]
    return float(np.mean(scores))


def aggregate_aspect_score(scores: list[float]) -> float:
    """Aggregate review-level aspect scores into a restaurant-level score."""

    valid_scores = [score for score in scores if pd.notna(score)]
    if len(valid_scores) == 0:
        return np.nan
    return float(np.mean(valid_scores))


def count_keyword_mentions(review: str, keywords: set[str]) -> int:
    """Count aspect keyword mentions using the notebook's boolean keyword checks."""

    review_lower = review.lower()
    return sum(keyword in review_lower for keyword in keywords)


def calculate_aspect_columns(
    df: pd.DataFrame,
    keywords: set[str],
    review_score_column: str,
    raw_score_column: str,
    final_score_column: str,
    mention_column: str,
    analyzer: SentimentIntensityAnalyzer | None = None,
) -> pd.DataFrame:
    """Add review-level, raw, normalized, and mention-count aspect features."""

    scored = df.copy()
    sia = analyzer or SentimentIntensityAnalyzer()
    scored[review_score_column] = scored["review_texts"].apply(
        lambda reviews: [calculate_aspect_score(review, keywords, sia) for review in reviews]
    )
    scored[raw_score_column] = scored[review_score_column].apply(aggregate_aspect_score)
    scored[final_score_column] = (scored[raw_score_column] + 1) / 2
    scored[mention_column] = scored["review_texts"].apply(
        lambda reviews: sum(count_keyword_mentions(review, keywords) for review in reviews)
    )
    return scored
