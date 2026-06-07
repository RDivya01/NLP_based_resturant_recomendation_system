"""Tests for review processing and score generation."""

from __future__ import annotations

import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer

from src.scoring.review_processing import clean_review_text, extract_reviews_and_ratings
from src.scoring.sentiment import get_sentiment_score


def test_extract_reviews_and_ratings() -> None:
    """Review extraction should parse notebook-style review tuples."""

    ratings, reviews = extract_reviews_and_ratings("[('Rated 4.0', 'Great food'), ('Rated 3.5', 'Good service')]")

    assert ratings == [4.0, 3.5]
    assert reviews == ["Great food", "Good service"]


def test_clean_review_text() -> None:
    """Review cleaning should remove notebook-defined noise."""

    assert clean_review_text("RATED <b>Great!</b> https://x.test") == "great!"


def test_sentiment_score_uses_vader() -> None:
    """Positive text should receive positive VADER compound sentiment."""

    analyzer = SentimentIntensityAnalyzer()
    assert get_sentiment_score("excellent delicious food", analyzer) > 0
