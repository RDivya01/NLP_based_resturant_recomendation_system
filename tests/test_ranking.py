"""Tests for restaurant ranking."""

from __future__ import annotations

import pandas as pd

from src.ranking.ranker import rank_restaurants


def test_rank_restaurants_adds_required_score_columns() -> None:
    """Ranking should expose weighted, confidence, and final scores."""

    df = pd.DataFrame(
        {
            "name": ["A", "B"],
            "food_score": [0.9, 0.6],
            "sentiment_score": [0.8, 0.7],
            "review_count": [100, 5],
        }
    )

    ranked = rank_restaurants(df, {"food_score": 0.7, "sentiment_score": 0.3})

    assert {"weighted_score", "confidence_score", "final_score"}.issubset(ranked.columns)
    assert ranked.iloc[0]["name"] == "A"
