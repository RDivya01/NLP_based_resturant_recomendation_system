"""Restaurant ranking engine."""

from __future__ import annotations

import numpy as np
import pandas as pd


def calculate_weighted_score(candidate_df: pd.DataFrame, weights: dict[str, float]) -> pd.Series:
    """Calculate the notebook weighted sum over available feature columns."""

    weighted_score = pd.Series(0.0, index=candidate_df.index)

    for feature, weight in weights.items():
        if feature in candidate_df.columns:
            weighted_score += candidate_df[feature].fillna(0.5) * weight

    return weighted_score


def calculate_confidence_score(candidate_df: pd.DataFrame) -> pd.Series:
    """Calculate confidence from review_count using log normalization."""

    if "review_count" not in candidate_df.columns or candidate_df.empty:
        return pd.Series(1.0, index=candidate_df.index)

    votes = candidate_df["votes"].fillna(0).clip(lower=0)
    confidence = np.log1p(votes)
    max_confidence = confidence.max()

    if max_confidence == 0:
        return pd.Series(1.0, index=candidate_df.index)

    return confidence / max_confidence


def rank_restaurants(candidate_df: pd.DataFrame, weights: dict[str, float]) -> pd.DataFrame:
    """Rank restaurants using weighted feature score and review-count confidence."""

    ranked = candidate_df.copy()
    ranked["weighted_score"] = calculate_weighted_score(ranked, weights)
    ranked["confidence_score"] = calculate_confidence_score(ranked)
    ranked["final_score"] = (
        ranked["weighted_score"]
        *
        (
            0.7 +
            0.3 * ranked["confidence_score"]
        )
    )
    return ranked.sort_values("final_score", ascending=False).reset_index(drop=True)


def get_top_n(ranked_df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """Return the top N ranked restaurants."""

    return ranked_df.head(n)
