"""One-pass aspect scoring for the full offline pipeline."""

from __future__ import annotations

import numpy as np
import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import sent_tokenize

from src.scoring.ambiance_score import AMBIANCE_KEYWORDS
from src.scoring.authenticity_score import AUTHENTICITY_KEYWORDS
from src.scoring.food_score import FOOD_KEYWORDS
from src.scoring.service_score import SERVICE_KEYWORDS
from src.scoring.value_score import VALUE_KEYWORDS


ASPECTS = {
    "food": {
        "keywords": FOOD_KEYWORDS,
        "review_column": "food_scores_review",
        "raw_column": "food_score_raw",
        "score_column": "food_score",
        "mention_column": "food_mentions",
    },
    "ambiance": {
        "keywords": AMBIANCE_KEYWORDS,
        "review_column": "ambiance_scores_review",
        "raw_column": "ambiance_score_raw",
        "score_column": "ambiance_score",
        "mention_column": "ambiance_mentions",
    },
    "authenticity": {
        "keywords": AUTHENTICITY_KEYWORDS,
        "review_column": "authenticity_scores_review",
        "raw_column": "authenticity_score_raw",
        "score_column": "authenticity_score",
        "mention_column": "authenticity_mentions",
    },
    "service": {
        "keywords": SERVICE_KEYWORDS,
        "review_column": "service_scores_review",
        "raw_column": "service_score_raw",
        "score_column": "service_score",
        "mention_column": "service_mentions",
    },
    "value": {
        "keywords": VALUE_KEYWORDS,
        "review_column": "value_scores_review",
        "raw_column": "value_for_money_score_raw",
        "score_column": "value_for_money_score",
        "mention_column": "value_mentions",
    },
}


def _tokenize_sentences(review: str) -> list[str]:
    """Tokenize sentences with a simple fallback when NLTK data is unavailable."""

    try:
        return sent_tokenize(review)
    except LookupError:
        return [sentence.strip() for sentence in review.split(".") if sentence.strip()]


def _score_review_aspects(
    review: str,
    analyzer: SentimentIntensityAnalyzer,
) -> tuple[dict[str, float], dict[str, int]]:
    """Score all configured aspects for one review using notebook formulas."""

    review_lower = review.lower()
    scores = {aspect: np.nan for aspect in ASPECTS}
    mentions = {
        aspect: sum(keyword in review_lower for keyword in config["keywords"])
        for aspect, config in ASPECTS.items()
    }

    active_aspects = [aspect for aspect, count in mentions.items() if count > 0]
    if not active_aspects:
        return scores, mentions

    sentences = _tokenize_sentences(review)
    for aspect in active_aspects:
        keywords = ASPECTS[aspect]["keywords"]
        aspect_sentences = [
            sentence
            for sentence in sentences
            if any(keyword in sentence.lower() for keyword in keywords)
        ]
        if aspect_sentences:
            vader_scores = [
                analyzer.polarity_scores(sentence)["compound"]
                for sentence in aspect_sentences
            ]
            scores[aspect] = float(np.mean(vader_scores))

    return scores, mentions


def _aggregate_review_scores(review_scores: list[float]) -> float:
    """Aggregate review-level aspect scores to a restaurant-level raw score."""

    valid_scores = [score for score in review_scores if pd.notna(score)]
    if not valid_scores:
        return np.nan
    return float(np.mean(valid_scores))


def add_all_aspect_scores(
    df: pd.DataFrame,
    analyzer: SentimentIntensityAnalyzer | None = None,
) -> pd.DataFrame:
    """Add all aspect scores and mention counts in one pass over review text."""

    scored = df.copy()
    sia = analyzer or SentimentIntensityAnalyzer()
    aspect_cache: dict[str, tuple[dict[str, float], dict[str, int]]] = {}

    for config in ASPECTS.values():
        scored[config["review_column"]] = [[] for _ in range(len(scored))]
        scored[config["mention_column"]] = 0

    for index, reviews in scored["review_texts"].items():
        row_scores = {aspect: [] for aspect in ASPECTS}
        row_mentions = {aspect: 0 for aspect in ASPECTS}

        for review in reviews:
            if review not in aspect_cache:
                aspect_cache[review] = _score_review_aspects(review, sia)
            review_scores, review_mentions = aspect_cache[review]
            for aspect in ASPECTS:
                row_scores[aspect].append(review_scores[aspect])
                row_mentions[aspect] += review_mentions[aspect]

        for aspect, config in ASPECTS.items():
            scored.at[index, config["review_column"]] = row_scores[aspect]
            scored.at[index, config["mention_column"]] = row_mentions[aspect]

    for config in ASPECTS.values():
        scored[config["raw_column"]] = scored[config["review_column"]].apply(_aggregate_review_scores)
        scored[config["score_column"]] = (scored[config["raw_column"]] + 1) / 2

    return scored
