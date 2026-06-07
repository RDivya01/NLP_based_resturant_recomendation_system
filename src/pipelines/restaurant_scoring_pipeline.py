"""Offline review intelligence and restaurant scoring pipeline."""

from __future__ import annotations

import nltk
import pandas as pd

from src.data.feature_store import load_cleaned_dataset, save_enriched_dataset
from src.scoring.aspect_scoring import add_all_aspect_scores
from src.scoring.popularity import calculate_popularity_score
from src.scoring.review_processing import add_review_processing_features
from src.scoring.sentiment import calculate_sentiment_scores
from src.utils.logger import get_logger


logger = get_logger(__name__)

ASPECT_COLUMNS = [
    "sentiment_score",
    "food_score",
    "ambiance_score",
    "service_score",
    "authenticity_score",
    "value_for_money_score",
]

DROP_COLUMNS = [
    "reviews_list",
    "value_for_money_score_raw",
    "value_scores_review",
    "service_score_raw",
    "service_scores_review",
    "authenticity_score_raw",
    "authenticity_scores_review",
    "ambiance_score_raw",
    "ambiance_scores_review",
    "food_score_raw",
    "food_scores_review",
    "review_sentiments",
    "review_ratings",
    "review_texts"
]


def ensure_nltk_resources() -> None:
    """Ensure the NLTK resources used by VADER and sentence tokenization exist."""

    for package in ("vader_lexicon", "punkt", "punkt_tab"):
        try:
            if package in {"punkt", "punkt_tab"}:
                nltk.data.find(f"tokenizers/{package}")
            else:
                nltk.data.find(f"sentiment/{package}.zip")
        except LookupError:
            nltk.download(package, quiet=True)


def run_restaurant_scoring_pipeline(cleaned_df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Generate data/processed/restaurants_enriched.csv from cleaned restaurants."""

    ensure_nltk_resources()
    scored = cleaned_df.copy() if cleaned_df is not None else load_cleaned_dataset()
    scored = add_review_processing_features(scored)
    scored = calculate_sentiment_scores(scored)
    scored = calculate_popularity_score(scored)
    scored = add_all_aspect_scores(scored)
    scored = scored.drop(columns=DROP_COLUMNS, errors="ignore")
    scored[ASPECT_COLUMNS] = scored[ASPECT_COLUMNS].fillna(0.5)
    save_enriched_dataset(scored)
    logger.info("Saved enriched dataset")
    return scored


if __name__ == "__main__":
    run_restaurant_scoring_pipeline()
