"""Review extraction and cleaning logic from the review notebook."""

from __future__ import annotations

import ast
import re

import numpy as np
import pandas as pd


def extract_reviews_and_ratings(review_list: object) -> pd.Series:
    """Extract numeric ratings and review texts from the raw reviews list field."""

    if pd.isna(review_list):
        return pd.Series([[], []])

    try:
        reviews = ast.literal_eval(str(review_list))
        ratings = []
        review_texts = []

        for review in reviews:
            if len(review) < 2:
                continue

            rating_text = review[0]
            review_text = review[1]
            match = re.search(r"(\d+(?:\.\d+)?)", rating_text)

            if match:
                ratings.append(float(match.group(1)))
            else:
                ratings.append(np.nan)

            review_texts.append(review_text)

        return pd.Series([ratings, review_texts])
    except (ValueError, SyntaxError, TypeError):
        return pd.Series([[], []])


def extract_review_ratings(review_list: object) -> list[float]:
    """Return only extracted review ratings."""

    return extract_reviews_and_ratings(review_list).iloc[0]


def extract_review_texts(review_list: object) -> list[str]:
    """Return only extracted review texts."""

    return extract_reviews_and_ratings(review_list).iloc[1]


def remove_rated(text: str) -> str:
    """Remove the RATED marker from review text."""

    return re.sub(r"\bRATED\b", "", text, flags=re.IGNORECASE)


def remove_html(text: str) -> str:
    """Remove HTML tags from review text."""

    return re.sub(r"<.*?>", " ", text)


def remove_urls(text: str) -> str:
    """Remove URLs from review text."""

    return re.sub(r"http\S+|www\S+", "", text)


def remove_emoji(text: str) -> str:
    """Remove emoji codepoint ranges used in the notebook."""

    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"
        "\U0001F300-\U0001F5FF"
        "\U0001F680-\U0001F6FF"
        "\U0001F1E0-\U0001F1FF"
        "]+",
        flags=re.UNICODE,
    )
    return emoji_pattern.sub("", text)


def remove_special_chars(text: str) -> str:
    """Remove characters outside the notebook's allowed review character set."""

    return re.sub(r"[^a-zA-Z0-9\s,.!?]", " ", text)


def normalize_spaces(text: str) -> str:
    """Normalize repeated whitespace."""

    return re.sub(r"\s+", " ", text).strip()


def clean_review_text(text: object) -> str:
    """Run the review text cleaning pipeline from the notebook."""

    cleaned = str(text)
    cleaned = remove_rated(cleaned)
    cleaned = remove_html(cleaned)
    cleaned = remove_urls(cleaned)
    cleaned = remove_emoji(cleaned)
    cleaned = remove_special_chars(cleaned)
    cleaned = cleaned.lower()
    cleaned = normalize_spaces(cleaned)
    return cleaned


def clean_review_texts(reviews: list[object]) -> list[str]:
    """Clean all review texts for one restaurant row."""

    return [clean_review_text(review) for review in reviews]


def calculate_review_count(reviews: list[object]) -> int:
    """Return the number of reviews in a cleaned review list."""

    return len(reviews)


def add_review_processing_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add review ratings, cleaned review texts, and review counts."""

    processed = df.copy()
    processed[["review_ratings", "review_texts"]] = processed["reviews_list"].apply(
        extract_reviews_and_ratings
    )
    processed["review_texts"] = processed["review_texts"].apply(clean_review_texts)
    processed["review_count"] = processed["review_texts"].apply(calculate_review_count)
    return processed
