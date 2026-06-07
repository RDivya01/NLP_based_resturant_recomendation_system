"""Online recommendation pipeline."""

from __future__ import annotations

from typing import Any

import pandas as pd
import ast
from src.data.feature_store import load_enriched_dataset
from src.nlp.intent_analyzer import analyze_intent
from src.nlp.query_parser import parse_user_query
from src.ranking.explainability import add_recommendation_reasons
from src.ranking.ranker import get_top_n, rank_restaurants
from src.retrieval.candidate_retriever import retrieve_candidates

def clean_phone(phone):

    if pd.isna(phone):
        return None

    return (
        str(phone)
        .replace("\r", "")
        .replace("\n", ", ")
        .strip()
    )

def recommend_restaurants(
    user_query: str,
    top_n: int = 10,
    restaurant_df: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Run query parsing, retrieval, ranking, and explainability for one request."""

    restaurants = restaurant_df.copy() if restaurant_df is not None else load_enriched_dataset()
    parsed_query = parse_user_query(user_query)
    intent = analyze_intent(parsed_query)
    candidates = retrieve_candidates(restaurants, parsed_query)
    ranked = rank_restaurants(candidates, intent["weights"])
    ranked = (
        ranked
        .sort_values(
            "final_score",
            ascending=False
        )
        .drop_duplicates(
            subset=["name"],
            keep="first"
        )
    )
    recommendations = add_recommendation_reasons(get_top_n(ranked, top_n))
    response_columns = [
        "name",
        "address",
        "phone",
        "location",
        "cuisines",
        "rest_type",
        "rate",
        "votes",
        "approx_cost(for two people)",
        "online_order",
        "book_table",
        "final_score",
        "reasons"
    ]
    recommendations = (
        recommendations[response_columns]
        .rename(
            columns={
                "rate": "rating",
                "approx_cost(for two people)": "cost_for_two",
                "final_score": "recommendation_score"
            }
        )
    )
    recommendations["cuisines"] = (
        recommendations["cuisines"]
        .apply(
            lambda x:
            ast.literal_eval(x)
            if isinstance(x, str)
            else x
        )
    )
    recommendations["recommendation_score"] = (
        recommendations["recommendation_score"]
        .round(3)
    )
    recommendations["phone"] = (
        recommendations["phone"]
        .apply(clean_phone)
    )
    recommendations["address"] = (
        recommendations["address"]
        .astype(str)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )
    return {
        "recommendations": recommendations.to_dict(orient="records"),
    }
