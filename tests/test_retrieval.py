"""Tests for candidate retrieval."""

from __future__ import annotations

import pandas as pd

from src.retrieval.candidate_retriever import retrieve_candidates


def test_retrieve_candidates_filters_location_cuisine_budget_and_flags() -> None:
    """Candidate retrieval should apply all notebook filters."""

    df = pd.DataFrame(
        {
            "name": ["A", "B"],
            "location": ["Indiranagar", "Jayanagar"],
            "cuisines": [["South Indian", "Chinese"], ["Italian"]],
            "approx_cost(for two people)": [700, 1000],
            "online_order": [1, 0],
            "book_table": [0, 1],
        }
    )

    result = retrieve_candidates(
        df,
        {
            "location": "indira",
            "cuisine": "south indian",
            "budget": 800,
            "online_order": True,
            "book_table": False,
        },
    )

    assert result["name"].tolist() == ["A"]
