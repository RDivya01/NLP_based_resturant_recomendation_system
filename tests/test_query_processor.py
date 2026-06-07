"""Tests for query parsing normalization and intent analysis."""

from __future__ import annotations

from src.nlp.intent_analyzer import analyze_intent
from src.nlp.query_parser import _normalize_parsed_query


def test_normalize_parsed_query_defaults_unknown_priority() -> None:
    """Unknown model priority should fall back to general."""

    parsed = _normalize_parsed_query(
        {
            "location": "Jayanagar",
            "cuisine": "South Indian",
            "budget": 800,
            "online_order": None,
            "book_table": None,
            "priority": "romantic",
        }
    )

    assert parsed["priority"] == "general"


def test_analyze_intent_uses_configured_weights() -> None:
    """Intent analyzer should read weights from config."""

    result = analyze_intent({"priority": "food"})

    assert result["intent"] == "food"
    assert result["weights"]["food_score"] == 0.40
