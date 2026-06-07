"""Intent analysis using configured ranking weights."""

from __future__ import annotations

from typing import Any

from src.config.intent_weights import INTENT_WEIGHTS


def analyze_intent(parsed_query: dict[str, Any]) -> dict[str, Any]:
    """Map parsed priority into configured ranking weights."""

    priority = parsed_query.get("priority", "general")
    weights = INTENT_WEIGHTS.get(priority, INTENT_WEIGHTS["general"])
    return {
        "intent": priority,
        "weights": weights,
    }
