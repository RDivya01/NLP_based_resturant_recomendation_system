"""Intent-specific ranking weights extracted from the NLP notebook."""

INTENT_WEIGHTS: dict[str, dict[str, float]] = {
    "food": {
        "food_score": 0.40,
        "sentiment_score": 0.20,
        "popularity_score": 0.15,
        "service_score": 0.10,
        "ambiance_score": 0.05,
        "authenticity_score": 0.05,
        "value_for_money_score": 0.05,
    },
    "authenticity": {
        "authenticity_score": 0.40,
        "food_score": 0.25,
        "sentiment_score": 0.15,
        "service_score": 0.10,
        "popularity_score": 0.10,
    },
    "ambiance": {
        "ambiance_score": 0.40,
        "food_score": 0.25,
        "service_score": 0.15,
        "sentiment_score": 0.10,
        "popularity_score": 0.10,
    },
    "service": {
        "service_score": 0.40,
        "food_score": 0.20,
        "sentiment_score": 0.15,
        "ambiance_score": 0.10,
        "popularity_score": 0.15,
    },
    "value_for_money": {
        "value_for_money_score": 0.40,
        "food_score": 0.20,
        "sentiment_score": 0.15,
        "popularity_score": 0.15,
        "service_score": 0.10,
    },
    "popularity": {
        "popularity_score": 0.40,
        "sentiment_score": 0.20,
        "food_score": 0.20,
        "service_score": 0.10,
        "ambiance_score": 0.10,
    },
    "general": {
        "food_score": 0.25,
        "sentiment_score": 0.20,
        "ambiance_score": 0.15,
        "service_score": 0.10,
        "authenticity_score": 0.10,
        "value_for_money_score": 0.10,
        "popularity_score": 0.10,
    },
}
