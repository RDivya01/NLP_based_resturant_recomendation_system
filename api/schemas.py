"""API request and response schemas."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

class RestaurantRecommendation(BaseModel):

    name: str

    address: str | None = None

    phone: str | None = None

    location: str | None = None

    cuisines: list[str] = []

    rest_type: str | None = None

    rating: float | None = None

    votes: int | None = None

    cost_for_two: float | None = None

    online_order: bool | None = None

    book_table: bool | None = None

    recommendation_score: float

    reasons: list[str] = []


class RecommendationRequest(BaseModel):
    """Request body for restaurant recommendations."""

    query: str = Field(..., min_length=1)
    top_n: int = Field(default=10, ge=1, le=50)


class RecommendationResponse(BaseModel):
    """Response body returned by the recommendation endpoint."""

    recommendations: list[RestaurantRecommendation]
