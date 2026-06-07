"""Thin API routes for the recommendation pipeline."""

from __future__ import annotations

from fastapi import APIRouter

from api.schemas import RecommendationRequest, RecommendationResponse
from src.pipelines.recommendation_pipeline import recommend_restaurants
import os

router = APIRouter()


@router.post("/recommend", response_model=RecommendationResponse)
def recommend(request: RecommendationRequest) -> RecommendationResponse:
    """Return restaurant recommendations for a natural-language query."""
    
    result = recommend_restaurants(request.query, top_n=request.top_n)
    return RecommendationResponse(**result)
