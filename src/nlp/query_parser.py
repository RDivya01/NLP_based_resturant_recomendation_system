"""OpenAI-backed natural language query parser."""

from __future__ import annotations

import json
import re
import time
from typing import Any

from openai import OpenAI
from pydantic import BaseModel, Field

from src.config.settings import get_settings


SUPPORTED_PRIORITIES = {
    "food",
    "ambiance",
    "service",
    "authenticity",
    "value_for_money",
    "popularity",
    "general",
}


class ParsedQuery(BaseModel):
    """Structured restaurant search filters extracted from a user query."""

    location: str | None = None
    cuisine: str | None = None
    budget: int | None = None
    online_order: bool | None = None
    book_table: bool | None = None
    priority: str = Field(default="general")


def _build_prompt(user_query: str) -> str:
    """Build the notebook prompt for extracting recommendation filters."""

    return f"""
You are an NLP engine for a restaurant recommendation system.

Extract the following fields from the user query.

Return ONLY valid JSON.

Fields:
- location
- cuisine
- budget
- online_order
- book_table
- priority

Priority can be one of:
food
ambiance
service
authenticity
value_for_money
popularity
general

If a field is not mentioned return null.

User Query:
{user_query}
"""


def _loads_json_response(response_text: str) -> dict[str, Any]:
    """Parse model JSON, removing markdown fences if present."""

    cleaned_text = re.sub(r"```json|```", "", response_text).strip()
    return json.loads(cleaned_text)


def _normalize_parsed_query(parsed: dict[str, Any]) -> dict[str, Any]:
    """Normalize parser output to the expected keys and supported priorities."""

    if parsed.get("priority") is None:
        parsed["priority"] = "general"

    model = ParsedQuery(**parsed)

    result = (
        model.model_dump()
        if hasattr(model, "model_dump")
        else model.dict()
    )

    priority = result.get("priority")
    result["priority"] = (
        priority
        if priority in SUPPORTED_PRIORITIES
        else "general"
    )

    return result


def parse_user_query(
    user_query: str,
    client: OpenAI | None = None,
    model: str | None = None,
    max_retries: int | None = None,
) -> dict[str, Any]:
    """Parse a natural language query into structured recommendation filters."""

    settings = get_settings()
    openai_client = client or OpenAI(api_key=settings.openai_api_key)
    model_name = model or settings.openai_model
    retries = max_retries or settings.openai_max_retries
    prompt = _build_prompt(user_query)

    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            response = openai_client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": "Return only JSON."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0,
                response_format={"type": "json_object"},
            )
            response_text = response.choices[0].message.content or "{}"
            return _normalize_parsed_query(_loads_json_response(response_text))
        except Exception as exc:  # pragma: no cover - retry path depends on OpenAI SDK/runtime.
            last_error = exc
            if attempt < retries - 1:
                time.sleep(2**attempt)

    raise RuntimeError("Failed to parse user query with OpenAI.") from last_error
