from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[2]

load_dotenv(
    PROJECT_ROOT / ".env",
    override=True
)


@dataclass(frozen=True)
class Settings:
    raw_dataset_path: Path
    cleaned_dataset_path: Path
    enriched_dataset_path: Path
    location_mapping_path: Path

    openai_api_key: str | None
    openai_model: str
    openai_max_retries: int


def get_settings() -> Settings:

    return Settings(
        raw_dataset_path=PROJECT_ROOT / "data" / "raw" / "zomato_bangalore.csv",
        cleaned_dataset_path=PROJECT_ROOT / "data" / "processed" / "restaurants_cleaned.csv",
        enriched_dataset_path=PROJECT_ROOT / "data" / "processed" / "restaurants_enriched.csv",
        location_mapping_path=PROJECT_ROOT / "data" / "processed" / "location_city_mapping.csv",

        openai_api_key=os.getenv("OPENAI_API_KEY"),

        openai_model=os.getenv(
            "OPENAI_MODEL",
            "gpt-4.1-mini"
        ),

        openai_max_retries=int(
            os.getenv(
                "OPENAI_MAX_RETRIES",
                "3"
            )
        )
    )