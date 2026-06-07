"""Offline data cleaning pipeline."""

from __future__ import annotations

import pandas as pd

from src.config.settings import get_settings
from src.data.cleaning import clean_dataset
from src.data.feature_store import save_cleaned_dataset
from src.data.imputation import impute_cleaned_dataset
from src.utils.io import write_csv
from src.utils.logger import get_logger


logger = get_logger(__name__)


def run_data_cleaning_pipeline(raw_df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Generate data/processed/restaurants_cleaned.csv from the raw Zomato dataset."""

    settings = get_settings()
    restaurant_df = raw_df if raw_df is not None else pd.read_csv(settings.raw_dataset_path)

    if {"address", "location", "listed_in(city)"}.issubset(restaurant_df.columns):
        location_mapping = restaurant_df[["address", "location", "listed_in(city)"]]
        write_csv(location_mapping, settings.location_mapping_path)

    cleaned = clean_dataset(restaurant_df)
    cleaned = impute_cleaned_dataset(cleaned)
    save_cleaned_dataset(cleaned)
    logger.info("Saved cleaned dataset to %s", settings.cleaned_dataset_path)
    return cleaned


if __name__ == "__main__":
    run_data_cleaning_pipeline()
