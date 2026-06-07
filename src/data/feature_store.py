"""CSV-backed feature-store helpers for cleaned and enriched datasets."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config.settings import get_settings
from src.utils.io import read_csv, write_csv


def load_cleaned_dataset(path: str | Path | None = None) -> pd.DataFrame:
    """Load the cleaned restaurant dataset."""

    settings = get_settings()
    return read_csv(path or settings.cleaned_dataset_path)


def load_enriched_dataset(path: str | Path | None = None) -> pd.DataFrame:
    """Load the enriched restaurant feature dataset."""

    settings = get_settings()
    return read_csv(path or settings.enriched_dataset_path)


def save_cleaned_dataset(df: pd.DataFrame, path: str | Path | None = None) -> None:
    """Save the cleaned restaurant dataset."""

    settings = get_settings()
    write_csv(df, path or settings.cleaned_dataset_path)


def save_enriched_dataset(df: pd.DataFrame, path: str | Path | None = None) -> None:
    """Save the enriched restaurant feature dataset."""

    settings = get_settings()
    write_csv(df, path or settings.enriched_dataset_path)
