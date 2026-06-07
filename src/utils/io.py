"""CSV input/output helpers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def read_csv(path: str | Path) -> pd.DataFrame:
    """Read a CSV file into a dataframe."""

    return pd.read_csv(path)


def write_csv(df: pd.DataFrame, path: str | Path) -> None:
    """Write a dataframe to CSV, creating parent directories when needed."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
