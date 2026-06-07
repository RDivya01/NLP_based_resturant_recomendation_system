"""Popularity scoring from the review intelligence notebook."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def calculate_popularity_score(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate normalized popularity score as rate * log1p(votes)."""

    scored = df.copy()
    scored["popularity_score"] = scored["rate"] * np.log1p(scored["votes"])
    scaler = MinMaxScaler()
    scored["popularity_score"] = scaler.fit_transform(scored[["popularity_score"]])
    return scored
