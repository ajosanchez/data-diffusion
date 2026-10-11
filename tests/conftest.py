"""Shared test fixtures."""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def toy_df() -> pd.DataFrame:
    """1000-row mixed-type frame without missing values."""
    rng = np.random.default_rng(7)
    n = 1000
    num = rng.normal(loc=0.0, scale=1.0, size=n)
    count = rng.integers(0, 10, size=n)
    flag = rng.integers(0, 2, size=n).astype(bool)
    color = rng.choice(["red", "green", "blue"], size=n, p=[0.5, 0.3, 0.2])
    day = pd.Series(
        pd.to_datetime("2023-01-01")
        + pd.to_timedelta(rng.integers(0, 730, size=n), unit="D"),
        name="day",
    )
    return pd.DataFrame(
        {"num": num, "count": count, "flag": flag, "color": color, "day": day}
    )
