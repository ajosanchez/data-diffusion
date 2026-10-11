"""User-facing generator API (v0 scaffold stub)."""

import pandas as pd


class TabularDiffusion:
    """XGBoost flow-matching generator for tabular DataFrames.

    Encodes mixed-type columns to numbers, trains per-(level, column) XGBoost
    regressors on the flow-matching path, then decodes sampled rows back to
    the original columns and dtypes.
    """

    def fit(self, df: pd.DataFrame) -> "TabularDiffusion":
        """Train the generator on a mixed-type DataFrame."""
        raise NotImplementedError

    def sample(self, n_rows: int, seed: int = 0) -> pd.DataFrame:
        """Sample `n_rows` synthetic rows matching the training dtypes."""
        raise NotImplementedError
