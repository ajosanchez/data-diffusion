"""High-level API: fit on a DataFrame, sample synthetic rows."""

from __future__ import annotations

from typing import Any

import pandas as pd

from data_diffusion.encoder import TabularEncoder
from data_diffusion.forestflow import (
    ForestFlowModel,
    fit_forest_flow,
    sample_forest_flow,
)


class TabularDiffusion:
    """Scikit-learn style wrapper: encode DataFrame → ForestFlow → decode.

    Args:
        n_t: Number of flow t levels in [0, 1).
        K: Duplication factor; each training row is paired with K noise draws.
        n_jobs: joblib workers for the per-(t, column) booster fits.
        seed: Seed for the fit-time noise draws and row resampling.
        rare_min_count: Categorical levels with fewer rows merge into "Other".
        rare_min_fraction: Categorical levels with a smaller row share than
            this merge into "Other".
        xgb_kwargs: Overrides for XGBoost booster parameters.
    """

    def __init__(
        self,
        n_t: int = 50,
        K: int = 100,
        n_jobs: int = 1,
        seed: int = 0,
        rare_min_count: int = 20,
        rare_min_fraction: float = 0.01,
        xgb_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self.n_t = n_t
        self.K = K
        self.n_jobs = n_jobs
        self.seed = seed
        self.rare_min_count = rare_min_count
        self.rare_min_fraction = rare_min_fraction
        self.xgb_kwargs = xgb_kwargs
        self.encoder_: TabularEncoder | None = None
        self.model_: ForestFlowModel | None = None
        self.feature_names_: list[str] = []

    def fit(self, df: pd.DataFrame) -> TabularDiffusion:
        """Infer column types, encode, and fit the ForestFlow models."""
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"fit expects a pandas DataFrame, got {type(df).__name__}")
        encoder = TabularEncoder(
            rare_min_count=self.rare_min_count, rare_min_fraction=self.rare_min_fraction
        )
        x = encoder.fit_transform(df)
        self.encoder_ = encoder
        self.feature_names_ = list(encoder.feature_names)
        self.model_ = fit_forest_flow(
            x,
            n_t=self.n_t,
            K=self.K,
            n_jobs=self.n_jobs,
            xgb_kwargs=self.xgb_kwargs,
            seed=self.seed,
        )
        return self

    def sample(self, n_rows: int, seed: int = 0) -> pd.DataFrame:
        """Draw `n_rows` synthetic rows with training columns and dtypes."""
        encoder = self.encoder_
        model = self.model_
        if encoder is None or model is None:
            raise ValueError("model is not fitted; call fit() first")
        x = sample_forest_flow(model, n_rows, seed=seed)
        return encoder.inverse_transform(x)
