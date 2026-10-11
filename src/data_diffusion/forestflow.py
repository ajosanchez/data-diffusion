"""ForestFlow core: XGBoost flow-matching on encoded numeric matrices.

Training duplicates each row K times, pairs it with fresh Gaussian noise z,
and fits one XGBoost booster per (t level, column) to predict the flow
velocity v = x - z from the interpolated state x_t = (1 - t) z + t x.
Sampling integrates the learned velocity field from pure noise to data space
with Euler steps over the training t grid. Uses the native xgboost.train API
(no scikit-learn dependency).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import xgboost as xgb

type ForestFlowModel = dict[str, Any]

_DEFAULT_XGB: dict[str, Any] = {
    "objective": "reg:squarederror",
    "max_depth": 6,
    "tree_method": "hist",
    "n_jobs": 1,
}


def fit_forest_flow(
    x: np.ndarray,
    n_t: int = 50,
    K: int = 100,
    xgb_kwargs: dict[str, Any] | None = None,
    seed: int = 0,
) -> ForestFlowModel:
    """Fit per-(t, column) XGBoost velocity models on encoded data.

    Args:
        x: Encoded numeric matrix of shape (n_rows, n_features); no NaN.
        n_t: Number of t levels on the flow grid in [0, 1).
        K: Duplication factor; each row is paired with K noise draws.
        xgb_kwargs: Overrides merged onto the default XGBoost parameters
            (`n_estimators` is accepted here and mapped to `num_boost_round`).
        seed: Seed for noise draws and row resampling.

    Returns:
        Model bundle: {"models": {(t, col): Booster}, "ts": list[float],
        "n_features": int}.
    """
    x = np.ascontiguousarray(x, dtype="float64")
    if x.ndim != 2:
        raise ValueError(f"x must be a 2D array, got shape {x.shape}")
    n_rows, n_features = x.shape
    if n_rows == 0 or n_features == 0:
        raise ValueError("cannot fit on an empty array")
    if not np.isfinite(x).all():
        raise ValueError("x contains NaN or infinite values; encode/impute first")
    if n_t < 1:
        raise ValueError(f"n_t must be >= 1, got {n_t}")
    if K < 1:
        raise ValueError(f"K must be >= 1, got {K}")

    kwargs = dict(xgb_kwargs or {})
    n_estimators = kwargs.pop("n_estimators", 100)
    params = {**_DEFAULT_XGB, **kwargs}
    rng = np.random.default_rng(seed)
    ts = [i / n_t for i in range(n_t)]
    models: dict[tuple[float, int], xgb.Booster] = {}
    for t in ts:
        z = rng.normal(size=(n_rows * K, n_features))
        row_idx = rng.integers(0, n_rows, size=n_rows * K)
        x_ref = x[row_idx]
        x_t = (1.0 - t) * z + t * x_ref
        v = x_ref - z
        for j in range(n_features):
            dtrain = xgb.DMatrix(x_t[:, [j]], label=v[:, j])
            models[(t, j)] = xgb.train(
                params, dtrain, num_boost_round=int(n_estimators)
            )
    return {"models": models, "ts": ts, "n_features": n_features}


def sample_forest_flow(
    bundle: ForestFlowModel, n_rows: int, seed: int = 0
) -> np.ndarray:
    """Integrate the learned velocity field from noise to data space.

    Args:
        bundle: Bundle returned by `fit_forest_flow`.
        n_rows: Number of samples to draw (must be positive).
        seed: Seed for the starting noise.

    Returns:
        Synthetic matrix of shape (n_rows, n_features) in encoded space.
    """
    if n_rows < 1:
        raise ValueError(f"n_rows must be positive, got {n_rows}")
    ts: list[float] = bundle["ts"]
    n_features: int = bundle["n_features"]
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n_rows, n_features))
    dt = 1.0 / len(ts)
    for t in ts:
        for j in range(n_features):
            model = bundle["models"][(t, j)]
            dmatrix = xgb.DMatrix(x[:, [j]])
            x[:, j] += dt * model.predict(dmatrix)
    return x
