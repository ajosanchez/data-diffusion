"""ForestFlow core: XGBoost flow-matching on encoded numeric matrices.

Training duplicates each row K times, pairs it with fresh Gaussian noise z,
and fits one XGBoost booster per (t level, column) to predict the clean data
value x_j from the full noisy interpolation x_t = (1 - t) z + t x (x0-
prediction with the complete state vector as input, ForestDiffusion-style;
univariate inputs cannot keep mutually exclusive one-hot blocks exclusive,
and regressing the velocity x - z is unstable for discrete columns because
its conditional variance never shrinks). Sampling integrates toward data
space with DDIM-style Euler steps over the training t grid:
x_{t+dt} = (1 - t - dt) * x0_hat + (t + dt) * x_hat, where
x0_hat = (x_t - t * x_hat) / (1 - t). Uses the native xgboost.train API
(no scikit-learn dependency).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import xgboost as xgb
from joblib import Parallel, delayed

type ForestFlowModel = dict[str, Any]

type _ModelKey = tuple[float, int]

_DEFAULT_XGB: dict[str, Any] = {
    "objective": "reg:squarederror",
    "max_depth": 6,
    "tree_method": "hist",
    "n_jobs": 1,
}


def _fit_level(
    x: np.ndarray,
    t_index: int,
    t: float,
    K: int,
    n_estimators: int,
    params: dict[str, Any],
    seed: int,
) -> list[tuple[_ModelKey, xgb.Booster]]:
    """Fit all per-column boosters for one t level; pure function of args.

    Noise is regenerated here from a deterministic child seed of (seed,
    t_index) so that serial and joblib paths produce identical models and
    only one level's (n*K, p) matrix is resident per worker.
    """
    rng = np.random.default_rng([seed, t_index])
    n_rows, n_features = x.shape
    z = rng.normal(size=(n_rows * K, n_features))
    row_idx = rng.integers(0, n_rows, size=n_rows * K)
    x_ref = x[row_idx]
    x_t = (1.0 - t) * z + t * x_ref
    fitted: list[tuple[_ModelKey, xgb.Booster]] = []
    for j in range(n_features):
        # Rows whose source value is NaN have NaN labels/inputs — exclude
        # them from this column's fit (NaN never enters sampling: noise is
        # finite and predictions are finite).
        mask = np.isfinite(x_t[:, j]) & np.isfinite(x_ref[:, j])
        dtrain = xgb.DMatrix(x_t[mask], label=x_ref[mask, j])
        fitted.append(
            ((t, j), xgb.train(dict(params), dtrain, num_boost_round=n_estimators))
        )
    return fitted


def fit_forest_flow(
    x: np.ndarray,
    n_t: int = 50,
    K: int = 100,
    n_jobs: int = 1,
    xgb_kwargs: dict[str, Any] | None = None,
    seed: int = 0,
) -> ForestFlowModel:
    """Fit per-(t, column) XGBoost x0-prediction models on encoded data.

    Args:
        x: Encoded numeric matrix of shape (n_rows, n_features); NaN is
            allowed (treated as missing per column), inf is not.
        n_t: Number of t levels on the flow grid in [0, 1).
        K: Duplication factor; each row is paired with K noise draws.
        n_jobs: joblib workers for the per-level fits; 1 is serial. Results
            are identical for any n_jobs at a fixed seed. XGBoost itself
            defaults to n_jobs=1 to avoid oversubscription inside workers.
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
    if np.isinf(x).any():
        raise ValueError("x contains infinite values")  # NaN is allowed (missing)
    if n_t < 1:
        raise ValueError(f"n_t must be >= 1, got {n_t}")
    if K < 1:
        raise ValueError(f"K must be >= 1, got {K}")

    kwargs = dict(xgb_kwargs or {})
    n_estimators = int(kwargs.pop("n_estimators", 100))
    params = {**_DEFAULT_XGB, **kwargs}
    ts = [i / n_t for i in range(n_t)]
    if n_jobs == 1:
        fitted = [
            result
            for t_index, t in enumerate(ts)
            for result in _fit_level(x, t_index, t, K, n_estimators, params, seed)
        ]
    else:
        fitted = [
            result
            for level in Parallel(n_jobs=n_jobs, backend="loky")(
                delayed(_fit_level)(x, t_index, t, K, n_estimators, params, seed)
                for t_index, t in enumerate(ts)
            )
            for result in level
        ]
    models: dict[_ModelKey, xgb.Booster] = dict(fitted)
    return {"models": models, "ts": ts, "n_features": n_features}


def sample_forest_flow(
    bundle: ForestFlowModel, n_rows: int, seed: int = 0
) -> np.ndarray:
    """Integrate the learned field from noise to data space.

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
        dmatrix = xgb.DMatrix(x)
        for j in range(n_features):
            x_hat = bundle["models"][(t, j)].predict(dmatrix)
            x0_hat = (x[:, j] - t * x_hat) / (1.0 - t)  # t < 1 on the grid
            x[:, j] = (1.0 - t - dt) * x0_hat + (t + dt) * x_hat
    return x
