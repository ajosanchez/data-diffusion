"""Tests for the TabularDiffusion high-level API."""

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from data_diffusion import TabularDiffusion

FAST: dict[str, object] = {
    "n_t": 5,
    "K": 8,
    "xgb_kwargs": {"n_estimators": 40, "max_depth": 4},
}


def _fast_model() -> TabularDiffusion:
    return TabularDiffusion(n_t=5, K=8, xgb_kwargs={"n_estimators": 40, "max_depth": 4})


def test_fit_returns_self_and_stores_state(toy_df: pd.DataFrame) -> None:
    model = _fast_model()
    assert model.fit(toy_df) is model
    assert model.encoder_ is not None
    assert model.model_ is not None
    assert model.model_["n_features"] == len(model.feature_names_)
    assert len(model.model_["ts"]) == 5
    assert len(model.model_["models"]) == 5 * len(model.feature_names_)


def test_sample_schema_matches_training(toy_df: pd.DataFrame) -> None:
    model = _fast_model().fit(toy_df)
    out = model.sample(200, seed=1)
    assert list(out.columns) == list(toy_df.columns)
    assert (out.dtypes == toy_df.dtypes).all()
    assert set(out["color"]) <= {"red", "green", "blue", "Other"}


def test_sample_dtypes_are_constrained(toy_df: pd.DataFrame) -> None:
    model = _fast_model().fit(toy_df)
    out = model.sample(500, seed=2)
    assert out["count"].dtype == np.dtype("int64")
    assert out["count"].between(0, 9).all()  # rounded and clipped to training range
    assert out["flag"].dtype == np.dtype("bool")
    assert out["day"].dtype == toy_df["day"].dtype
    assert out["day"].between(toy_df["day"].min(), toy_df["day"].max()).all()
    lo, hi = toy_df["num"].min(), toy_df["num"].max()
    finite = out["num"].dropna()
    assert ((finite >= lo - 1e-9) & (finite <= hi + 1e-9)).all()


def test_sample_is_deterministic_per_seed(toy_df: pd.DataFrame) -> None:
    model = _fast_model().fit(toy_df)
    a = model.sample(100, seed=7)
    b = model.sample(100, seed=7)
    c = model.sample(100, seed=8)
    assert_frame_equal(a, b)
    assert not a["num"].equals(c["num"])


def test_sample_before_fit_raises() -> None:
    with pytest.raises(ValueError, match="fit"):
        TabularDiffusion().sample(10)


def test_fit_rejects_non_dataframe() -> None:
    with pytest.raises(TypeError, match="DataFrame"):
        TabularDiffusion().fit("not a dataframe")  # ty: ignore[invalid-argument-type]


def test_rare_level_settings_forwarded_to_encoder() -> None:
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "num": rng.normal(size=400),
            "color": ["red"] * 396 + ["gold", "gold", "teal", "teal"],
        }
    )
    model = TabularDiffusion(
        n_t=2, K=2, rare_min_count=10, xgb_kwargs={"n_estimators": 10}
    ).fit(df)
    assert "color=Other" in model.feature_names_
    out = model.sample(50, seed=0)
    assert set(out["color"]) <= {"red", "Other"}
