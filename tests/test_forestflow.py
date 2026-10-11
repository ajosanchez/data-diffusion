"""Tests for the ForestFlow core: bundle structure, sampling, determinism."""

import numpy as np
import pytest

from data_diffusion.forestflow import fit_forest_flow, sample_forest_flow


@pytest.fixture
def normal_data() -> np.ndarray:
    rng = np.random.default_rng(42)
    return rng.normal(loc=0.0, scale=1.0, size=(2000, 2))


def test_bundle_structure(normal_data: np.ndarray) -> None:
    bundle = fit_forest_flow(
        normal_data,
        n_t=8,
        K=10,
        xgb_kwargs={"n_estimators": 40, "max_depth": 4},
        seed=0,
    )
    assert bundle["n_features"] == 2
    assert np.allclose(bundle["ts"], np.arange(8) / 8)
    assert len(bundle["models"]) == 8 * 2  # one model per (t level, column)
    # t levels appear exactly once as keys
    assert len({t for t, _ in bundle["models"]}) == 8


def test_sample_shape_and_finiteness(normal_data: np.ndarray) -> None:
    bundle = fit_forest_flow(
        normal_data,
        n_t=8,
        K=10,
        xgb_kwargs={"n_estimators": 40, "max_depth": 4},
        seed=0,
    )
    s = sample_forest_flow(bundle, 100, seed=1)
    assert s.shape == (100, 2)
    assert np.isfinite(s).all()


def test_sampling_is_deterministic_per_seed(normal_data: np.ndarray) -> None:
    bundle = fit_forest_flow(
        normal_data,
        n_t=8,
        K=10,
        xgb_kwargs={"n_estimators": 40, "max_depth": 4},
        seed=0,
    )
    a = sample_forest_flow(bundle, 200, seed=7)
    b = sample_forest_flow(bundle, 200, seed=7)
    c = sample_forest_flow(bundle, 200, seed=8)
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)


def test_fit_is_deterministic_per_seed(normal_data: np.ndarray) -> None:
    b1 = fit_forest_flow(
        normal_data, n_t=4, K=5, xgb_kwargs={"n_estimators": 10}, seed=3
    )
    b2 = fit_forest_flow(
        normal_data, n_t=4, K=5, xgb_kwargs={"n_estimators": 10}, seed=3
    )
    s1 = sample_forest_flow(b1, 50, seed=0)
    s2 = sample_forest_flow(b2, 50, seed=0)
    np.testing.assert_array_equal(s1, s2)


def test_recovers_standard_normal_moments() -> None:
    rng = np.random.default_rng(1)
    x = rng.normal(loc=5.0, scale=2.0, size=(3000, 1))
    bundle = fit_forest_flow(
        x, n_t=10, K=20, xgb_kwargs={"n_estimators": 64, "max_depth": 5}, seed=0
    )
    s = sample_forest_flow(bundle, 2000, seed=3)
    assert abs(s.mean() - 5.0) < 0.15
    assert 1.6 < s.std() < 2.4


def test_user_xgb_kwargs_reach_models(normal_data: np.ndarray) -> None:
    bundle = fit_forest_flow(
        normal_data, n_t=2, K=2, xgb_kwargs={"n_estimators": 55}, seed=0
    )
    model = next(iter(bundle["models"].values()))
    assert model.num_boosted_rounds() == 55


def test_single_row_training_data() -> None:
    x = np.ones((1, 3))
    bundle = fit_forest_flow(x, n_t=2, K=5, xgb_kwargs={"n_estimators": 10}, seed=0)
    s = sample_forest_flow(bundle, 10, seed=0)
    assert s.shape == (10, 3)
    assert np.isfinite(s).all()


def test_fit_rejects_non_2d_input() -> None:
    with pytest.raises(ValueError, match="2-?D|2D|two-dimensional"):
        fit_forest_flow(np.zeros(10))


def test_fit_allows_nan_but_rejects_inf() -> None:
    x_nan = np.array([[1.0, 2.0], [np.nan, 3.0]])
    bundle = fit_forest_flow(x_nan, n_t=2, K=2, xgb_kwargs={"n_estimators": 5})
    out = sample_forest_flow(bundle, 10, seed=0)
    assert np.isfinite(out).all()
    x_inf = np.array([[1.0, 2.0], [np.inf, 3.0]])
    with pytest.raises(ValueError, match="infinite"):
        fit_forest_flow(x_inf, n_t=2, K=2)


def test_fit_rejects_empty_input() -> None:
    with pytest.raises(ValueError, match="empty"):
        fit_forest_flow(np.zeros((0, 3)))
    with pytest.raises(ValueError, match="empty"):
        fit_forest_flow(np.zeros((5, 0)))


def test_fit_rejects_nonpositive_counts(normal_data: np.ndarray) -> None:
    with pytest.raises(ValueError, match="n_t"):
        fit_forest_flow(normal_data, n_t=0)
    with pytest.raises(ValueError, match="K"):
        fit_forest_flow(normal_data, K=0)


def test_parallel_fit_matches_sequential_fit(normal_data: np.ndarray) -> None:
    serial = fit_forest_flow(
        normal_data, n_t=4, K=5, xgb_kwargs={"n_estimators": 20}, n_jobs=1, seed=5
    )
    parallel = fit_forest_flow(
        normal_data, n_t=4, K=5, xgb_kwargs={"n_estimators": 20}, n_jobs=4, seed=5
    )
    s1 = sample_forest_flow(serial, 100, seed=0)
    s2 = sample_forest_flow(parallel, 100, seed=0)
    np.testing.assert_array_equal(s1, s2)


def test_sample_rejects_nonpositive_n(normal_data: np.ndarray) -> None:
    bundle = fit_forest_flow(
        normal_data,
        n_t=8,
        K=10,
        xgb_kwargs={"n_estimators": 40, "max_depth": 4},
        seed=0,
    )
    with pytest.raises(ValueError, match="n_rows|positive"):
        sample_forest_flow(bundle, 0)
