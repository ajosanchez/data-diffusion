"""End-to-end integration: full pipeline fidelity on a 1k-row mixed fixture."""

import numpy as np
import pandas as pd
import pytest

from data_diffusion import TabularDiffusion

N = 1000
KS_THRESHOLD = 0.15
TV_THRESHOLD = 0.15


def _make_df() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    num = rng.normal(0.0, 1.0, N)
    skew = rng.lognormal(0.0, 0.5, N)
    count = rng.integers(0, 10, N)
    flag = rng.integers(0, 2, N).astype(bool)
    color = rng.choice(
        ["red", "green", "blue", "purple"], N, p=[0.45, 0.30, 0.24, 0.01]
    )
    day = pd.Series(
        pd.to_datetime("2022-01-01")
        + pd.to_timedelta(rng.integers(0, 900, N), unit="D"),
        name="day",
    )
    df = pd.DataFrame(
        {
            "num": num,
            "skew": skew,
            "count": count,
            "flag": flag,
            "color": color,
            "day": day,
        }
    )
    df.loc[rng.choice(N, 50, replace=False), "num"] = np.nan
    df.loc[rng.choice(N, 40, replace=False), "day"] = pd.NaT
    return df


def ks_statistic(a: np.ndarray, b: np.ndarray) -> float:
    """Two-sample Kolmogorov-Smirnov statistic (supremum ECDF distance)."""
    a, b = np.sort(np.asarray(a)), np.sort(np.asarray(b))
    grid = np.concatenate([a, b])
    cdf_a = np.searchsorted(a, grid, side="right") / len(a)
    cdf_b = np.searchsorted(b, grid, side="right") / len(b)
    return float(np.max(np.abs(cdf_a - cdf_b)))


def tv_distance(p: dict[str, float], q: dict[str, float]) -> float:
    """Total variation distance between two discrete distributions."""
    keys = set(p) | set(q)
    return 0.5 * sum(abs(p.get(k, 0.0) - q.get(k, 0.0)) for k in keys)


def frequencies(values: np.ndarray) -> dict[str, float]:
    labels, counts = np.unique(values, return_counts=True)
    return {
        str(label): int(count) / len(values)
        for label, count in zip(labels.tolist(), counts.tolist())
    }


@pytest.fixture(scope="module")
def data() -> pd.DataFrame:
    return _make_df()


@pytest.fixture(scope="module")
def synthetic(data: pd.DataFrame) -> pd.DataFrame:
    model = TabularDiffusion(
        n_t=15, K=20, n_jobs=4, xgb_kwargs={"n_estimators": 80, "max_depth": 6}
    )
    model.fit(data)
    return model.sample(N, seed=123)


def test_output_schema_and_domains(data: pd.DataFrame, synthetic: pd.DataFrame) -> None:
    assert list(synthetic.columns) == list(data.columns)
    assert (synthetic.dtypes == data.dtypes).all()
    assert synthetic.notna().all().all()  # finite noise in, finite decode out
    assert (
        synthetic["count"]
        .between(int(data["count"].min()), int(data["count"].max()))
        .all()
    )
    assert synthetic["day"].between(data["day"].min(), data["day"].max()).all()
    lo, hi = data["skew"].min(), data["skew"].max()
    assert synthetic["skew"].between(lo - 1e-9, hi + 1e-9).all()
    merged = data["color"].where(data["color"].isin(["red", "green", "blue"]), "Other")
    assert set(synthetic["color"]) <= set(merged)


@pytest.mark.parametrize(
    ("column", "numeric"),
    [("num", True), ("skew", True), ("count", True), ("day", True)],
)
def test_numeric_marginals_ks(
    data: pd.DataFrame, synthetic: pd.DataFrame, column: str, numeric: bool
) -> None:
    train = data[column].dropna()
    sample = synthetic[column].dropna()
    if column == "day":
        train = train.view("int64") if hasattr(train, "view") else train.astype("int64")
        sample = sample.astype("int64")
    stat = ks_statistic(train.to_numpy(), sample.to_numpy())
    assert stat < KS_THRESHOLD, f"{column}: KS = {stat:.3f}"


def test_categorical_marginals_tv(data: pd.DataFrame, synthetic: pd.DataFrame) -> None:
    merged = data["color"].where(data["color"].isin(["red", "green", "blue"]), "Other")
    stat = tv_distance(
        frequencies(merged.to_numpy()), frequencies(synthetic["color"].to_numpy())
    )
    assert stat < TV_THRESHOLD, f"color: TV = {stat:.3f}"
    flag_stat = tv_distance(
        frequencies(data["flag"].to_numpy()), frequencies(synthetic["flag"].to_numpy())
    )
    assert flag_stat < TV_THRESHOLD, f"flag: TV = {flag_stat:.3f}"
