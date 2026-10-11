"""Tests for TabularEncoder: type inference, encoding, exact round-trip."""

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from data_diffusion.encoder import TabularEncoder


@pytest.fixture
def mixed_df() -> pd.DataFrame:
    """500 rows: float with NaN, int64, bool, categorical, datetime with NaT."""
    rng = np.random.default_rng(0)
    n = 500
    num = rng.normal(loc=10.0, scale=2.0, size=n)
    num[::25] = np.nan
    ints = rng.integers(-5, 6, size=n)
    flags = rng.integers(0, 2, size=n).astype(bool)
    cats = rng.choice(["red", "green", "blue"], size=n)
    day = pd.Series(
        pd.to_datetime("2024-01-01")
        + pd.to_timedelta(rng.integers(0, 365, size=n), unit="D"),
        name="day",
    )
    day.iloc[::50] = pd.NaT
    return pd.DataFrame(
        {"num": num, "count": ints, "flag": flags, "color": cats, "day": day}
    )


def test_roundtrip_exact_on_mixed_fixture(mixed_df: pd.DataFrame) -> None:
    enc = TabularEncoder()
    out = enc.inverse_transform(enc.fit_transform(mixed_df))
    assert list(out.columns) == list(mixed_df.columns)
    assert (out.dtypes == mixed_df.dtypes).all()
    assert_frame_equal(out, mixed_df)


def test_encoded_numeric_column_spans_full_scaled_range(mixed_df: pd.DataFrame) -> None:
    x = TabularEncoder().fit_transform(mixed_df)
    finite = x[:, 0][~np.isnan(x[:, 0])]
    assert finite.min() == pytest.approx(-1.0)
    assert finite.max() == pytest.approx(1.0)


def test_nans_pass_through_encoding_and_decode(mixed_df: pd.DataFrame) -> None:
    enc = TabularEncoder()
    x = enc.fit_transform(mixed_df)
    assert np.isnan(x[0, 0])  # num has NaN at row 0
    out = enc.inverse_transform(x)
    assert out["num"].isna().sum() == 20
    assert out["day"].isna().sum() == 10


def test_feature_names_layout(mixed_df: pd.DataFrame) -> None:
    enc = TabularEncoder()
    enc.fit(mixed_df)
    assert enc.feature_names == [
        "num",
        "count",
        "flag",
        "color=blue",
        "color=green",
        "color=red",
        "day",
    ]


def test_rare_levels_merge_to_other() -> None:
    colors = ["red"] * 400 + ["gold"] * 5 + ["teal"] * 5
    df = pd.DataFrame({"color": colors})
    enc = TabularEncoder()
    x = enc.fit_transform(df)
    assert enc.feature_names == ["color=Other", "color=red"]
    out = enc.inverse_transform(x)
    assert (out["color"] == "red").sum() == 400
    assert (out["color"] == "Other").sum() == 10


def test_all_rare_levels_collapse_to_single_other() -> None:
    df = pd.DataFrame({"color": ["a", "b", "c", "d", "e"] * 4})
    enc = TabularEncoder()
    x = enc.fit_transform(df)
    assert enc.feature_names == ["color=Other"]
    out = enc.inverse_transform(x)
    assert (out["color"] == "Other").all()


def test_single_level_is_never_merged() -> None:
    df = pd.DataFrame({"color": ["red"] * 3})
    enc = TabularEncoder()
    out = enc.inverse_transform(enc.fit_transform(df))
    assert enc.feature_names == ["color=red"]
    assert (out["color"] == "red").all()


def test_single_row_roundtrip() -> None:
    df = pd.DataFrame(
        {
            "num": [3.5],
            "count": [7],
            "flag": [True],
            "color": ["red"],
            "day": pd.to_datetime(["2024-05-05"]),
        }
    )
    enc = TabularEncoder()
    out = enc.inverse_transform(enc.fit_transform(df))
    assert_frame_equal(out, df)


def test_constant_column_roundtrip() -> None:
    df = pd.DataFrame({"const": [2.5] * 50, "vary": np.linspace(0.0, 1.0, 50)})
    enc = TabularEncoder()
    out = enc.inverse_transform(enc.fit_transform(df))
    assert_frame_equal(out, df)


def test_unseen_level_maps_to_other_at_transform() -> None:
    train = pd.DataFrame({"color": ["red"] * 400 + ["gold"] * 5})
    enc = TabularEncoder()
    enc.fit(train)
    x = enc.transform(pd.DataFrame({"color": ["teal", "red"]}))
    out = enc.inverse_transform(x)
    assert out["color"].tolist() == ["Other", "red"]


def test_unseen_level_raises_when_no_other_exists() -> None:
    train = pd.DataFrame({"color": ["red", "green"] * 200})
    enc = TabularEncoder()
    enc.fit(train)
    with pytest.raises(ValueError, match="unseen"):
        enc.transform(pd.DataFrame({"color": ["blue", "red"]}))


def test_empty_dataframe_raises() -> None:
    with pytest.raises(ValueError, match="empty"):
        TabularEncoder().fit(pd.DataFrame({"a": pd.Series([], dtype="float64")}))
    with pytest.raises(ValueError, match="empty"):
        TabularEncoder().fit(pd.DataFrame())


def test_all_nan_column_raises() -> None:
    df = pd.DataFrame({"ok": [1.0, 2.0], "allnan": [np.nan, np.nan]})
    with pytest.raises(ValueError, match="all-NaN"):
        TabularEncoder().fit(df)


def test_tz_aware_datetime_raises() -> None:
    df = pd.DataFrame(
        {"t": pd.to_datetime(["2024-01-01", "2024-06-01"]).tz_localize("UTC")}
    )
    with pytest.raises(ValueError, match="tz-aware"):
        TabularEncoder().fit(df)


def test_unsupported_dtype_raises() -> None:
    df = pd.DataFrame({"obj": [[1, 2], [3], [4]] * 10})
    with pytest.raises(ValueError, match="unsupported dtype"):
        TabularEncoder().fit(df)


def test_transform_before_fit_raises() -> None:
    with pytest.raises(ValueError, match="fit"):
        TabularEncoder().transform(pd.DataFrame({"a": [1.0]}))


def test_inverse_transform_rejects_wrong_shape(mixed_df: pd.DataFrame) -> None:
    enc = TabularEncoder()
    enc.fit(mixed_df)
    with pytest.raises(ValueError, match="shape"):
        enc.inverse_transform(np.zeros((5, len(enc.feature_names) + 1)))
