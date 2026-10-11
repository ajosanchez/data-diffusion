"""Reversible tabular encoder: mixed-dtype DataFrame to float matrix and back.

Numeric, integer, and datetime columns are min-max scaled to [-1, 1]; booleans
map to {0, 1}; categoricals are one-hot encoded after merging rare levels into
"Other". NaN/NaT values pass through as NaN in the encoded matrix and are
restored on decode.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

type ColKind = (
    str  # one of "float" | "integer" | "boolean" | "datetime" | "categorical"
)

_EPOCH = pd.Timestamp("1970-01-01")


@dataclass
class _ColumnSpec:
    """Layout and scaling state for one source column."""

    name: str
    kind: ColKind
    vmin: float = 0.0
    vmax: float = 0.0
    levels: list[str] = field(default_factory=list)


class TabularEncoder:
    """Fit/transform/inverse_transform encoder driven by inferred column types.

    Args:
        rare_min_count: Categorical levels with fewer than this many rows are
            merged into "Other".
        rare_min_fraction: Categorical levels with a smaller share of rows than
            this are merged into "Other".
    """

    def __init__(
        self, rare_min_count: int = 20, rare_min_fraction: float = 0.01
    ) -> None:
        self.rare_min_count = rare_min_count
        self.rare_min_fraction = rare_min_fraction
        self.columns: list[_ColumnSpec] = []
        self.feature_names: list[str] = []
        self._fitted = False

    def fit(self, df: pd.DataFrame) -> TabularEncoder:
        """Infer column kinds and record scaling/level state."""
        self._check_not_empty(df)
        self.columns = []
        self.feature_names = []
        for column_name, series in df.items():
            name = str(column_name)
            spec = self._infer(name, series)
            self.columns.append(spec)
            if spec.kind == "categorical":
                self.feature_names.extend(f"{name}={level}" for level in spec.levels)
            else:
                self.feature_names.append(str(name))
        self._fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Encode a DataFrame to the fitted float matrix."""
        self._require_fitted()
        self._check_not_empty(df)
        parts: list[np.ndarray] = []
        offset = 0
        for spec in self.columns:
            series = df[spec.name]
            if spec.kind == "categorical":
                block = self._encode_categorical(series, spec)
                parts.append(block)
                offset += len(spec.levels)
            elif spec.kind == "boolean":
                parts.append(series.to_numpy(dtype="float64"))
                offset += 1
            elif spec.kind == "datetime":
                parts.append(self._scale(self._to_days(series), spec))
                offset += 1
            else:
                parts.append(self._scale(series, spec))
                offset += 1
        return np.column_stack(parts)

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        """Fit on `df` and return its encoded matrix."""
        return self.fit(df).transform(df)

    def inverse_transform(self, x: np.ndarray) -> pd.DataFrame:
        """Decode an encoded matrix back to a DataFrame with training dtypes."""
        self._require_fitted()
        x = np.asarray(x, dtype="float64")
        if x.ndim != 2 or x.shape[1] != len(self.feature_names):
            raise ValueError(
                f"expected matrix of shape (n, {len(self.feature_names)}), got shape {x.shape}"
            )
        data: dict[str, np.ndarray] = {}
        offset = 0
        for spec in self.columns:
            width = len(spec.levels) if spec.kind == "categorical" else 1
            block = x[:, offset : offset + width]
            data[str(spec.name)] = self._decode_column(block, spec)
            offset += width
        return pd.DataFrame(data)

    def _require_fitted(self) -> None:
        if not self._fitted:
            raise ValueError("encoder is not fitted; call fit() first")

    @staticmethod
    def _check_not_empty(df: pd.DataFrame) -> None:
        if df.shape[0] == 0 or df.shape[1] == 0:
            raise ValueError("cannot fit or transform an empty DataFrame")

    def _infer(self, name: str, series: pd.Series) -> _ColumnSpec:
        dtype = series.dtype
        if pd.api.types.is_bool_dtype(dtype):
            return _ColumnSpec(name, "boolean", 0.0, 1.0)
        if pd.api.types.is_datetime64_any_dtype(dtype):
            if getattr(dtype, "tz", None) is not None:
                raise ValueError(
                    f"column {name!r}: tz-aware datetimes are not supported"
                )
            days = self._to_days(series)
            if days.notna().sum() == 0:
                raise ValueError(f"column {name!r} is all-NaN/NaT")
            return _ColumnSpec(name, "datetime", float(days.min()), float(days.max()))
        if pd.api.types.is_integer_dtype(dtype):
            return _ColumnSpec(
                name, "integer", float(series.min()), float(series.max())
            )
        if pd.api.types.is_float_dtype(dtype):
            if series.isna().all():
                raise ValueError(f"column {name!r} is all-NaN/NaT")
            return _ColumnSpec(name, "float", float(series.min()), float(series.max()))
        if (
            isinstance(dtype, pd.CategoricalDtype)
            or pd.api.types.is_string_dtype(dtype)
            or pd.api.types.is_object_dtype(dtype)
        ):
            return self._infer_categorical(name, series)
        raise ValueError(f"column {name!r} has unsupported dtype {dtype}")

    def _infer_categorical(self, name: str, series: pd.Series) -> _ColumnSpec:
        non_null = series.dropna()
        if len(non_null) and not all(isinstance(v, str) for v in non_null):
            raise ValueError(f"column {name!r} has unsupported dtype {series.dtype}")
        counts = series.dropna().astype(str).value_counts()
        total = int(counts.sum())
        if len(counts) == 1:  # a single level carries no information to merge away
            return _ColumnSpec(name, "categorical", levels=[str(counts.index[0])])
        surviving = [
            str(level)
            for level, count in counts.items()
            if count >= self.rare_min_count and count / total >= self.rare_min_fraction
        ]
        rare = len(counts) - len(surviving)
        if not surviving:
            levels = ["Other"]
        elif rare:
            levels = sorted([*surviving, "Other"])
        else:
            levels = sorted(surviving)
        return _ColumnSpec(name, "categorical", levels=levels)

    def _encode_categorical(self, series: pd.Series, spec: _ColumnSpec) -> np.ndarray:
        isna = series.isna()
        known = series.isin(spec.levels) | isna
        if not known.all():
            if "Other" not in spec.levels:
                unseen = sorted(set(series[~known].astype(str)))
                raise ValueError(f"column {spec.name!r}: unseen levels {unseen}")
            series = series.astype(object).copy()
            series[~known] = "Other"
        codes = pd.Categorical(series.astype(str), categories=spec.levels).codes
        block = np.zeros((len(series), len(spec.levels)))
        rows = np.flatnonzero(~isna.to_numpy())
        block[rows, codes[rows]] = 1.0
        if isna.any():
            block[isna.to_numpy(), :] = np.nan  # missing row = entirely NaN block
        return block

    def _decode_column(self, block: np.ndarray, spec: _ColumnSpec) -> np.ndarray:
        if spec.kind == "categorical":
            missing = np.isnan(block).all(axis=1)
            idx = np.full(block.shape[0], -1)
            present = ~missing
            idx[present] = np.nanargmax(block[present], axis=1)
            return np.array(
                [spec.levels[i] if i >= 0 else None for i in idx], dtype="object"
            )
        unscaled = self._unscale(block[:, 0], spec)
        if spec.kind == "boolean":
            return unscaled > 0.5
        if spec.kind == "integer":
            return np.clip(np.rint(unscaled), spec.vmin, spec.vmax).astype("int64")
        if spec.kind == "datetime":
            days = pd.Series(np.rint(unscaled))
            return (_EPOCH + pd.to_timedelta(days, unit="D")).to_numpy()
        return unscaled  # float

    def _scale(self, series: pd.Series, spec: _ColumnSpec) -> np.ndarray:
        return self._scale_values(series.to_numpy(dtype="float64"), spec)

    def _scale_values(self, values: np.ndarray, spec: _ColumnSpec) -> np.ndarray:
        if spec.vmax <= spec.vmin:  # constant (or single-value) column
            return np.where(np.isnan(values), np.nan, 0.0)
        return (values - spec.vmin) / (spec.vmax - spec.vmin) * 2.0 - 1.0

    def _unscale(self, values: np.ndarray, spec: _ColumnSpec) -> np.ndarray:
        if spec.vmax <= spec.vmin:
            return np.where(np.isnan(values), np.nan, spec.vmin)
        clipped = np.clip(values, -1.0, 1.0)
        return (clipped + 1.0) / 2.0 * (spec.vmax - spec.vmin) + spec.vmin

    @staticmethod
    def _to_days(series: pd.Series) -> pd.Series:
        return (series - _EPOCH).dt.total_seconds() / 86400.0
