# Spec: data-diffusion — Synthetic Tabular Data via XGBoost Flow Matching

## Objective

A pip-installable Python package that trains a tabular data generator on a
pandas DataFrame using XGBoost-based flow matching (the "ForestFlow" technique,
Jolicoeur-Martineau et al., AISTATS 2024) and samples a synthetic dataset that
resembles the original without copying it — a reusable, shareable approximation,
not PII scrubbing.

Primary user: a data scientist who has a DataFrame and wants a synthetic
version to share or iterate on.

**v0 scope:** encode → fit → sample → decode, end to end, in-package. The
privacy evaluation suite (duplicates, DCR, membership inference, Anonymeter)
and the constraints/repair module are explicitly out of scope and deferred to
later phases.

**User stories**

- `pip install data-diffusion` works in a clean environment (Python ≥ 3.12).
- `TabularDiffusion().fit(df)` trains on a mixed-type DataFrame (numeric,
  integer, categorical, boolean, datetime, NaNs) without manual preprocessing.
- `model.sample(n)` returns a DataFrame with the same columns and dtypes.
- Mixed types are handled transparently: categoricals one-hot encoded with
  rare-level merging, dates as epoch days, numerics min-max scaled — all
  decoded back after sampling.

## Tech Stack

- Python ≥ 3.12 (matches AGENTS.md; Python 3.12+ syntax such as `type` alias
  statements is allowed)
- Runtime dependencies: `numpy`, `pandas`, `xgboost`, `joblib` (parallel model
  training; user-approved). Min-max scaling and one-hot encoding implemented
  with NumPy/pandas.
- License: MIT (set in `pyproject.toml` and `LICENSE`)
- Dev dependencies (already in project): `pytest`, `ruff`, `ty`
- Packaging: `uv` + PEP 621 `pyproject.toml`, src layout, built with `uv build`

## Commands

```
Install/sync: uv sync
Test:         uv run pytest
Format:       uv run ruff format .
Lint:         uv run ruff check .
Type check:   uv run ty check
Build wheel:  uv build
```

## Project Structure

```
src/data_diffusion/
  __init__.py        # public API: TabularDiffusion
  encoder.py         # TabularEncoder: infer types, one-hot + rare-level merge,
                     # min-max scaling, datetime epoch encoding, decode
  forestflow.py      # fit_forest_flow / sample_forest_flow (NumPy + XGBoost core,
                     # per-(level, column) fits parallelized with joblib's loky
                     # backend, n_jobs configurable; XGBoost internal threads
                     # limited per worker to avoid oversubscription)
  model.py           # TabularDiffusion: user-facing fit/sample API
tests/
  test_encoder.py    # round-trip, type inference, rare merging, edge cases
  test_forestflow.py # velocity targets, Euler sampling shapes, noise→data path
  test_model.py      # end-to-end fit/sample on small mixed-type DataFrames
SPEC.md              # this file
pyproject.toml
```

## Code Style

Follow `AGENTS.md`: plain functions over classes (the XGBoost core is two
free functions; `TabularEncoder` and `TabularDiffusion` are the only classes,
justified by genuine fit/transform state), full type annotations (Python 3.12
syntax), compact docstrings.

```python
def fit_forest_flow(
    x: np.ndarray, n_t: int = 50, k: int = 100, seed: int = 0, **xgb_kwargs: Any
) -> ForestFlowModel:
    """Fit per-level, per-column XGBoost regressors on the flow-matching path.

    Args:
        x: Encoded training matrix, scaled to roughly [-1, 1].
        n_t: Number of time levels (one model set per level).
        k: Duplication factor pairing each row with fresh Gaussian noise.
        seed: RNG seed for noise pairing.
        **xgb_kwargs: Forwarded to every XGBRegressor.

    Returns:
        Fitted model bundle (per-level regressors + time grid).
    """
```

## Testing Strategy

- Framework: pytest, test-first (red → green per module).
- Levels:
  - **Unit (encoder):** exact round-trip `decode(encode(df)) == df` on mixed
    types including dates and NaNs; type inference table; rare-category
    merging threshold; error cases (all-NaN column, empty DataFrame,
    unknown dtype).
  - **Unit (forestflow):** velocity target `v* = x - z` correctness; sampled
    matrix shapes; loss decreases vs. shuffled-target control on a tiny
    synthetic matrix.
  - **Integration (model):** `fit` + `sample` on a small mixed-type fixture
    (~1k rows, 6 columns incl. datetime, categorical with NaNs): output has
    identical columns/dtypes, no NaNs where training data had none, numeric
    values within clipped training ranges, categorical levels ⊆ training
    levels, and per-column marginals roughly match (KS statistic threshold on
    the fixture).
- Edge cases required: single-row training set, single-column DataFrame,
  all-constant column, high-cardinality categorical after rare-merge.

## Boundaries

- **Always:** run `uv run pytest`, `uv run ruff format .`, `uv run ruff check .`,
  `uv run ty check` before declaring a task done; TDD red-green for each
  module; keep runtime deps minimal (ask before adding any).
- **Ask first:** adding any runtime dependency; changing the public API
  (`TabularDiffusion.fit/sample` signature); supporting Python < 3.12;
  publishing to PyPI.
- **Never:** model direct identifier columns (out of scope, user drops them);
  claim the output is anonymous/private in docs without the privacy eval
  phase; commit generated datasets or model artifacts.

## Success Criteria

1. `uv build` produces a wheel/sdist that installs cleanly into a fresh
   virtualenv (`uv pip install dist/*.whl` in a temp env), with only
   numpy/pandas/xgboost/joblib as runtime deps, requires-python >= 3.10,
   and MIT license metadata.
2. Encoder round-trip is exact on the test fixture, including datetimes and
   NaN patterns.
3. End-to-end integration test passes: `sample(n)` returns a DataFrame with
   identical columns, dtypes, categorical domains, and clipped numeric ranges.
4. On the integration fixture, generated marginals match training marginals
   (KS statistic < 0.15 per numeric column; total variation < 0.15 per
   categorical column) — sanity-level targets; the doc's ≥ 0.90 fidelity
   scoring belongs to the later evaluation phase.
5. All four checks (ty, ruff format, ruff check, pytest) pass with no errors.
6. README documents install + the two-call API.

## Resolved Decisions

1. License: MIT.
2. Python floor: ≥ 3.12 (was briefly 3.10+; reverted to match AGENTS.md and
   the project venv).
3. Parallelism: joblib (loky backend) parallelizes the p × n_t model fits;
   `n_jobs` exposed as an `TabularDiffusion` parameter.
