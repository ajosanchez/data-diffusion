# Task List: data-diffusion v0

Ordered by dependency. Full rationale in `tasks/plan.md`; spec in `SPEC.md`.

- [x] Task 1: Packaging scaffold
  - Acceptance: `pyproject.toml` has requires-python ">=3.12", MIT license
    metadata, runtime deps numpy/pandas/xgboost/joblib; `LICENSE` (MIT) exists;
    `src/data_diffusion/__init__.py` exports a stub `TabularDiffusion`;
    `uv sync` and `uv build` succeed.
  - Verify: `uv build` then install wheel in a temp venv and
    `python -c "import data_diffusion"`.
  - Files: `pyproject.toml`, `LICENSE`, `src/data_diffusion/__init__.py`,
    `src/data_diffusion/model.py`, `ruff.toml` (target py312).

- [x] Task 2: TabularEncoder
  - Acceptance: infers column kinds (numeric/integer/categorical/boolean/
    datetime); merges rare categories into "Other"; encode→decode round-trip
    is exact on mixed fixture incl. datetimes and NaNs; errors on empty df and
    all-NaN column; constant column round-trips.
  - Verify: `uv run pytest tests/test_encoder.py` green after failing first (red→green).
  - Files: `src/data_diffusion/encoder.py`, `tests/test_encoder.py`.

- [ ] Task 3: ForestFlow core (sequential)
  - Acceptance: `fit_forest_flow` builds path points `x_t = (1-t)·z + t·x` and
    velocity targets `v* = x - z` correctly; `sample_forest_flow` returns
    shape (n, p) and moves noise toward data (training-marginal distance
    shrinks vs. pure-noise control); deterministic under fixed seed.
  - Verify: `uv run pytest tests/test_forestflow.py` green (red→green).
  - Files: `src/data_diffusion/forestflow.py`, `tests/test_forestflow.py`.

- [ ] Task 4: Parallel training via joblib
  - Acceptance: model fits distributed over (level, column) with
    `joblib.Parallel(backend="loky")`; `n_jobs` parameter; XGBoost internal
    `n_jobs=1` inside workers; output bundle identical (same predictions) for
    `n_jobs=1` and `n_jobs=4` at fixed seed.
  - Verify: equality test `n_jobs=1` vs `n_jobs=4`; small timing sanity check.
  - Files: `src/data_diffusion/forestflow.py`, `tests/test_forestflow.py`.

- [ ] Task 5: TabularDiffusion API
  - Acceptance: `fit(df)` stores encoder + model; `sample(n, seed)` returns
    DataFrame with identical columns/dtypes; categorical values ⊆ training
    domains (plus "Other"); integers/counts rounded and clipped; datetimes
    cast to datetime64; float columns within training range; params `n_t`,
    `K`, `n_jobs`, `rare_level_threshold`, `xgb_kwargs` plumbed through.
  - Verify: `uv run pytest tests/test_model.py` green (red→green).
  - Files: `src/data_diffusion/model.py`, `src/data_diffusion/__init__.py`,
    `tests/test_model.py`.

- [ ] Task 6: Integration fidelity + package check
  - Acceptance: on ~1k-row mixed fixture, per-column marginals match:
    KS < 0.15 (numeric), TV < 0.15 (categorical); `uv build` wheel installs
    into a fresh venv with only the four runtime deps.
  - Verify: `uv run pytest tests/test_integration.py`; clean-venv install + import.
  - Files: `tests/test_integration.py`, `tests/conftest.py` (fixture),
    possibly `pyproject.toml`.

- [ ] Task 7: README and final checks
  - Acceptance: README shows install + two-call usage and carries a
    "not anonymized / not PII-scrubbed" disclaimer; all AGENTS.md checks pass.
  - Verify: `uv run ty check && uv run ruff format . && uv run ruff check . && uv run pytest`.
  - Files: `README.md`.

- [ ] Task 8: Spec/plan housekeeping
  - Acceptance: checkboxes updated as tasks complete; spec updated if any
    decision changed during implementation.
  - Verify: review of `SPEC.md` / `tasks/` diff.
  - Files: `SPEC.md`, `tasks/todo.md`, `tasks/plan.md`.
