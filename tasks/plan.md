# Plan: data-diffusion v0

Implements `SPEC.md`. Python ≥ 3.12, runtime deps numpy/pandas/xgboost/joblib,
MIT. TDD throughout (`test-driven-development` skill); this plan follows
`planning-and-task-breakdown` mechanics.

## Components and dependency order

```
1 packaging-scaffold ──→ 2 tabular-encoder ──→ 3 forestflow-core ──→ 4 parallel-training ──→ 5 model-api ──→ 6 integration-and-package ──→ 7 readme
```

- **1 packaging-scaffold** — Everything depends on an installable src-layout
  package. Update `pyproject.toml` (requires-python >= 3.10, MIT metadata,
  runtime deps via `uv add numpy pandas xgboost joblib`), add `LICENSE`,
  `src/data_diffusion/` skeleton, stub `TabularDiffusion`.
- **2 tabular-encoder** — `TabularEncoder`: type inference
  (numeric/integer/categorical/boolean/datetime), rare-category merge
  (threshold param, default merges levels with < 1% or < 20 rows into
  "Other"), one-hot, min-max scale to [-1, 1], datetime → epoch-days float,
  NaN passthrough. Pure reverse `decode`. No dependency on 3–5.
- **3 forestflow-core** — `fit_forest_flow` / `sample_forest_flow` free
  functions on NumPy arrays. Sequential first; correctness over speed.
  Dataclass-free: returns a `dict[str, Any]` model bundle
  (`{"models": dict[tuple[int,int], XGBRegressor], "ts": np.ndarray,
  "n_features": int}`).
- **4 parallel-training** — Refactor 3's per-(level, column) fits through
  `joblib.Parallel(backend="loky")`, `n_jobs` parameter. Inside each worker,
  XGBoost `n_jobs=1` (threads) to avoid oversubscription — each loky process
  is already single-threaded CPU-bound work. Refactor is transparent: same
  bundle, same predictions (XGBoost is deterministic given fixed seeds).
- **5 model-api** — `TabularDiffusion` class: `fit(df)` → infer types →
  encode → fit core → store encoder + bundle; `sample(n, seed)` → Euler
  integration from noise → decode → clip numerics/integers to training
  ranges, round integers, argmax one-hot blocks, cast datetimes. Exposes
  `n_t`, `K`, `xgb_kwargs`, `n_jobs`, `rare_level_threshold`.
- **6 integration-and-package** — End-to-end integration test on a ~1k-row
  mixed-type fixture (KS < 0.15, TV < 0.15, dtype/domain/range equality);
  `uv build` + clean-venv wheel install check.
- **7 readme** — Install, two-call API example, explicit "not PII scrubbing /
  not anonymous" disclaimer.

Tasks 2 and 3 could proceed in parallel after 1 (different files, no shared
interface until 5 wires them), but both are small enough that sequential is
cheaper than coordinating; keep sequential.

## Risks and mitigations

| Risk | Mitigation |
|---|---|
| Memory: `Xt` per level is K·n·p float64 — at 50k×20×100 ≈ 800 MB | Build `Xt` one level at a time inside the fit loop and release it; keep `K` configurable; document memory ≈ 8·K·n·p bytes. Chunked/batched training deferred. |
| Model count blow-up: p grows with one-hot expansion | Rare-level merge before one-hot; `n_t` default 50 but docs recommend 10–20 for flow; expose both. |
| Oversubscription: loky processes × XGBoost threads | Force `n_jobs=1` on XGBRegressor inside workers; user-facing `n_jobs` controls only process count. |
| Min-max scaling degenerate on constant columns | Range = 0 → map to 0.0 and remember to emit constant column on decode. |
| Round-trip drift: float scaling + datetime days | float64 throughout; round to day on decode; clip to observed training range on decode. |
| XGBoost version drift changes behavior | Pin `xgboost` minimum in pyproject (`>=3.0`); dev lockfile pins exact. |

## Verification checkpoints

1. After task 2: encoder round-trip test green (exact, incl. dates/NaNs).
2. After task 4: parallel fit produces identical predictions to sequential
   fit on a fixed seed; `n_jobs=1` equals `n_jobs=4` output exactly.
3. After task 5: end-to-end `fit`/`sample` shape/dtype/domain invariants green.
4. After task 6: integration fidelity thresholds green + clean-venv install
   of built wheel works.
5. Final: all four AGENTS.md checks pass.
