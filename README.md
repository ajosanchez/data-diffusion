# data-diffusion

Synthetic tabular data with an XGBoost diffusion model (ForestFlow, after
Jolicoeur-Martineau et al., AISTATS 2024). Train on a pandas DataFrame,
sample a synthetic DataFrame with identical columns, dtypes, and domains.

## Disclaimer

**This is not anonymization.** `data-diffusion` does not scrub, redact, or
remove PII, and synthetic output is **not** guaranteed to be private,
de-identified, or safe to share. "Anonymized" here only means *generic* —
rows are generated, not copied. Apply a real privacy evaluation
(membership-inference risk, distance-to-closest-record, etc.) before
treating any output as safe for release.

## Install

```bash
uv add data-diffusion
# or
pip install data-diffusion
```

Requires Python ≥ 3.12. Runtime dependencies: `numpy`, `pandas`, `xgboost`,
`joblib` — no GPU needed; training runs on CPUs in parallel.

## Usage

Two calls: `fit` on a DataFrame, `sample` from the fitted model.

```python
import pandas as pd
from data_diffusion import TabularDiffusion

df = pd.read_csv("data.csv")  # numeric, bool, datetime, string/categorical

model = TabularDiffusion(n_t=50, K=100, n_jobs=4)
model.fit(df)

synthetic = model.sample(10_000, seed=0)
```

Column types are inferred automatically: bool → binary, int → rounded and
clipped integers, float → continuous, datetime → dates, string/categorical
→ one-hot with rare levels merged into `"Other"`. Missing values (NaN/NaT)
are supported in training data; sampled rows are complete (missingness is
not modeled).

## Parameters

| Parameter | Default | Meaning |
| --- | --- | --- |
| `n_t` | 50 | Number of flow levels on the t grid in [0, 1) |
| `K` | 100 | Duplication factor: noise draws per training row |
| `n_jobs` | 1 | joblib workers for the per-level booster fits |
| `seed` | 0 | Fit-time noise seed (sampling seed goes to `sample`) |
| `rare_min_count` | 20 | Categorical levels with fewer rows merge into `"Other"` |
| `rare_min_fraction` | 0.01 | …or a smaller share than this |
| `xgb_kwargs` | `None` | Overrides for XGBoost booster parameters |

Results are deterministic for a fixed `(seed, n_t, K, data, xgb_kwargs)`
regardless of `n_jobs`.

## How it works

1. **Encode** — every column becomes float features scaled to a common
   range; categoricals become exclusive one-hot blocks with rare-level
   merging; datetimes become epoch days.
2. **Fit** — each training row is paired with K Gaussian noise draws to
   form noisy interpolations `x_t = (1−t)·z + t·x`; one XGBoost booster per
   (t level, column) predicts the clean value from the full noisy state.
3. **Sample** — start from Gaussian noise and integrate toward data space
   with DDIM-style Euler steps over the training t grid.
4. **Decode** — invert the encoding: round/clip integers, cast datetimes,
   take `argmax` over one-hot blocks.

## Development

```bash
uv sync          # install with dev dependencies
uv run pytest    # test suite
uv run ty check && uv run ruff format . && uv run ruff check .
```

## License

MIT — see [LICENSE](LICENSE).
