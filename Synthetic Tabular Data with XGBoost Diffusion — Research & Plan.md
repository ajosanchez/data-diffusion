# Synthetic Tabular Data with XGBoost Diffusion — Research & Plan

Oct 9, 2026 · @Alex Sanchez

## Summary

Yes, this works and is an established technique: XGBoost as the denoiser inside a diffusion or flow-matching model ("Forest Diffusion" / "ForestFlow", AISTATS 2024). On a 27-dataset benchmark it beat deep-learning generators such as TabDDPM and CTGAN on fidelity, and it trains on CPUs with no GPU ([paper](https://arxiv.org/abs/2309.09968v3)).

**Recommended approach:** build on the maintained `synthyverse` library, which wraps ForestDiffusion plus a newer XGBoost diffusion generator (XGenBoost, 2026) and ships fidelity and privacy-attack metrics. Alongside it, write a \~40-line ForestFlow from scratch in one notebook so you understand and can tune the mechanism. The original ForestDiffusion repo was archived (read-only) on April 1, 2026, so treat it as a pinned dependency, not a moving target.

**The catch:** synthetic is not automatically anonymous. These models can memorize rows, especially rare ones, and none of them carry a formal privacy guarantee. The plan therefore spends as much effort measuring privacy (distance-to-closest-record, membership inference, singling-out/linkability/inference attacks) as it does generating data.

**Dates** are not supported natively by any of these libraries; they are encoded as numbers (epoch days, or an anchor date plus deltas) and decoded after sampling.

## How XGBoost diffusion works

The model learns to turn Gaussian noise into a plausible row, one small step at a time, with XGBoost regressors predicting each step instead of a neural network. ForestFlow (the flow-matching variant, generally better than the VP-diffusion variant) uses a straight path between a noise row z and a real row x:

```latex
x_t = (1 - t)\,z + t\,x, \qquad v^{*} = x - z, \qquad t \in [0, 1]
```

1. Encode the table to numbers (scale floats, one-hot categoricals) and duplicate it K times (default 100), pairing each copy with fresh noise.
2. For each of n\_t time levels (default 50) and each encoded column, fit one XGBoost regressor that predicts that column of v\* from x\_t. That is p × n\_t models, trained in parallel across CPU cores.
3. To sample, start from pure noise and take n\_t Euler steps: x ← x + (1/n\_t) · v̂(x, t), using the models for that level.
4. Decode: invert the scaling, argmax the one-hot blocks, round integers.

Two properties matter for your use case. XGBoost handles missing values natively, so it can train on incomplete data without imputing first ([paper](https://arxiv.org/html/2309.09968v3)). And because it is tree-based, the duplication by K is how it averages over noise; the library's data iterator (`n_batch`) avoids holding all K copies in memory, which let the authors train on 11M rows × 21 features in about 1.5 hours ([repo](https://github.com/SamsungSAILMontreal/ForestDiffusion)).

The newer XGenBoost diffusion model (XGenB-DF) changes two things: categoricals get their own multinomial diffusion with native XGBoost categorical splits (no one-hot), and a dropout on inputs reduces memorization. It ranked best on the same 27-dataset benchmark for fidelity ([XGenBoost](https://arxiv.org/html/2603.06904v1)).

## Handling mixed column types

Every column passes through one reversible encoder (`fit` / `transform` / `inverse_transform`) driven by a schema file, so the generator only ever sees numbers and every output is decoded back to the original dtypes.

| Column kind | Encode for training | Decode after sampling | Notes |
| --- | --- | --- | --- |
| Direct identifier (ID, name, email, free text) | Drop | Regenerate surrogate (UUID, faker) | Never model these; they are pure leakage |
| Categorical, ≥3 levels | Pass in `cat_indexes` (ForestDiffusion one-hots it) or `discrete_features` (synthyverse) | Argmax back to label | Merge rare levels (e.g. < 1% or < 20 rows) into "Other" first: fewer models, less re-identification risk |
| Binary | `bin_indexes` | Threshold at 0.5 |  |
| Integer / ordinal / counts | `int_indexes` | Round, clip to observed range |  |
| Float | Min-max scale (library does this); optionally quantile-transform skewed columns | Inverse transform, clip to training range | Clipping stops impossible extremes |
| Single date | Days since a fixed epoch, as a float | Round to day, add epoch, cast to `datetime64` | Use seconds if time of day matters |
| Related dates (start/end, admit/discharge) | Anchor date as epoch days + each other date as a delta in days from the anchor | Anchor + delta | Model durations, not raw dates, so end ≥ start holds; clip deltas at 0 |
| Missing values | Leave as NaN | Model can emit NaN where the pattern was learned | XGBoost learns split directions for NaN |

Derive calendar features (weekday, month, quarter) **after** decoding, from the generated date, rather than generating them as separate columns that could contradict it. Business rules the encoding cannot guarantee (e.g. `discount ≤ price`) go in a constraints module that repairs or rejects violating rows and reports the violation rate.

## Privacy: what "anonymized" really means here

A generative model gives you rows that are not copies by construction, but not rows that are safe by construction. High-fidelity generators sit on a fidelity–privacy trade-off: in the XGenBoost benchmark, the forest/XGBoost diffusion models ranked near the top on fidelity and near the bottom on distance-to-closest-record, while weaker models like CTGAN and TVAE looked "more private" largely because they were less faithful ([XGenBoost](https://arxiv.org/html/2603.06904v1)).

**Why tree diffusion can memorize:** at low noise levels each XGBoost model sees K slightly noised copies of every real row clustered together, and can learn those clusters, which amounts to memorizing the row. Rare categories and outliers are the most exposed.

**How to measure it** (on a holdout set the generator never saw):

- **Exact and near duplicates** — share of synthetic rows identical to, or within a tiny distance of, a training row.
- **DCR** — are synthetic rows closer to training rows than to holdout rows? About 50/50 means no sign of overfitting. It is a proxy, not proof.
- **Membership inference** — can an attacker tell whether a given real row was in training? synthyverse includes DOMIAS and classifier/ensemble MIA ([docs](https://synthyverse.readthedocs.io/en/latest/)).
- **Anonymeter** — attack-based singling-out, linkability and inference risks, the three GDPR anonymization criteria, each compared against a control set ([Anonymeter](https://arxiv.org/abs/2211.10459)).

**Levers that reduce risk:** drop direct identifiers; merge rare categories; clip to training ranges; lower model capacity (fewer trees, higher `min_child_weight`, `subsample`); input dropout (XGenBoost); and as a last pass, reject synthetic rows closer to a real row than a chosen threshold.

If you need a formal guarantee (regulated health or financial data, external release), none of these methods provide differential privacy; use a DP generator such as PrivBayes for that release and keep XGBoost diffusion for internal use.

## Alternatives and baselines

Run at least one baseline from each row below through the same evaluation, so you can see whether XGBoost diffusion is earning its training cost on your data.

| Method | Engine | Role in this project | Watch out for |
| --- | --- | --- | --- |
| ForestFlow / ForestDiffusion (`ForestDiffusion` package) | XGBoost, flow or VP diffusion | Primary candidate | One-hot blows up model count with high-cardinality categoricals; repo archived Apr 2026 |
| XGBoost diffusion, XGenB-DF (`XGBDiffusionGenerator`, XGB-DDPM in synthyverse) | XGBoost, Gaussian + multinomial diffusion | Primary candidate | Expensive on wide or large tables |
| XGenB-AR (autoregressive, synthyverse) | XGBoost classifiers per feature | Fallback for large tables; trained in about 3 min on 1.66M rows ([XGenBoost](https://arxiv.org/html/2603.06904v1)) | Quantizes numerics, so slightly weaker correlations |
| TabDDPM, TabSyn | Neural diffusion (GPU) | Deep-learning comparison | Tuning sensitive; TabDDPM produced NaNs on some datasets |
| CTGAN, TVAE | GAN / VAE | Low-fidelity, lower-risk reference | Lower fidelity; `ctgan` is under the Business Source License |
| SMOTE | Interpolation between real rows | "Leaky" reference: shows what a privacy failure looks like in your metrics | Not for release |
| Gaussian copula / univariate | Statistical | Floor: anything worse is not worth it | Misses interactions |

## Implementation plan

Reusable logic lives in a small `src/` package; notebooks only orchestrate, plot and record decisions. Every generator sits behind one interface (`fit(df, schema)` / `sample(n)`), so swapping ForestFlow for a baseline is a one-line change.

&#91;embedded content: synthetic data pipeline · 8 steps, 1 tuning loop\]

The 20% holdout never trains the generator and only feeds evaluation, which is what makes the DCR and membership-inference numbers meaningful; a run that fails any gate loops back to retraining.

```
synth-tabular/
  config/schema.yaml        # column roles, quasi-identifiers, sensitive cols, target
  data/raw/  data/splits/   # train.parquet, holdout.parquet (holdout never trains)
  src/synth/
    schema.py               # load/validate schema, infer defaults
    encoders.py             # TabularEncoder: dates, rare-category merge, scaling
    constraints.py          # repair/reject rules, violation report
    generators.py           # ForestFlowScratch, ForestDiffusionGen, SynthyverseGen
    eval_fidelity.py        # shape/trend, C2ST, TSTR
    eval_privacy.py         # duplicates, DCR, MIA, Anonymeter wrappers
  notebooks/  (below)
  runs/                     # one folder per run: config, model, sample, metrics.json
```

1. **00\_env** — Python 3.11 venv; `pip install ForestDiffusion synthyverse anonymeter xgboost pandas pyarrow scikit-learn`. Pin versions in `requirements.txt`. Run on a many-core CPU machine, not Colab (2 weak cores).
2. **01\_profile\_schema** — profile every column (dtype, cardinality, missingness, ranges); write `schema.yaml`; mark identifiers, quasi-identifiers and sensitive columns; make the 80/20 train/holdout split.
3. **02\_encode** — build and unit-test `TabularEncoder`: round-trip `inverse_transform(transform(df)) == df` on the training set, including dates and NaNs.
4. **03\_forestflow\_scratch** — implement the core below on a 5k-row sample to learn how n\_t, K and tree depth change samples and memorization.
5. **04\_forestdiffusion** — train `ForestDiffusionModel` with `cat_indexes`, `bin_indexes`, `int_indexes`, `diffusion_type='flow'`, `n_batch` for memory; log wall time and peak RAM; save the model.
6. **05\_synthyverse** — train `XGBDiffusionGenerator` and two baselines from the table above via `fit(X, discrete_features)`.
7. **06\_fidelity\_utility** — compare all runs on the fidelity and utility metrics in the next section.
8. **07\_privacy** — run duplicates, DCR, MIA and Anonymeter on every run against the holdout.
9. **08\_tune\_release** — sweep the privacy levers, pick the config on the fidelity–privacy frontier, apply the distance filter and constraints, generate the release table and a one-page report of its metrics.

The from-scratch core (notebook 03), assuming `X` is already encoded and scaled to roughly \[-1, 1\]:

```python
import numpy as np
from xgboost import XGBRegressor


def fit_forest_flow(X, n_t=50, K=100, seed=0, **xgb_kw):
    rng = np.random.default_rng(seed)
    X1 = np.repeat(X, K, axis=0)  # data, duplicated K times
    X0 = rng.standard_normal(X1.shape)  # paired noise
    ts = np.arange(n_t) / n_t
    models = {}
    for i, t in enumerate(ts):
        Xt = (1 - t) * X0 + t * X1  # point on the straight path
        V = X1 - X0  # target velocity
        for j in range(X.shape[1]):
            m = XGBRegressor(n_estimators=100, max_depth=7, **xgb_kw)
            models[i, j] = m.fit(Xt, V[:, j])
    return models, ts


def sample_forest_flow(models, ts, n, p, seed=1):
    x = np.random.default_rng(seed).standard_normal((n, p))
    for i, t in enumerate(ts):  # Euler steps from noise to data
        v = np.column_stack([models[i, j].predict(x) for j in range(p)])
        x = x + v / len(ts)
    return x
```

## Evaluation

A release candidate must pass all three groups; the thresholds are starting targets to adjust once you see your baselines.

| Group | Metric | Question it answers | Starting target |
| --- | --- | --- | --- |
| Fidelity | Shape (KS / total variation per column) | Do single-column distributions match? | ≥ 0.90 |
| Fidelity | Trend (pairwise correlation / contingency) | Do pairwise relationships match? | ≥ 0.90 |
| Fidelity | Detection (C2ST: XGBoost real-vs-synthetic AUC) | Can a classifier tell them apart? | ≤ 0.65 |
| Fidelity | Constraint violation rate | Are business rules (date order, ranges) kept? | 0% after repair |
| Utility | TSTR: train on synthetic, test on real holdout | Does a model trained on it still work? | Within 5% of train-on-real |
| Privacy | Exact-duplicate rate | Did it copy rows outright? | 0 |
| Privacy | DCR share closer to train than holdout | Is it overfitting to training rows? | ≤ 0.55 |
| Privacy | Membership inference AUC (DOMIAS, classifier MIA) | Can an attacker detect training members? | ≤ 0.55 |
| Privacy | Anonymeter singling-out, linkability, inference risk | GDPR-style re-identification risk | Near 0, confidence interval overlapping 0 |

Run each metric over several sampling seeds and report mean ± spread; a single sample can look better or worse than the model really is. Use Gower-style distances (min-max scaled numerics, categoricals one-hot at ½, L1) for DCR so categoricals do not dominate, as the XGenBoost evaluation does.

## Risks, limits and open decisions

**Risks**

- **Training cost scales with width.** ForestDiffusion fits p × n\_t models, where p counts one-hot columns; 20 raw columns that expand to 60 encoded columns at n\_t = 50 is 3,000 models. Mitigate with rare-category merging, flow with n\_t = 10–20 (the authors found flow works with few levels), or XGenB-AR for large tables.
- **Memory.** Without the data iterator the table is held K times; set `n_batch` > 0 and lower `n_jobs` if you hit out-of-memory errors.
- **Maintenance.** The original repo is archived; pin `ForestDiffusion` and prefer synthyverse for anything you will keep running.
- **False sense of safety.** Passing DCR alone is not anonymization. Treat the privacy notebook as a gate, and get a privacy or legal review before sharing outside the team if the source data is regulated.
- **Single table only.** These methods do not preserve relationships across multiple linked tables or repeated rows per entity (e.g. many transactions per customer).

**Open decisions**

- [ ] Dataset size: rows, raw columns, and largest categorical cardinality (decides ForestFlow vs XGenB-AR)
- [ ] Which columns are quasi-identifiers and which are sensitive (drives Anonymeter setup)
- [ ] Is there a target column for the utility (TSTR) test?
- [ ] Who receives the synthetic data: internal team, partner, or public? (decides whether a formal DP method is required)
- [ ] Does the data have one row per entity, or repeated rows per entity?

## Sources

- [Jolicoeur-Martineau, Fatras, Kachman — Generating and Imputing Tabular Data via Diffusion and Flow-based Gradient-Boosted Trees (AISTATS 2024)](https://arxiv.org/html/2309.09968v3)
- [ForestDiffusion repository and README (archived Apr 1, 2026)](https://github.com/SamsungSAILMontreal/ForestDiffusion)
- [ForestDiffusion on PyPI](https://pypi.org/project/ForestDiffusion)
- [Achterberg et al. — XGenBoost: Synthesizing Small and Large Tabular Datasets with XGBoost (Mar 2026)](https://arxiv.org/html/2603.06904v1)
- [synthyverse documentation](https://synthyverse.readthedocs.io/en/latest/)
- [synthyverse installation](https://synthyverse.readthedocs.io/en/latest/getting_started.html)
- [synthyverse XGBDiffusionGenerator](https://synthyverse.readthedocs.io/en/latest/api/generators/xgbdiffusion.html)
- [synthyverse ForestDiffusionGenerator](https://synthyverse.readthedocs.io/en/stable/api/generators/forestdiffusion.html)
- [Giomi et al. — Anonymeter: A Unified Framework for Quantifying Privacy Risk in Synthetic Data (PETS 2023)](https://arxiv.org/abs/2211.10459)
