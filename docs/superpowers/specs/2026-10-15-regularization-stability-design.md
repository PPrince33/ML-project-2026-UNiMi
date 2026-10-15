# Regularization and Stability — Design Spec

**Course:** Statistical Methods for Machine Learning, A.Y. 2025/26, Università degli Studi di Milano
**Assignment:** 3 — Regularization and Stability
**Reference paper:** O. Bousquet & A. Elisseeff, *Stability and Generalization*, JMLR 2, 2002
**Target deadline:** 30/10/2026
**Repository:** https://github.com/PPrince33/ML-project-2026-UNiMi
**Date:** 2026-10-15

---

## 1. Goal

Empirically study how the strength of L2 regularization (λ) affects
**algorithmic stability** and **generalization**, using ridge regression and
unregularized least squares (OLS) implemented from scratch, on one real and one
synthetic regression dataset.

The expected findings to test (not assume):

1. Increasing λ increases stability (prediction changes under leave-one-out shrink).
2. More stable models have a smaller generalization gap (test error − train error).
3. Stability alone does not minimise test error: test error is U-shaped in λ
   because large λ adds bias. The best λ balances the stability term
   (∝ 1/(λn) in the paper) against regularization bias.

Success = correct, leak-free, reproducible methodology; clear plots; a discussion
with positive and negative findings. Accuracy and speed are not graded.

## 2. Constraints (course rules)

- Python; numpy for linear algebra, matplotlib for plots. All learning
  algorithms written from scratch. `sklearn` is used **only** to load the
  Diabetes dataset.
- Train/test split with no leakage: every statistic (standardization, centering)
  is fitted on training data only.
- Fixed seeds; the notebook runs top to bottom in under a minute.
- LaTeX report, 5–10 pages, PDF committed to the repo, containing the mandatory
  plagiarism declaration verbatim.
- The report prose is written by the student. Claude provides the LaTeX
  skeleton, figures and bullet-point notes only.

## 3. Datasets

### 3.1 Synthetic (controlled setting, known ground truth)

- Features: x ∼ N(0, Σ), d = 30, Σᵢⱼ = 0.9^|i−j| (strongly correlated
  neighbouring features → ill-conditioned XᵀX).
- Targets: y = xᵀw* + ε, ε ∼ N(0, σ²), σ = 1. w* ∼ N(0, I_d), drawn once from
  the master seed and fixed for all repeats.
- Main experiment: n_train = 40, n_test = 2000 (fresh draws; test MSE ≈ true
  risk). Bayes-optimal MSE = σ² = 1, used as a reference line.
- Each repeat draws a new training set and a new test set; w* stays fixed.

### 3.2 Real: Diabetes (sklearn `load_diabetes`, raw/unscaled version)

- 442 samples, 10 features, target = disease progression after one year.
- Per repeat: random permutation → 142 test points, and n_train = 50 drawn from
  the remaining 300.
- Contrast run: n_train = 300 (all remaining points), expected to show nearly
  flat curves (regularization matters little when n ≫ d). Reported as a
  negative/contrast result.
- Loaded with `scaled=False` so that all standardization is our own,
  training-only code.

### 3.3 Shared settings

- λ grid: 30 log-spaced values in [10⁻⁵, 10²]. OLS (λ = 0) is shown as a
  separate reference (dashed line / marker).
- Repeats: R = 20 per configuration; all curves are mean ± 1 std across repeats.
- Master seed in `config.py`; all randomness via `numpy.random.default_rng`,
  with child generators derived deterministically per repeat.

### 3.4 Dataset-size extension (synthetic only)

- n_train ∈ {35, 40, 50, 75, 100, 150, 200, 300, 500}, d = 30.
- λ ∈ {0 (OLS), 10⁻³, 10⁻², 10⁻¹, 1}.
- Expected: ridge stability decays roughly like 1/n; OLS instability spikes as
  n → d and recovers as n grows.

## 4. Models

Objective (the paper's normalization, so λ is comparable across n):

    minimise (1/n) Σᵢ (yᵢ − wᵀxᵢ − b)² + λ‖w‖²

Pipeline inside `fit(X, y)` (the whole pipeline is "the learning algorithm"):

1. Compute μ_X, s_X (per-feature mean/std) and ȳ **from the given training data**.
   Features with zero std are left unscaled (std set to 1) to avoid division by zero.
2. Z = (X − μ_X) / s_X, t = y − ȳ.
3. Solve for w:
   - Ridge: (ZᵀZ + nλI) w = Zᵀt via `np.linalg.solve` (no explicit inverse).
   - OLS: w = pinv(Z) t (minimum-norm solution; works when ZᵀZ is singular).
4. Intercept b = ȳ (unpenalised; follows from centering).

`predict(X)` = ((X − μ_X) / s_X) w + ȳ.

Because standardization lives inside the model, leave-one-out refits recompute
μ_X, s_X, ȳ on S∖i. This avoids leaking the removed point through the scaling
statistics.

## 5. Stability measurement

For a training set S = {z₁, …, z_n}, a fixed test set T = {x'₁, …, x'_m}, and
model factory `make_model` (returns a fresh untrained model):

1. f_S = make_model().fit(S).
2. For each i = 1..n: f_{S∖i} = make_model().fit(S without zᵢ) (brute-force refit).
3. Per removed point i:
   - Δᵢ^pred = (1/m) Σⱼ |f_S(x'ⱼ) − f_{S∖i}(x'ⱼ)|
   - Δᵢ^loss = (1/m) Σⱼ |ℓ(f_S, z'ⱼ) − ℓ(f_{S∖i}, z'ⱼ)|, ℓ = squared error
   - Mᵢ^pred = maxⱼ |f_S(x'ⱼ) − f_{S∖i}(x'ⱼ)|
4. Reported metrics:
   - **mean prediction change** = meanᵢ Δᵢ^pred (average stability, primary metric)
   - **max prediction change** = maxᵢ Mᵢ^pred (empirical proxy for uniform stability β)
   - **mean loss change** = meanᵢ Δᵢ^loss

Remove-one (not replace-one) is used. Correctness is verified against the
closed-form leave-one-out formula in tests (§8).

## 6. Experiments (`experiments.py`)

- `lambda_sweep(dataset, n_train, lambdas, R, seed)` → for each repeat and each
  λ (plus OLS): train MSE, test MSE, gap = test − train, the three stability
  metrics. Returns arrays of shape (R, n_lambdas) plus the OLS arrays of shape (R,).
- `size_sweep(n_values, lambdas, R, seed)` → same metrics indexed by (R, n, λ).

Runs:

| Run | Dataset | n_train | Purpose |
|---|---|---|---|
| A | Synthetic | 40 | main |
| B | Diabetes | 50 | main |
| C | Diabetes | 300 | contrast (negative result) |
| D | Synthetic | 35…500 | dataset-size extension |

## 7. Figures and table (`plotting.py`, saved as PDF to `figures/`)

1. Stability (mean prediction change) vs λ, log–log, ±std bands, OLS dashed,
   dotted ∝ 1/λ reference. Panels: Synthetic | Diabetes.
2. Train & test MSE vs λ, OLS dashed, best-λ marker; synthetic panel shows
   Bayes MSE = 1.
3. Stability vs generalization gap, one point per λ, coloured by log λ.
4. Size extension: stability vs n_train (log–log) per λ, 1/n reference, marker at n = d.
5. Diabetes n_train = 300 contrast: stability and test MSE vs λ.
6. Metric agreement: max prediction change and mean loss change vs λ.

Table: OLS vs best-λ ridge per dataset — train MSE, test MSE, gap, mean
prediction change (mean ± std over R), and R² for Diabetes.

`plotting.py` only draws; it never computes metrics.

## 8. Testing (`tests/`, pytest)

- **Ridge:** matches a direct normal-equation solve; λ → 0 matches OLS on
  full-rank data; λ → ∞ gives w → 0 and predictions → training mean of y.
- **OLS:** recovers w* exactly on noise-free full-rank data; with d > n returns
  the minimum-norm solution without error.
- **LOO correctness:** `models.py` exposes a core solver
  `ridge_solve(Z, t, alpha)` that solves (ZᵀZ + αI) w = Zᵀt for an *absolute*
  penalty α (the model calls it with α = n·λ, n = current training size). With
  α held fixed, brute-force refits on S∖i match the closed-form leave-one-out
  update (Sherman–Morrison) to ~1e-10. (The fixed α is required because the
  shortcut assumes an unchanged penalty matrix; the experiments themselves use
  α = (n−1)λ for S∖i, consistent with the paper's normalization.)
- **No leakage:** perturbing test targets/features leaves fitted parameters
  unchanged; scaler statistics equal those computed on the training rows only.
- **Reproducibility:** the same seed gives bit-identical experiment outputs.
- **Sanity:** stability metrics ≥ 0.

## 9. Repository layout

```
regstab/            config.py, data.py, models.py, stability.py, experiments.py, plotting.py
tests/              test_models.py, test_stability.py, test_data.py, test_experiments.py
notebooks/          assignment3.ipynb
figures/            generated PDFs (committed, used by the report)
report/             main.tex, refs.bib, NOTES.md
docs/superpowers/   specs/, plans/
STUDY_GUIDE.md      concepts, figure walkthrough, ~15 likely oral questions + answers
README.md, requirements.txt, .gitignore
```

## 10. Documentation and study material

- Code comments explain the *why* and the maths (normal equations, nλ scaling,
  solve vs inv, why standardization is inside the model, the definition of each metric).
- `STUDY_GUIDE.md`: the paper's idea in plain language; uniform vs average stability;
  why λ stabilizes (eigenvalues of ZᵀZ + nλI are ≥ nλ, bounding how much one point
  can move w); the bias–stability trade-off; a walkthrough of each figure; likely
  oral questions with model answers.
- `report/NOTES.md`: per-section bullet notes with the actual numbers, for the
  student to write up in their own words.
- `report/main.tex`: section skeleton, verbatim declaration, figure includes,
  Bousquet & Elisseeff citation. Compiled on Overleaf or a local MacTeX
  (pdflatex is not installed on this machine).

## 11. Out of scope

L1/lasso, the dimensionality extension, replace-one stability, kernel ridge,
hyperparameter selection by cross-validation (λ is swept, not tuned), result caching.

## 12. Git

Local commits per milestone. Pushing to `origin` happens only after explicit
approval from the student.
