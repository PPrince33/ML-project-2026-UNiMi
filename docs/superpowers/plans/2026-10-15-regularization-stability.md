# Regularization and Stability Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a from-scratch numpy study of how ridge regularization strength λ affects leave-one-out stability and generalization on a synthetic dataset and on Diabetes, with figures, tests, a study guide and a LaTeX report skeleton.

**Architecture:** A small package `regstab/` with one responsibility per module: config → data → models → stability → experiments → plotting. A single notebook orchestrates runs A–D and writes PDFs to `figures/`. Standardization lives inside each model's `fit`, so leave-one-out refits recompute all statistics.

**Tech Stack:** Python 3.12, numpy 1.26, matplotlib 3.8, sklearn (only `load_diabetes`), pytest 7.4, Jupyter/nbconvert.

**Spec:** `docs/superpowers/specs/2026-10-15-regularization-stability-design.md`

## Global Constraints

- All learning algorithms from scratch in numpy; `sklearn` imported **only** in `regstab/data.py` for `load_diabetes(return_X_y=True, scaled=False)`.
- Objective: (1/n)‖y − Xw − b‖² + λ‖w‖² → ridge solves (ZᵀZ + nλI)w = Zᵀt with `np.linalg.solve`; never `np.linalg.inv`.
- OLS solves w = `np.linalg.pinv(Z) @ t`.
- All standardization statistics come from the data passed to `fit` (training only). A zero-std feature gets scale 1.
- Randomness only through `np.random.SeedSequence` / `np.random.default_rng`; master seed `SEED = 2026` in `regstab/config.py`.
- Every module and function carries comments explaining *why* and the maths (user studies this for an oral exam).
- Notebook runs top to bottom in < 60 s from the `notebooks/` directory.
- `report/main.tex` contains the plagiarism declaration verbatim (exact text in Task 7).
- Commit after every task; never push without explicit user approval.
- Tests run with `python -m pytest` from the repo root (`pytest.ini` sets `pythonpath = .`).

## Review Focus

1. **Zero-variance feature in a small training subset** (Diabetes `sex` is binary): `fit` must not produce NaN/inf; the constant column contributes nothing. → test in Task 1.
2. **OLS when n is barely above d** (size sweep n = 35, LOO sets of 34, d = 30): stability metrics must be finite, even if huge, so log plots don't break. → test in Task 3.
3. **Test set must be shared across all λ and OLS within one repeat**, otherwise curves are incomparable. Expected: with R = 1, ridge at λ = 1e-10 and OLS give the same test MSE to ~1e-6 relative. → test in Task 4.
4. **Large-λ plateau:** the intercept ȳ is unpenalized, so as λ → ∞ prediction change tends to meanᵢ|ȳ − ȳ₋ᵢ| > 0, not 0. Plots on log axes must not assume values reach 0, and the 1/λ reference must not be presented as matching the plateau. → test in Task 3 (positive floor), noted in Task 7.
5. **Notebook executed from `notebooks/`** (nbconvert's default cwd): imports and figure paths must resolve. `config.FIG_DIR` is absolute, from `Path(__file__)`. → verified by execution in Task 6.

---

### Task 1: Scaffolding, config and models

**Files:**
- Create: `requirements.txt`, `pytest.ini`, `regstab/__init__.py`, `regstab/config.py`, `regstab/models.py`
- Test: `tests/test_models.py`

**Interfaces:**
- Produces (`regstab/config.py`):
  `SEED = 2026`; `LAMBDAS = np.logspace(-5, 2, 30)`; `R = 20`;
  `SYN_D = 30`; `SYN_RHO = 0.9`; `SYN_SIGMA = 1.0`; `SYN_N_TRAIN = 40`; `SYN_N_TEST = 2000`;
  `DIAB_N_TEST = 142`; `DIAB_N_TRAIN = 50`; `DIAB_N_TRAIN_FULL = 300`;
  `SIZE_N_VALUES = (35, 40, 50, 75, 100, 150, 200, 300, 500)`; `SIZE_LAMBDAS = np.array([1e-3, 1e-2, 1e-1, 1.0])`;
  `ROOT = Path(__file__).resolve().parents[1]`; `FIG_DIR = ROOT / "figures"`.
- Produces (`regstab/models.py`):
  - `ridge_solve(Z: ndarray(n,d), t: ndarray(n,), alpha: float) -> ndarray(d,)`: solves (ZᵀZ + αI)w = Zᵀt.
  - `ols_solve(Z, t) -> ndarray(d,)`: pinv.
  - `class RidgeRegression(lam: float)` and `class LeastSquares()`, both with `.fit(X, y) -> self`, `.predict(X) -> ndarray(m,)`, and fitted attributes `mu_`, `scale_`, `y_mean_`, `w_`. Ridge calls `ridge_solve(Z, t, alpha=n*lam)` with n = `len(y)` at fit time. Share the standardize/centre/predict logic through a private base class `_LinearModel` with an abstract `_solve(Z, t)`.
  - `mse(y_true, y_pred) -> float`.

- [ ] **Step 1: Create scaffolding**

`requirements.txt`: `numpy==1.26.4`, `matplotlib==3.8.4`, `scikit-learn>=1.1`, `pytest>=7`, `jupyter`, `nbconvert`.
`pytest.ini`: `[pytest]` / `pythonpath = .` / `testpaths = tests`. `regstab/__init__.py`: one-line docstring. Write `config.py` with the values above, each with a comment giving its reason (e.g. `SYN_N_TRAIN = 40  # just above d=30: OLS is ill-conditioned here`).

- [ ] **Step 2: Write failing tests in `tests/test_models.py`**

```python
def test_ridge_solve_matches_normal_equations():      # random Z (20,5), t, alpha=0.7
    assert np.allclose(ridge_solve(Z, t, 0.7), np.linalg.lstsq(Z.T@Z + 0.7*np.eye(5), Z.T@t, rcond=None)[0])

def test_ridge_tiny_lambda_matches_ols():             # full-rank X (50,5)
    assert np.allclose(RidgeRegression(1e-10).fit(X,y).predict(Xte), LeastSquares().fit(X,y).predict(Xte), atol=1e-6)

def test_ridge_huge_lambda_predicts_training_mean():  # lam=1e8
    m = RidgeRegression(1e8).fit(X, y)
    assert np.allclose(m.w_, 0, atol=1e-6) and np.allclose(m.predict(Xte), y.mean(), atol=1e-4)

def test_ols_recovers_true_weights_noise_free():      # y = 3 + X@w_star, X (100,4)
    assert np.allclose(LeastSquares().fit(X, y).predict(Xte), 3 + Xte@w_star, atol=1e-8)

def test_ols_min_norm_when_d_greater_than_n():        # Z (5,10)
    w = ols_solve(Z, t); assert np.allclose(Z@w, t) and np.allclose(w, np.linalg.pinv(Z)@t)

def test_scaler_uses_training_rows_only():
    m = RidgeRegression(0.1).fit(X, y)
    assert np.allclose(m.mu_, X.mean(0)) and np.allclose(m.scale_, X.std(0)) and m.y_mean_ == pytest.approx(y.mean())

def test_constant_feature_is_finite():                # Review Focus 1: X[:,2] = 1.0
    m = RidgeRegression(0.1).fit(X, y); assert np.all(np.isfinite(m.predict(Xte)))
    assert m.scale_[2] == 1.0
    q = LeastSquares().fit(X, y).predict(Xte); assert np.all(np.isfinite(q))

def test_mse():
    assert mse(np.array([1., 2.]), np.array([1., 4.])) == 2.0
```

- [ ] **Step 3: Run tests, verify they fail**

Run: `python -m pytest tests/test_models.py -v` → FAIL (ImportError).

- [ ] **Step 4: Implement `regstab/models.py`**

`std` with `ddof=0`. Module docstring explains the objective, why `nλ` (paper's normalization: λ comparable across n, gives the 1/(λn) stability rate), why `solve` not `inv` (fewer flops, backward-stable), why the intercept is unpenalized (centring), why standardization is inside `fit` (the algorithm = whole pipeline; LOO must recompute statistics).

- [ ] **Step 5: Run tests, verify they pass**

Run: `python -m pytest tests/test_models.py -v` → 8 passed.

- [ ] **Step 6: Commit**

```bash
git add requirements.txt pytest.ini regstab tests/test_models.py
git commit -m "feat: config and from-scratch ridge/OLS models"
```

---

### Task 2: Data generation and loading

**Files:**
- Create: `regstab/data.py`
- Test: `tests/test_data.py`

**Interfaces:**
- Consumes: `config` constants.
- Produces:
  - `toeplitz_cov(d: int, rho: float) -> ndarray(d,d)` with Σᵢⱼ = ρ^|i−j|.
  - `make_true_weights(d: int, rng: Generator) -> ndarray(d,)`: w* ∼ N(0, I).
  - `make_synthetic(n: int, w_star, rho: float, sigma: float, rng) -> (X (n,d), y (n,))`: X = G Lᵀ with L = cholesky(Σ), G standard normal; y = X w* + σε.
  - `load_diabetes_raw() -> (X (442,10), y (442,))`: `load_diabetes(return_X_y=True, scaled=False)`.
  - `diabetes_split(X, y, n_train: int, n_test: int, rng) -> (X_tr, y_tr, X_te, y_te)`: one permutation; first `n_test` indices are test, next `n_train` are train (disjoint by construction).

- [ ] **Step 1: Write failing tests**

```python
def test_toeplitz_cov():  S = toeplitz_cov(4, 0.9); assert S[0,3] == pytest.approx(0.9**3) and np.allclose(S, S.T)
def test_synthetic_shapes_and_covariance():           # n=200_000, d=5, sigma=0
    X, y = make_synthetic(200_000, w, 0.9, 0.0, rng); assert X.shape == (200_000, 5)
    assert np.allclose(np.cov(X.T), toeplitz_cov(5, 0.9), atol=0.02) and np.allclose(y, X@w)
def test_synthetic_deterministic():                    # same seed twice → np.array_equal
def test_diabetes_raw_is_unscaled():   X, y = load_diabetes_raw(); assert X.shape == (442,10) and X[:,0].max() > 1
def test_diabetes_split_sizes_and_disjoint():
    Xtr, ytr, Xte, yte = diabetes_split(X, y, 50, 142, rng)
    assert len(Xtr) == 50 and len(Xte) == 142
    assert not ({tuple(r) for r in Xtr} & {tuple(r) for r in Xte})   # rows are unique in Diabetes
```

- [ ] **Step 2: Run, verify fail** — `python -m pytest tests/test_data.py -v` → FAIL (ImportError).

- [ ] **Step 3: Implement `regstab/data.py`** — comments: why correlated features (ill-conditioned XᵀX ⇒ unstable OLS), why fresh large test set (≈ true risk), why `scaled=False` (sklearn's pre-scaled version uses all 442 rows = leakage), why the Diabetes subsample is small.

- [ ] **Step 4: Run, verify pass** — 5 passed.

- [ ] **Step 5: Commit** — `git commit -m "feat: synthetic generator and Diabetes split"`

---

### Task 3: Leave-one-out stability

**Files:**
- Create: `regstab/stability.py`
- Test: `tests/test_stability.py`

**Interfaces:**
- Consumes: any model with `.fit/.predict`; `models.mse`, `models.ridge_solve`, `RidgeRegression`, `LeastSquares`.
- Produces: `loo_stability(make_model: Callable[[], Model], X_tr, y_tr, X_te, y_te) -> dict[str, float]` with keys
  `train_mse`, `test_mse` (of f_S), `mean_pred_change`, `max_pred_change`, `mean_loss_change`, defined exactly as spec §5. It fits f_S once and f_{S∖i} for each i via `np.delete(X_tr, i, axis=0)`. Returning f_S's errors avoids a second fit in experiments.

- [ ] **Step 1: Write failing tests**

```python
class MeanModel:   # predicts the training mean of y — LOO effect computable by hand
    def fit(self, X, y): self.m = y.mean(); return self
    def predict(self, X): return np.full(len(X), self.m)

def test_known_values_with_mean_model():
    # y_tr = [0,3,6]: ȳ=3, LOO means 4.5, 3, 1.5 → |Δ| = 1.5, 0, 1.5
    r = loo_stability(MeanModel, np.zeros((3,1)), np.array([0.,3.,6.]), np.zeros((2,1)), np.array([3.,5.]))
    assert r["mean_pred_change"] == pytest.approx(1.0) and r["max_pred_change"] == pytest.approx(1.5)
    # loss change per test point: i=0 → |(3-4.5)²-0|=2.25, |(5-4.5)²-4|=3.75 → 3.0; i=1 → 0; i=2 → |2.25|,|12.25-4|=8.25 → 5.25
    assert r["mean_loss_change"] == pytest.approx((3.0 + 0 + 5.25) / 3)
    assert r["test_mse"] == pytest.approx((0 + 4) / 2)

def test_loo_matches_sherman_morrison():   # core solver, fixed alpha = 2.0, Z (30,6)
    A_inv = np.linalg.inv(Z.T@Z + 2.0*np.eye(6))   # inv allowed in the test as an independent reference
    for i in range(30):
        brute = ridge_solve(np.delete(Z,i,0), np.delete(t,i), 2.0)
        z = Z[i]; Ai = A_inv + np.outer(A_inv@z, z@A_inv) / (1 - z@A_inv@z)
        assert np.allclose(brute, Ai @ (Z.T@t - z*t[i]), atol=1e-10)

def test_identical_training_points_zero_change():   # 10 copies of the same row and target
    assert loo_stability(lambda: RidgeRegression(0.1), ...)["mean_pred_change"] == pytest.approx(0, abs=1e-12)

def test_metric_ordering_and_nonnegative():   # synthetic, n=40
    assert 0 <= r["mean_pred_change"] <= r["max_pred_change"] and r["mean_loss_change"] >= 0

def test_more_regularization_more_stable():   # synthetic n=40, d=30
    assert stab(lam=10.0) < stab(lam=1e-3) < stab_ols

def test_ols_near_square_is_finite():         # Review Focus 2: n=31, d=30
    assert all(np.isfinite(v) for v in loo_stability(LeastSquares, ...).values())

def test_test_targets_do_not_affect_fit():    # no leakage: y_te only enters test_mse / loss metrics
    a = loo_stability(lambda: RidgeRegression(0.1), X, y, Xte, yte)
    b = loo_stability(lambda: RidgeRegression(0.1), X, y, Xte, yte + 100.0)
    for k in ("train_mse", "mean_pred_change", "max_pred_change"): assert a[k] == b[k]
    assert a["test_mse"] != b["test_mse"]

def test_large_lambda_plateau_is_positive():  # Review Focus 4: lam=1e6
    r = loo_stability(lambda: RidgeRegression(1e6), X, y, Xte, yte)
    loo_means = [np.delete(y, i).mean() for i in range(len(y))]
    assert r["mean_pred_change"] == pytest.approx(np.mean(np.abs(y.mean() - np.array(loo_means))), rel=1e-3)
```

- [ ] **Step 2: Run, verify fail** — `python -m pytest tests/test_stability.py -v` → FAIL.

- [ ] **Step 3: Implement `regstab/stability.py`** — docstring maps each metric to the paper: max ↔ uniform stability β (Def. 6), mean ↔ average/hypothesis stability; explain remove-one vs replace-one, and why the test set is fixed.

- [ ] **Step 4: Run, verify pass** — 8 passed.

- [ ] **Step 5: Commit** — `git commit -m "feat: leave-one-out stability metrics"`

---

### Task 4: Experiments

**Files:**
- Create: `regstab/experiments.py`
- Test: `tests/test_experiments.py`

**Interfaces:**
- Consumes: `data.*`, `RidgeRegression`, `LeastSquares`, `loo_stability`, `config`.
- Produces:
  - `METRICS = ("train_mse", "test_mse", "gap", "mean_pred_change", "max_pred_change", "mean_loss_change", "test_r2")`
  - `lambda_sweep(dataset: Literal["synthetic","diabetes"], n_train: int, lambdas: ndarray, R: int, seed: int) -> dict` with
    `{"dataset", "n_train", "lambdas", "ridge": {metric: ndarray(R, L)}, "ols": {metric: ndarray(R,)}}`.
  - `size_sweep(n_values: Sequence[int], lambdas: ndarray, R: int, seed: int) -> dict` with
    `{"n_values": ndarray(N,), "lambdas", "ridge": {metric: ndarray(R, N, L)}, "ols": {metric: ndarray(R, N)}}` (synthetic, d = `SYN_D`).
  - `mean_std(a: ndarray, axis=0) -> (ndarray, ndarray)`.
- Seeding (fixed decision): `ss = np.random.SeedSequence(seed)`; `w_ss, *rep_ss = ss.spawn(R + 1)`; `w_star = make_true_weights(SYN_D, default_rng(w_ss))`; repeat r uses `default_rng(rep_ss[r])` to draw **one** train/test set shared by every λ and by OLS (for `size_sweep`, one test set per repeat and a fresh training set per n). `gap = test_mse − train_mse`; `test_r2 = 1 − test_mse / var(y_te)`.

- [ ] **Step 1: Write failing tests** (use `R=2`, 3 lambdas, small sizes for speed)

```python
def test_lambda_sweep_shapes():   res = lambda_sweep("synthetic", 40, np.array([1e-3,1e-1,1.]), R=2, seed=0)
    assert res["ridge"]["test_mse"].shape == (2,3) and res["ols"]["gap"].shape == (2,)
    assert set(res["ridge"]) == set(METRICS)
def test_reproducible():          # two calls, same seed → np.array_equal for every metric (ridge and ols)
def test_different_seed_differs():
def test_gap_definition():        assert np.allclose(r["gap"], r["test_mse"] - r["train_mse"])
def test_shared_test_set():       # Review Focus 3, both datasets, R=1
    res = lambda_sweep(ds, n, np.array([1e-10]), R=1, seed=0)
    assert res["ridge"]["test_mse"][0,0] == pytest.approx(res["ols"]["test_mse"][0], rel=1e-6)
def test_size_sweep_shapes():     res = size_sweep([35, 50], np.array([1e-2, 1.]), R=2, seed=0)
    assert res["ridge"]["mean_pred_change"].shape == (2,2,2) and res["ols"]["test_mse"].shape == (2,2)
```

- [ ] **Step 2: Run, verify fail.**
- [ ] **Step 3: Implement `regstab/experiments.py`** — comments: why one test set per repeat, why SeedSequence.spawn (independent, reproducible streams), why R = 20.
- [ ] **Step 4: Run, verify pass** — 6 passed. Then time the full runs: `python -c "import time,regstab.experiments as e,regstab.config as c; t=time.time(); e.lambda_sweep('synthetic',40,c.LAMBDAS,20,c.SEED); e.lambda_sweep('diabetes',50,c.LAMBDAS,20,c.SEED); e.lambda_sweep('diabetes',300,c.LAMBDAS,20,c.SEED); e.size_sweep(c.SIZE_N_VALUES,c.SIZE_LAMBDAS,20,c.SEED); print(time.time()-t)"` → < 60 s.
- [ ] **Step 5: Commit** — `git commit -m "feat: lambda and dataset-size sweeps"`

---

### Task 5: Plotting and summary table

Load the `dataviz` skill before writing this module.

**Files:**
- Create: `regstab/plotting.py`
- Test: `tests/test_plotting.py`

**Interfaces:**
- Consumes: result dicts from Task 4; `mean_std`.
- Produces (each saves a PDF to `path` and returns the `Figure`; computes nothing beyond mean/std/argmin):
  - `plot_stability_vs_lambda(results: dict[str, dict], path)` — Figure 1; one panel per key (panel title = key); OLS as a dashed horizontal line; dotted c/λ reference anchored at the panel's median λ.
  - `plot_error_vs_lambda(results, path, bayes_mse: dict[str, float | None])` — Figure 2; best λ (argmin of mean test MSE) marked.
  - `plot_stability_vs_gap(results, path)` — Figure 3; colour = log10 λ, colourbar.
  - `plot_size_sweep(size_res, d: int, path)` — Figure 4; 1/n reference; vertical marker at n = d.
  - `plot_contrast(res_small, res_full, path)` — Figure 5.
  - `plot_metric_agreement(results, path)` — Figure 6.
  - `summary_rows(results) -> list[dict]` — per dataset and per model (OLS, best-λ ridge): keys `dataset, model, lam, train_mse, test_mse, gap, mean_pred_change, test_r2`, each value a `(mean, std)` tuple except `dataset/model/lam`.
  - `rows_to_latex(rows) -> str` — a `tabular` environment; R² column shown as "–" for synthetic.
- Shared style: log x-axis for λ, ±1 std shaded bands, the same colour for a dataset/λ across figures, axis labels with units ("MSE", "mean |Δ prediction|").

- [ ] **Step 1: Write failing tests** — build a fake result dict (R=3, L=4) with a known test-MSE minimum at index 2.

```python
def test_each_plot_writes_pdf(tmp_path):   # call all six functions → each file exists and size > 1000 bytes
def test_summary_picks_best_lambda():      assert ridge_row["lam"] == lambdas[2]
def test_rows_to_latex():                  s = rows_to_latex(rows); assert "\\begin{tabular}" in s and "OLS" in s
```

- [ ] **Step 2: Run, verify fail.**
- [ ] **Step 3: Implement `regstab/plotting.py`** (`matplotlib.use("Agg")` at the top so tests run headless).
- [ ] **Step 4: Run, verify pass** — 3 passed; full suite `python -m pytest` all green.
- [ ] **Step 5: Commit** — `git commit -m "feat: figures and summary table"`

---

### Task 6: Notebook, figures and README

**Files:**
- Create: `notebooks/assignment3.ipynb`, `README.md`
- Create (generated): `figures/fig1_stability_vs_lambda.pdf` … `figures/fig6_metric_agreement.pdf`, `figures/summary_table.tex`

**Interfaces:**
- Consumes: everything above.

- [ ] **Step 1: Write the notebook** (author it with `nbformat` from a scratch script). First cell: `sys.path.insert(0, str(Path.cwd().parent))` with a comment explaining why. Sections, each a markdown cell explaining the *idea* followed by a code cell: 1 Setup & config · 2 Datasets (shapes, condition number of ZᵀZ for synthetic vs Diabetes-50 vs Diabetes-300, which explains everything downstream) · 3 Run A/B/C/D · 4 Figures 1–6 (each followed by a "what to notice" markdown cell left as a short placeholder that Task 7 fills with observed numbers) · 5 Summary table (printed, and written to `figures/summary_table.tex`).
- [ ] **Step 2: Execute from `notebooks/`** (Review Focus 5)

Run: `cd notebooks && time jupyter nbconvert --to notebook --execute --inplace assignment3.ipynb`
Expected: exits 0, real time < 60 s, 6 PDFs and `summary_table.tex` exist in `figures/`.

- [ ] **Step 3: Inspect each figure** (convert to PNG in the scratchpad and view). Check: stability decreasing in λ on both datasets; OLS line above all ridge curves; U-shaped test MSE on synthetic; flat curves in Figure 5; plateau visible at large λ. If something contradicts the spec's expectations, record it and investigate before moving on; do not tune parameters to force a result.
- [ ] **Step 4: Write `README.md`** — project summary, install (`pip install -r requirements.txt`), commands (`python -m pytest`, the nbconvert line), layout, links to spec/report/study guide.
- [ ] **Step 5: Commit** — `git add notebooks README.md figures && git commit -m "feat: experiment notebook and generated figures"`

---

### Task 7: Study guide, report skeleton and notes

**Files:**
- Create: `STUDY_GUIDE.md`, `report/main.tex`, `report/refs.bib`, `report/NOTES.md`
- Modify: `notebooks/assignment3.ipynb` (fill the "what to notice" cells with actual numbers), re-execute.

- [ ] **Step 1: `STUDY_GUIDE.md`** — plain language: the paper's idea (stability instead of hypothesis-class complexity); uniform vs average stability; why λ stabilizes (eigenvalues of ZᵀZ + nλI ≥ nλ ⇒ ‖w_S − w_{S∖i}‖ is O(1/(λn))); the bias–stability trade-off; why standardization is inside the model; the large-λ plateau from the unpenalized intercept; a walkthrough of each figure with the actual numbers; ~15 likely oral questions with model answers; a 5-minute talk outline.
- [ ] **Step 2: `report/main.tex` + `refs.bib`** — `article` class, sections: Introduction · Background (stability, ridge) · Datasets · Implementation · Experimental setup · Results (Figures 1–6 via `\includegraphics{../figures/...}`, `\input{../figures/summary_table.tex}`) · Discussion (positive and negative) · Conclusion · Declaration. Every section body is a `% TODO (you): ...` comment pointing to the matching `NOTES.md` section; no prose. `refs.bib` contains Bousquet & Elisseeff 2002 (JMLR 2:499–526). Declaration paragraph, verbatim:

  > I declare that this material, which I now submit for assessment, is entirely my own work and has not been taken from the work of others, save and to the extent that such work has been cited and acknowledged within the text of my work. I understand that plagiarism, collusion, and copying are grave and serious offences in the university and accept the penalties that would be imposed should I engage in plagiarism, collusion or copying. This assignment, or any part of it, has not been previously submitted by me or any other person for assessment on this or any other course of study.

- [ ] **Step 3: `report/NOTES.md`** — per report section: bullet facts and the actual numbers from the executed notebook (OLS vs best λ, stability ratios, plateau value, size-sweep slope estimate), plus positive and negative findings. Bullets only.
- [ ] **Step 4: Verify**

Run: `grep -c "has not been previously submitted by me or any other person" report/main.tex` → `1`; re-run the nbconvert command → exit 0; `python -m pytest` → all pass; spot-check three numbers in NOTES.md against the notebook output.

- [ ] **Step 5: Commit** — `git commit -m "docs: study guide, report skeleton and notes"`
