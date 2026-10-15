# Study guide: Regularization and Stability

Everything you need to explain this project at the oral exam, in plain language. The numbers quoted come from the committed run (seed 2026, R = 20 repetitions). If you change anything and re-run, re-check them in the notebook.

---

## 1. The 60-second version

> Classical learning theory bounds the generalization error by the *complexity of the hypothesis class*. Bousquet & Elisseeff (2002) take a different route: they look at the *learning algorithm* and ask how much its output changes when one training point is removed. If no single point can change the learned function much (the algorithm is **stable**), the training error must be close to the test error. For ridge regression the stability constant shrinks like **1/(λn)**: more regularization or more data means more stability, which means a smaller generalization gap.
>
> I implemented ridge and OLS from scratch, measured stability by literally refitting the model *n* times with each training point removed, and compared it with the train/test errors on a synthetic dataset and on Diabetes.
>
> **Results:** (1) Regularization improves stability by up to 6× (synthetic) and the generalization gap shrinks with it. The worst-case stability metric and the gap are almost perfectly rank-correlated (Spearman 0.996–1.0). (2) Stability is **not** the same as low test error: the test error is U-shaped in λ, because a very stable model can be uniformly bad (underfitting). (3) The *average* prediction change is itself U-shaped, because the unpenalized intercept and underfitting residuals put a floor under it. (4) At fixed λ, stability improves like 1/n, as the theory predicts.

---

## 2. Concepts you must be able to explain

### 2.1 Generalization gap
- **Training error** R̂(f) = average loss on the training set; **true risk** R(f) = expected loss on new data (estimated by the test error).
- **Gap** = R(f) − R̂(f). An algorithm *generalizes* if the gap is small. The gap is what theory bounds, *not* the test error itself.

### 2.2 Stability (the paper's idea)
- S = {z₁,…,zₙ}; S^{\i} = S with point i removed. f_S and f_{S\i} are the functions learned on the two sets.
- **Uniform stability β**: for every i and every input x, |ℓ(f_S, z) − ℓ(f_{S\i}, z)| ≤ β. It is a *worst case* over points and inputs.
- **Main theorem (informally):** with probability ≥ 1 − δ,
  R(f_S) ≤ R̂(f_S) + 2β + (4nβ + M)·√(ln(1/δ) / (2n)),
  where M bounds the loss. If β = O(1/n), the right-hand side converges to the training error, so the gap goes to 0. (Check the exact constants in the paper before quoting them.)
- **Intuition:** if removing any single point barely changes the model, the model cannot have *memorized* that point. Its error on a training point is then close to its error on an unseen point, which is exactly what a small gap means.

### 2.3 Why ridge is stable: the key derivation
Ridge with our normalization minimizes (1/n)‖y − Zw‖² + λ‖w‖², whose solution is

  w = (ZᵀZ + nλI)⁻¹ Zᵀy = A⁻¹ Zᵀy.

- ZᵀZ is positive semidefinite (eigenvalues ≥ 0), so **every eigenvalue of A is ≥ nλ**, hence ‖A⁻¹‖ ≤ 1/(nλ).
- Removing point i is a rank-one change of A. The Sherman–Morrison formula gives the exact identity (for a fixed penalty):

  **w_S − w_{S\i} = A⁻¹ zᵢ · (yᵢ − f_{S\i}(xᵢ))**

  i.e. *(influence of point i)* × *(the leave-one-out residual of point i)*.
- With bounded inputs and residuals: ‖w_S − w_{S\i}‖ ≤ ‖zᵢ‖·|rᵢ|/(nλ) = **O(1/(λn))**. That is the 1/(λn) rate.
- **OLS** (λ = 0): A = ZᵀZ, whose smallest eigenvalue can be close to 0 (few points, correlated features). Dividing by it **amplifies** small data changes. Our synthetic training set has condition number ≈ 5 500, so OLS is unstable there.

### 2.4 Bias–stability trade-off
Large λ means very stable, but the model is pulled towards "predict the mean", so it underfits and both train and test errors rise. Small λ means less bias, but the model is unstable. The best test error sits in between. This is the stability version of the bias–variance trade-off.

---

## 3. Implementation choices (and how to defend each)

| Choice | Why |
|---|---|
| **From scratch:** `np.linalg.solve` on the normal equations | Course rule. `solve` instead of `inv`: fewer operations and numerically more accurate. |
| **OLS with `pinv`** | Works even when ZᵀZ is singular (d > n, constant feature), returning the minimum-norm solution. |
| **λ normalization** (1/n)‖·‖² + λ‖w‖² ⇒ penalty nλ | Same convention as the paper, so λ means the same thing for any n and the 1/(λn) rate is directly comparable. |
| **Standardization inside `fit`** | The learning algorithm is the whole pipeline. In a leave-one-out refit the mean/std are recomputed without point i; otherwise the removed point would still leak in through the scaling statistics. It also guarantees no test-set leakage. |
| **Unpenalized intercept** (centre y) | Penalizing the intercept would pull predictions towards 0, which is meaningless for a target around 150. |
| **Remove-one, brute-force refits** | The literal definition, easy to verify. Checked against the exact Sherman–Morrison formula in the tests (to 1e-10). |
| **Fixed test set per repetition** | f_S and every f_{S\i} are compared on the same points, so differences come only from removing zᵢ. |
| **Three metrics** | mean \|Δprediction\| (average stability), max \|Δprediction\| (worst case, closest to the paper's uniform β), mean \|Δloss\| (the paper's bounds are on the loss). |
| **Synthetic: d = 30, ρ = 0.9, n = 40** | Correlated features and n just above d give an ill-conditioned ZᵀZ, the regime where regularization matters. Known Bayes error σ² = 1. |
| **Diabetes, n_train = 50 (+300 contrast)** | With all 300 points and d = 10, OLS is already fairly stable. A small sample makes the effect visible; the 300 run is the honest contrast. |
| **`scaled=False` when loading Diabetes** | sklearn's default version is standardized using all 442 rows (test rows included), which would be leakage. |
| **R = 20 repetitions, mean ± std** | One split could be lucky; the bands show the variability. |
| **SeedSequence.spawn** | One master seed gives independent, reproducible streams per repetition. |

---

## 4. The results, figure by figure

**Conditioning (notebook section 2):** condition number of ZᵀZ is ≈ 5 470 (synthetic, n = 40), ≈ 1 185 (Diabetes, n = 50) and ≈ 505 (Diabetes, n = 300). The smallest eigenvalue is 0.10, 0.18 and 2.42 respectively. This explains everything below.

**Figure 1 – stability vs λ.**
- Small λ: ridge = OLS (mean \|Δpred\| 0.44 synthetic, 2.75 Diabetes).
- It decreases to a minimum near λ ≈ 3.6: 0.073 (6× better than OLS) and 1.15 (2.4× better).
- Then it **rises slightly** (0.092 and 1.30 at λ = 100). See §5.1.
- The decay is much slower than the 1/λ slope guide (fitted log-log slope ≈ −0.18 synthetic, −0.11 Diabetes). The theory gives an *upper bound*, and it is not tight here.

**Figure 2 – errors vs λ.**
- Training error grows monotonically with λ; test error is **U-shaped**.
- Best λ ≈ 0.014 (synthetic): test MSE 2.46 vs 3.97 for OLS. Best λ ≈ 0.39 (Diabetes): 3 411 vs 3 823 (R² 0.43 vs 0.36).
- At λ = 100 both errors explode (21.9 synthetic): underfitting.
- Best ridge beats OLS in 18/20 repetitions (synthetic) and 15/20 (Diabetes).

**Figure 3 – stability vs gap.**
- More stable means a smaller gap. Spearman rank correlation between mean \|Δpred\| and the gap across λ: 0.94 (synthetic), 0.97 (Diabetes).
- It is exactly 1.0 on the part of the curve before the stability minimum.
- With the **worst-case** metric (max \|Δpred\|) the correlation is 0.996 / 1.000: the uniform-stability proxy tracks the gap almost perfectly, as the theorem suggests.

**Figure 4 – dataset size (synthetic).**
- At fixed λ, stability falls roughly like 1/n: fitted log-log slopes for n ≥ 50 are −1.03 (λ = 1), −1.10 (λ = 0.1), −1.23 (λ = 0.01), −1.35 (λ = 0.001). Strongly regularized ridge matches the 1/n rate best.
- **OLS explodes near n = d**: 1.26 at n = 35, 0.43 at n = 40, 0.0076 at n = 500.
- Test error: OLS 10.0 at n = 35 vs 1.06 at n = 500, approaching the Bayes error of 1. Large λ (= 1) stays biased even at n = 500 (test MSE 4.2).

**Figure 5 – Diabetes, n = 50 vs n = 300.**
- With 300 points, OLS stability is already 0.38 (vs 2.75 at n = 50), and the best ridge improves test MSE only from 3 054 to 3 031 (beating OLS in just 11/20 repetitions, essentially a coin flip).
- Ridge still halves the prediction change (minimum 0.19), but this does not translate into better test error.
- **Regularization matters when data is scarce or ill-conditioned.**

**Figure 6 – other metrics.**
- **max \|Δpred\| decreases monotonically**: 6.8 to 0.43 synthetic, 70 to 3.7 Diabetes, a 16–19× improvement.
- mean \|Δloss\| is U-shaped like the mean prediction change, with its minimum near λ ≈ 0.7–2.

---

## 5. The surprises: discuss these openly (they are your "negative aspects")

### 5.1 Why is the *average* prediction change U-shaped?
From §2.3, the change is **influence × leave-one-out residual**.
- **Influence** (how much point i can move the model) decreases monotonically with λ. Checked numerically on one synthetic draw: 0.23 to 0.025.
- It cannot go below **1/n**: the intercept is not penalized, and removing a point always shifts the mean of y by (ȳ − yᵢ)/(n−1).
- The **leave-one-out residual** is small when the model fits well, and large when it underfits (λ large). On the same draw: 2.4 at λ = 1e-4, 1.3 at 0.1, 3.6 at 100.
- Their product is U-shaped. At λ → ∞ the stability converges exactly to mean\|ȳ − ȳ₋ᵢ\|, which a unit test checks.
- **The max metric stays monotone** because the worst-case point is a high-leverage point, whose influence λ keeps cutting.

### 5.2 Stability ≠ good test error
The most stable models (large λ) have the smallest gap but the **worst** test error. A stable algorithm guarantees "test ≈ train", not "test is low". Both terms of the trade-off matter.

### 5.3 The bound's rate is not attained in λ
The observed decay in λ (slope ≈ −0.1 to −0.2) is much flatter than 1/λ. The theorem is a worst-case upper bound; real data is far from the worst case. In n, by contrast, the 1/n rate is matched well.

### 5.4 Other limitations
- Stability is measured on a finite test set, not as a sup over all inputs, so we only estimate β.
- Remove-one only; the paper also studies replace-one.
- The "best λ" in the table is chosen on the test set. That is descriptive, not a proper model-selection procedure (which would use cross-validation).
- Diabetes is small; the n = 50 subsample is our choice, and we show the n = 300 contrast.
- Only L2; L1 (lasso) is not studied.

---

## 6. Likely oral questions, with answers

1. **What is algorithmic stability?** How much the learned function changes when one training point is removed (or replaced). Uniform stability is the worst case over the removed point and over all inputs.
2. **Why does stability imply generalization?** If no single point can change the output much, the model cannot depend strongly on any individual training point. Its loss on a training point then behaves like its loss on a fresh point, and concentration inequalities (McDiarmid) turn this into a bound on the gap.
3. **How does it differ from VC-dimension bounds?** VC bounds depend on the hypothesis class (uniform over all functions in it); stability depends on the algorithm. Ridge can be stable even in infinite-dimensional (kernel) spaces where VC bounds are useless.
4. **Why does regularization make ridge stable?** The eigenvalues of ZᵀZ + nλI are at least nλ, so the inverse has norm at most 1/(nλ), and one point can move w by at most O(1/(λn)).
5. **Why is OLS unstable on your synthetic data?** n = 40 is close to d = 30 and the features are correlated (ρ = 0.9), so ZᵀZ has eigenvalues near 0 (condition number ≈ 5 500). Inverting amplifies small changes.
6. **How exactly did you measure stability?** Fit on S, then refit n times on S without point i, predict on a fixed test set, and average (or take the max of) the absolute prediction differences. Repeated 20 times.
7. **Why standardize inside the model?** Because the learning algorithm includes preprocessing. Recomputing the statistics without point i avoids leaking it back in, and fitting them on training data only avoids test leakage.
8. **Why is λ multiplied by n?** We average the loss (1/n), as in the paper. Setting the gradient to zero gives ZᵀZ + nλI. This keeps λ comparable across training-set sizes.
9. **Why did you subsample Diabetes to 50 points?** With 300 points and 10 features, OLS is already well-conditioned and the curves are flat (Figure 5). A small sample shows the regime the paper is about. The 300 run is reported as the contrast.
10. **Your stability curve goes back up at large λ. Is that a bug?** No. The change equals influence × leave-one-out residual. The influence keeps falling but is floored at 1/n by the unpenalized intercept, while the residual grows when the model underfits. The limit value mean\|ȳ − ȳ₋ᵢ\| is verified by a unit test, and the worst-case metric is monotone as expected.
11. **Does more stability always mean better test error?** No. Test error is U-shaped. Stability controls the gap, not the bias. Very large λ is very stable but underfits.
12. **What would happen with more data?** Stability improves like 1/n (Figure 4, slopes ≈ −1.0 to −1.35), OLS becomes stable too, and the benefit of regularization shrinks.
13. **How do you know your implementation is correct?** 32 unit tests, including: ridge matches the normal equations; λ → 0 equals OLS; λ → ∞ predicts the mean; OLS recovers the true weights without noise; brute-force leave-one-out matches Sherman–Morrison to 1e-10; a hand-computed stability example; no-leakage and reproducibility checks.
14. **Why `solve` instead of `inv`?** Solving the system directly is cheaper and more accurate than forming the inverse and multiplying.
15. **What are the limitations?** We estimate stability on a test sample rather than as a supremum, use only remove-one and L2, pick the "best λ" on the test set (descriptive only), and the decay in λ is much slower than the bound's rate.
16. **What would you do next?** Replace-one stability, L1 vs L2 (lasso is expected to be less stable because its selected features flip), varying d, and choosing λ by cross-validation.

---

## 7. A 5-minute talk outline (no slides)

1. **Question (30 s):** the paper's idea in one sentence; what I wanted to check (λ → stability → gap).
2. **Setup (1 min):** ridge and OLS from scratch; the two datasets and why they are ill-conditioned (condition numbers); the leave-one-out refits and the three metrics; 20 repetitions.
3. **Main result (1.5 min):** Figure 1 + Figure 3: stability improves up to 6×, the gap shrinks with it, and the rank correlation is ≈ 1 for the worst-case metric. Figure 4: the 1/n rate.
4. **Nuance (1.5 min):** Figure 2: test error is U-shaped, so stable ≠ accurate. The U-shape of average stability, explained by influence × residual and the intercept floor. Figure 5: with plenty of data, regularization barely matters.
5. **Wrap-up (30 s):** stability explains *when* models generalize; regularization buys stability at the price of bias; correctness is backed by tests.
