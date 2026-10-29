# Notes for the report

All numbers: seed 2026, R = 20 repetitions, mean over repetitions (± std where given). Every number below is printed by `notebooks/assignment3.ipynb` (summary table in §5, everything else in §6 "Every statistic quoted"). Re-check them there if you change anything. Section numbers match `main.tex`. For explanations in plain language, see `STUDY_GUIDE.md`.

## Abstract
- Question: does stronger L2 regularization make a learner more stable, and does stability come with better generalization?
- Method: ridge vs OLS from scratch; leave-one-out refits; 3 stability metrics; synthetic (d=30, n=40) and Diabetes (n=50; contrast n=300); 20 repetitions; dataset-size extension.
- Findings: (1) stability improves with λ up to ~6× (synthetic) and the gap shrinks with it (over the 45 (n, λ) settings of the size sweep, stability ranks the gap with Spearman 0.98–0.99); (2) the test error is U-shaped, so stable ≠ accurate; (3) at fixed λ stability decays ≈ 1/n; (4) with plenty of data (Diabetes n=300) regularization barely matters.

## 1 Introduction
- Classical bounds: complexity of the hypothesis class (VC dimension, Rademacher complexity). Bousquet & Elisseeff: properties of the *algorithm* instead.
- Stable algorithm ⇒ small generalization gap; for regularized algorithms (ridge) β = O(1/(λn)).
- Research questions: RQ1 λ vs stability; RQ2 stability vs gap/test error; RQ3 effect of n; RQ4 does it hold on real data?

## 2 Background
### 2.1 Stability
- Notation S, S^{\i}; uniform stability β = sup over training sets S of size n, removed indices i and examples z = (x, y) of |ℓ(f_S,z) − ℓ(f_{S\i},z)|.
- Bound: R ≤ R̂ + 2β + (4nβ + M)√(ln(1/δ)/(2n)). Verify the constants and the theorem number in the paper before writing them.
- Consequence: β = O(1/n) ⇒ gap → 0.
### 2.2 Ridge
- Objective (1/n)Σ(yᵢ − wᵀzᵢ)² + λ‖w‖² (the paper's normalization) ⇒ w = (ZᵀZ + nλI)⁻¹Zᵀy.
- Eigenvalues of ZᵀZ + nλI are ≥ nλ ⇒ ‖A⁻¹‖ ≤ 1/(nλ).
- Sherman–Morrison: w_S − w_{S\i} = A⁻¹zᵢ · rᵢ^{LOO}, exact for a fixed penalty and fixed scaling (our refits re-standardise and use (n−1)λ, so approximate) ⇒ influence × leave-one-out residual ⇒ O(1/(λn)).
- OLS = λ=0; unstable when ZᵀZ has small eigenvalues.

## 3 Datasets
- Synthetic: x ~ N(0, Σ), Σᵢⱼ = 0.9^|i−j|, d = 30; y = xᵀw* + ε, ε ~ N(0,1); w* ~ N(0,I), fixed. n_train = 40, fresh n_test = 2000. Bayes MSE = 1.
- Diabetes (Efron et al. 2004): 442 × 10, target = progression after one year. Loaded unscaled (`scaled=False`): sklearn's default is pre-standardized with all rows (leakage). Per repetition: 142 test, n_train = 50 from the remaining 300; contrast run n_train = 300.
- Condition number of ZᵀZ (one draw): synthetic 5 470 (smallest eigenvalue 0.097); Diabetes n=50: 1 185 (0.18); Diabetes n=300: 505 (2.42).
- Why the subsample: with n=300 ≫ d=10 OLS is already stable (see Figure 5); being honest about this choice is a positive.

## 4 Implementation
- numpy only; sklearn only to load data. Package `regstab/` (config, data, models, stability, experiments, plotting).
- Ridge: `np.linalg.solve` on (ZᵀZ + nλI)w = Zᵀt (not `inv`: cheaper, more accurate). OLS: `pinv` (min-norm when singular).
- Standardization + centring inside `fit` from the data given; unpenalized intercept = ȳ; zero-std features scaled by 1.
- LOO: n brute-force refits per (repetition, λ), fixed test set; ~400k fits in total, ~90 s.
- Metrics: mean |Δpred| (average stability, main), max |Δpred| (worst case, uniform-β proxy), mean |Δloss| (squared loss).
- Testing: 41 pytest tests, e.g. normal equations; λ→0 = OLS; λ→∞ = mean; noise-free recovery; min-norm d>n; core solver's LOO vs Sherman–Morrison (fixed penalty, 1e-10); hand-computed MeanModel example for the full pipeline; leakage on test targets and features (verified by injecting a leak); constant feature in train but not test; reproducibility; large-λ plateau value; influence × residual decomposition.

## 5 Experimental setup
- λ grid: 30 log-spaced values, 1e-5 … 1e2; OLS separately.
- R = 20 repetitions; one train/test draw per repetition shared by all λ; SeedSequence(2026).spawn.
- Runs: A synthetic n=40; B Diabetes n=50; C Diabetes n=300; D synthetic n ∈ {35,40,50,75,100,150,200,300,500} × λ ∈ {0, 1e-3, 1e-2, 1e-1, 1}.
- Reported: train/test MSE, gap = test − train, R² (Diabetes), 3 stability metrics, mean ± std.

## 6 Results
### Figure 1 (stability vs λ)
- Small λ ≡ OLS: 0.441 (synthetic), 2.75 (Diabetes).
- Minimum at λ ≈ 3.6: 0.073 (6.1× better) and 1.15 (2.4× better).
- Then a slight rise to 0.092 / 1.30 at λ=100 (U-shape, explained in the Discussion).
- Decay slope on log-log axes ≈ −0.18 / −0.11 (fitted over λ ∈ [10⁻³, 3.6]), much flatter than 1/λ.
### Figure 2 (errors vs λ)
- Train error increases monotonically; test error is U-shaped.
- Synthetic: best λ = 0.014, test 2.46 ± 0.58 vs OLS 3.97 ± 1.12; train 0.38 vs 0.24; Bayes 1. λ=100: test 21.9.
- Diabetes n=50: best λ = 0.39, test 3 411 ± 440 vs OLS 3 823 ± 599; R² 0.43 vs 0.36.
- Best ridge beats OLS in 18/20 (synthetic) and 15/20 (Diabetes) repetitions.
### Figure 3 (stability vs gap)
- Gap: synthetic OLS 3.73 → 1.42 at λ=100; Diabetes 1 727 → 315.
- Spearman with the gap, across λ: mean |Δpred| 0.94 / 0.97; max |Δpred| 0.996 / 1.000; mean |Δloss| 0.68 / 0.92 (synthetic / Diabetes n=50).
- Caveat: stability and gap are both (nearly) monotone in λ, so the across-λ rank correlation is partly automatic: consistent with the theory, weak as a test.
- Stronger check (size sweep, 45 (n, λ) configurations, two parameters vary): Spearman 0.989 (mean), 0.985 (max), 0.983 (loss).
- Across repetitions at fixed λ: no positive relation (median −0.47 synthetic, −0.55 Diabetes, mean metric). The theorem bounds the expected gap, not individual draws; single-draw gaps are noisy.
### Figure 4 (dataset size)
- Fitted log-log slope of stability vs n (n ≥ 50): λ=1 −1.03; λ=0.1 −1.10; λ=0.01 −1.23; λ=0.001 −1.35; OLS −1.38. The theory predicts −1 at fixed λ.
- OLS: 1.26 at n=35 (≈ d), 0.43 at n=40, 0.0076 at n=500. Test MSE OLS 10.0 → 1.06 (Bayes 1).
- λ=1 stays biased: test 5.05 → 4.18.
### Figure 5 (Diabetes n=50 vs n=300)
- OLS stability 2.75 → 0.38. Ridge minimum 0.19 at n=300 (still ~2× better than OLS).
- Test MSE: OLS 3 054 vs best ridge 3 031 (λ = 0.042); ridge better in only 11/20 repetitions; R² 0.489 vs 0.493.
- Gap 233 (OLS) → 125 (λ=100).
### Figure 6 (other metrics)
- max |Δpred| monotone decreasing: 6.84 → 0.43 synthetic (16×); 70.1 → 3.69 Diabetes (19×).
- mean |Δloss| U-shaped: minimum at λ ≈ 0.67 (synthetic, 0.28 vs OLS 1.56) and λ ≈ 2.0 (Diabetes, 126 vs 278).
### Table
- `figures/summary_table.tex`: OLS vs best-λ ridge for the three runs (λ chosen on the test set: descriptive only).

## 7 Discussion
Positive
- RQ1/RQ2 confirmed in the regime the theory covers: more λ ⇒ more stable ⇒ smaller gap, on both datasets.
- The worst-case (uniform-β proxy) metric is monotone in λ; stability ranks the gap almost perfectly across the 45 (n, λ) settings (Spearman ≈ 0.98–0.99).
- 1/n rate in the size sweep reproduced (slopes ≈ −1 to −1.35).
- Real data (Diabetes n=50) behaves like the synthetic data.
Negative / nuances
- Stability ≠ low test error: large λ is most stable but underfits (test MSE ×9 synthetic at λ=100).
- Average stability is U-shaped: ≈ influence × residual; measured on run A, repetition 1: influence 0.236 → 0.0252 (≈ 1/n = 0.025, floor from the unpenalized intercept), |LOO residual| 2.24 → 1.31 (λ=0.1) → 3.71 (λ=100, underfitting). Plateau = mean |ȳ − ȳ₋ᵢ| (unit-tested). Not covered by the theorem's penalized-everything setting.
- The decay in λ is much slower than 1/λ: (a) for small/medium λ the bound is loose, since a direction only enters the 1/(nλ) regime once nλ exceeds its eigenvalue of ZᵀZ (eigenvalues up to several hundred ⇒ λ ≳ 10); (b) the plateau would eventually exceed any C/λ bound, i.e. the theorem's assumptions are not met (unpenalized intercept, unbounded squared loss with Gaussian noise).
- With plenty of data (n=300) regularization changes test error negligibly (11/20).
- Limitations: stability estimated on a finite test set (the max is a lower estimate of β, not a sup); theorem assumptions not fully met (unpenalized intercept, unbounded loss); remove-one only; best λ chosen on the test set; small dataset; L2 only.

## 8 Conclusion
- Regularization buys stability, and stability controls the generalization gap, but stability alone does not choose λ: the best λ balances stability against bias.
- Matters most when data is scarce or ill-conditioned.
- Future work: replace-one stability; L1 vs L2; varying d; λ by cross-validation.
