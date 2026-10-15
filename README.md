# Regularization and Stability

Project for **Statistical Methods for Machine Learning** (A.Y. 2025/26, Università degli Studi di Milano), Assignment 3.

An empirical study of how the strength λ of L2 regularization affects the **leave-one-out stability** and the **generalization** of linear regression, following O. Bousquet & A. Elisseeff, *Stability and Generalization*, JMLR 2002.

- Ridge regression and ordinary least squares are implemented **from scratch** in numpy (`regstab/models.py`).
- Stability is measured by refitting the model once for every removed training point and measuring how much its predictions on a fixed test set change (`regstab/stability.py`).
- Datasets: a synthetic correlated-Gaussian linear model (d = 30) and the Diabetes dataset (442 patients, 10 features). `scikit-learn` is used **only** to load Diabetes.
- Extension: the effect of the training-set size on stability.

## Reproduce

```bash
pip install -r requirements.txt
python -m pytest
cd notebooks && jupyter nbconvert --to notebook --execute --inplace assignment3.ipynb
```

The notebook regenerates every figure in `figures/` and the table `figures/summary_table.tex`, in about 1.5 minutes. All randomness derives from `SEED` in `regstab/config.py`, so results are bit-identical across runs.

## Layout

| Path | Content |
|---|---|
| `regstab/config.py` | every experimental setting (seed, λ grid, sizes) |
| `regstab/data.py` | synthetic generator, Diabetes loading and train/test split |
| `regstab/models.py` | `RidgeRegression`, `LeastSquares` (standardisation inside `fit`) |
| `regstab/stability.py` | `loo_stability`: leave-one-out refits and the three stability metrics |
| `regstab/experiments.py` | λ sweep and dataset-size sweep over 20 repetitions |
| `regstab/plotting.py` | Figures 1–6 and the summary table |
| `regstab/analysis.py` | the statistics quoted in the report (rank correlations, fitted slopes, influence × residual) |
| `tests/` | pytest suite (model maths, LOO correctness, leakage, reproducibility) |
| `notebooks/assignment3.ipynb` | runs everything and explains each figure |
| `report/` | LaTeX report |
| `STUDY_GUIDE.md` | concepts, figure walkthrough and likely oral-exam questions |
| `docs/superpowers/` | design spec and implementation plan |
