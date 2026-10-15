"""Every experimental setting lives here, so the whole study can be changed
(or audited) from a single file. Nothing in this module computes anything.

Reproducibility: all randomness in the project is derived from SEED through
numpy's SeedSequence (see experiments.py), so the same SEED always gives
bit-identical results.
"""
from pathlib import Path

import numpy as np

# Master random seed. Every train/test split, synthetic draw and true weight
# vector is derived from this single number.
SEED = 2026

# Regularization strengths for ridge: 30 values evenly spaced on a LOG scale
# from 1e-5 (almost no regularization, close to OLS) to 1e2 (very strong,
# weights pushed almost to zero). A log grid is used because the effect of
# lambda is multiplicative: going 0.001 -> 0.01 matters as much as 1 -> 10.
# OLS (lambda = 0) is handled separately because 0 cannot sit on a log axis.
LAMBDAS = np.logspace(-5, 2, 30)

# Number of independent repetitions (new random split / new synthetic draw).
# Every curve is reported as mean +/- std over these R runs, so a single
# lucky or unlucky split cannot drive the conclusions.
R = 20

# ---------------- Synthetic dataset ----------------
SYN_D = 30          # number of features
SYN_RHO = 0.9       # feature correlation: Sigma_ij = 0.9^|i-j|. Strong correlation
                    # makes X^T X nearly singular -> OLS becomes unstable.
SYN_SIGMA = 1.0     # std of the label noise; Bayes-optimal test MSE = SYN_SIGMA^2 = 1
SYN_N_TRAIN = 40    # just above d = 30: the regime where OLS is ill-conditioned
SYN_N_TEST = 2000   # large fresh test set -> test MSE is a near-exact estimate of the true risk

# ---------------- Diabetes dataset (442 patients, 10 features) ----------------
DIAB_N_TEST = 142       # held-out test points per repetition (~1/3 of the data)
DIAB_N_TRAIN = 50       # small training subsample: makes regularization matter
DIAB_N_TRAIN_FULL = 300 # contrast run: all non-test points (n >> d, OLS already stable)

# ---------------- Dataset-size extension (synthetic only) ----------------
SIZE_N_VALUES = (35, 40, 50, 75, 100, 150, 200, 300, 500)  # all > d = 30
SIZE_LAMBDAS = np.array([1e-3, 1e-2, 1e-1, 1.0])           # OLS is added separately

# ---------------- Paths ----------------
# Absolute paths computed from this file's location, so the code works no
# matter which directory it is run from (e.g. from inside notebooks/).
ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "figures"
