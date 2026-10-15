"""The experiments: sweep the regularisation strength (and, for the extension,
the training-set size), measuring errors and stability, over R repetitions.

REPRODUCIBLE RANDOMNESS
-----------------------
np.random.SeedSequence(seed).spawn(k) turns ONE master seed into k
statistically independent child seeds. We use:
    child 0      -> the synthetic true weights w* (fixed for the whole run)
    child 1..R   -> one per repetition (data split / data draw)
So the same seed always reproduces the same numbers, and repetitions are
independent of each other.

ONE TEST SET PER REPETITION
---------------------------
Inside a repetition, every lambda (and OLS) is trained on the SAME training set
and evaluated on the SAME test set. Differences along a curve are then due to
lambda only, not to a different random split.
"""
import numpy as np

from . import config
from .data import diabetes_split, load_diabetes_raw, make_synthetic, make_true_weights
from .models import LeastSquares, RidgeRegression
from .stability import loo_stability

# Everything recorded for each (repetition, lambda) cell.
#   gap     = test_mse - train_mse : the generalisation gap that stability bounds
#   test_r2 = 1 - test_mse / Var(y_test) : fraction of target variance explained
METRICS = ("train_mse", "test_mse", "gap",
           "mean_pred_change", "max_pred_change", "mean_loss_change", "test_r2")


def _evaluate(make_model, X_tr, y_tr, X_te, y_te):
    """Run the leave-one-out evaluation and add the derived metrics."""
    r = loo_stability(make_model, X_tr, y_tr, X_te, y_te)
    r["gap"] = r["test_mse"] - r["train_mse"]
    r["test_r2"] = 1.0 - r["test_mse"] / np.var(y_te)
    return r


def _seeds(seed, R):
    """Split the master seed into (seed for w*, list of R per-repetition seeds)."""
    w_ss, *rep_ss = np.random.SeedSequence(seed).spawn(R + 1)
    return w_ss, rep_ss


def lambda_sweep(dataset, n_train, lambdas, R, seed):
    """Errors and stability of ridge (for each lambda) and of OLS.

    dataset : "synthetic" or "diabetes"
    Returns {"dataset", "n_train", "lambdas",
             "ridge": {metric: array (R, L)}, "ols": {metric: array (R,)}}
    """
    L = len(lambdas)
    ridge = {m: np.empty((R, L)) for m in METRICS}
    ols = {m: np.empty(R) for m in METRICS}

    w_ss, rep_ss = _seeds(seed, R)
    if dataset == "synthetic":
        w_star = make_true_weights(config.SYN_D, np.random.default_rng(w_ss))
    elif dataset == "diabetes":
        X_all, y_all = load_diabetes_raw()
    else:
        raise ValueError(f"unknown dataset {dataset!r}")

    for r in range(R):
        rng = np.random.default_rng(rep_ss[r])
        # Draw this repetition's train/test data ONCE; reused by every model below.
        if dataset == "synthetic":
            X_tr, y_tr = make_synthetic(n_train, w_star, config.SYN_RHO, config.SYN_SIGMA, rng)
            X_te, y_te = make_synthetic(config.SYN_N_TEST, w_star, config.SYN_RHO, config.SYN_SIGMA, rng)
        else:
            X_tr, y_tr, X_te, y_te = diabetes_split(X_all, y_all, n_train, config.DIAB_N_TEST, rng)

        for j, lam in enumerate(lambdas):
            res = _evaluate(lambda lam=lam: RidgeRegression(lam), X_tr, y_tr, X_te, y_te)
            for m in METRICS:
                ridge[m][r, j] = res[m]

        res = _evaluate(LeastSquares, X_tr, y_tr, X_te, y_te)
        for m in METRICS:
            ols[m][r] = res[m]

    return {"dataset": dataset, "n_train": n_train, "lambdas": np.asarray(lambdas),
            "ridge": ridge, "ols": ols}


def size_sweep(n_values, lambdas, R, seed):
    """Dataset-size extension (synthetic, d = config.SYN_D).

    For each repetition: one fixed test set, and for every n a FRESH training
    set of size n. Theory predicts ridge stability decays like 1/n.
    Returns {"n_values", "lambdas",
             "ridge": {metric: array (R, N, L)}, "ols": {metric: array (R, N)}}
    """
    N, L = len(n_values), len(lambdas)
    ridge = {m: np.empty((R, N, L)) for m in METRICS}
    ols = {m: np.empty((R, N)) for m in METRICS}

    w_ss, rep_ss = _seeds(seed, R)
    w_star = make_true_weights(config.SYN_D, np.random.default_rng(w_ss))

    for r in range(R):
        rng = np.random.default_rng(rep_ss[r])
        X_te, y_te = make_synthetic(config.SYN_N_TEST, w_star, config.SYN_RHO, config.SYN_SIGMA, rng)
        for k, n in enumerate(n_values):
            X_tr, y_tr = make_synthetic(n, w_star, config.SYN_RHO, config.SYN_SIGMA, rng)
            for j, lam in enumerate(lambdas):
                res = _evaluate(lambda lam=lam: RidgeRegression(lam), X_tr, y_tr, X_te, y_te)
                for m in METRICS:
                    ridge[m][r, k, j] = res[m]
            res = _evaluate(LeastSquares, X_tr, y_tr, X_te, y_te)
            for m in METRICS:
                ols[m][r, k] = res[m]

    return {"n_values": np.asarray(n_values), "lambdas": np.asarray(lambdas),
            "ridge": ridge, "ols": ols}


def mean_std(a, axis=0):
    """Mean and standard deviation across repetitions (axis 0 by default)."""
    return a.mean(axis=axis), a.std(axis=axis)
