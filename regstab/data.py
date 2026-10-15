"""Datasets: a controlled synthetic regression problem and the real Diabetes data.

Every function takes an explicit numpy Generator `rng`, so the caller decides
the randomness and results are reproducible (no hidden global random state).

This module only produces RAW arrays. All preprocessing (standardisation,
centring) happens inside the models, fitted on training data only.
"""
import numpy as np
from sklearn.datasets import load_diabetes  # used ONLY to load the data file


# ---------------------------------------------------------------------------
# Synthetic data
# ---------------------------------------------------------------------------

def toeplitz_cov(d, rho):
    """Covariance matrix Sigma with Sigma_ij = rho^|i-j|.

    Each feature has variance 1; neighbouring features are strongly correlated
    (rho = 0.9) and correlation decays with distance. Correlated features make
    X^T X nearly singular (small eigenvalues), which is exactly the situation
    where unregularised least squares becomes unstable.
    """
    idx = np.arange(d)
    return rho ** np.abs(idx[:, None] - idx[None, :])


def make_true_weights(d, rng):
    """The 'ground truth' weight vector w* ~ N(0, I_d).

    Drawn ONCE and kept fixed across repetitions: every repetition samples a
    new dataset from the same underlying linear model.
    """
    return rng.normal(size=d)


def make_synthetic(n, w_star, rho, sigma, rng):
    """Draw n points from the linear model y = x.w* + eps.

    x   ~ N(0, Sigma) with Sigma = toeplitz_cov(d, rho)
    eps ~ N(0, sigma^2), independent noise

    Sampling x ~ N(0, Sigma): take G with i.i.d. N(0,1) entries and the Cholesky
    factor L of Sigma (Sigma = L L^T); then X = G L^T has rows with covariance
    L I L^T = Sigma.

    Because we know w* and sigma, the best possible test MSE (Bayes risk) is
    exactly sigma^2: any model's excess over it is estimation error.
    """
    d = len(w_star)
    L = np.linalg.cholesky(toeplitz_cov(d, rho))
    X = rng.normal(size=(n, d)) @ L.T
    y = X @ w_star + sigma * rng.normal(size=n)
    return X, y


# ---------------------------------------------------------------------------
# Real data: Diabetes (Efron et al., 2004), shipped with scikit-learn
# ---------------------------------------------------------------------------

def load_diabetes_raw():
    """Return (X, y): 442 patients x 10 baseline features, y = disease progression.

    scaled=False matters: sklearn's default version is already standardised
    using all 442 rows, i.e. using test-set statistics (data leakage).
    We load the raw values and standardise inside the models instead.
    """
    X, y = load_diabetes(return_X_y=True, scaled=False)
    return X.astype(float), y.astype(float)


def diabetes_split(X, y, n_train, n_test, rng):
    """Random disjoint train/test split, with a SMALL training subsample.

    One random permutation of the rows: the first n_test indices form the test
    set, the next n_train form the training set. The two sets cannot overlap.

    Why a small n_train (50 in the main run)? With all 300 remaining points and
    only d = 10 features, X^T X is very well-conditioned, OLS is already stable,
    and regularisation changes almost nothing. A small training set puts us in
    the regime where stability and regularisation actually matter. (The
    n_train = 300 run is kept as a contrast.)
    """
    perm = rng.permutation(len(y))
    test_idx = perm[:n_test]
    train_idx = perm[n_test:n_test + n_train]
    return X[train_idx], y[train_idx], X[test_idx], y[test_idx]
