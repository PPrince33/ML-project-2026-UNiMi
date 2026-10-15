"""From-scratch linear regression models: ridge regression and ordinary least squares.

THE OBJECTIVE
-------------
Ridge regression minimises, over weights w and intercept b,

    (1/n) * sum_i (y_i - w.x_i - b)^2  +  lam * ||w||^2

i.e. the average squared error on the n training points plus a penalty on the
size of the weights. This is the normalisation used by Bousquet & Elisseeff
(2002): the data term is an AVERAGE (1/n), so lam means the same thing whatever
the training-set size is. With this choice the paper's stability bound for
ridge decreases like 1/(lam * n), which is what we compare against.

THE CLOSED-FORM SOLUTION
------------------------
After centring (see below) the intercept disappears, and setting the gradient
of the objective to zero gives the "normal equations"

    (2/n) Z^T (Z w - t) + 2 lam w = 0      <=>      (Z^T Z + n*lam*I) w = Z^T t

so ridge is just a d x d linear system with penalty alpha = n * lam.
OLS is the special case lam = 0:  Z^T Z w = Z^T t.

WHY REGULARISATION STABILISES
-----------------------------
All eigenvalues of Z^T Z are >= 0, so all eigenvalues of Z^T Z + n*lam*I are
>= n*lam > 0. The matrix is therefore always invertible and its inverse has
norm <= 1/(n*lam): a small change in the data (e.g. removing one point) can
change w by at most a bounded amount. Without the penalty (OLS), when Z^T Z
has eigenvalues close to 0 (few points, correlated features), dividing by them
amplifies tiny data changes into huge weight changes -> instability.

WHY STANDARDISATION IS INSIDE fit()
-----------------------------------
Features are standardised (zero mean, unit variance) so that one penalty lam
treats all features equally regardless of their units. The statistics
(mean, std, mean of y) are computed from the data passed to fit() ONLY:
  * on the training set -> no information from the test set leaks in;
  * in a leave-one-out refit on S without point i -> the statistics are
    recomputed without that point too. The "learning algorithm" whose stability
    we measure is the whole pipeline (scaling + solving), not just the solver.

WHY THE INTERCEPT IS NOT PENALISED
----------------------------------
Penalising b would pull predictions towards 0, which is meaningless (the
Diabetes target is around 150). Centring y and X makes the optimal b exactly
mean(y), so we fit w on centred data and add mean(y) back when predicting.
"""
import numpy as np


def ridge_solve(Z, t, alpha):
    """Solve the ridge normal equations (Z^T Z + alpha*I) w = Z^T t.

    Z     : (n, d) already-centred feature matrix
    t     : (n,)   already-centred targets
    alpha : absolute penalty (the models pass alpha = n * lam)

    We call np.linalg.solve instead of computing the inverse explicitly:
    solving a linear system needs fewer operations and is numerically more
    accurate than forming (Z^T Z + alpha I)^{-1} and then multiplying.
    """
    d = Z.shape[1]
    A = Z.T @ Z + alpha * np.eye(d)   # d x d, symmetric positive definite when alpha > 0
    return np.linalg.solve(A, Z.T @ t)


def ols_solve(Z, t):
    """Ordinary least squares via the Moore-Penrose pseudo-inverse: w = pinv(Z) t.

    When Z^T Z is invertible this equals (Z^T Z)^{-1} Z^T t. When it is not
    (e.g. d > n, or a constant feature), there are infinitely many minimisers
    and pinv returns the one with the smallest norm ||w||, instead of crashing.
    pinv is computed through the SVD, discarding numerically-zero singular values.
    """
    return np.linalg.pinv(Z) @ t


def mse(y_true, y_pred):
    """Mean squared error: (1/m) * sum_j (y_j - yhat_j)^2."""
    return float(np.mean((y_true - y_pred) ** 2))


class _LinearModel:
    """Shared pipeline for both models: standardise -> centre -> solve -> predict.

    Subclasses only decide HOW to solve for w on the prepared data (_solve).
    Fitted attributes (sklearn-style trailing underscore):
        mu_     : per-feature training mean
        scale_  : per-feature training std (1 for constant features)
        y_mean_ : training mean of y  (= the intercept)
        w_      : weights in the standardised feature space
    """

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        # 1) Standardisation statistics, from THIS data only (no leakage).
        self.mu_ = X.mean(axis=0)
        scale = X.std(axis=0)            # population std (ddof=0)
        # A constant column has std 0; dividing by 0 would give NaN. We use 1
        # instead: the centred column is then all zeros and simply gets no weight.
        scale[scale == 0] = 1.0
        self.scale_ = scale

        # 2) Centre the target; its mean becomes the (unpenalised) intercept.
        self.y_mean_ = y.mean()

        Z = (X - self.mu_) / self.scale_
        t = y - self.y_mean_

        # 3) Solve for the weights (ridge or OLS, decided by the subclass).
        self.w_ = self._solve(Z, t)
        return self

    def predict(self, X):
        # Apply the SAME training statistics to new data, then undo the centring.
        Z = (np.asarray(X, dtype=float) - self.mu_) / self.scale_
        return Z @ self.w_ + self.y_mean_

    def _solve(self, Z, t):
        raise NotImplementedError


class RidgeRegression(_LinearModel):
    """Ridge regression with penalty lam (paper's normalisation, see module docstring)."""

    def __init__(self, lam):
        self.lam = lam

    def _solve(self, Z, t):
        n = Z.shape[0]   # current training-set size: n for S, n-1 for S without point i
        return ridge_solve(Z, t, alpha=n * self.lam)


class LeastSquares(_LinearModel):
    """Unregularised least squares (OLS), i.e. ridge with lam = 0, solved with pinv."""

    def _solve(self, Z, t):
        return ols_solve(Z, t)
