"""Tests for the from-scratch ridge and least-squares models.

Each test pins one mathematical property we rely on in the report.
"""
import numpy as np
import pytest

from regstab.models import LeastSquares, RidgeRegression, mse, ols_solve, ridge_solve


@pytest.fixture
def data():
    """A small, well-conditioned regression problem (n=50, d=5)."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(50, 5)) * np.array([1.0, 2.0, 0.5, 3.0, 1.0]) + 4.0
    y = X @ np.array([1.0, -2.0, 0.5, 0.0, 3.0]) + 10.0 + rng.normal(size=50)
    Xte = rng.normal(size=(20, 5)) + 4.0
    return X, y, Xte


def test_ridge_solve_matches_normal_equations():
    # Fails if ridge_solve uses the wrong system, e.g. forgets the alpha*I term.
    rng = np.random.default_rng(1)
    Z, t = rng.normal(size=(20, 5)), rng.normal(size=20)
    expected = np.linalg.lstsq(Z.T @ Z + 0.7 * np.eye(5), Z.T @ t, rcond=None)[0]
    assert np.allclose(ridge_solve(Z, t, 0.7), expected)


def test_ridge_tiny_lambda_matches_ols(data):
    # As lambda -> 0 the penalty vanishes, so ridge must coincide with OLS.
    X, y, Xte = data
    p_ridge = RidgeRegression(1e-10).fit(X, y).predict(Xte)
    p_ols = LeastSquares().fit(X, y).predict(Xte)
    assert np.allclose(p_ridge, p_ols, atol=1e-6)


def test_ridge_huge_lambda_predicts_training_mean(data):
    # Infinite penalty forces w -> 0; the unpenalized intercept remains = mean(y).
    X, y, Xte = data
    m = RidgeRegression(1e8).fit(X, y)
    assert np.allclose(m.w_, 0, atol=1e-6)
    assert np.allclose(m.predict(Xte), y.mean(), atol=1e-4)


def test_ols_recovers_true_weights_noise_free():
    # Without noise and with n > d, least squares must find the exact function.
    rng = np.random.default_rng(2)
    w_star = np.array([1.5, -2.0, 0.3, 4.0])
    X, Xte = rng.normal(size=(100, 4)), rng.normal(size=(10, 4))
    y = 3.0 + X @ w_star
    assert np.allclose(LeastSquares().fit(X, y).predict(Xte), 3.0 + Xte @ w_star, atol=1e-8)


def test_ols_min_norm_when_d_greater_than_n():
    # d > n: infinitely many exact fits; pinv must return the minimum-norm one.
    rng = np.random.default_rng(3)
    Z, t = rng.normal(size=(5, 10)), rng.normal(size=5)
    w = ols_solve(Z, t)
    assert np.allclose(Z @ w, t)
    assert np.allclose(w, np.linalg.pinv(Z) @ t)


def test_scaler_uses_training_rows_only(data):
    # Standardization statistics must come exactly from the data given to fit().
    X, y, _ = data
    m = RidgeRegression(0.1).fit(X, y)
    assert np.allclose(m.mu_, X.mean(0))
    assert np.allclose(m.scale_, X.std(0))
    assert m.y_mean_ == pytest.approx(y.mean())


def test_constant_feature_is_finite(data):
    # A zero-variance column (possible for the binary Diabetes 'sex' feature in a
    # small subsample) must not cause a division by zero.
    X, y, Xte = data
    X = X.copy()
    X[:, 2] = 1.0
    m = RidgeRegression(0.1).fit(X, y)
    assert np.all(np.isfinite(m.predict(Xte)))
    assert m.scale_[2] == 1.0
    q = LeastSquares().fit(X, y).predict(Xte)
    assert np.all(np.isfinite(q))


def test_constant_non_representable_feature_is_ignored(data):
    # A feature constant in the TRAINING set (e.g. binary 'sex' in a small subsample) but
    # varying in the test set. A constant 0.1 has a float std of ~3e-17, not exactly 0;
    # it must still be treated as constant, so it gets no weight and the predictions
    # equal those of the model trained without that column.
    X, y, Xte = data
    Xc, Xtec = X.copy(), Xte.copy()
    Xc[:, 2] = 0.1
    Xtec[:, 2] = np.where(np.arange(len(Xte)) % 2 == 0, 0.1, 0.2)
    with_col = RidgeRegression(0.1).fit(Xc, y).predict(Xtec)
    without = RidgeRegression(0.1).fit(np.delete(X, 2, 1), y).predict(np.delete(Xte, 2, 1))
    assert np.allclose(with_col, without)


def test_mse():
    assert mse(np.array([1.0, 2.0]), np.array([1.0, 4.0])) == 2.0
