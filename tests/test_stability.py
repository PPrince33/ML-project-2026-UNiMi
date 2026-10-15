"""Tests for the leave-one-out stability metrics."""
import numpy as np
import pytest

from regstab.data import make_synthetic, make_true_weights
from regstab.models import LeastSquares, RidgeRegression, ridge_solve
from regstab.stability import loo_stability


class MeanModel:
    """Predicts the training mean of y. Its leave-one-out behaviour can be computed by hand."""

    def fit(self, X, y):
        self.m = y.mean()
        return self

    def predict(self, X):
        return np.full(len(X), self.m)


def synthetic(n, d=30, seed=0, n_test=200):
    rng = np.random.default_rng(seed)
    w = make_true_weights(d, rng)
    X, y = make_synthetic(n, w, 0.9, 1.0, rng)
    Xte, yte = make_synthetic(n_test, w, 0.9, 1.0, rng)
    return X, y, Xte, yte


def test_known_values_with_mean_model():
    # y_tr = [0, 3, 6]: full mean 3; leave-one-out means 4.5, 3, 1.5 -> |change| = 1.5, 0, 1.5
    r = loo_stability(MeanModel, np.zeros((3, 1)), np.array([0.0, 3.0, 6.0]),
                      np.zeros((2, 1)), np.array([3.0, 5.0]))
    assert r["mean_pred_change"] == pytest.approx(1.0)
    assert r["max_pred_change"] == pytest.approx(1.5)
    # Full-model losses on y_te=[3,5]: 0 and 4.
    # i=0 (pred 4.5): losses 2.25, 0.25 -> |diff| 2.25, 3.75 -> mean 3.0
    # i=1 (pred 3):   identical          -> 0
    # i=2 (pred 1.5): losses 2.25, 12.25 -> |diff| 2.25, 8.25 -> mean 5.25
    assert r["mean_loss_change"] == pytest.approx((3.0 + 0.0 + 5.25) / 3)
    assert r["test_mse"] == pytest.approx((0 + 4) / 2)
    assert r["train_mse"] == pytest.approx((9 + 0 + 9) / 3)


def test_loo_matches_sherman_morrison():
    # Brute-force refits on S without point i must equal the exact rank-one
    # downdate of (Z^T Z + alpha I)^{-1}. Alpha is held fixed (see spec, section 8).
    rng = np.random.default_rng(1)
    Z, t, alpha = rng.normal(size=(30, 6)), rng.normal(size=30), 2.0
    A_inv = np.linalg.inv(Z.T @ Z + alpha * np.eye(6))  # inv is fine here: independent reference
    for i in range(30):
        brute = ridge_solve(np.delete(Z, i, 0), np.delete(t, i), alpha)
        z = Z[i]
        Ai_inv = A_inv + np.outer(A_inv @ z, z @ A_inv) / (1 - z @ A_inv @ z)
        assert np.allclose(brute, Ai_inv @ (Z.T @ t - z * t[i]), atol=1e-10)


def test_identical_training_points_zero_change():
    # If every training point is the same, removing one changes nothing.
    X = np.tile([1.0, 2.0, 3.0], (10, 1))
    y = np.full(10, 5.0)
    Xte = np.random.default_rng(2).normal(size=(5, 3))
    r = loo_stability(lambda: RidgeRegression(0.1), X, y, Xte, np.zeros(5))
    assert r["mean_pred_change"] == pytest.approx(0, abs=1e-12)


def test_metric_ordering_and_nonnegative():
    X, y, Xte, yte = synthetic(40)
    r = loo_stability(lambda: RidgeRegression(0.01), X, y, Xte, yte)
    assert 0 <= r["mean_pred_change"] <= r["max_pred_change"]
    assert r["mean_loss_change"] >= 0


def test_more_regularization_more_stable():
    X, y, Xte, yte = synthetic(40)
    stab = lambda make: loo_stability(make, X, y, Xte, yte)["mean_pred_change"]
    assert stab(lambda: RidgeRegression(10.0)) < stab(lambda: RidgeRegression(1e-3)) < stab(LeastSquares)


def test_ols_near_square_is_finite():
    # n = 31, d = 30: each leave-one-out set is (nearly) singular after centring.
    X, y, Xte, yte = synthetic(31)
    r = loo_stability(LeastSquares, X, y, Xte, yte)
    assert all(np.isfinite(v) for v in r.values())


def test_test_targets_do_not_affect_fit():
    # No leakage: y_te only enters test_mse and the loss-change metric.
    X, y, Xte, yte = synthetic(40)
    a = loo_stability(lambda: RidgeRegression(0.1), X, y, Xte, yte)
    b = loo_stability(lambda: RidgeRegression(0.1), X, y, Xte, yte + 100.0)
    for k in ("train_mse", "mean_pred_change", "max_pred_change"):
        assert a[k] == b[k]
    assert a["test_mse"] != b["test_mse"]


def test_test_features_do_not_affect_fit():
    # No leakage (spec section 8): perturbing the test FEATURES leaves everything that
    # depends only on the training data unchanged (the fitted model, hence train_mse).
    X, y, Xte, yte = synthetic(40)
    a = loo_stability(lambda: RidgeRegression(0.1), X, y, Xte, yte)
    b = loo_stability(lambda: RidgeRegression(0.1), X, y, Xte * 3.0 + 5.0, yte)
    assert a["train_mse"] == b["train_mse"]
    assert a["test_mse"] != b["test_mse"]
    m1 = RidgeRegression(0.1).fit(X, y)
    m2 = RidgeRegression(0.1).fit(X, y)
    m2.predict(Xte * 3.0 + 5.0)   # predicting must not modify the fitted parameters
    assert np.array_equal(m1.w_, m2.w_) and np.array_equal(m1.mu_, m2.mu_)


def test_large_lambda_plateau_is_positive():
    # With w ~ 0 only the unpenalised intercept mean(y) moves when a point is
    # removed, so stability plateaus at mean_i |ybar - ybar_{-i}| > 0.
    X, y, Xte, yte = synthetic(40)
    r = loo_stability(lambda: RidgeRegression(1e6), X, y, Xte, yte)
    loo_means = np.array([np.delete(y, i).mean() for i in range(len(y))])
    plateau = np.mean(np.abs(y.mean() - loo_means))
    assert plateau > 0
    assert r["mean_pred_change"] == pytest.approx(plateau, rel=1e-3)
