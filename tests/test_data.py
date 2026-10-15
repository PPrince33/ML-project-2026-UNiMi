"""Tests for synthetic data generation and the Diabetes train/test split."""
import numpy as np
import pytest

from regstab.data import (
    diabetes_split,
    load_diabetes_raw,
    make_synthetic,
    make_true_weights,
    toeplitz_cov,
)


def test_toeplitz_cov():
    S = toeplitz_cov(4, 0.9)
    assert S[0, 3] == pytest.approx(0.9**3)
    assert S[2, 2] == pytest.approx(1.0)
    assert np.allclose(S, S.T)


def test_synthetic_shapes_and_covariance():
    # With many samples the empirical covariance must match Sigma_ij = rho^|i-j|,
    # and with sigma = 0 the labels must be exactly X w*.
    rng = np.random.default_rng(0)
    w = make_true_weights(5, rng)
    X, y = make_synthetic(200_000, w, 0.9, 0.0, rng)
    assert X.shape == (200_000, 5) and y.shape == (200_000,)
    assert np.allclose(np.cov(X.T), toeplitz_cov(5, 0.9), atol=0.02)
    assert np.allclose(y, X @ w)


def test_synthetic_deterministic():
    w = make_true_weights(3, np.random.default_rng(7))
    X1, y1 = make_synthetic(10, w, 0.9, 1.0, np.random.default_rng(42))
    X2, y2 = make_synthetic(10, w, 0.9, 1.0, np.random.default_rng(42))
    assert np.array_equal(X1, X2) and np.array_equal(y1, y2)


def test_diabetes_raw_is_unscaled():
    # The pre-scaled sklearn version was standardised using ALL 442 rows (leakage);
    # the raw version has e.g. age in years (max > 1).
    X, y = load_diabetes_raw()
    assert X.shape == (442, 10) and y.shape == (442,)
    assert X[:, 0].max() > 1


def test_diabetes_split_sizes_and_disjoint():
    X, y = load_diabetes_raw()
    Xtr, ytr, Xte, yte = diabetes_split(X, y, 50, 142, np.random.default_rng(0))
    assert len(Xtr) == len(ytr) == 50 and len(Xte) == len(yte) == 142
    # Rows are unique in Diabetes, so no shared row means no shared patient.
    assert not ({tuple(r) for r in Xtr} & {tuple(r) for r in Xte})
