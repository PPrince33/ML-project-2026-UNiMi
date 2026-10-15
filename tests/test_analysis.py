"""Tests for the statistics quoted in the report (regstab/analysis.py)."""
import numpy as np
import pytest

from regstab.analysis import (
    across_reps_spearman,
    influence_and_residual,
    loglog_slope,
    pooled_spearman,
    spearman,
)
from regstab.models import RidgeRegression


class MeanModel:
    def fit(self, X, y):
        self.m = y.mean()
        return self

    def predict(self, X):
        return np.full(len(X), self.m)


def test_spearman_known_values():
    assert spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)
    # one swapped pair: 1 - 6*sum(d^2)/(n(n^2-1)) = 1 - 6*2/60 = 0.8
    assert spearman([1, 2, 3, 4], [1, 3, 2, 4]) == pytest.approx(0.8)


def test_loglog_slope_power_law():
    x = np.array([1.0, 10.0, 100.0])
    assert loglog_slope(x, 3.0 * x**-2) == pytest.approx(-2.0)


def test_influence_of_mean_model_is_one_over_n():
    # Mean model: |delta pred| = |y_i - ybar|/(n-1) and LOO residual = n(y_i - ybar)/(n-1),
    # so influence = 1/n exactly (also as a ratio of averages).
    y = np.array([0.0, 1.0, 2.0, 5.0, 7.0])
    r = influence_and_residual(MeanModel, np.zeros((5, 1)), y, np.zeros((3, 1)))
    assert r["mean_influence"] == pytest.approx(1 / 5)
    loo_resid = [abs(y[i] - np.delete(y, i).mean()) for i in range(5)]
    assert r["mean_abs_residual"] == pytest.approx(np.mean(loo_resid))


def test_influence_shrinks_with_lambda():
    rng = np.random.default_rng(0)
    X, y, Xte = rng.normal(size=(30, 5)), rng.normal(size=30), rng.normal(size=(50, 5))
    small = influence_and_residual(lambda: RidgeRegression(1e-3), X, y, Xte)["mean_influence"]
    large = influence_and_residual(lambda: RidgeRegression(1e3), X, y, Xte)["mean_influence"]
    assert large < small


def test_influence_monotone_and_robust_to_tiny_residuals():
    # Run-A-like draw (same seeds as experiment A, repetition 1). A point whose LOO
    # residual is ~0 must not dominate the average: influence must decrease with lambda,
    # and stability = influence * residual must hold exactly on average.
    from regstab.data import make_synthetic, make_true_weights
    from regstab.stability import loo_stability
    w_ss, rep0 = np.random.SeedSequence(2026).spawn(2)
    w = make_true_weights(30, np.random.default_rng(w_ss))
    rng = np.random.default_rng(rep0)
    X, y = make_synthetic(40, w, 0.9, 1.0, rng)
    Xte, yte = make_synthetic(500, w, 0.9, 1.0, rng)
    infl = []
    for lam in [1e-4, 1e-2, 1e-1, 1.0, 10.0, 100.0]:
        make = lambda lam=lam: RidgeRegression(lam)
        d = influence_and_residual(make, X, y, Xte)
        stab = loo_stability(make, X, y, Xte, yte)["mean_pred_change"]
        assert d["mean_influence"] * d["mean_abs_residual"] == pytest.approx(stab)
        infl.append(d["mean_influence"])
    assert all(a > b for a, b in zip(infl, infl[1:]))


def test_across_reps_spearman_per_column():
    a = np.array([[1.0, 1.0], [2.0, 3.0], [3.0, 2.0]])
    b = np.array([[1.0, 3.0], [2.0, 2.0], [3.0, 1.0]])
    assert np.allclose(across_reps_spearman(a, b), [1.0, -0.5])


def test_pooled_spearman_uses_all_configs():
    # ridge (R=2, N=2, L=2) + ols (R=2, N=2): 6 configurations, gap = 2 * stability
    stab_r = np.arange(8, dtype=float).reshape(2, 2, 2)
    stab_o = np.array([[10.0, 11.0], [10.0, 11.0]])
    size_res = {"ridge": {"mean_pred_change": stab_r, "gap": 2 * stab_r},
                "ols": {"mean_pred_change": stab_o, "gap": 2 * stab_o}}
    rho, k = pooled_spearman(size_res, "mean_pred_change")
    assert rho == pytest.approx(1.0) and k == 6
