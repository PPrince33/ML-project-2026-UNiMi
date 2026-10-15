"""Tests for the lambda sweep and the dataset-size sweep (small sizes for speed)."""
import numpy as np
import pytest

from regstab.experiments import METRICS, lambda_sweep, mean_std, size_sweep

LAMS = np.array([1e-3, 1e-1, 1.0])


def test_lambda_sweep_shapes():
    res = lambda_sweep("synthetic", 40, LAMS, R=2, seed=0)
    assert res["ridge"]["test_mse"].shape == (2, 3)
    assert res["ols"]["gap"].shape == (2,)
    assert set(res["ridge"]) == set(METRICS) and set(res["ols"]) == set(METRICS)
    assert np.array_equal(res["lambdas"], LAMS)


def test_reproducible():
    a = lambda_sweep("diabetes", 50, LAMS, R=2, seed=5)
    b = lambda_sweep("diabetes", 50, LAMS, R=2, seed=5)
    for m in METRICS:
        assert np.array_equal(a["ridge"][m], b["ridge"][m])
        assert np.array_equal(a["ols"][m], b["ols"][m])


def test_different_seed_differs():
    a = lambda_sweep("synthetic", 40, LAMS, R=2, seed=0)
    b = lambda_sweep("synthetic", 40, LAMS, R=2, seed=1)
    assert not np.array_equal(a["ridge"]["test_mse"], b["ridge"]["test_mse"])


def test_gap_definition():
    r = lambda_sweep("synthetic", 40, LAMS, R=2, seed=0)["ridge"]
    assert np.allclose(r["gap"], r["test_mse"] - r["train_mse"])


@pytest.mark.parametrize("ds,n", [("synthetic", 40), ("diabetes", 50)])
def test_shared_test_set(ds, n):
    # Ridge with a negligible lambda equals OLS only if both saw the same data.
    res = lambda_sweep(ds, n, np.array([1e-10]), R=1, seed=0)
    assert res["ridge"]["test_mse"][0, 0] == pytest.approx(res["ols"]["test_mse"][0], rel=1e-6)


def test_size_sweep_shapes():
    res = size_sweep([35, 50], np.array([1e-2, 1.0]), R=2, seed=0)
    assert res["ridge"]["mean_pred_change"].shape == (2, 2, 2)
    assert res["ols"]["test_mse"].shape == (2, 2)
    assert np.array_equal(res["n_values"], [35, 50])


def test_mean_std():
    m, s = mean_std(np.array([[1.0, 2.0], [3.0, 4.0]]))
    assert np.allclose(m, [2.0, 3.0]) and np.allclose(s, [1.0, 1.0])
