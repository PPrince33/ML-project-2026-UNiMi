"""Smoke tests for the figures and the summary table (fake results, fast)."""
import numpy as np
import pytest

from regstab.experiments import METRICS
from regstab.plotting import (
    plot_contrast,
    plot_error_vs_lambda,
    plot_metric_agreement,
    plot_size_sweep,
    plot_stability_vs_gap,
    plot_stability_vs_lambda,
    rows_to_latex,
    summary_rows,
)

LAMS = np.array([1e-3, 1e-2, 1e-1, 1.0])


def fake_sweep(dataset, best=2, R=3, seed=0):
    """A lambda_sweep-shaped dict whose mean test MSE is minimal at index `best`."""
    rng = np.random.default_rng(seed)
    L = len(LAMS)
    ridge = {m: rng.uniform(0.5, 1.5, size=(R, L)) for m in METRICS}
    ridge["test_mse"] = 2.0 + np.abs(np.arange(L) - best)[None, :] + 0.01 * rng.random((R, L))
    ridge["mean_pred_change"] = 1.0 / LAMS[None, :] * (1 + 0.1 * rng.random((R, L)))
    ols = {m: rng.uniform(2.0, 3.0, size=R) for m in METRICS}
    return {"dataset": dataset, "n_train": 40, "lambdas": LAMS, "ridge": ridge, "ols": ols}


def fake_size(R=3):
    rng = np.random.default_rng(1)
    n_values = np.array([35, 50, 100])
    ridge = {m: rng.uniform(0.1, 1.0, size=(R, 3, len(LAMS))) for m in METRICS}
    ols = {m: rng.uniform(1.0, 5.0, size=(R, 3)) for m in METRICS}
    return {"n_values": n_values, "lambdas": LAMS, "ridge": ridge, "ols": ols}


@pytest.fixture
def results():
    return {"Synthetic": fake_sweep("synthetic"), "Diabetes": fake_sweep("diabetes", best=1, seed=3)}


def test_each_plot_writes_pdf(tmp_path, results):
    paths = [tmp_path / f"f{k}.pdf" for k in range(6)]
    plot_stability_vs_lambda(results, paths[0])
    plot_error_vs_lambda(results, paths[1], bayes_mse={"Synthetic": 1.0, "Diabetes": None})
    plot_stability_vs_gap(results, paths[2])
    plot_size_sweep(fake_size(), 30, paths[3])
    plot_contrast(results["Diabetes"], fake_sweep("diabetes", seed=9), paths[4])
    plot_metric_agreement(results, paths[5])
    for p in paths:
        assert p.exists() and p.stat().st_size > 1000


def test_summary_picks_best_lambda(results):
    rows = summary_rows(results)
    ridge_syn = next(r for r in rows if r["dataset"] == "Synthetic" and r["model"] != "OLS")
    ridge_dia = next(r for r in rows if r["dataset"] == "Diabetes" and r["model"] != "OLS")
    assert ridge_syn["lam"] == LAMS[2] and ridge_dia["lam"] == LAMS[1]
    assert len(rows) == 4
    mean, std = ridge_syn["test_mse"]
    assert mean == pytest.approx(results["Synthetic"]["ridge"]["test_mse"][:, 2].mean())


def test_rows_to_latex(results):
    s = rows_to_latex(summary_rows(results))
    assert "\\begin{tabular}" in s and "OLS" in s and "\\end{tabular}" in s
