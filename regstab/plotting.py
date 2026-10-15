"""Figures and the summary table. This module only DRAWS: it receives the
result dictionaries produced by experiments.py and computes nothing beyond
means, standard deviations and the argmin of the test error.

Conventions shared by every figure (so the reader learns them once):
  * x-axis lambda on a log scale (the grid is log-spaced);
  * solid line = mean over the R repetitions, shaded band = +/- 1 std;
  * DASHED line = OLS (lambda = 0), which cannot be placed on a log axis,
    so it is drawn as a horizontal reference with its own +/- 1 std band;
  * DOTTED gray line = a theoretical reference (1/lambda, 1/n, Bayes error);
  * blue = test / main quantity, orange = train / second series;
  * when lambda itself is encoded by colour it uses one blue ramp,
    light = weak regularisation, dark = strong.
"""
import functools
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

from .experiments import mean_std

# ---- palette (validated: adjacent CVD dE 24.7, normal-vision dE 33.6, >= 3:1 on white) ----
BLUE = "#2a78d6"      # categorical slot 1: test error / main series
ORANGE = "#eb6834"    # categorical slot 2: train error / second series
REF = "#52514e"       # references (OLS, theory)
INK = "#0b0b0b"
MUTED = "#8a8985"
GRID = "#e6e5e1"
# One-hue sequential ramp (light -> dark) for encoding lambda as a magnitude.
BLUE_RAMP = ["#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
LAMBDA_CMAP = LinearSegmentedColormap.from_list("lambda_blues", BLUE_RAMP)

STYLE = {
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9, "legend.fontsize": 8,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "lines.linewidth": 1.6, "legend.frameon": False,
    "savefig.dpi": 200, "pdf.fonttype": 42,
}


def _styled(fn):
    """Apply STYLE only while a figure is drawn (does not change global settings)."""
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        with plt.rc_context(STYLE):
            return fn(*args, **kwargs)
    return wrapper


def _save(fig, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    return fig


def _band(ax, x, runs, color, label, **kw):
    """Mean curve with a +/- 1 std band; `runs` has the repetitions on axis 0."""
    m, s = mean_std(runs)
    ax.plot(x, m, color=color, label=label, **kw)
    ax.fill_between(x, np.maximum(m - s, 1e-300), m + s, color=color, alpha=0.15, linewidth=0)
    return m


def _ols_line(ax, runs, color, label):
    """OLS as a dashed horizontal reference with a light +/- 1 std band."""
    m, s = mean_std(runs)
    ax.axhline(m, color=color, linestyle="--", linewidth=1.2, label=label)
    ax.axhspan(max(m - s, 1e-300), m + s, color=color, alpha=0.07, linewidth=0)


def _panels(n, width=3.5, height=2.8):
    fig, axes = plt.subplots(1, n, figsize=(width * n, height), squeeze=False)
    return fig, axes[0]


def _readable_log_y(ax):
    """Label log-y ticks at 1, 2, 5 x 10^k as plain numbers, so values can be read
    off even when the axis spans less than one decade."""
    lo, hi = ax.get_ylim()
    # 1-2-5 labels for narrow axes; only powers of ten when the axis spans > 2 decades.
    subs = (1.0, 2.0, 5.0) if np.log10(hi / lo) <= 2 else (1.0,)
    ax.yaxis.set_major_locator(LogLocator(base=10, subs=subs))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_formatter(NullFormatter())


def _legend_below(fig, axes, ncol):
    """One shared legend under the panels, so it never covers the curves."""
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=ncol, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.1, 1, 1))


# ---------------------------------------------------------------------------
# Figure 1
# ---------------------------------------------------------------------------
@_styled
def plot_stability_vs_lambda(results, path):
    """Mean |change in prediction| under leave-one-out vs lambda (log-log), one panel per dataset."""
    fig, axes = _panels(len(results))
    for ax, (name, res) in zip(axes, results.items()):
        lam = res["lambdas"]
        m = _band(ax, lam, res["ridge"]["mean_pred_change"], BLUE, "ridge")
        _ols_line(ax, res["ols"]["mean_pred_change"], REF, "OLS ($\\lambda=0$)")
        # Slope guide for the theoretical rate 1/lambda: a short segment (one decade
        # either side of the grid's midpoint) through the curve. It shows the SLOPE the
        # bound predicts; the bound is an upper bound, so the curve may decay slower.
        k = len(lam) // 2
        seg = (lam >= lam[k] / 10) & (lam <= lam[k] * 10)
        ax.plot(lam[seg], m[k] * lam[k] / lam[seg], color=REF, linestyle=":", linewidth=1.4,
                label="slope of $1/\\lambda$ (theory)")
        ax.set(xscale="log", yscale="log", xlabel="$\\lambda$",
               ylabel="mean $|\\Delta$ prediction$|$", title=f"{name} ($n={res['n_train']}$)")
        _readable_log_y(ax)
        ax.legend(loc="lower left")
    fig.tight_layout()
    return _save(fig, path)


# ---------------------------------------------------------------------------
# Figure 2
# ---------------------------------------------------------------------------
@_styled
def plot_error_vs_lambda(results, path, bayes_mse):
    """Train and test MSE vs lambda, OLS references, best lambda marked."""
    fig, axes = _panels(len(results))
    for ax, (name, res) in zip(axes, results.items()):
        lam = res["lambdas"]
        m_test = _band(ax, lam, res["ridge"]["test_mse"], BLUE, "test (ridge)")
        _band(ax, lam, res["ridge"]["train_mse"], ORANGE, "train (ridge)")
        _ols_line(ax, res["ols"]["test_mse"], BLUE, "test (OLS)")
        _ols_line(ax, res["ols"]["train_mse"], ORANGE, "train (OLS)")
        if bayes_mse.get(name) is not None:
            ax.axhline(bayes_mse[name], color=REF, linestyle=":", linewidth=1.2,
                       label="Bayes error $\\sigma^2$")
        b = int(np.argmin(m_test))
        ax.plot(lam[b], m_test[b], "o", color=BLUE, markersize=6,
                markeredgecolor="white", markeredgewidth=1.2, zorder=5)
        ax.annotate(f"best $\\lambda$ = {lam[b]:.2g}", (lam[b], m_test[b]),
                    xytext=(0, -14), textcoords="offset points", ha="center",
                    fontsize=8, color=INK)
        ax.set(xscale="log", yscale="log", xlabel="$\\lambda$", ylabel="MSE",
               title=f"{name} ($n={res['n_train']}$)")
        _readable_log_y(ax)
    # The first panel may have the extra Bayes-error entry; use the panel with most entries.
    richest = max(axes, key=lambda a: len(a.get_legend_handles_labels()[0]))
    _legend_below(fig, [richest], ncol=5)
    return _save(fig, path)


# ---------------------------------------------------------------------------
# Figure 3
# ---------------------------------------------------------------------------
@_styled
def plot_stability_vs_gap(results, path):
    """One point per lambda: x = stability, y = generalisation gap (test - train)."""
    fig, axes = _panels(len(results))
    for ax, (name, res) in zip(axes, results.items()):
        lam = res["lambdas"]
        sx, sx_sd = mean_std(res["ridge"]["mean_pred_change"])
        gy, gy_sd = mean_std(res["ridge"]["gap"])
        ax.errorbar(sx, gy, yerr=gy_sd, fmt="none", ecolor=GRID, elinewidth=1, zorder=1)
        sc = ax.scatter(sx, gy, c=lam, cmap=LAMBDA_CMAP, norm=LogNorm(lam.min(), lam.max()),
                        s=28, edgecolors="white", linewidths=0.8, zorder=3, label="ridge")
        ox, _ = mean_std(res["ols"]["mean_pred_change"])
        og, _ = mean_std(res["ols"]["gap"])
        ax.scatter([ox], [og], marker="X", s=60, color=REF, edgecolors="white",
                   linewidths=0.8, zorder=4, label="OLS")
        ax.set(xlabel="mean $|\\Delta$ prediction$|$ (less = more stable)",
               ylabel="gap = test MSE $-$ train MSE", title=f"{name} ($n={res['n_train']}$)")
        ax.legend(loc="upper left")
        fig.colorbar(sc, ax=ax, label="$\\lambda$", pad=0.02)
    fig.tight_layout()
    return _save(fig, path)


# ---------------------------------------------------------------------------
# Figure 4
# ---------------------------------------------------------------------------
@_styled
def plot_size_sweep(size_res, d, path):
    """Dataset-size extension: stability and test MSE vs n, one line per lambda."""
    n = size_res["n_values"]
    lams = size_res["lambdas"]
    shades = [LAMBDA_CMAP(v) for v in np.linspace(0.15, 1.0, len(lams))]  # light = small lambda
    fig, axes = _panels(2)
    for ax, metric, ylabel in [(axes[0], "mean_pred_change", "mean $|\\Delta$ prediction$|$"),
                               (axes[1], "test_mse", "test MSE")]:
        for j, lam in enumerate(lams):
            _band(ax, n, size_res["ridge"][metric][:, :, j], shades[j],
                  f"ridge $\\lambda$={lam:g}", marker="o", markersize=3)
        _band(ax, n, size_res["ols"][metric], REF, "OLS", linestyle="--", marker="o", markersize=3)
        ax.axvline(d, color=MUTED, linewidth=1)
        ax.annotate(f"$n = d = {d}$", (d, 1), xycoords=("data", "axes fraction"),
                    xytext=(3, -10), textcoords="offset points", fontsize=8, color=MUTED)
        ax.set(xscale="log", yscale="log", xlabel="training-set size $n$", ylabel=ylabel)
        _readable_log_y(ax)
    # 1/n reference anchored at the largest n of the strongest-regularised curve.
    m_last, _ = mean_std(size_res["ridge"]["mean_pred_change"][:, :, -1])
    axes[0].plot(n, m_last[-1] * n[-1] / n, color=REF, linestyle=":", linewidth=1.2,
                 label="$\\propto 1/n$ (theory)")
    axes[0].set_title("Stability vs $n$ (synthetic, $d=%d$)" % d)
    axes[1].set_title("Test error vs $n$")
    _legend_below(fig, axes, ncol=6)
    return _save(fig, path)


# ---------------------------------------------------------------------------
# Figure 5
# ---------------------------------------------------------------------------
@_styled
def plot_contrast(res_small, res_full, path):
    """Same dataset, small vs large training set: regularisation matters much less when n >> d."""
    fig, axes = _panels(2)
    for ax, metric, ylabel in [(axes[0], "mean_pred_change", "mean $|\\Delta$ prediction$|$"),
                               (axes[1], "test_mse", "test MSE")]:
        for res, color in [(res_small, BLUE), (res_full, ORANGE)]:
            lab = f"$n={res['n_train']}$"
            _band(ax, res["lambdas"], res["ridge"][metric], color, f"ridge, {lab}")
            _ols_line(ax, res["ols"][metric], color, f"OLS, {lab}")
        ax.set(xscale="log", yscale="log", xlabel="$\\lambda$", ylabel=ylabel)
        _readable_log_y(ax)
    axes[0].set_title("Stability: small vs large training set")
    axes[1].set_title("Test error: small vs large training set")
    axes[0].legend(loc="lower left", fontsize=7)
    fig.tight_layout()
    return _save(fig, path)


# ---------------------------------------------------------------------------
# Figure 6
# ---------------------------------------------------------------------------
@_styled
def plot_metric_agreement(results, path):
    """The two other stability metrics vs lambda: rows = metric, columns = dataset."""
    rows = [("max_pred_change", "max $|\\Delta$ prediction$|$"),
            ("mean_loss_change", "mean $|\\Delta$ loss$|$")]
    fig, axes = plt.subplots(2, len(results), figsize=(3.5 * len(results), 5.2), squeeze=False)
    for c, (name, res) in enumerate(results.items()):
        for r, (metric, ylabel) in enumerate(rows):
            ax = axes[r, c]
            _band(ax, res["lambdas"], res["ridge"][metric], BLUE, "ridge")
            _ols_line(ax, res["ols"][metric], REF, "OLS")
            ax.set(xscale="log", yscale="log", xlabel="$\\lambda$", ylabel=ylabel)
            _readable_log_y(ax)
            if r == 0:
                ax.set_title(f"{name} ($n={res['n_train']}$)")
            ax.legend(loc="lower left")
    fig.tight_layout()
    return _save(fig, path)


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------
_TABLE_METRICS = ("train_mse", "test_mse", "gap", "mean_pred_change", "test_r2")


def summary_rows(results):
    """OLS vs best-lambda ridge per dataset; each metric is a (mean, std) tuple.

    "Best" = lambda with the lowest MEAN test MSE over the repetitions. (This is
    a descriptive summary of the sweep, not model selection: choosing lambda on
    the test set would be optimistic, so the table is read as "how good can
    ridge get", not as an unbiased estimate.) test_r2 is None for synthetic data.
    """
    rows = []
    for name, res in results.items():
        b = int(np.argmin(res["ridge"]["test_mse"].mean(axis=0)))
        for model, lam, src in [("OLS", 0.0, res["ols"]),
                                ("Ridge (best $\\lambda$)", float(res["lambdas"][b]),
                                 {m: res["ridge"][m][:, b] for m in res["ridge"]})]:
            row = {"dataset": name, "model": model, "lam": lam}
            for m in _TABLE_METRICS:
                row[m] = mean_std(src[m])
            if res.get("dataset") == "synthetic":
                row["test_r2"] = None
            rows.append(row)
    return rows


def rows_to_latex(rows):
    """Format summary_rows as a LaTeX tabular (needs \\usepackage{booktabs})."""
    def pm(v, digits=3):
        return "--" if v is None else f"{v[0]:.{digits}g} $\\pm$ {v[1]:.2g}"

    lines = [
        "\\begin{tabular}{llrrrrrr}",
        "\\toprule",
        "Dataset & Model & $\\lambda$ & Train MSE & Test MSE & Gap & Stability & Test $R^2$ \\\\",
        "\\midrule",
    ]
    for r in rows:
        lam = "0" if r["model"] == "OLS" else f"{r['lam']:.2g}"
        lines.append(" & ".join([r["dataset"], r["model"], lam, pm(r["train_mse"]), pm(r["test_mse"]),
                                 pm(r["gap"]), pm(r["mean_pred_change"]), pm(r["test_r2"], 2)])
                     + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines)
