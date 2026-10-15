"""Statistics quoted in the report and study guide, so every number has code behind it.

Nothing here is needed to RUN the experiments; these functions only summarise
the result dictionaries produced by experiments.py.
"""
import numpy as np


def _ranks(v):
    """Rank of each entry (0 = smallest). Ties are not averaged: our values are
    continuous measurements, so exact ties do not occur in practice."""
    return np.argsort(np.argsort(np.asarray(v)))


def spearman(a, b):
    """Spearman rank correlation = Pearson correlation of the ranks.

    +1: b always increases when a increases (not necessarily linearly);
    -1: always decreases; ~0: no monotone relationship.
    CAUTION: if a and b are both monotone functions of one parameter (e.g. lambda),
    their Spearman correlation is +-1 automatically, so it tests little.
    """
    return float(np.corrcoef(_ranks(a), _ranks(b))[0, 1])


def loglog_slope(x, y):
    """Least-squares slope of log(y) against log(x).

    If y ~ C * x^p then the slope is p: e.g. -1 means y ~ 1/x.
    The caller chooses the window of x values to fit (it must be reported).
    """
    return float(np.polyfit(np.log(x), np.log(y), 1)[0])


def influence_and_residual(make_model, X_tr, y_tr, X_te):
    """Split the leave-one-out prediction change into two factors.

    For ridge with a FIXED penalty and FIXED scaling, Sherman-Morrison gives exactly
        f_S(x) - f_{S\\i}(x) = [x-dependent influence of point i] * r_i,
        r_i = y_i - f_{S\\i}(x_i)   (leave-one-out residual of point i).
    Our models re-standardise on S\\i and use (n-1)*lam, so the identity is only
    approximate; here we MEASURE the two factors on average:
        mean_abs_residual = mean_i |y_i - f_{S\\i}(x_i)|
        mean_influence    = mean_pred_change / mean_abs_residual
    so that  mean_pred_change = mean_influence * mean_abs_residual  exactly.
    (A ratio of averages, not an average of per-point ratios: a point whose
    residual is ~0 would otherwise divide by ~0 and dominate the mean.)
    """
    full = make_model().fit(X_tr, y_tr)
    pred_full = full.predict(X_te)
    change, residual = [], []
    for i in range(len(y_tr)):
        m = make_model().fit(np.delete(X_tr, i, axis=0), np.delete(y_tr, i))
        residual.append(abs(y_tr[i] - m.predict(X_tr[i:i + 1])[0]))
        change.append(np.abs(pred_full - m.predict(X_te)).mean())
    return {"mean_influence": float(np.mean(change) / np.mean(residual)),
            "mean_abs_residual": float(np.mean(residual))}


def across_reps_spearman(a, b):
    """For each column (one lambda), Spearman correlation across the R repetitions.

    a, b : arrays (R, L). This asks a DIFFERENT question from the across-lambda
    correlation: at a fixed lambda, are the random draws that happen to be more
    stable also the ones with a smaller gap?
    """
    return np.array([spearman(a[:, j], b[:, j]) for j in range(a.shape[1])])


def pooled_spearman(size_res, metric):
    """Spearman correlation between stability `metric` and the gap over ALL
    (n, lambda) configurations of the size sweep, ridge and OLS together.

    Stability and gap vary along TWO parameters here, so a high correlation is
    not automatic (unlike the correlation along lambda alone).
    Returns (rho, number_of_configurations).
    """
    stab = np.concatenate([size_res["ridge"][metric].mean(0).ravel(),
                           size_res["ols"][metric].mean(0).ravel()])
    gap = np.concatenate([size_res["ridge"]["gap"].mean(0).ravel(),
                          size_res["ols"]["gap"].mean(0).ravel()])
    return spearman(stab, gap), len(stab)
