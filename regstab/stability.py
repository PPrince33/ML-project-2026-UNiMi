"""Leave-one-out (remove-one) stability of a learning algorithm.

THE IDEA (Bousquet & Elisseeff, 2002)
-------------------------------------
Classical generalisation bounds look at how rich the hypothesis class is.
The stability approach instead asks: how much does the OUTPUT of the learning
algorithm change if we remove one training point? If no single point can
change the learned function much, the algorithm cannot be "memorising"
individual points, and its training error must be close to its test error.

For a training set S = {z_1, ..., z_n} and S^{\\i} = S without z_i, the paper
defines (among others):
  * uniform stability beta:  sup over training sets S, removed indices i and ALL examples z of
        |loss(f_S, z) - loss(f_{S\\i}, z)|  (or of the prediction difference)
    -> the generalisation gap is bounded by 2*beta + (4*n*beta + M)*sqrt(ln(1/delta)/(2n));
       when beta = O(1/n) this is O(1/sqrt(n)). (Check the exact constants in the paper.)
    For ridge regression beta <= C / (lam * n).
  * weaker average notions (hypothesis stability), where the change is
    averaged over i and over the data distribution instead of maximised.

WHAT WE MEASURE (empirical versions)
------------------------------------
We cannot take a sup over all x, so we use a FIXED test set T = {x'_1..x'_m}
as a sample of inputs. For every removed point i:

    Delta_i^pred = (1/m) sum_j |f_S(x'_j) - f_{S\\i}(x'_j)|
    M_i^pred     = max_j       |f_S(x'_j) - f_{S\\i}(x'_j)|
    Delta_i^loss = (1/m) sum_j |l(f_S, z'_j) - l(f_{S\\i}, z'_j)|,  l = squared error

and report
    mean_pred_change = mean_i Delta_i^pred   (average stability; our main metric)
    max_pred_change  = max_i  M_i^pred       (worst case: proxy for uniform stability beta)
    mean_loss_change = mean_i Delta_i^loss   (stability measured on the loss, as in the paper's bounds)

The test set is the SAME for f_S and every f_{S\\i}, so differences come only
from removing z_i. We use REMOVE-one; the paper also studies replace-one
(swap z_i for a fresh point), which requires an extra pool of data.

Each f_{S\\i} is obtained by literally refitting the model from scratch on
S^{\\i} (n refits). That is the most direct implementation of the definition
and includes re-computing the standardisation statistics.
"""
import numpy as np

from .models import mse


def loo_stability(make_model, X_tr, y_tr, X_te, y_te):
    """Fit on S and on every S^{\\i}; return errors of f_S and the stability metrics.

    make_model : callable returning a FRESH unfitted model with .fit/.predict
                 (e.g. `lambda: RidgeRegression(0.1)` or the class `LeastSquares`).
                 A factory is used so every refit starts from a clean model.

    Returns a dict with:
        train_mse, test_mse                         -- of the full model f_S
        mean_pred_change, max_pred_change,
        mean_loss_change                            -- see module docstring
    (f_S's errors are returned here so the experiments do not fit it twice.)
    """
    n = len(y_tr)

    # The full model f_S and its predictions/losses on the fixed test set.
    full = make_model().fit(X_tr, y_tr)
    pred_full = full.predict(X_te)
    loss_full = (y_te - pred_full) ** 2

    mean_pred = np.empty(n)   # Delta_i^pred
    max_pred = np.empty(n)    # M_i^pred
    mean_loss = np.empty(n)   # Delta_i^loss

    for i in range(n):
        # Retrain from scratch on S without point i.
        model_i = make_model().fit(np.delete(X_tr, i, axis=0), np.delete(y_tr, i))
        pred_i = model_i.predict(X_te)

        diff = np.abs(pred_full - pred_i)
        mean_pred[i] = diff.mean()
        max_pred[i] = diff.max()
        mean_loss[i] = np.abs(loss_full - (y_te - pred_i) ** 2).mean()

    return {
        "train_mse": mse(y_tr, full.predict(X_tr)),
        "test_mse": float(loss_full.mean()),
        "mean_pred_change": float(mean_pred.mean()),
        "max_pred_change": float(max_pred.max()),
        "mean_loss_change": float(mean_loss.mean()),
    }
