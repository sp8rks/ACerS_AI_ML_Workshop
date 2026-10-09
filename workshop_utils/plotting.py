"""Small plotting helpers used across sessions."""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score


def parity_plot(y_true, y_pred, label="", units="", ax=None):
    """Predicted-vs-actual plot with MAE and R^2 in the legend."""
    if ax is None:
        _, ax = plt.subplots(figsize=(4.5, 4.5))
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    mae, r2 = mean_absolute_error(y_true, y_pred), r2_score(y_true, y_pred)
    ax.scatter(y_true, y_pred, s=8, alpha=0.4,
               label=f"{label} MAE={mae:.3g}{' ' + units if units else ''}, R²={r2:.2f}")
    lo, hi = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
    ax.plot([lo, hi], [lo, hi], "k--", lw=1)
    ax.set_xlabel(f"Actual {units}".strip())
    ax.set_ylabel(f"Predicted {units}".strip())
    ax.legend(loc="upper left", fontsize=8)
    return ax
