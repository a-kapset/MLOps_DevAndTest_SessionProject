"""Метрики бинарной классификации и ROC-кривая."""

import numpy as np
import numpy.typing as npt
import pandas as pd
from matplotlib.figure import Figure
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, roc_curve

THRESHOLD = 0.5


def compute_metrics(
    y_true: pd.Series, y_proba: npt.NDArray[np.float64], threshold: float = THRESHOLD
) -> dict[str, float]:
    """Считает ROC-AUC по вероятностям и Precision, Recall, F1 для класса «дефолт» при пороге threshold.

    Вероятность, равная порогу, относится к классу «дефолт».
    """
    y_pred = (y_proba >= threshold).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred)),
    }


def plot_roc_curve(y_true: pd.Series, y_proba: npt.NDArray[np.float64]) -> Figure:
    """Строит ROC-кривую с диагональю случайного классификатора и значением ROC-AUC в легенде."""
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    fig = Figure(figsize=(5, 5), layout="constrained")
    ax = fig.subplots()
    ax.plot(fpr, tpr, label=f"модель, ROC-AUC = {roc_auc_score(y_true, y_proba):.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", label="случайный классификатор")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC-кривая")
    ax.legend(loc="lower right")
    return fig
