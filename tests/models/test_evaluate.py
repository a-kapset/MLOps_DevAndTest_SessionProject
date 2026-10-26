"""Тесты метрик качества."""

import numpy as np
import pandas as pd
import pytest

from credit_default.models.evaluate import compute_metrics


def test_compute_metrics_match_hand_calculation() -> None:
    """Ловит ROC-AUC по меткам вместо вероятностей, перепутанные Precision и Recall, строгое сравнение с порогом.

    Анализ граничных значений: вероятность 0.5 равна порогу и относится к классу «дефолт».
    Ожидания посчитаны вручную: TP = 2, FP = 2, FN = 1; ROC-AUC = 5/9 (верно упорядочены 5 из 9 пар).
    """
    y_true = pd.Series([1, 1, 1, 0, 0, 0])
    y_proba = np.array([0.9, 0.5, 0.3, 0.6, 0.7, 0.1])

    metrics = compute_metrics(y_true, y_proba)

    assert metrics == pytest.approx({"roc_auc": 5 / 9, "precision": 1 / 2, "recall": 2 / 3, "f1": 4 / 7})
