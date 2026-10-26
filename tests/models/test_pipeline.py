"""Тесты сборки Pipeline."""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from credit_default.features.build_features import FEATURES
from credit_default.models.pipeline import CATEGORICAL, NUMERIC, build_pipeline


def test_pipeline_predicts_for_unseen_category() -> None:
    """Ловит ошибку инференса на категории, которой не было в обучении, и Pipeline без шага доменных признаков.

    Разбиение на классы эквивалентности: известная и неизвестная категория EDUCATION.
    """
    rng = np.random.default_rng(0)
    raw_columns = [column for column in NUMERIC if column not in FEATURES] + CATEGORICAL
    train = pd.DataFrame(rng.integers(1, 4, size=(40, len(raw_columns))), columns=raw_columns)
    target = pd.Series(np.arange(40) % 2)
    unseen = train.head(1).assign(EDUCATION=99)

    proba = build_pipeline(LogisticRegression()).fit(train, target).predict_proba(unseen)

    assert proba.shape == (1, 2)
