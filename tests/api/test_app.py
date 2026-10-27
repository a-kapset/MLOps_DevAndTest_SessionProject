"""Тесты REST API модели."""

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from credit_default.api.app import app
from credit_default.models.pipeline import build_pipeline

CLIENT: dict[str, Any] = {
    "LIMIT_BAL": 50000.0,
    "SEX": 2,
    "EDUCATION": 3,
    "MARRIAGE": 1,
    "AGE": 43,
    **{f"PAY_{i}": 0 for i in range(1, 7)},
    **{f"BILL_AMT{i}": 20000.0 for i in range(1, 7)},
    **{f"PAY_AMT{i}": 1000.0 for i in range(1, 7)},
}


@pytest.fixture
def model() -> Pipeline:
    """Pipeline, обученный на синтетических клиентах с колонками обучающего CSV."""
    rng = np.random.default_rng(0)
    rows = pd.DataFrame([CLIENT] * 40)
    rows["PAY_1"] = rng.integers(-2, 4, size=40)
    rows["BILL_AMT1"] = rng.uniform(0, 50000, size=40)
    target = (rows["PAY_1"] > 0).astype(int)
    return build_pipeline(LogisticRegression()).fit(rows, target)


@pytest.fixture
def client(model: Pipeline, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    """Клиент приложения, которое при старте загружает модель из временного файла."""
    path = tmp_path / "model.joblib"
    joblib.dump(model, path)
    monkeypatch.setenv("MODEL_PATH", str(path))
    with TestClient(app) as test_client:
        yield test_client


@pytest.mark.parametrize("pay_status", [0, 3])
def test_predict_returns_model_probability_and_class(client: TestClient, model: Pipeline, pay_status: int) -> None:
    """Ловит вероятность не того класса, класс, не согласованный с вероятностью, имена признаков не как при обучении.

    Разбиение на классы эквивалентности: клиент без просрочки и с просрочкой, вероятность ниже и выше порога;
    ожидание считается той же моделью напрямую.
    """
    request = CLIENT | {"PAY_1": pay_status}
    expected = float(model.predict_proba(pd.DataFrame([request]))[0, 1])

    response = client.post("/predict", json=request)

    assert response.json() == {"prediction": int(expected >= 0.5), "probability": pytest.approx(expected)}


@pytest.mark.parametrize(
    "change",
    [
        {"LIMIT_BAL": 0},
        {"AGE": 17},
        {"EDUCATION": 5},
        {"MARRIAGE": 0},
        {"PAY_1": 9},
        {"PAY_AMT1": -1},
        {"ID": 1},
    ],
)
def test_predict_rejects_out_of_contract_input(client: TestClient, change: dict[str, float]) -> None:
    """Ловит ослабленную схему: деление на нулевой лимит, коды категорий, которых модель не видела, лишние поля.

    Анализ граничных значений: каждое значение на шаг за границей допустимого диапазона.
    """
    response = client.post("/predict", json=CLIENT | change)

    assert response.status_code == 422


def test_startup_fails_without_model(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Ловит запуск сервиса без модели, который отвечал бы ошибкой 500 на каждый запрос.

    Переход состояний: старт приложения при отсутствующем файле модели.
    """
    monkeypatch.setenv("MODEL_PATH", str(tmp_path / "missing.joblib"))

    with pytest.raises(RuntimeError, match="Модель не найдена"), TestClient(app):
        pass
