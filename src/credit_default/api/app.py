"""Приложение FastAPI: загрузка модели при старте, проверка состояния и прогноз дефолта."""

import os
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, Request

from credit_default.api.schemas import ClientFeatures, Prediction
from credit_default.config import MODEL_PATH, THRESHOLD


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Загружает модель один раз при старте; путь задаёт переменная MODEL_PATH.

    Без файла модели сервис не стартует, а не отвечает ошибкой на каждый запрос.
    """
    path = Path(os.environ.get("MODEL_PATH", MODEL_PATH))
    if not path.is_file():
        raise RuntimeError(f"Модель не найдена: {path}. Получите её через dvc pull или dvc repro.")
    app.state.model = joblib.load(path)
    yield


app = FastAPI(title="Credit Default API", lifespan=lifespan)


@app.get("/health")
def health() -> dict[str, str]:
    """Сообщает, что сервис запущен и модель загружена."""
    return {"status": "ok"}


@app.post("/predict")
def predict(client: ClientFeatures, request: Request) -> Prediction:
    """Возвращает вероятность дефолта клиента и класс при пороге THRESHOLD."""
    features = pd.DataFrame([client.model_dump()])
    probability = float(request.app.state.model.predict_proba(features)[0, 1])
    return Prediction(prediction=1 if probability >= THRESHOLD else 0, probability=probability)
