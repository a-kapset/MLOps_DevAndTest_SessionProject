"""Схемы запроса и ответа API.

Поля запроса повторяют имена колонок очищенных данных: Pipeline выбирает признаки по имени.
Категории принимаются только в кодах, которые модель видела после очистки.
"""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

PayStatus = Annotated[int, Field(ge=-2, le=8)]
PayAmount = Annotated[float, Field(ge=0)]


class ClientFeatures(BaseModel):
    """Признаки одного клиента: лимит, демография, статусы платежей, суммы счетов и платежей за полгода."""

    model_config = ConfigDict(extra="forbid")

    LIMIT_BAL: float = Field(gt=0)
    SEX: Literal[1, 2]
    EDUCATION: Literal[1, 2, 3, 4]
    MARRIAGE: Literal[1, 2, 3]
    AGE: int = Field(ge=18, le=100)
    PAY_1: PayStatus
    PAY_2: PayStatus
    PAY_3: PayStatus
    PAY_4: PayStatus
    PAY_5: PayStatus
    PAY_6: PayStatus
    BILL_AMT1: float
    BILL_AMT2: float
    BILL_AMT3: float
    BILL_AMT4: float
    BILL_AMT5: float
    BILL_AMT6: float
    PAY_AMT1: PayAmount
    PAY_AMT2: PayAmount
    PAY_AMT3: PayAmount
    PAY_AMT4: PayAmount
    PAY_AMT5: PayAmount
    PAY_AMT6: PayAmount


class Prediction(BaseModel):
    """Прогноз: класс (1 означает дефолт) и вероятность дефолта."""

    prediction: Literal[0, 1]
    probability: float = Field(ge=0, le=1)
