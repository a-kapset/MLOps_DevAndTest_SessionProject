"""Тесты доменных признаков."""

import pandas as pd

from credit_default.features.build_features import PAY_STATUS, add_features


def make_frame(pay_rows: list[list[int]], bills: list[float], limits: list[float]) -> pd.DataFrame:
    """Строит DataFrame с колонками, которые нужны признакам."""
    frame = pd.DataFrame(pay_rows, columns=PAY_STATUS)
    return frame.assign(BILL_AMT1=bills, LIMIT_BAL=limits)


def test_max_delay_by_payment_status() -> None:
    """Ловит ошибку в MAX_DELAY: отрицательное значение без просрочки или учёт только последнего месяца.

    Разбиение на классы эквивалентности: только коды −2 и −1 → 0; есть просрочка → максимум за шесть месяцев.
    """
    frame = make_frame([[-2, -1, -2, -1, -2, -1], [0, 3, 1, -1, 0, 2]], bills=[0, 0], limits=[1, 1])

    assert add_features(frame)["MAX_DELAY"].tolist() == [0, 3]


def test_limit_utilization_keeps_sign() -> None:
    """Ловит перепутанные числитель и знаменатель или потерянный знак у переплаты.

    Разбиение на классы эквивалентности: положительный и отрицательный счёт.
    """
    frame = make_frame([[0] * 6, [0] * 6], bills=[25_000, -5_000], limits=[100_000, 50_000])

    assert add_features(frame)["LIMIT_UTILIZATION"].tolist() == [0.25, -0.1]
