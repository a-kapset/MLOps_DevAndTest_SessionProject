"""Доменные признаки по лимиту и истории платежей."""

import pandas as pd

PAY_STATUS = [f"PAY_{i}" for i in range(1, 7)]
FEATURES = ["LIMIT_UTILIZATION", "MAX_DELAY"]


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Добавляет долю использованного лимита в последнем месяце и максимальную просрочку за полгода.

    Коды PAY −2, −1 и 0 означают отсутствие просрочки и дают MAX_DELAY = 0.
    LIMIT_BAL > 0 гарантируют валидация данных и схема запроса API.
    """
    return df.assign(
        LIMIT_UTILIZATION=df["BILL_AMT1"] / df["LIMIT_BAL"], MAX_DELAY=df[PAY_STATUS].max(axis=1).clip(lower=0)
    )
