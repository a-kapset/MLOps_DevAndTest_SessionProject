"""Стадия prepare: загрузка исходного CSV, очистка и разбиение на train и test."""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from credit_default.config import RAW_DATA_PATH, TARGET, TEST_PATH, TRAIN_PATH, load_params

RENAME_COLUMNS = {"PAY_0": "PAY_1", "default.payment.next.month": TARGET}
# Коды, которых нет в описании датасета, относим к категории «прочее»
EDUCATION_RECODE = {0: 4, 5: 4, 6: 4}
MARRIAGE_RECODE = {0: 3}


def load_raw(path: Path) -> pd.DataFrame:
    """Читает исходный CSV с данными клиентов."""
    return pd.read_csv(path)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Приводит имена колонок к единому виду, перекодирует неописанные категории и удаляет дубликаты."""
    cleaned = df.rename(columns=RENAME_COLUMNS).drop(columns="ID")
    cleaned["EDUCATION"] = cleaned["EDUCATION"].replace(EDUCATION_RECODE)
    cleaned["MARRIAGE"] = cleaned["MARRIAGE"].replace(MARRIAGE_RECODE)
    return cleaned.drop_duplicates().reset_index(drop=True)


def split(df: pd.DataFrame, test_size: float, random_state: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Делит данные на train и test со стратификацией по целевой переменной."""
    train, test = train_test_split(df, test_size=test_size, random_state=random_state, stratify=df[TARGET])
    return train, test


def main() -> None:
    """Выполняет стадию prepare: читает параметры, очищает данные и сохраняет выборки."""
    params = load_params("prepare")
    data = clean(load_raw(RAW_DATA_PATH))
    train, test = split(data, test_size=params["test_size"], random_state=params["random_state"])
    TRAIN_PATH.parent.mkdir(parents=True, exist_ok=True)
    train.to_csv(TRAIN_PATH, index=False)
    test.to_csv(TEST_PATH, index=False)
    print(f"prepare: train={len(train)}, test={len(test)}, default_rate={data[TARGET].mean():.4f}")


if __name__ == "__main__":
    main()
