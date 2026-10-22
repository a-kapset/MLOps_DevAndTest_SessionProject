"""Тесты очистки и разбиения данных стадии prepare."""

from typing import Any

import pandas as pd

from credit_default.config import TARGET
from credit_default.data.make_dataset import clean, split

RAW_COLUMNS = [
    "ID",
    "LIMIT_BAL",
    "SEX",
    "EDUCATION",
    "MARRIAGE",
    "AGE",
    *[f"PAY_{i}" for i in (0, 2, 3, 4, 5, 6)],
    *[f"BILL_AMT{i}" for i in range(1, 7)],
    *[f"PAY_AMT{i}" for i in range(1, 7)],
    "default.payment.next.month",
]
RENAMED = {"PAY_0": "PAY_1", "default.payment.next.month": TARGET}


def make_raw(rows: list[dict[str, Any]]) -> pd.DataFrame:
    """Строит сырой DataFrame со всеми колонками исходного CSV; не заданные значения равны 1."""
    records = [{column: 1 for column in RAW_COLUMNS} | {"ID": i} | row for i, row in enumerate(rows)]
    return pd.DataFrame(records, columns=RAW_COLUMNS)


def test_clean_recodes_undocumented_education() -> None:
    """Ловит неверную перекодировку EDUCATION (разбиение на классы эквивалентности: 0/5/6 → 4, 1–4 без изменений)."""
    # AGE различается, чтобы перекодированные строки не схлопнулись как дубликаты
    raw = make_raw([{"EDUCATION": code, "AGE": code} for code in range(7)])

    assert clean(raw)["EDUCATION"].tolist() == [4, 1, 2, 3, 4, 4, 4]


def test_clean_recodes_undocumented_marriage() -> None:
    """Ловит неверную перекодировку MARRIAGE (разбиение на классы эквивалентности: 0 → 3, 1–3 без изменений)."""
    # AGE различается, чтобы перекодированные строки не схлопнулись как дубликаты
    raw = make_raw([{"MARRIAGE": code, "AGE": code} for code in range(4)])

    assert clean(raw)["MARRIAGE"].tolist() == [3, 1, 2, 3]


def test_clean_produces_expected_columns() -> None:
    """Ловит расхождение схемы с последующими стадиями: остался ID, не переименованы PAY_0 или целевая колонка."""
    expected = [RENAMED.get(column, column) for column in RAW_COLUMNS if column != "ID"]

    assert clean(make_raw([{}])).columns.tolist() == expected


def test_clean_drops_rows_differing_only_by_id() -> None:
    """Ловит дубликаты, которые пережили очистку из-за того, что дедупликация выполнена до удаления ID."""
    raw = make_raw([{}, {}])

    assert len(clean(raw)) == 1


def test_clean_keeps_rows_differing_only_by_target() -> None:
    """Ловит слишком широкую дедупликацию (граничный случай: строки совпадают во всём, кроме целевой переменной)."""
    raw = make_raw([{"default.payment.next.month": 0}, {"default.payment.next.month": 1}])

    assert len(clean(raw)) == 2


def test_split_preserves_class_ratio() -> None:
    """Ловит разбиение без стратификации: доля дефолтов в test должна совпадать с исходной."""
    data = pd.DataFrame({"x": range(100), TARGET: [1] * 22 + [0] * 78})

    _, test = split(data, test_size=0.5, random_state=0)

    assert test[TARGET].sum() == 11


def test_split_partitions_rows_without_overlap() -> None:
    """Ловит утечку и потерю строк: train и test не пересекаются и вместе дают все исходные строки."""
    data = pd.DataFrame({"x": range(100), TARGET: [1] * 22 + [0] * 78})

    train, test = split(data, test_size=0.2, random_state=0)

    assert sorted([*train.index, *test.index]) == list(data.index)
