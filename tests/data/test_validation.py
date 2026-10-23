"""Тесты набора правил Great Expectations для исходного CSV."""

from collections.abc import Callable

import pandas as pd
import pytest

from credit_default.data.validation import RAW_COLUMNS, summarize, validate

FLOAT_COLUMNS = {"LIMIT_BAL", *[f"BILL_AMT{i}" for i in range(1, 7)], *[f"PAY_AMT{i}" for i in range(1, 7)]}


def make_valid_raw() -> pd.DataFrame:
    """Строит две корректные строки исходного CSV; AGE на границах допустимого диапазона 18 и 100."""
    base = {column: 1.0 if column in FLOAT_COLUMNS else 1 for column in RAW_COLUMNS}
    rows = [base | {"ID": 1, "AGE": 18}, base | {"ID": 2, "AGE": 100}]
    return pd.DataFrame(rows, columns=RAW_COLUMNS)


def failed_checks(df: pd.DataFrame) -> set[tuple[str, str]]:
    """Возвращает пары (правило, колонка) для проверок, которые не прошли."""
    return {(item["expectation"], item["column"]) for item in summarize(validate(df))["results"] if not item["success"]}


def test_valid_data_passes() -> None:
    """Ловит ложные срабатывания правил на корректных данных, включая границы AGE (анализ граничных значений)."""
    assert failed_checks(make_valid_raw()) == set()


@pytest.mark.parametrize(
    ("column", "value", "expected"),
    [
        pytest.param("LIMIT_BAL", None, ("expect_column_values_to_not_be_null", "LIMIT_BAL"), id="null"),
        pytest.param("AGE", 17, ("expect_column_values_to_be_between", "AGE"), id="age_below_min"),
        pytest.param("AGE", 101, ("expect_column_values_to_be_between", "AGE"), id="age_above_max"),
        pytest.param("LIMIT_BAL", 0.0, ("expect_column_values_to_be_between", "LIMIT_BAL"), id="zero_limit"),
        pytest.param("PAY_AMT1", -1.0, ("expect_column_values_to_be_between", "PAY_AMT1"), id="negative_payment"),
        pytest.param("EDUCATION", 7, ("expect_column_values_to_be_in_set", "EDUCATION"), id="unknown_category"),
        pytest.param("PAY_0", 9, ("expect_column_values_to_be_in_set", "PAY_0"), id="unknown_pay_status"),
        pytest.param("ID", 1, ("expect_column_values_to_be_unique", "ID"), id="duplicate_id"),
    ],
)
def test_anomaly_fails_its_rule(column: str, value: float | None, expected: tuple[str, str]) -> None:
    """Ловит аномалию, которую набор правил пропустил бы (классы эквивалентности по правилам, границы AGE и сумм)."""
    df = make_valid_raw()
    df.loc[1, column] = value

    assert expected in failed_checks(df)


@pytest.mark.parametrize(
    "change",
    [
        pytest.param(lambda df: df.drop(columns="AGE"), id="missing_column"),
        pytest.param(lambda df: df.assign(EXTRA=1), id="extra_column"),
    ],
)
def test_schema_change_fails(change: Callable[[pd.DataFrame], pd.DataFrame]) -> None:
    """Ловит изменение схемы источника: пропавшую или новую колонку (классы эквивалентности «неверная схема»)."""
    assert ("expect_table_columns_to_match_set", "") in failed_checks(change(make_valid_raw()))
