"""Стадия validate: проверка исходного CSV набором правил Great Expectations."""

import json
import sys
from typing import Any

import great_expectations as gx
import great_expectations.expectations as gxe
import pandas as pd
from great_expectations.core import ExpectationSuiteValidationResult
from great_expectations.data_context.types.base import ProgressBarsConfig

from credit_default.config import RAW_DATA_PATH, VALIDATION_REPORT_PATH

RAW_TARGET = "default.payment.next.month"
PAY_STATUS = ["PAY_0", *[f"PAY_{i}" for i in range(2, 7)]]
BILL_AMOUNTS = [f"BILL_AMT{i}" for i in range(1, 7)]
PAY_AMOUNTS = [f"PAY_AMT{i}" for i in range(1, 7)]
RAW_COLUMNS = [
    "ID",
    "LIMIT_BAL",
    "SEX",
    "EDUCATION",
    "MARRIAGE",
    "AGE",
    *PAY_STATUS,
    *BILL_AMOUNTS,
    *PAY_AMOUNTS,
    RAW_TARGET,
]

CATEGORY_SETS = {
    "SEX": [1, 2],
    "EDUCATION": list(range(7)),
    "MARRIAGE": list(range(4)),
    RAW_TARGET: [0, 1],
} | {column: list(range(-2, 9)) for column in PAY_STATUS}


def build_suite() -> gx.ExpectationSuite:
    """Собирает набор правил для исходного CSV: схема, полнота, типы, уникальность, диапазоны, категории."""
    suite = gx.ExpectationSuite(name="raw_credit_data")
    suite.add_expectation(
        gxe.ExpectTableColumnsToMatchSet(column_set=RAW_COLUMNS, exact_match=True, meta={"dimension": "completeness"})
    )
    for column in RAW_COLUMNS:
        suite.add_expectation(gxe.ExpectColumnValuesToNotBeNull(column=column, meta={"dimension": "completeness"}))
        suite.add_expectation(
            gxe.ExpectColumnValuesToBeInTypeList(
                column=column, type_list=["int64", "float64"], meta={"dimension": "validity"}
            )
        )
    suite.add_expectation(gxe.ExpectColumnValuesToBeUnique(column="ID", meta={"dimension": "uniqueness"}))
    suite.add_expectation(
        gxe.ExpectColumnValuesToBeBetween(column="AGE", min_value=18, max_value=100, meta={"dimension": "consistency"})
    )
    suite.add_expectation(
        gxe.ExpectColumnValuesToBeBetween(
            column="LIMIT_BAL", min_value=0, strict_min=True, meta={"dimension": "consistency"}
        )
    )
    for column in PAY_AMOUNTS:
        suite.add_expectation(
            gxe.ExpectColumnValuesToBeBetween(column=column, min_value=0, meta={"dimension": "consistency"})
        )
    for column, values in CATEGORY_SETS.items():
        suite.add_expectation(
            gxe.ExpectColumnValuesToBeInSet(column=column, value_set=values, meta={"dimension": "validity"})
        )
    return suite


def validate(df: pd.DataFrame) -> ExpectationSuiteValidationResult:
    """Проверяет DataFrame набором правил во временном контексте GE без файлов на диске."""
    context = gx.get_context(mode="ephemeral")
    context.variables.progress_bars = ProgressBarsConfig(globally=False)
    batch = (
        context.data_sources.add_pandas("raw")
        .add_dataframe_asset("credit")
        .add_batch_definition_whole_dataframe("all")
        .get_batch(batch_parameters={"dataframe": df})
    )
    return batch.validate(build_suite())


def summarize(result: ExpectationSuiteValidationResult) -> dict[str, Any]:
    """Сводит результат GE к отчёту: итог и по строке на каждую проверку в стабильном порядке."""
    checks = []
    for check in result.results:
        config = check.expectation_config
        assert config is not None
        checks.append(
            {
                "expectation": config.type,
                "column": config.kwargs.get("column", ""),
                "dimension": config.meta["dimension"],
                "success": check.success,
                "unexpected_count": check.result.get("unexpected_count"),
            }
        )
    checks.sort(key=lambda item: (item["column"], item["expectation"]))
    failed = sum(not item["success"] for item in checks)
    return {"success": result.success, "checks": len(checks), "failed": failed, "results": checks}


def main() -> None:
    """Выполняет стадию validate: пишет отчёт и завершается с кодом 1, если хотя бы одна проверка не прошла."""
    report = summarize(validate(pd.read_csv(RAW_DATA_PATH)))
    VALIDATION_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    VALIDATION_REPORT_PATH.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"validate: success={report['success']}, checks={report['checks']}, failed={report['failed']}")
    for item in report["results"]:
        if not item["success"]:
            print(f"FAILED {item['expectation']} column={item['column']}")
    if not report["success"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
