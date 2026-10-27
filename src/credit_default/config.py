"""Пути проекта и загрузка параметров из params.yaml."""

from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PARAMS_PATH = PROJECT_ROOT / "params.yaml"

RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "UCI_Credit_Card.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
TRAIN_PATH = PROCESSED_DIR / "train.csv"
TEST_PATH = PROCESSED_DIR / "test.csv"

REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
VALIDATION_REPORT_PATH = REPORTS_DIR / "validation.json"
METRICS_PATH = REPORTS_DIR / "metrics.json"
ROC_CURVE_PATH = REPORTS_DIR / "roc_curve.png"

MODEL_PATH = PROJECT_ROOT / "models" / "model.joblib"

TARGET = "default"
THRESHOLD = 0.5


def load_params(section: str) -> dict[str, Any]:
    """Возвращает раздел params.yaml, относящийся к одной стадии DVC."""
    with PARAMS_PATH.open(encoding="utf-8") as file:
        params: dict[str, dict[str, Any]] = yaml.safe_load(file)
    return params[section]
