"""Стадия train: подбор гиперпараметров GridSearchCV, трекинг экспериментов в MLflow и сохранение лучшей модели."""

import json
from dataclasses import dataclass
from typing import Any

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from dotenv import load_dotenv
from sklearn.base import ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline

from credit_default.config import (
    METRICS_PATH,
    MODEL_PATH,
    PROJECT_ROOT,
    ROC_CURVE_PATH,
    TARGET,
    TEST_PATH,
    TRAIN_PATH,
    load_params,
)
from credit_default.models.evaluate import compute_metrics, plot_roc_curve
from credit_default.models.pipeline import build_pipeline

EXPERIMENT_NAME = "credit-default"
STEP_PREFIX = "classifier__"
SKOPS_TRUSTED_TYPES = ["credit_default.features.build_features.add_features", "numpy.dtype", "sklearn.tree._tree.Tree"]


@dataclass
class SearchResult:
    """Итог подбора для одного семейства моделей: лучший Pipeline, его параметры и метрики."""

    name: str
    model: Pipeline
    params: dict[str, Any]
    cv_roc_auc: float
    test_metrics: dict[str, float]


def make_classifier(name: str, random_state: int) -> ClassifierMixin:
    """Создаёт классификатор по имени из params.yaml с фиксированными, не перебираемыми параметрами."""
    if name == "logreg":
        return LogisticRegression(max_iter=1000)
    if name == "random_forest":
        return RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=random_state)
    raise ValueError(f"Неизвестная модель: {name}")


def search_model(
    name: str,
    grid: dict[str, list[Any]],
    params: dict[str, Any],
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> SearchResult:
    """Перебирает сетку GridSearchCV по ROC-AUC на стратифицированной кросс-валидации и логирует её в MLflow.

    Родительский run хранит лучшие параметры, CV и тестовые метрики, ROC-кривую и модель;
    вложенные runs — параметры и CV ROC-AUC каждого кандидата сетки.
    """
    x_train, y_train = train.drop(columns=TARGET), train[TARGET]
    x_test, y_test = test.drop(columns=TARGET), test[TARGET]
    search = GridSearchCV(
        build_pipeline(make_classifier(name, params["random_state"])),
        {STEP_PREFIX + key: values for key, values in grid.items()},
        scoring="roc_auc",
        cv=StratifiedKFold(n_splits=params["cv_folds"], shuffle=True, random_state=params["random_state"]),
    )
    with mlflow.start_run(run_name=name):
        search.fit(x_train, y_train)
        results = search.cv_results_
        for i, candidate in enumerate(results["params"]):
            with mlflow.start_run(run_name=f"{name}-{i + 1}", nested=True):
                mlflow.log_params({key.removeprefix(STEP_PREFIX): value for key, value in candidate.items()})
                mlflow.log_metrics(
                    {"cv_roc_auc": results["mean_test_score"][i], "cv_roc_auc_std": results["std_test_score"][i]}
                )
        best_params = {key.removeprefix(STEP_PREFIX): value for key, value in search.best_params_.items()}
        y_proba = search.best_estimator_.predict_proba(x_test)[:, 1]
        test_metrics = compute_metrics(y_test, y_proba)
        mlflow.log_params({"model": name, "cv_folds": params["cv_folds"], **best_params})
        mlflow.log_metrics({"cv_roc_auc": search.best_score_, **{f"test_{k}": v for k, v in test_metrics.items()}})
        mlflow.log_figure(plot_roc_curve(y_test, y_proba), "roc_curve.png")
        mlflow.sklearn.log_model(
            search.best_estimator_,
            name="model",
            input_example=x_test.head(3),
            skops_trusted_types=SKOPS_TRUSTED_TYPES,
        )
    return SearchResult(name, search.best_estimator_, best_params, float(search.best_score_), test_metrics)


def save_best(best: SearchResult, test: pd.DataFrame) -> None:
    """Сохраняет лучшую модель в joblib, её метрики в JSON и ROC-кривую на тестовой выборке в PNG."""
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best.model, MODEL_PATH)
    report = {"model": best.name, "params": best.params, "cv_roc_auc": best.cv_roc_auc, "test": best.test_metrics}
    METRICS_PATH.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n")
    y_proba = best.model.predict_proba(test.drop(columns=TARGET))[:, 1]
    plot_roc_curve(test[TARGET], y_proba).savefig(ROC_CURVE_PATH)


def main() -> None:
    """Выполняет стадию train: подбор для каждой модели из params.yaml, выбор лучшей по CV ROC-AUC."""
    load_dotenv(PROJECT_ROOT / ".env")
    params = load_params("train")
    train, test = pd.read_csv(TRAIN_PATH), pd.read_csv(TEST_PATH)
    mlflow.set_experiment(EXPERIMENT_NAME)
    results = [search_model(name, grid, params, train, test) for name, grid in params["grids"].items()]
    best = max(results, key=lambda result: result.cv_roc_auc)
    save_best(best, test)
    print(f"train: best={best.name}, cv_roc_auc={best.cv_roc_auc:.4f}, test_roc_auc={best.test_metrics['roc_auc']:.4f}")


if __name__ == "__main__":
    main()
