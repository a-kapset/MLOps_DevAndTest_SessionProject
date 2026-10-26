"""Сборка sklearn Pipeline: доменные признаки, предобработка колонок и классификатор."""

from sklearn.base import ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from credit_default.features.build_features import FEATURES, PAY_STATUS, add_features

CATEGORICAL = ["SEX", "EDUCATION", "MARRIAGE"]
NUMERIC = [
    "LIMIT_BAL",
    "AGE",
    *PAY_STATUS,
    *[f"BILL_AMT{i}" for i in range(1, 7)],
    *[f"PAY_AMT{i}" for i in range(1, 7)],
    *FEATURES,
]


def build_pipeline(classifier: ClassifierMixin) -> Pipeline:
    """Собирает Pipeline: признаки → медиана и масштабирование для чисел, мода и one-hot для категорий → модель.

    Неизвестная категория на инференсе кодируется нулями, а не вызывает ошибку.
    """
    numeric = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    categorical = Pipeline(
        [("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]
    )
    preprocessor = ColumnTransformer([("num", numeric, NUMERIC), ("cat", categorical, CATEGORICAL)])
    return Pipeline(
        [("features", FunctionTransformer(add_features)), ("preprocessor", preprocessor), ("classifier", classifier)]
    )
