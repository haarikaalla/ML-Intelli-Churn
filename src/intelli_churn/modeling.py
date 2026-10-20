from __future__ import annotations

from collections.abc import Mapping

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .feature_engineering import CustomerFeatureEngineer
from .features import (
    CATEGORICAL_FEATURES,
    ENGINEERED_CATEGORICAL_FEATURES,
    ENGINEERED_NUMERIC_FEATURES,
    NUMERIC_FEATURES,
)


def build_preprocessor() -> ColumnTransformer:
    numeric = Pipeline(
        [("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
    )
    categorical = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        [
            ("numeric", numeric, list(NUMERIC_FEATURES + ENGINEERED_NUMERIC_FEATURES)),
            (
                "categorical",
                categorical,
                list(CATEGORICAL_FEATURES + ENGINEERED_CATEGORICAL_FEATURES),
            ),
        ],
        sparse_threshold=0,
    )


def candidate_estimators(random_state: int = 42) -> Mapping[str, object]:
    return {
        "logistic_regression": LogisticRegression(
            max_iter=2500, class_weight="balanced", random_state=random_state
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=240,
            min_samples_leaf=4,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=random_state,
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            max_iter=180,
            learning_rate=0.06,
            max_leaf_nodes=24,
            l2_regularization=1.0,
            class_weight="balanced",
            random_state=random_state,
        ),
    }


def build_pipeline(estimator: object) -> Pipeline:
    return Pipeline(
        [
            ("business_features", CustomerFeatureEngineer()),
            ("preprocess", build_preprocessor()),
            ("model", estimator),
        ]
    )
