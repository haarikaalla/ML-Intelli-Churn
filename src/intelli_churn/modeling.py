from __future__ import annotations

from collections.abc import Mapping

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

from .feature_engineering import CustomerFeatureEngineer
from .features import (
    CATEGORICAL_FEATURES,
    ENGINEERED_CATEGORICAL_FEATURES,
    ENGINEERED_NUMERIC_FEATURES,
    NUMERIC_FEATURES,
)


class BalancedMLPClassifier(MLPClassifier):
    """MLP that derives balanced sample weights for minority churn cases."""

    def fit(self, X, y, sample_weight=None):
        weights = compute_sample_weight(class_weight="balanced", y=y) if sample_weight is None else sample_weight
        return super().fit(X, y, sample_weight=weights)


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
    """Return classical and neural binary-classification candidates.

    Both neural candidates use the same scaled/encoded inputs as the classical
    estimators. Early stopping controls overfitting and keeps local training
    practical while still making the architecture comparison meaningful.
    """
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
        "shallow_neural_network": BalancedMLPClassifier(
            hidden_layer_sizes=(24,),
            activation="relu",
            solver="adam",
            alpha=0.001,
            learning_rate_init=0.002,
            max_iter=300,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=15,
            random_state=random_state,
        ),
        "multilayer_perceptron": BalancedMLPClassifier(
            hidden_layer_sizes=(64, 32, 16),
            activation="relu",
            solver="adam",
            alpha=0.002,
            learning_rate_init=0.001,
            max_iter=400,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=18,
            random_state=random_state,
        ),
    }


def candidate_architectures() -> Mapping[str, dict[str, object]]:
    """Human-readable model cards shared by training reports and the dashboard."""
    return {
        "logistic_regression": {
            "family": "Linear baseline",
            "architecture": "Scaled features → sigmoid output",
            "depth": 1,
        },
        "random_forest": {
            "family": "Bagged trees",
            "architecture": "240 balanced decision trees → probability average",
            "depth": "adaptive",
        },
        "hist_gradient_boosting": {
            "family": "Boosted trees",
            "architecture": "Sequential histogram trees → logistic loss",
            "depth": "adaptive",
        },
        "shallow_neural_network": {
            "family": "Shallow neural network",
            "architecture": "Input → Dense(24, ReLU) → Sigmoid probability",
            "depth": 1,
        },
        "multilayer_perceptron": {
            "family": "Deep MLP",
            "architecture": "Input → Dense(64) → Dense(32) → Dense(16) → Sigmoid probability",
            "depth": 3,
        },
    }


def build_pipeline(estimator: object) -> Pipeline:
    return Pipeline(
        [
            ("business_features", CustomerFeatureEngineer()),
            ("preprocess", build_preprocessor()),
            ("model", estimator),
        ]
    )
