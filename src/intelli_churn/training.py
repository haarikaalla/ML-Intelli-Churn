from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .features import (
    CATEGORICAL_FEATURES,
    MODEL_FEATURES,
    NUMERIC_FEATURES,
    TARGET,
    validate_training_frame,
)


def build_pipeline() -> Pipeline:
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("encode", OneHotEncoder(handle_unknown="ignore")),
    ])
    processor = ColumnTransformer([
        ("numeric", numeric, list(NUMERIC_FEATURES)),
        ("categorical", categorical, list(CATEGORICAL_FEATURES)),
    ])
    return Pipeline([
        ("features", processor),
        ("model", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)),
    ])


def train_and_evaluate(frame: pd.DataFrame, random_state: int = 42) -> tuple[Pipeline, dict[str, float]]:
    frame = validate_training_frame(frame)
    train, test = train_test_split(
        frame,
        test_size=0.2,
        random_state=random_state,
        stratify=frame[TARGET],
    )
    pipeline = build_pipeline()
    pipeline.fit(train[list(MODEL_FEATURES)], train[TARGET])
    probabilities = pipeline.predict_proba(test[list(MODEL_FEATURES)])[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    metrics = {
        "roc_auc": float(roc_auc_score(test[TARGET], probabilities)),
        "pr_auc": float(average_precision_score(test[TARGET], probabilities)),
        "precision": float(precision_score(test[TARGET], predictions, zero_division=0)),
        "recall": float(recall_score(test[TARGET], predictions, zero_division=0)),
        "f1": float(f1_score(test[TARGET], predictions, zero_division=0)),
        "brier_score": float(brier_score_loss(test[TARGET], probabilities)),
        "test_rows": float(len(test)),
        "positive_rate": float(test[TARGET].mean()),
    }
    return pipeline, metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate the IntelliChurn pipeline")
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--output", type=Path, default=Path("artifacts"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    model, metrics = train_and_evaluate(pd.read_csv(args.dataset))
    joblib.dump(model, args.output / "model.joblib")
    (args.output / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
