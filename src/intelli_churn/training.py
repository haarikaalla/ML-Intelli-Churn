from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split

from .artifacts import ModelBundle, ModelRegistry, fingerprint_file, generated_version
from .config import TrainingConfig
from .data_quality import profile_dataset
from .features import MODEL_FEATURES, TARGET, validate_training_frame
from .modeling import build_pipeline, candidate_estimators
from .monitoring import build_reference_profile


def _select_threshold(y_true, probabilities, config: TrainingConfig) -> tuple[float, list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    for threshold in [step / 100 for step in range(5, 91)]:
        predictions = (probabilities >= threshold).astype(int)
        _tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()
        cost = (
            fp * config.false_positive_cost
            + fn * config.false_negative_cost
            + tp * config.true_positive_cost
        )
        rows.append(
            {
                "threshold": threshold,
                "cost": float(cost),
                "recall": float(recall_score(y_true, predictions, zero_division=0)),
                "precision": float(precision_score(y_true, predictions, zero_division=0)),
            }
        )
    feasible = [row for row in rows if row["recall"] >= config.min_recall]
    winner = min(feasible or rows, key=lambda row: (row["cost"], -row["recall"]))
    return float(winner["threshold"]), sorted(rows, key=lambda row: row["cost"])[:10]


def _evaluate(y_true, probabilities, threshold: float, config: TrainingConfig) -> dict[str, Any]:
    predictions = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()
    model_cost = (
        fp * config.false_positive_cost
        + fn * config.false_negative_cost
        + tp * config.true_positive_cost
    )
    no_model_cost = int((y_true == 1).sum()) * config.false_negative_cost
    return {
        "roc_auc": round(float(roc_auc_score(y_true, probabilities)), 6),
        "pr_auc": round(float(average_precision_score(y_true, probabilities)), 6),
        "precision": round(float(precision_score(y_true, predictions, zero_division=0)), 6),
        "recall": round(float(recall_score(y_true, predictions, zero_division=0)), 6),
        "f1": round(float(f1_score(y_true, predictions, zero_division=0)), 6),
        "brier_score": round(float(brier_score_loss(y_true, probabilities)), 6),
        "threshold": threshold,
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "estimated_campaign_cost": round(float(model_cost), 2),
        "estimated_cost_without_model": round(float(no_model_cost), 2),
        "estimated_savings": round(float(no_model_cost - model_cost), 2),
        "estimated_savings_percent": round(
            float((no_model_cost - model_cost) / no_model_cost * 100), 2
        )
        if no_model_cost
        else 0.0,
        "test_rows": len(y_true),
        "positive_rate": round(float(y_true.mean()), 6),
    }


def _candidate_selection(X, y, config: TrainingConfig):
    folds = StratifiedKFold(
        n_splits=config.cv_folds, shuffle=True, random_state=config.random_state
    )
    reports: list[dict[str, Any]] = []
    pipelines = {}
    scoring = {"pr_auc": "average_precision", "roc_auc": "roc_auc"}
    for name, estimator in candidate_estimators(config.random_state).items():
        pipeline = build_pipeline(estimator)
        results = cross_validate(pipeline, X, y, cv=folds, scoring=scoring, n_jobs=1)
        reports.append(
            {
                "model": name,
                "cv_pr_auc_mean": round(float(results["test_pr_auc"].mean()), 6),
                "cv_pr_auc_std": round(float(results["test_pr_auc"].std()), 6),
                "cv_roc_auc_mean": round(float(results["test_roc_auc"].mean()), 6),
            }
        )
        pipelines[name] = pipeline
    reports.sort(key=lambda row: row["cv_pr_auc_mean"], reverse=True)
    return clone(pipelines[reports[0]["model"]]), reports


def train_bundle(
    frame: pd.DataFrame,
    data_fingerprint: str = "in-memory",
    config: TrainingConfig | None = None,
) -> tuple[ModelBundle, dict[str, Any]]:
    config = config or TrainingConfig()
    quality = profile_dataset(frame)
    cleaned = validate_training_frame(frame)
    development, test = train_test_split(
        cleaned,
        test_size=config.test_size,
        random_state=config.random_state,
        stratify=cleaned[TARGET],
    )
    train, validation = train_test_split(
        development,
        test_size=config.validation_size,
        random_state=config.random_state,
        stratify=development[TARGET],
    )
    X_train, y_train = train[list(MODEL_FEATURES)], train[TARGET]
    selected, candidates = _candidate_selection(X_train, y_train, config)
    calibrated = CalibratedClassifierCV(estimator=selected, method="sigmoid", cv=3)
    calibrated.fit(X_train, y_train)
    validation_probabilities = calibrated.predict_proba(validation[list(MODEL_FEATURES)])[:, 1]
    threshold, threshold_analysis = _select_threshold(
        validation[TARGET].to_numpy(), validation_probabilities, config
    )
    test_features = test[list(MODEL_FEATURES)]
    test_probabilities = calibrated.predict_proba(test_features)[:, 1]
    metrics = _evaluate(test[TARGET].to_numpy(), test_probabilities, threshold, config)
    importance = permutation_importance(
        calibrated,
        test_features,
        test[TARGET],
        scoring="average_precision",
        n_repeats=5,
        random_state=config.random_state,
        n_jobs=1,
    )
    importance_rows = sorted(
        [
            {
                "feature": feature,
                "importance_mean": round(float(mean), 6),
                "importance_std": round(float(std), 6),
            }
            for feature, mean, std in zip(
                MODEL_FEATURES, importance.importances_mean, importance.importances_std, strict=True
            )
        ],
        key=lambda row: row["importance_mean"],
        reverse=True,
    )
    model_version = generated_version(data_fingerprint)
    bundle = ModelBundle(
        estimator=calibrated,
        model_version=model_version,
        trained_at=datetime.now(UTC).isoformat(),
        threshold=threshold,
        metrics=metrics,
        candidate_scores=candidates,
        feature_importance=importance_rows,
        reference_profile=build_reference_profile(train[list(MODEL_FEATURES)]),
        data_fingerprint=data_fingerprint,
        training_config=config.to_dict(),
    )
    report = {
        "model_version": model_version,
        "selected_model": candidates[0]["model"],
        "data_quality": quality,
        "split_rows": {"train": len(train), "validation": len(validation), "test": len(test)},
        "candidate_scores": candidates,
        "threshold_analysis_top_10": threshold_analysis,
        "test_metrics": metrics,
        "feature_importance": importance_rows,
    }
    return bundle, report


def train_and_evaluate(
    frame: pd.DataFrame, random_state: int = 42
) -> tuple[object, dict[str, Any]]:
    """Compatibility-friendly programmatic entry point."""
    config = TrainingConfig(random_state=random_state, cv_folds=3)
    bundle, _ = train_bundle(frame, config=config)
    return bundle.estimator, bundle.metrics


def main() -> None:  # pragma: no cover - thin CLI adapter
    parser = argparse.ArgumentParser(description="Train, compare, calibrate and register IntelliChurn")
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--registry", type=Path, default=Path("artifacts/registry"))
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--min-recall", type=float, default=0.45)
    args = parser.parse_args()
    config = TrainingConfig(cv_folds=args.cv_folds, min_recall=args.min_recall)
    bundle, report = train_bundle(
        pd.read_csv(args.dataset), fingerprint_file(args.dataset), config=config
    )
    model_path = ModelRegistry(args.registry).register(bundle, report)
    output = {"artifact": str(model_path), **report}
    print(json.dumps(output, indent=2))


if __name__ == "__main__":  # pragma: no cover
    main()
