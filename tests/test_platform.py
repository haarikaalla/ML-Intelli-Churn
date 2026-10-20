from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from intelli_churn import api
from intelli_churn.artifacts import ModelRegistry
from intelli_churn.batch import score_file
from intelli_churn.config import TrainingConfig
from intelli_churn.contracts import CustomerFeatures
from intelli_churn.demo_data import generate_customers
from intelli_churn.monitoring import (
    build_reference_profile,
    drift_report,
    population_stability_index,
)
from intelli_churn.predictor import Predictor
from intelli_churn.training import train_bundle


@pytest.fixture(scope="session")
def customers() -> pd.DataFrame:
    return generate_customers(rows=360, seed=9)


@pytest.fixture(scope="session")
def trained(customers):
    return train_bundle(
        customers,
        data_fingerprint="sha256:test",
        config=TrainingConfig(cv_folds=3, random_state=7),
    )


def payload_from(frame: pd.DataFrame) -> CustomerFeatures:
    raw = frame.drop(columns=["customerID", "Churn"]).iloc[0].to_dict()
    return CustomerFeatures.model_validate(raw)


def test_training_selects_calibrates_and_reports_decision_metrics(trained):
    bundle, report = trained
    assert hasattr(bundle.estimator, "predict_proba")
    assert 0.05 <= bundle.threshold <= 0.90
    assert len(bundle.candidate_scores) == 5
    assert all(
        {"family", "architecture", "fit_seconds_mean"}.issubset(candidate)
        for candidate in bundle.candidate_scores
    )
    assert {"roc_auc", "pr_auc", "recall", "brier_score", "estimated_savings"}.issubset(
        bundle.metrics
    )
    assert report["selected_model"] in {
        "logistic_regression",
        "random_forest",
        "hist_gradient_boosting",
        "shallow_neural_network",
        "multilayer_perceptron",
    }


def test_registry_round_trip_and_prediction_contract(tmp_path, trained, customers):
    bundle, report = trained
    model_path = ModelRegistry(tmp_path).register(bundle, report)
    assert model_path.exists()
    predictor = Predictor.from_registry(tmp_path)
    prediction = predictor.predict(payload_from(customers))
    assert 0 <= prediction.churn_probability <= 1
    assert prediction.model_version == bundle.model_version
    assert prediction.risk_band in {"low", "medium", "high"}
    assert prediction.recommended_action
    assert Predictor.from_artifact(model_path).version == bundle.model_version


def test_batch_scoring_creates_actionable_csv(tmp_path, trained, customers):
    registry = tmp_path / "registry"
    ModelRegistry(registry).register(*trained)
    source = tmp_path / "customers.csv"
    output = tmp_path / "retention_queue.csv"
    batch = customers.head(12).copy()
    batch["TotalCharges"] = batch["TotalCharges"].astype(object)
    batch.loc[batch.index[0], "TotalCharges"] = " "
    batch.to_csv(source, index=False)
    summary = score_file(source, output, registry)
    scored = pd.read_csv(output)
    assert summary["input_rows"] == 12
    assert {"churn_probability", "risk_band", "recommended_action", "model_version"}.issubset(
        scored.columns
    )


def test_api_liveness_readiness_and_prediction(monkeypatch, trained, customers):
    predictor = Predictor(trained[0])
    monkeypatch.setattr(api, "get_predictor", lambda: predictor)
    client = TestClient(api.app)
    assert client.get("/health/live").status_code == 200
    ready = client.get("/health/ready")
    assert ready.status_code == 200
    response = client.post("/v1/predict", json=payload_from(customers).model_dump())
    assert response.status_code == 200
    assert response.headers["X-Request-ID"]
    assert response.json()["model_version"] == trained[0].model_version
    assert client.get("/v1/model").json()["metrics"]["roc_auc"] >= 0
    batch_payload = {
        "customers": [payload_from(customers).model_dump(), payload_from(customers).model_dump()]
    }
    assert client.post("/v1/predict/batch", json=batch_payload).json()["count"] == 2
    drift_payload = {
        "customers": [
            CustomerFeatures.model_validate(
                customers.drop(columns=["customerID", "Churn"]).iloc[index].to_dict()
            ).model_dump()
            for index in range(12)
        ]
    }
    assert client.post("/v1/monitor/drift", json=drift_payload).status_code == 200
    assert client.get("/metrics").status_code == 200


def test_drift_monitor_detects_numeric_and_categorical_shift(customers):
    features = customers.drop(columns=["customerID", "Churn"])
    reference = build_reference_profile(features)
    stable = drift_report(features.copy(), reference)
    shifted = features.copy()
    shifted["tenure"] = shifted["tenure"] + 100
    shifted["Contract"] = "Two year"
    changed = drift_report(shifted, reference)
    assert stable["status"] == "stable"
    assert "tenure" in changed["drifted_features"]
    assert "Contract" in changed["drifted_features"]


def test_population_stability_index_detects_shift():
    reference = np.linspace(0, 1, 1000)
    stable = population_stability_index(reference, reference + 0.001)
    shifted = population_stability_index(reference, reference + 2)
    assert shifted > stable


def test_schema_failures_are_explicit(customers):
    incomplete = customers.drop(columns=["Contract"])
    with pytest.raises(ValueError, match="Contract"):
        train_bundle(incomplete, config=TrainingConfig(cv_folds=2))

    invalid = payload_from(customers).model_dump()
    invalid["tenure"] = -1
    response = TestClient(api.app).post("/v1/predict", json=invalid)
    assert response.status_code == 422
