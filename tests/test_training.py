import numpy as np
import pandas as pd

from intelli_churn.features import CATEGORICAL_FEATURES, NUMERIC_FEATURES
from intelli_churn.monitoring import population_stability_index
from intelli_churn.training import train_and_evaluate


def synthetic_frame(rows=240):
    rng = np.random.default_rng(42)
    frame = pd.DataFrame({
        "SeniorCitizen": rng.integers(0, 2, rows),
        "tenure": rng.integers(1, 72, rows),
        "MonthlyCharges": rng.uniform(20, 120, rows),
        "TotalCharges": rng.uniform(20, 8000, rows),
    })
    values = {
        "gender": ["Female", "Male"], "Partner": ["Yes", "No"],
        "Dependents": ["Yes", "No"], "PhoneService": ["Yes", "No"],
        "MultipleLines": ["Yes", "No", "No phone service"],
        "InternetService": ["DSL", "Fiber optic", "No"],
        "OnlineSecurity": ["Yes", "No"], "OnlineBackup": ["Yes", "No"],
        "DeviceProtection": ["Yes", "No"], "TechSupport": ["Yes", "No"],
        "StreamingTV": ["Yes", "No"], "StreamingMovies": ["Yes", "No"],
        "Contract": ["Month-to-month", "One year", "Two year"],
        "PaperlessBilling": ["Yes", "No"],
        "PaymentMethod": ["Electronic check", "Credit card"],
    }
    for name in CATEGORICAL_FEATURES:
        frame[name] = rng.choice(values[name], rows)
    risk = (frame["Contract"] == "Month-to-month").astype(int) + (frame["tenure"] < 18).astype(int)
    frame["Churn"] = np.where(risk + rng.binomial(1, 0.25, rows) >= 2, "Yes", "No")
    assert set(NUMERIC_FEATURES).issubset(frame)
    return frame


def test_pipeline_trains_and_reports_decision_metrics():
    model, metrics = train_and_evaluate(synthetic_frame())
    assert hasattr(model, "predict_proba")
    assert {"roc_auc", "pr_auc", "recall", "precision", "brier_score"}.issubset(metrics)
    assert 0 <= metrics["roc_auc"] <= 1


def test_population_stability_index_detects_shift():
    reference = np.linspace(0, 1, 1000)
    stable = population_stability_index(reference, reference + 0.001)
    shifted = population_stability_index(reference, reference + 2)
    assert shifted > stable


def test_missing_schema_fails_fast():
    frame = synthetic_frame().drop(columns=["Contract"])
    try:
        train_and_evaluate(frame)
    except ValueError as exc:
        assert "Contract" in str(exc)
    else:
        raise AssertionError("missing feature should fail")
