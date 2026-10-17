from __future__ import annotations

import pandas as pd

TARGET = "Churn"
IDENTIFIER_COLUMNS = ("customerID",)
NUMERIC_FEATURES = ("SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges")
CATEGORICAL_FEATURES = (
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
)
MODEL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def validate_training_frame(frame: pd.DataFrame) -> pd.DataFrame:
    missing = sorted(set(MODEL_FEATURES + (TARGET,)) - set(frame.columns))
    if missing:
        raise ValueError(f"Dataset is missing required columns: {', '.join(missing)}")
    cleaned = frame.copy()
    cleaned["TotalCharges"] = pd.to_numeric(cleaned["TotalCharges"], errors="coerce")
    if cleaned[TARGET].dtype == object:
        cleaned[TARGET] = cleaned[TARGET].map({"Yes": 1, "No": 0})
    if cleaned[TARGET].isna().any():
        raise ValueError("Churn labels must be Yes/No or 1/0")
    cleaned[TARGET] = cleaned[TARGET].astype(int)
    if cleaned[TARGET].nunique() != 2:
        raise ValueError("Training data must contain both churn classes")
    return cleaned
