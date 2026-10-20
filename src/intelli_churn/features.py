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
ENGINEERED_NUMERIC_FEATURES = (
    "AverageMonthlySpend",
    "ChargeToTenureRatio",
    "ServiceCount",
    "IsMonthToMonth",
    "HasFiberInternet",
)
ENGINEERED_CATEGORICAL_FEATURES = ("TenureBand",)


def validate_inference_frame(frame: pd.DataFrame) -> pd.DataFrame:
    missing = sorted(set(MODEL_FEATURES) - set(frame.columns))
    if missing:
        raise ValueError(f"Input data is missing required columns: {', '.join(missing)}")
    cleaned = frame.loc[:, list(MODEL_FEATURES)].copy()
    for column in NUMERIC_FEATURES:
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")
    required_numeric = ["SeniorCitizen", "tenure", "MonthlyCharges"]
    if cleaned[required_numeric].isna().any().any():
        raise ValueError("SeniorCitizen, tenure and MonthlyCharges must be numeric")
    if (cleaned[list(NUMERIC_FEATURES)].dropna() < 0).any().any():
        raise ValueError("tenure and charge fields cannot contain negative values")
    return cleaned


def validate_training_frame(frame: pd.DataFrame) -> pd.DataFrame:
    missing = sorted(set(MODEL_FEATURES + (TARGET,)) - set(frame.columns))
    if missing:
        raise ValueError(f"Dataset is missing required columns: {', '.join(missing)}")
    cleaned = frame.copy()
    if cleaned.empty:
        raise ValueError("Training data cannot be empty")
    cleaned.loc[:, list(MODEL_FEATURES)] = validate_inference_frame(cleaned)
    if cleaned[TARGET].dtype == object:
        cleaned[TARGET] = cleaned[TARGET].map({"Yes": 1, "No": 0})
    if cleaned[TARGET].isna().any():
        raise ValueError("Churn labels must be Yes/No or 1/0")
    cleaned[TARGET] = cleaned[TARGET].astype(int)
    if cleaned[TARGET].nunique() != 2:
        raise ValueError("Training data must contain both churn classes")
    return cleaned
