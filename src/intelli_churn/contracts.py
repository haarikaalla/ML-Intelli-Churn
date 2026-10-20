from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class CustomerFeatures(BaseModel):
    model_config = ConfigDict(extra="forbid")

    gender: Literal["Female", "Male"]
    SeniorCitizen: int = Field(ge=0, le=1)
    Partner: Literal["Yes", "No"]
    Dependents: Literal["Yes", "No"]
    tenure: int = Field(ge=0, le=120)
    PhoneService: Literal["Yes", "No"]
    MultipleLines: Literal["Yes", "No", "No phone service"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: Literal["Yes", "No", "No internet service"]
    OnlineBackup: Literal["Yes", "No", "No internet service"]
    DeviceProtection: Literal["Yes", "No", "No internet service"]
    TechSupport: Literal["Yes", "No", "No internet service"]
    StreamingTV: Literal["Yes", "No", "No internet service"]
    StreamingMovies: Literal["Yes", "No", "No internet service"]
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: Literal["Yes", "No"]
    PaymentMethod: Literal[
        "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
    ]
    MonthlyCharges: float = Field(ge=0)
    TotalCharges: float = Field(ge=0)


class Prediction(BaseModel):
    churn_probability: float
    prediction: Literal["churn", "stay"]
    risk_band: Literal["low", "medium", "high"]
    recommended_action: str
    decision_threshold: float
    model_version: str


class BatchPredictionRequest(BaseModel):
    customers: list[CustomerFeatures] = Field(min_length=1, max_length=500)


class BatchPredictionResponse(BaseModel):
    predictions: list[Prediction]
    count: int


class DriftRequest(BaseModel):
    customers: list[CustomerFeatures] = Field(min_length=10, max_length=10_000)
