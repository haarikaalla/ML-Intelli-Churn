from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

SERVICE_COLUMNS = (
    "PhoneService",
    "MultipleLines",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
)


class CustomerFeatureEngineer(TransformerMixin, BaseEstimator):
    """Deterministic business features shared by training and inference."""

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        frame = X.copy()
        tenure = pd.to_numeric(frame["tenure"], errors="coerce").fillna(0)
        total = pd.to_numeric(frame["TotalCharges"], errors="coerce")
        monthly = pd.to_numeric(frame["MonthlyCharges"], errors="coerce")
        frame["AverageMonthlySpend"] = total / np.maximum(tenure, 1)
        frame["ChargeToTenureRatio"] = monthly / np.maximum(tenure, 1)
        frame["ServiceCount"] = sum(
            frame[column].astype(str).str.lower().eq("yes").astype(int)
            for column in SERVICE_COLUMNS
        )
        frame["TenureBand"] = pd.cut(
            tenure,
            bins=[-1, 6, 12, 24, 48, np.inf],
            labels=["0-6m", "7-12m", "13-24m", "25-48m", "49m+"],
        ).astype(str)
        frame["IsMonthToMonth"] = frame["Contract"].eq("Month-to-month").astype(int)
        frame["HasFiberInternet"] = frame["InternetService"].eq("Fiber optic").astype(int)
        return frame
