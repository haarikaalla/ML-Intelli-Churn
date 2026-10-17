from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

from .contracts import CustomerFeatures, Prediction


class Predictor:
    def __init__(self, model_path: str | Path, version: str = "1.0.0", threshold: float = 0.5):
        self.pipeline = joblib.load(model_path)
        self.version = version
        self.threshold = threshold

    def predict(self, features: CustomerFeatures) -> Prediction:
        probability = float(self.pipeline.predict_proba(pd.DataFrame([features.model_dump()]))[0, 1])
        return Prediction(
            churn_probability=round(probability, 6),
            prediction="churn" if probability >= self.threshold else "stay",
            model_version=self.version,
        )
