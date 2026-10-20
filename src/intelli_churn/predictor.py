from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

from .artifacts import ModelBundle, ModelRegistry
from .contracts import CustomerFeatures, Prediction
from .features import validate_inference_frame


class Predictor:
    def __init__(self, bundle: ModelBundle):
        self.bundle = bundle
        self.pipeline = bundle.estimator
        self.version = bundle.model_version
        self.threshold = bundle.threshold

    @classmethod
    def from_registry(cls, registry: str | Path, version: str = "latest") -> Predictor:
        return cls(ModelRegistry(registry).load(version))

    @classmethod
    def from_artifact(cls, path: str | Path) -> Predictor:
        bundle = joblib.load(path)
        if not isinstance(bundle, ModelBundle):
            raise TypeError("Expected a version 2 IntelliChurn ModelBundle")
        return cls(bundle)

    def predict(self, features: CustomerFeatures) -> Prediction:
        return self.predict_frame(pd.DataFrame([features.model_dump()]))[0]

    def predict_many(self, features: list[CustomerFeatures]) -> list[Prediction]:
        return self.predict_frame(pd.DataFrame([item.model_dump() for item in features]))

    def predict_frame(self, frame: pd.DataFrame) -> list[Prediction]:
        probabilities = self.pipeline.predict_proba(validate_inference_frame(frame))[:, 1]
        return [self._response(float(probability)) for probability in probabilities]

    def _response(self, probability: float) -> Prediction:
        risk_band = "high" if probability >= 0.70 else "medium" if probability >= self.threshold else "low"
        actions = {
            "high": "Route to the retention team for a priority personalized offer.",
            "medium": "Trigger a proactive service check-in and monitor engagement.",
            "low": "Continue the standard customer-success journey.",
        }
        return Prediction(
            churn_probability=round(probability, 6),
            prediction="churn" if probability >= self.threshold else "stay",
            risk_band=risk_band,
            recommended_action=actions[risk_band],
            decision_threshold=self.threshold,
            model_version=self.version,
        )
