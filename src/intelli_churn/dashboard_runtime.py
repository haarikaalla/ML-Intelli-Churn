from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import requests

from .artifacts import ModelBundle, ModelRegistry
from .config import TrainingConfig
from .contracts import CustomerFeatures
from .demo_data import generate_customers
from .predictor import Predictor
from .training import train_bundle


def _model_card(bundle: ModelBundle) -> dict:
    """Expose the same model metadata contract as the FastAPI endpoint."""
    return {
        "model_version": bundle.model_version,
        "trained_at": bundle.trained_at,
        "threshold": bundle.threshold,
        "metrics": bundle.metrics,
        "candidate_scores": bundle.candidate_scores,
        "top_features": bundle.feature_importance[:10],
        "data_fingerprint": bundle.data_fingerprint,
    }


@dataclass
class DashboardRuntime:
    """One dashboard interface backed by either FastAPI or a local model."""

    source: Literal["api", "local-demo"]
    model_card: dict
    api_url: str | None = None
    predictor: Predictor | None = None

    @classmethod
    def from_api(cls, api_url: str, timeout: float = 3.0) -> DashboardRuntime:
        response = requests.get(f"{api_url.rstrip('/')}/v1/model", timeout=timeout)
        response.raise_for_status()
        return cls(source="api", model_card=response.json(), api_url=api_url.rstrip("/"))

    @classmethod
    def from_local_registry(
        cls,
        registry: str | Path,
        *,
        demo_rows: int = 800,
        demo_seed: int = 42,
    ) -> DashboardRuntime:
        registry = Path(registry)
        try:
            predictor = Predictor.from_registry(registry)
        except FileNotFoundError:
            frame = generate_customers(rows=demo_rows, seed=demo_seed)
            fingerprint = f"synthetic:dashboard-demo-v1-{demo_rows}-{demo_seed}"
            bundle, report = train_bundle(
                frame,
                data_fingerprint=fingerprint,
                config=TrainingConfig(cv_folds=3, random_state=demo_seed),
            )
            ModelRegistry(registry).register(bundle, report)
            predictor = Predictor(bundle)
        return cls(
            source="local-demo",
            model_card=_model_card(predictor.bundle),
            predictor=predictor,
        )

    def predict_one(self, payload: dict) -> dict:
        if self.source == "api":
            response = requests.post(f"{self.api_url}/v1/predict", json=payload, timeout=15)
            response.raise_for_status()
            return response.json()
        if self.predictor is None:
            raise RuntimeError("Local dashboard runtime has no predictor")
        customer = CustomerFeatures.model_validate(payload)
        return self.predictor.predict(customer).model_dump()

    def predict_many(self, records: list[dict], batch_size: int = 500) -> list[dict]:
        predictions: list[dict] = []
        for offset in range(0, len(records), batch_size):
            batch = records[offset : offset + batch_size]
            if self.source == "api":
                response = requests.post(
                    f"{self.api_url}/v1/predict/batch",
                    json={"customers": batch},
                    timeout=60,
                )
                response.raise_for_status()
                predictions.extend(response.json()["predictions"])
            else:
                if self.predictor is None:
                    raise RuntimeError("Local dashboard runtime has no predictor")
                customers = [CustomerFeatures.model_validate(row) for row in batch]
                predictions.extend(
                    prediction.model_dump()
                    for prediction in self.predictor.predict_many(customers)
                )
        return predictions
