from __future__ import annotations

import os
import time
import uuid
from functools import lru_cache

import pandas as pd
from fastapi import FastAPI, HTTPException, Request, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

from .contracts import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    CustomerFeatures,
    DriftRequest,
    Prediction,
)
from .monitoring import drift_report
from .predictor import Predictor

REQUESTS = Counter(
    "intelli_churn_http_requests_total", "HTTP requests", ["method", "path", "status"]
)
LATENCY = Histogram(
    "intelli_churn_http_request_duration_seconds", "HTTP request duration", ["method", "path"]
)
PREDICTIONS = Counter("intelli_churn_predictions_total", "Predictions", ["outcome", "risk_band"])

app = FastAPI(
    title="IntelliChurn Decision API",
    version="2.0.0",
    description="Calibrated churn risk, retention decisions and drift diagnostics.",
)


@app.middleware("http")
async def observability(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    started = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        REQUESTS.labels(request.method, request.url.path, "500").inc()
        raise
    elapsed = time.perf_counter() - started
    REQUESTS.labels(request.method, request.url.path, str(response.status_code)).inc()
    LATENCY.labels(request.method, request.url.path).observe(elapsed)
    response.headers["X-Request-ID"] = request_id
    return response


@lru_cache
def get_predictor() -> Predictor:
    if artifact := os.environ.get("MODEL_PATH"):
        return Predictor.from_artifact(artifact)
    registry = os.environ.get("MODEL_REGISTRY", "artifacts/registry")
    version = os.environ.get("MODEL_VERSION", "latest")
    return Predictor.from_registry(registry, version)


@app.get("/health/live")
def liveness():
    return {"status": "ok", "service": "intelli-churn"}


@app.get("/health/ready")
def readiness():
    try:
        predictor = get_predictor()
        return {"status": "ready", "model_version": predictor.version}
    except (FileNotFoundError, TypeError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/v1/model")
def model_card():
    try:
        bundle = get_predictor().bundle
        return {
            "model_version": bundle.model_version,
            "trained_at": bundle.trained_at,
            "threshold": bundle.threshold,
            "metrics": bundle.metrics,
            "candidate_scores": bundle.candidate_scores,
            "top_features": bundle.feature_importance[:10],
            "data_fingerprint": bundle.data_fingerprint,
        }
    except (FileNotFoundError, TypeError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/v1/predict", response_model=Prediction)
def predict(payload: CustomerFeatures):
    try:
        result = get_predictor().predict(payload)
        PREDICTIONS.labels(result.prediction, result.risk_band).inc()
        return result
    except (FileNotFoundError, TypeError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/v1/predict/batch", response_model=BatchPredictionResponse)
def predict_batch(payload: BatchPredictionRequest):
    try:
        predictions = get_predictor().predict_many(payload.customers)
        for result in predictions:
            PREDICTIONS.labels(result.prediction, result.risk_band).inc()
        return BatchPredictionResponse(predictions=predictions, count=len(predictions))
    except (FileNotFoundError, TypeError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/v1/monitor/drift")
def monitor_drift(payload: DriftRequest):
    try:
        current = pd.DataFrame([customer.model_dump() for customer in payload.customers])
        return drift_report(current, get_predictor().bundle.reference_profile)
    except (FileNotFoundError, TypeError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/metrics", include_in_schema=False)
def metrics():
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
