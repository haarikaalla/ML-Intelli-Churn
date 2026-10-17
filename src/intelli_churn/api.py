import os
from functools import lru_cache

from fastapi import FastAPI, HTTPException

from .contracts import CustomerFeatures, Prediction
from .predictor import Predictor

app = FastAPI(title="IntelliChurn API", version="1.0.0")


@lru_cache
def get_predictor() -> Predictor:
    model_path = os.environ.get("MODEL_PATH", "artifacts/model.joblib")
    if not os.path.exists(model_path):
        raise FileNotFoundError("Model artifact not found; run intelli-churn-train first")
    return Predictor(model_path, version=os.environ.get("MODEL_VERSION", "1.0.0"))


@app.get("/health")
def health():
    try:
        predictor = get_predictor()
        return {"status": "ok", "model_version": predictor.version}
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/v1/predict", response_model=Prediction)
def predict(payload: CustomerFeatures):
    try:
        return get_predictor().predict(payload)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
