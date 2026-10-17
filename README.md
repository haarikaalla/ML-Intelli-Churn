# IntelliChurn — reproducible churn intelligence service

IntelliChurn is now a real train-to-serve ML system rather than a notebook-style
demo. A single scikit-learn pipeline owns imputation, scaling, categorical encoding
and classification, eliminating training/serving skew. The service reports decision
metrics suitable for imbalanced churn data and exposes validated inference through
a versioned FastAPI contract.

## What is implemented

- strict input and dataset contracts;
- stratified evaluation with ROC-AUC, PR-AUC, precision, recall, F1 and calibration;
- one serialized preprocessing/model pipeline;
- validated `/v1/predict` and health endpoints;
- population-stability-index drift utility;
- synthetic, deterministic tests that require no private dataset;
- packaging, Docker and GitHub Actions CI.

## Train and run

```bash
python -m pip install -e '.[dev]'
intelli-churn-train data/WA_Fn-UseC_-Telco-Customer-Churn.csv
uvicorn intelli_churn.api:app --reload
pytest
```

Raw datasets, local environments, MLflow databases and generated model artifacts are
intentionally excluded from Git. Production evolution should add an artifact registry,
scheduled drift jobs, model approval gates and an out-of-time validation dataset.
