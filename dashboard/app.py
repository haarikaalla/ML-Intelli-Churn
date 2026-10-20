from __future__ import annotations

import os

import pandas as pd
import requests
import streamlit as st

API_URL = os.environ.get("API_URL", "http://localhost:8000")

st.set_page_config(page_title="IntelliChurn", page_icon="📡", layout="wide")
st.title("IntelliChurn Operations Console")
st.caption("Turn calibrated churn risk into an explainable retention queue.")

try:
    model = requests.get(f"{API_URL}/v1/model", timeout=10).json()
    metrics = model["metrics"]
    columns = st.columns(4)
    columns[0].metric("ROC-AUC", f"{metrics['roc_auc']:.3f}")
    columns[1].metric("PR-AUC", f"{metrics['pr_auc']:.3f}")
    columns[2].metric("Recall", f"{metrics['recall']:.1%}")
    columns[3].metric("Estimated savings", f"{metrics['estimated_savings_percent']:.1f}%")
    st.caption(f"Model {model['model_version']} · decision threshold {model['threshold']:.2f}")
except (requests.RequestException, KeyError, ValueError):
    st.warning("The model API is not ready. Train a model and start the API service.")

uploaded = st.file_uploader("Upload customer CSV", type="csv")
if uploaded:
    customers = pd.read_csv(uploaded)
    st.write(f"Loaded {len(customers):,} customers")
    if st.button("Create retention queue", type="primary"):
        records = customers.to_dict(orient="records")
        responses = []
        for offset in range(0, len(records), 500):
            response = requests.post(
                f"{API_URL}/v1/predict/batch",
                json={"customers": records[offset : offset + 500]},
                timeout=60,
            )
            response.raise_for_status()
            responses.extend(response.json()["predictions"])
        result = pd.concat([customers.reset_index(drop=True), pd.DataFrame(responses)], axis=1)
        high_risk = result[result["risk_band"].eq("high")].sort_values(
            "churn_probability", ascending=False
        )
        left, right = st.columns(2)
        left.metric("Customers scored", f"{len(result):,}")
        right.metric("High-risk customers", f"{len(high_risk):,}")
        st.dataframe(high_risk, use_container_width=True)
        st.download_button(
            "Download scored queue",
            result.to_csv(index=False),
            "intelli_churn_retention_queue.csv",
            "text/csv",
        )
