from __future__ import annotations

import os

import pandas as pd
import requests
import streamlit as st

API_URL = os.environ.get("API_URL", "http://localhost:8000")

st.set_page_config(page_title="IntelliChurn", page_icon="◉", layout="wide")
st.markdown(
    """
    <style>
      .stApp {color: #20243a; background: linear-gradient(145deg, #f6f8fc 0%, #eef1fb 100%);}
      .stApp h2, .stApp h3, .stApp h4 {color: #20243a;}
      .block-container {max-width: 1320px; padding-top: 2.2rem;}
      .hero {padding: 1.8rem 2rem; border-radius: 22px; color: white;
             background: linear-gradient(120deg, #14182b, #34305f 62%, #594bb0); margin-bottom: 1.4rem;}
      .hero small {letter-spacing: .14em; color: #bfb8ff; font-weight: 700;}
      .hero h1 {margin: .35rem 0; font-size: 2.6rem;}
      .hero p {max-width: 760px; color: #dddaf4; margin-bottom: .25rem;}
      .model-card {min-height: 150px; padding: 1.15rem; border: 1px solid #e2e4ef;
                   border-radius: 16px; background: white; box-shadow: 0 10px 28px rgba(31,35,66,.06);}
      .model-card strong {color: #4d3da4; font-size: 1rem;}
      .model-card p {color: #687086; font-size: .86rem; line-height: 1.45;}
      .winner {display:inline-block; padding:.2rem .55rem; border-radius:999px;
               color:#176b4b; background:#ddf6eb; font-size:.72rem; font-weight:700;}
      div[data-testid="stMetric"] {background:white; border:1px solid #e2e4ef; padding:1rem;
                                    border-radius:15px; box-shadow:0 8px 25px rgba(31,35,66,.05);}
    </style>
    <section class="hero">
      <small>TELECOM RETENTION INTELLIGENCE</small>
      <h1>IntelliChurn Model Lab</h1>
      <p>Compare classical binary classifiers, a shallow neural network and a deep multilayer
      perceptron—then turn the selected calibrated model into a prioritized retention queue.</p>
    </section>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(ttl=30)
def load_model_card() -> dict:
    response = requests.get(f"{API_URL}/v1/model", timeout=10)
    response.raise_for_status()
    return response.json()


model = None
try:
    model = load_model_card()
except (requests.RequestException, KeyError, ValueError):
    st.warning(
        "The model service is not ready. Train a model and start the FastAPI service to unlock "
        "live metrics, architecture comparison and scoring."
    )

overview_tab, comparison_tab, queue_tab = st.tabs(
    ["Performance overview", "Architecture comparison", "Retention queue"]
)

with overview_tab:
    if model:
        metrics = model["metrics"]
        columns = st.columns(5)
        columns[0].metric("ROC-AUC", f"{metrics['roc_auc']:.3f}")
        columns[1].metric("PR-AUC", f"{metrics['pr_auc']:.3f}")
        columns[2].metric("Recall", f"{metrics['recall']:.1%}")
        columns[3].metric("Brier score", f"{metrics['brier_score']:.3f}")
        columns[4].metric("Estimated savings", f"{metrics['estimated_savings_percent']:.1f}%")
        st.caption(
            f"Model {model['model_version']} · operating threshold {model['threshold']:.2f} · "
            "illustrative cost scenario, not guaranteed realized savings"
        )

        left, right = st.columns([1.15, 1])
        with left:
            st.subheader("Decision outcomes")
            confusion = metrics["confusion_matrix"]
            st.dataframe(
                pd.DataFrame(
                    [[confusion["tn"], confusion["fp"]], [confusion["fn"], confusion["tp"]]],
                    index=["Actual retained", "Actual churn"],
                    columns=["Predicted retained", "Predicted churn"],
                ),
                width="stretch",
            )
        with right:
            st.subheader("Strongest global drivers")
            features = pd.DataFrame(model.get("top_features", [])).head(8)
            if not features.empty:
                st.bar_chart(features.set_index("feature")["importance_mean"], width="stretch")

with comparison_tab:
    st.subheader("Five architectures, one evaluation contract")
    st.caption(
        "Every candidate receives the same engineered features and stratified folds. Mean CV "
        "PR-AUC selects the winner because churn is the minority class; ROC-AUC is shown as a "
        "secondary ranking metric."
    )
    if model:
        candidates = pd.DataFrame(model["candidate_scores"])
        candidates["selected"] = ["Winner" if index == 0 else "" for index in range(len(candidates))]
        display = candidates.rename(
            columns={
                "model": "Model",
                "family": "Family",
                "architecture": "Architecture",
                "cv_pr_auc_mean": "CV PR-AUC",
                "cv_pr_auc_std": "PR-AUC std",
                "cv_roc_auc_mean": "CV ROC-AUC",
                "fit_seconds_mean": "Mean fit seconds",
                "selected": "Status",
            }
        )
        st.dataframe(
            display[
                [
                    "Status",
                    "Model",
                    "Family",
                    "Architecture",
                    "CV PR-AUC",
                    "CV ROC-AUC",
                    "Mean fit seconds",
                ]
            ],
            hide_index=True,
            width="stretch",
        )
        chart = display.set_index("Model")[["CV PR-AUC", "CV ROC-AUC"]]
        st.bar_chart(chart)

        st.markdown("#### Neural network designs")
        shallow, deep = st.columns(2)
        with shallow:
            st.markdown(
                """<div class="model-card"><span class="winner">FAST, INTERPRETABLE BASELINE</span>
                <h4>Shallow neural network</h4><strong>Input → Dense 24 ReLU → Sigmoid</strong>
                <p>One hidden layer learns non-linear interactions while keeping training fast and
                reducing overfitting risk on a medium-sized tabular telecom dataset.</p></div>""",
                unsafe_allow_html=True,
            )
        with deep:
            st.markdown(
                """<div class="model-card"><span class="winner">HIGHER CAPACITY</span>
                <h4>Multilayer perceptron</h4><strong>Input → 64 ReLU → 32 ReLU → 16 ReLU → Sigmoid</strong>
                <p>Three hidden layers test whether additional representation capacity improves
                minority-class ranking. L2 regularization and early stopping control complexity.</p></div>""",
                unsafe_allow_html=True,
            )
        st.info(
            "Neural networks are candidates, not automatic winners. The pipeline promotes whichever "
            "architecture achieves the strongest cross-validated PR-AUC, then calibrates its probability."
        )

with queue_tab:
    st.subheader("Build an actionable retention queue")
    st.caption("Upload contract-compatible customer data. Scoring is sent to the versioned FastAPI service.")
    uploaded = st.file_uploader("Upload customer CSV", type="csv")
    if uploaded:
        customers = pd.read_csv(uploaded)
        st.write(f"Loaded **{len(customers):,}** customers")
        if st.button("Create retention queue", type="primary", disabled=model is None):
            records = customers.to_dict(orient="records")
            responses = []
            progress = st.progress(0, text="Scoring customers…")
            for offset in range(0, len(records), 500):
                response = requests.post(
                    f"{API_URL}/v1/predict/batch",
                    json={"customers": records[offset : offset + 500]},
                    timeout=60,
                )
                response.raise_for_status()
                responses.extend(response.json()["predictions"])
                progress.progress(min((offset + 500) / len(records), 1.0), text="Scoring customers…")
            progress.empty()
            result = pd.concat([customers.reset_index(drop=True), pd.DataFrame(responses)], axis=1)
            high_risk = result[result["risk_band"].eq("high")].sort_values(
                "churn_probability", ascending=False
            )
            left, right = st.columns(2)
            left.metric("Customers scored", f"{len(result):,}")
            right.metric("High-risk customers", f"{len(high_risk):,}")
            st.dataframe(high_risk, hide_index=True, width="stretch")
            st.download_button(
                "Download full scored queue",
                result.to_csv(index=False),
                "intelli_churn_retention_queue.csv",
                "text/csv",
            )

with st.sidebar:
    st.markdown("### IntelliChurn")
    st.caption("Model governance console")
    st.write("● API connected" if model else "○ API unavailable")
    if model:
        st.write(f"**Version**  `{model['model_version']}`")
        st.write(f"**Candidates compared**  {len(model['candidate_scores'])}")
        st.write(f"**Selected**  {model['candidate_scores'][0]['family']}")
