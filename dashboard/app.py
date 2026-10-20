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


def predict_customer(payload: dict) -> dict:
    """Request one validated decision from the production inference path."""
    response = requests.post(f"{API_URL}/v1/predict", json=payload, timeout=15)
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

story_tab, prediction_tab, overview_tab, comparison_tab, queue_tab = st.tabs(
    [
        "Project story",
        "Live prediction",
        "Performance overview",
        "Architecture comparison",
        "Retention queue",
    ]
)

with story_tab:
    st.subheader("From customer signals to a governed retention decision")
    st.write(
        "Telecom churn is costly because the business often discovers dissatisfaction only after a "
        "customer leaves. IntelliChurn identifies customers who may need attention, explains the "
        "evidence behind the deployed model and converts risk into a reviewable retention action."
    )

    challenge, intelligence, outcome = st.columns(3)
    with challenge:
        st.markdown(
            """<div class="model-card"><strong>1 · Business challenge</strong>
            <h4>Find preventable churn early</h4><p>Use relationship, service, contract and billing
            signals to rank customers before their next renewal decision.</p></div>""",
            unsafe_allow_html=True,
        )
    with intelligence:
        st.markdown(
            """<div class="model-card"><strong>2 · ML intelligence</strong>
            <h4>Compare, calibrate, validate</h4><p>Five classical and neural architectures compete
            under the same leakage-safe cross-validation and held-out evaluation contract.</p></div>""",
            unsafe_allow_html=True,
        )
    with outcome:
        st.markdown(
            """<div class="model-card"><strong>3 · Human decision</strong>
            <h4>Prioritize the right action</h4><p>Calibrated risk, an optimized threshold and an
            action recommendation support—not replace—the retention specialist.</p></div>""",
            unsafe_allow_html=True,
        )

    st.markdown("#### End-to-end system flow")
    st.info(
        "Customer data → schema and quality checks → shared feature pipeline → five-model "
        "comparison → probability calibration → cost-aware threshold → versioned model → "
        "FastAPI → Streamlit decision console → monitored human action"
    )

    evidence, walkthrough = st.columns([1, 1.25])
    with evidence:
        st.markdown("#### Evidence from the active model")
        if model:
            selected = model["candidate_scores"][0]
            st.write(f"**Selected architecture:** {selected['model'].replace('_', ' ').title()}")
            st.write(f"**Held-out customers:** {model['metrics']['test_rows']:,}")
            st.write(f"**Held-out PR-AUC:** {model['metrics']['pr_auc']:.3f}")
            st.write(f"**Churn recall:** {model['metrics']['recall']:.1%}")
            st.caption(
                "These values come from the currently loaded model artifact. Results from generated "
                "demo data demonstrate the workflow and must not be presented as real-world impact."
            )
        else:
            st.write("Start the API to load evidence from the active model artifact.")
    with walkthrough:
        st.markdown("#### A clear three-minute faculty walkthrough")
        st.markdown(
            """
            1. **Problem:** retention teams need an early, prioritized signal—not just a yes/no label.
            2. **Method:** show that five models receive the same features and validation procedure.
            3. **Evidence:** explain PR-AUC, recall, calibration and the untouched test set.
            4. **Demo:** change one customer's details and request a live prediction through the API.
            5. **Engineering:** finish with model versioning, drift checks, tests, CI and containers.
            """
        )

    with st.expander("What information does the model use?"):
        st.dataframe(
            pd.DataFrame(
                [
                    ("Relationship", "tenure, partner, dependents, senior-citizen indicator"),
                    ("Services", "phone, internet, security, backup, support and streaming"),
                    ("Commercial", "contract term, billing method and payment method"),
                    ("Spend", "monthly and total charges"),
                ],
                columns=["Signal group", "Examples"],
            ),
            hide_index=True,
            width="stretch",
        )
        st.caption(
            "The model does not use customer names. In a real deployment, consent, access controls, "
            "fairness review and intervention auditing remain required."
        )

with prediction_tab:
    st.subheader("Score one customer through the live API")
    st.caption(
        "This form uses the same strict data contract and serialized feature pipeline as batch scoring."
    )
    with st.form("single_customer_prediction"):
        relationship, services, commercial = st.columns(3)
        with relationship:
            st.markdown("##### Customer relationship")
            gender = st.selectbox("Gender", ["Female", "Male"])
            senior = st.selectbox("Senior citizen", ["No", "Yes"])
            partner = st.selectbox("Partner", ["Yes", "No"])
            dependents = st.selectbox("Dependents", ["No", "Yes"])
            tenure = st.slider("Tenure (months)", 0, 72, 12)
        with services:
            st.markdown("##### Services")
            phone = st.selectbox("Phone service", ["Yes", "No"])
            multiple_lines = st.selectbox(
                "Multiple lines", ["No", "Yes"] if phone == "Yes" else ["No phone service"]
            )
            internet = st.selectbox("Internet service", ["Fiber optic", "DSL", "No"])
            internet_options = ["No", "Yes"] if internet != "No" else ["No internet service"]
            security = st.selectbox("Online security", internet_options)
            backup = st.selectbox("Online backup", internet_options)
            protection = st.selectbox("Device protection", internet_options)
            support = st.selectbox("Tech support", internet_options)
            streaming_tv = st.selectbox("Streaming TV", internet_options)
            streaming_movies = st.selectbox("Streaming movies", internet_options)
        with commercial:
            st.markdown("##### Contract and billing")
            contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
            paperless = st.selectbox("Paperless billing", ["Yes", "No"])
            payment = st.selectbox(
                "Payment method",
                [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)",
                ],
            )
            monthly = st.number_input("Monthly charges", min_value=0.0, value=75.0, step=5.0)
            total = st.number_input(
                "Total charges", min_value=0.0, value=float(monthly * max(tenure, 1)), step=25.0
            )
        submitted = st.form_submit_button(
            "Predict churn risk", type="primary", disabled=model is None, width="stretch"
        )

    if submitted:
        payload = {
            "gender": gender,
            "SeniorCitizen": int(senior == "Yes"),
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": phone,
            "MultipleLines": multiple_lines if phone == "Yes" else "No phone service",
            "InternetService": internet,
            "OnlineSecurity": security if internet != "No" else "No internet service",
            "OnlineBackup": backup if internet != "No" else "No internet service",
            "DeviceProtection": protection if internet != "No" else "No internet service",
            "TechSupport": support if internet != "No" else "No internet service",
            "StreamingTV": streaming_tv if internet != "No" else "No internet service",
            "StreamingMovies": streaming_movies if internet != "No" else "No internet service",
            "Contract": contract,
            "PaperlessBilling": paperless,
            "PaymentMethod": payment,
            "MonthlyCharges": monthly,
            "TotalCharges": total,
        }
        try:
            prediction = predict_customer(payload)
            probability = float(prediction["churn_probability"])
            result, risk, threshold = st.columns(3)
            result.metric("Decision", prediction["prediction"].upper())
            risk.metric("Churn probability", f"{probability:.1%}")
            threshold.metric("Decision threshold", f"{prediction['decision_threshold']:.1%}")
            st.progress(probability, text=f"Risk band: {prediction['risk_band'].upper()}")
            st.success(f"Recommended action: {prediction['recommended_action']}")
            st.caption(
                f"Served by model {prediction['model_version']}. This is decision support, not an "
                "automatic customer-treatment decision."
            )
            with st.expander("Show API response"):
                st.json(prediction)
        except (requests.RequestException, KeyError, ValueError) as exc:
            st.error(f"Prediction could not be completed. Check that the API is running. Details: {exc}")

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
        st.caption(
            "ROC-AUC measures overall ranking, PR-AUC emphasizes the churn minority class, recall "
            "shows how many churners were found, and Brier score measures probability accuracy "
            "where lower is better."
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
    st.divider()
    st.caption("Suggested demo path")
    st.write("Project story → Live prediction → Performance → Architectures")
