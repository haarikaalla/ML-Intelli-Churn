import streamlit as st 
import pandas as pd
import pickle
import joblib

# Load model using joblib
model = joblib.load("models/churn_model.pkl")

# Load columns (still with pickle)
with open("models/model_columns.pkl", "rb") as f:
    model_columns = pickle.load(f)


st.title("📊 Customer Churn Prediction App")
st.write("Enter customer details to predict churn")

# ---------------- INPUTS ---------------- #
gender = st.selectbox("Gender", ["Male", "Female"], key="gender")
senior = st.selectbox("Senior Citizen", [0, 1], key="senior")
partner = st.selectbox("Has Partner?", ["Yes", "No"], key="partner")
dependents = st.selectbox("Has Dependents?", ["Yes", "No"], key="dependents")
tenure = st.slider("Tenure (months)", 0, 72, 12, key="tenure")
monthly = st.number_input("Monthly Charges", 0.0, 200.0, key="monthly")
total = st.number_input("Total Charges", 0.0, 10000.0, key="total")
contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"], key="contract")
internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"], key="internet")

# ---------------- PREDICTION ---------------- #
if st.button("Predict Churn"):

    input_data = pd.DataFrame([{
        "gender": gender,
        "SeniorCitizen": senior,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "MonthlyCharges": monthly,
        "TotalCharges": total,
        "Contract": contract,
        "InternetService": internet
    }])

    # Encoding like training
    input_data = pd.get_dummies(input_data)

    # Add missing columns
    for col in model_columns:
        if col not in input_data.columns:
            input_data[col] = 0

    input_data = input_data[model_columns]

    prediction = model.predict(input_data)[0]

    if prediction == 1:
        st.error("⚠ Customer is likely to CHURN")
    else:
        st.success("✅ Customer is likely to STAY")
