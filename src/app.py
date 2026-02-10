import streamlit as st
import pandas as pd
import joblib

# Load model and columns
model = joblib.load("../models/churn_model.pkl")
model_columns = joblib.load("../models/model_columns.pkl")

st.title("📉 Customer Churn Prediction App")

st.write("Enter customer details:")

# User Inputs
gender = st.selectbox("Gender", ["Male", "Female"])
senior = st.selectbox("Senior Citizen", [0, 1])
partner = st.selectbox("Has Partner?", ["Yes", "No"])
dependents = st.selectbox("Has Dependents?", ["Yes", "No"])
tenure = st.slider("Tenure (months)", 1, 72)
monthly = st.number_input("Monthly Charges", 0.0, 200.0)
total = st.number_input("Total Charges", 0.0, 10000.0)

contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])

# Create dataframe
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

# Encoding
input_data = pd.get_dummies(input_data)

for col in model_columns:
    if col not in input_data.columns:
        input_data[col] = 0

input_data = input_data[model_columns]

# Predict
if st.button("Predict Churn"):
    prediction = model.predict(input_data)[0]
    if prediction == 1:
        st.error("⚠ Customer is likely to CHURN")
    else:
        st.success("✅ Customer is NOT likely to churn")
