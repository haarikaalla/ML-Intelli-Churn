import pandas as pd
import joblib
import os

# ================= PATH HANDLING =================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

model_path = os.path.join(BASE_DIR, "models", "churn_model.pkl")
columns_path = os.path.join(BASE_DIR, "models", "model_columns.pkl")
data_path = os.path.join(BASE_DIR, "data", "WA_Fn-UseC_-Telco-Customer-Churn.csv")

# ================= LOAD MODEL =================
model = joblib.load(model_path)
model_columns = joblib.load(columns_path)

# ================= LOAD DATA =================
df = pd.read_csv(data_path)

# ================= SAME CLEANING AS TRAINING =================
if "customerID" in df.columns:
    df.drop("customerID", axis=1, inplace=True)

df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

# Remove target column if present
if "Churn" in df.columns:
    df = df.drop("Churn", axis=1)

# ================= ENCODING =================
df = pd.get_dummies(df, drop_first=True)

# Add any missing columns
for col in model_columns:
    if col not in df.columns:
        df[col] = 0

# Keep same column order
df = df[model_columns]

# ================= PREDICTION =================
predictions = model.predict(df)

df["Prediction"] = ["CHURN" if p == 1 else "NO CHURN" for p in predictions]

print(df[["Prediction"]].head(10))
import os
os.makedirs("models", exist_ok=True)


