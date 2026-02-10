import pandas as pd
import os
import joblib   # ✅ THIS LINE WAS MISSING

# 🔹 Load dataset
df = pd.read_csv("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")

# 🔹 Remove customerID
df.drop("customerID", axis=1, inplace=True)

# 🔹 Convert TotalCharges
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

# 🔹 Target column
df["Churn"] = df["Churn"].map({"No": 0, "Yes": 1})

# Split features & target
X = df.drop("Churn", axis=1)
y = df["Churn"]

# Encoding
X = pd.get_dummies(X, drop_first=True)

# Train test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Logistic Regression
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(max_iter=5000)
model.fit(X_train, y_train)

# Metrics
from sklearn.metrics import accuracy_score
print("Accuracy:", accuracy_score(y_test, model.predict(X_test)))

# ✅ Create models folder
os.makedirs("models", exist_ok=True)

# ✅ Save model
joblib.dump(model, "models/churn_model.pkl")

# ✅ Save column structure
joblib.dump(X.columns.tolist(), "models/model_columns.pkl")

print("Model and columns saved successfully!")

import matplotlib.pyplot as plt
import pandas as pd

importance = model.coef_[0]
features = X.columns

feat_imp = pd.DataFrame({"Feature": features, "Importance": importance})
feat_imp = feat_imp.sort_values(by="Importance", ascending=False)

print(feat_imp.head(10))
