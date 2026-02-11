import pandas as pd
import os
import joblib
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

#  Set MLflow experiment
mlflow.set_experiment("Churn_Prediction")

# 🔹 Load dataset
df = pd.read_csv("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")

# 🔹 Remove customerID
df.drop("customerID", axis=1, inplace=True)

# 🔹 Convert TotalCharges
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

# 🔹 Target column
df["Churn"] = df["Churn"].map({"No": 0, "Yes": 1})

# 🔹 Split features & target
X = df.drop("Churn", axis=1)
y = df["Churn"]

# 🔹 Encoding
X = pd.get_dummies(X, drop_first=True)

# 🔹 Train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

#  TRAIN + LOG WITH MLFLOW
with mlflow.start_run():

    model = LogisticRegression(max_iter=5000)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    print("Accuracy:", acc)

    # MLflow logging
    mlflow.log_param("model_type", "LogisticRegression")
    mlflow.log_metric("accuracy", acc)
    mlflow.sklearn.log_model(model, "churn-model")

# Create models folder
os.makedirs("models", exist_ok=True)

#  Save model
joblib.dump(model, "models/churn_model.pkl")

#  Save column structure
joblib.dump(X.columns.tolist(), "models/model_columns.pkl")

print("Model and columns saved successfully!")

# 🔹 Feature importance (coefficients)
importance = model.coef_[0]
features = X.columns

feat_imp = pd.DataFrame({"Feature": features, "Importance": importance})
feat_imp = feat_imp.sort_values(by="Importance", ascending=False)

print(feat_imp.head(10))
