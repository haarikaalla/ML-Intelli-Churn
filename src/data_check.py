import pandas as pd

# Load data
df = pd.read_csv("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")

print("Before Cleaning:\n", df.dtypes)

# 🔹 Remove customerID (not useful)
df.drop("customerID", axis=1, inplace=True)

# 🔹 Convert TotalCharges to numeric
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

# 🔹 Check missing values
print("\nMissing values:\n", df.isnull().sum())

# 🔹 Fill missing values
df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

# 🔹 Convert target column to 0/1
df["Churn"] = df["Churn"].map({"No": 0, "Yes": 1})

print("\nAfter Cleaning:\n", df.dtypes)
print("\nSample Data:\n", df.head())
