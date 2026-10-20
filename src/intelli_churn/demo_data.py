from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def generate_customers(rows: int = 2500, seed: int = 42) -> pd.DataFrame:
    """Create contract-compatible synthetic data for a no-download demo."""
    rng = np.random.default_rng(seed)
    tenure = rng.integers(0, 73, rows)
    contract = rng.choice(
        ["Month-to-month", "One year", "Two year"], rows, p=[0.55, 0.25, 0.20]
    )
    internet = rng.choice(["DSL", "Fiber optic", "No"], rows, p=[0.35, 0.45, 0.20])
    monthly = np.clip(
        20 + (internet == "DSL") * 30 + (internet == "Fiber optic") * 55 + rng.normal(0, 12, rows),
        18,
        125,
    )
    frame = pd.DataFrame(
        {
            "customerID": [f"DEMO-{index:06d}" for index in range(rows)],
            "gender": rng.choice(["Female", "Male"], rows),
            "SeniorCitizen": rng.binomial(1, 0.16, rows),
            "Partner": rng.choice(["Yes", "No"], rows),
            "Dependents": rng.choice(["Yes", "No"], rows, p=[0.30, 0.70]),
            "tenure": tenure,
            "PhoneService": rng.choice(["Yes", "No"], rows, p=[0.90, 0.10]),
            "InternetService": internet,
            "Contract": contract,
            "PaperlessBilling": rng.choice(["Yes", "No"], rows, p=[0.60, 0.40]),
            "PaymentMethod": rng.choice(
                [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)",
                ],
                rows,
            ),
            "MonthlyCharges": monthly.round(2),
            "TotalCharges": np.maximum(monthly * np.maximum(tenure, 1) + rng.normal(0, 50, rows), 0).round(2),
        }
    )
    frame["MultipleLines"] = np.where(
        frame["PhoneService"].eq("No"),
        "No phone service",
        rng.choice(["Yes", "No"], rows),
    )
    for column in [
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
    ]:
        frame[column] = np.where(
            frame["InternetService"].eq("No"),
            "No internet service",
            rng.choice(["Yes", "No"], rows),
        )
    logits = (
        -1.8
        + 1.25 * frame["Contract"].eq("Month-to-month")
        + 0.65 * frame["InternetService"].eq("Fiber optic")
        + 0.55 * frame["PaymentMethod"].eq("Electronic check")
        - 0.025 * frame["tenure"]
        - 0.45 * frame["TechSupport"].eq("Yes")
    )
    probability = 1 / (1 + np.exp(-logits))
    frame["Churn"] = np.where(rng.random(rows) < probability, "Yes", "No")
    return frame


def main() -> None:  # pragma: no cover - thin CLI adapter
    parser = argparse.ArgumentParser(description="Generate demo data for the IntelliChurn pipeline")
    parser.add_argument("--rows", type=int, default=2500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("data/demo_telco_churn.csv"))
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    generate_customers(args.rows, args.seed).to_csv(args.output, index=False)
    print(f"Generated {args.rows} customers at {args.output}")


if __name__ == "__main__":  # pragma: no cover
    main()
