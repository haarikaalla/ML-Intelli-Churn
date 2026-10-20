from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .features import MODEL_FEATURES
from .predictor import Predictor


def score_file(
    input_path: Path, output_path: Path, registry: Path, version: str = "latest"
) -> dict[str, object]:
    frame = pd.read_csv(input_path)
    missing = sorted(set(MODEL_FEATURES) - set(frame.columns))
    if missing:
        raise ValueError(f"Input data is missing required columns: {', '.join(missing)}")
    predictor = Predictor.from_registry(registry, version)
    predictions = predictor.predict_frame(frame[list(MODEL_FEATURES)])
    scored = frame.copy()
    scored["churn_probability"] = [item.churn_probability for item in predictions]
    scored["churn_prediction"] = [item.prediction for item in predictions]
    scored["risk_band"] = [item.risk_band for item in predictions]
    scored["recommended_action"] = [item.recommended_action for item in predictions]
    scored["model_version"] = predictor.version
    output_path.parent.mkdir(parents=True, exist_ok=True)
    scored.to_csv(output_path, index=False)
    return {
        "input_rows": len(frame),
        "output": str(output_path),
        "model_version": predictor.version,
        "high_risk_customers": int((scored["risk_band"] == "high").sum()),
    }


def main() -> None:  # pragma: no cover - thin CLI adapter
    parser = argparse.ArgumentParser(description="Score a customer CSV with IntelliChurn")
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, default=Path("outputs/scored_customers.csv"))
    parser.add_argument("--registry", type=Path, default=Path("artifacts/registry"))
    parser.add_argument("--version", default="latest")
    args = parser.parse_args()
    print(json.dumps(score_file(args.input, args.output, args.registry, args.version), indent=2))


if __name__ == "__main__":  # pragma: no cover
    main()
