from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from .artifacts import ModelBundle, ModelRegistry
from .features import CATEGORICAL_FEATURES, MODEL_FEATURES, NUMERIC_FEATURES

EPSILON = 1e-6


def population_stability_index(reference, current, bins: int = 10) -> float:
    """Compute PSI using reference quantiles and numerical safeguards."""
    reference = np.asarray(reference, dtype=float)
    current = np.asarray(current, dtype=float)
    reference = reference[np.isfinite(reference)]
    current = current[np.isfinite(current)]
    if reference.size == 0 or current.size == 0:
        raise ValueError("reference and current samples cannot be empty")
    boundaries = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    if boundaries.size < 2:
        return 0.0
    boundaries[0], boundaries[-1] = -np.inf, np.inf
    ref_counts, _ = np.histogram(reference, boundaries)
    cur_counts, _ = np.histogram(current, boundaries)
    ref_share = np.clip(ref_counts / reference.size, EPSILON, None)
    cur_share = np.clip(cur_counts / current.size, EPSILON, None)
    return float(np.sum((cur_share - ref_share) * np.log(cur_share / ref_share)))


def jensen_shannon_divergence(reference: dict[str, float], current: dict[str, float]) -> float:
    categories = sorted(set(reference) | set(current))
    p = np.array([reference.get(key, 0.0) for key in categories], dtype=float)
    q = np.array([current.get(key, 0.0) for key in categories], dtype=float)
    p = np.clip(p / max(p.sum(), EPSILON), EPSILON, None)
    q = np.clip(q / max(q.sum(), EPSILON), EPSILON, None)
    midpoint = (p + q) / 2
    return float(0.5 * np.sum(p * np.log(p / midpoint)) + 0.5 * np.sum(q * np.log(q / midpoint)))


def build_reference_profile(frame: pd.DataFrame) -> dict[str, Any]:
    profile: dict[str, Any] = {"rows": len(frame), "numeric": {}, "categorical": {}}
    for column in NUMERIC_FEATURES:
        values = pd.to_numeric(frame[column], errors="coerce").dropna().to_numpy(dtype=float)
        profile["numeric"][column] = values.tolist()
    for column in CATEGORICAL_FEATURES:
        shares = frame[column].fillna("__missing__").astype(str).value_counts(normalize=True)
        profile["categorical"][column] = {
            str(category): float(share) for category, share in shares.items()
        }
    return profile


def drift_report(current: pd.DataFrame, reference_profile: dict[str, Any]) -> dict[str, Any]:
    missing = sorted(set(MODEL_FEATURES) - set(current.columns))
    if missing:
        raise ValueError(f"Current data is missing required columns: {', '.join(missing)}")
    feature_results: list[dict[str, Any]] = []
    for column, reference in reference_profile["numeric"].items():
        values = pd.to_numeric(current[column], errors="coerce").dropna().to_numpy(dtype=float)
        score = population_stability_index(reference, values)
        feature_results.append(_drift_result(column, "psi", score, 0.10, 0.25))
    for column, reference in reference_profile["categorical"].items():
        shares = current[column].fillna("__missing__").astype(str).value_counts(normalize=True)
        score = jensen_shannon_divergence(reference, shares.to_dict())
        feature_results.append(_drift_result(column, "js_divergence", score, 0.05, 0.10))
    drifted = [item["feature"] for item in feature_results if item["status"] != "stable"]
    return {
        "status": "drift_detected" if drifted else "stable",
        "current_rows": len(current),
        "reference_rows": int(reference_profile["rows"]),
        "drifted_features": drifted,
        "features": feature_results,
    }


def _drift_result(
    feature: str, metric: str, score: float, warning_threshold: float, alert_threshold: float
) -> dict[str, Any]:
    status = "alert" if score >= alert_threshold else "warning" if score >= warning_threshold else "stable"
    return {"feature": feature, "metric": metric, "score": round(score, 6), "status": status}


def main() -> None:  # pragma: no cover - thin CLI adapter
    parser = argparse.ArgumentParser(description="Compare current customer data with training data")
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--registry", type=Path, default=Path("artifacts/registry"))
    parser.add_argument("--model", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    bundle = joblib.load(args.model) if args.model else ModelRegistry(args.registry).load()
    if not isinstance(bundle, ModelBundle):
        raise TypeError("Monitoring requires a version 2 ModelBundle")
    report = drift_report(pd.read_csv(args.dataset), bundle.reference_profile)
    rendered = json.dumps(report, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered)


if __name__ == "__main__":  # pragma: no cover
    main()
