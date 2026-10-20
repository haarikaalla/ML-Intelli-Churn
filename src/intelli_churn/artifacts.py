from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib


@dataclass
class ModelBundle:
    """Versioned inference artifact containing the model and its operating contract."""

    estimator: Any
    model_version: str
    trained_at: str
    threshold: float
    metrics: dict[str, Any]
    candidate_scores: list[dict[str, Any]]
    feature_importance: list[dict[str, Any]]
    reference_profile: dict[str, Any]
    data_fingerprint: str
    training_config: dict[str, Any]
    schema_version: str = "2.0"

    def metadata(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "model_version": self.model_version,
            "trained_at": self.trained_at,
            "threshold": self.threshold,
            "metrics": self.metrics,
            "candidate_scores": self.candidate_scores,
            "feature_importance": self.feature_importance,
            "data_fingerprint": self.data_fingerprint,
            "training_config": self.training_config,
        }


def fingerprint_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return f"sha256:{digest.hexdigest()}"


def generated_version(data_fingerprint: str) -> str:
    timestamp = datetime.now(UTC).strftime("%Y%m%d.%H%M%S")
    return f"{timestamp}-{data_fingerprint.split(':')[-1][:8]}"


class ModelRegistry:
    """Small filesystem registry with immutable versions and an atomic latest pointer."""

    def __init__(self, root: str | Path = "artifacts/registry") -> None:
        self.root = Path(root)

    def register(self, bundle: ModelBundle, reports: dict[str, Any]) -> Path:
        version_dir = self.root / bundle.model_version
        version_dir.mkdir(parents=True, exist_ok=False)
        model_path = version_dir / "model.joblib"
        joblib.dump(bundle, model_path)
        (version_dir / "metadata.json").write_text(
            json.dumps(bundle.metadata(), indent=2), encoding="utf-8"
        )
        (version_dir / "training_report.json").write_text(
            json.dumps(reports, indent=2), encoding="utf-8"
        )
        pointer = {
            "model_version": bundle.model_version,
            "model_path": str(model_path.relative_to(self.root)),
            "updated_at": datetime.now(UTC).isoformat(),
        }
        self.root.mkdir(parents=True, exist_ok=True)
        temporary = self.root / "latest.json.tmp"
        temporary.write_text(json.dumps(pointer, indent=2), encoding="utf-8")
        temporary.replace(self.root / "latest.json")
        return model_path

    def resolve(self, version: str = "latest") -> Path:
        if version == "latest":
            pointer = self.root / "latest.json"
            if not pointer.exists():
                raise FileNotFoundError("No registered model; run intelli-churn-train first")
            stored = Path(json.loads(pointer.read_text(encoding="utf-8"))["model_path"])
            return stored if stored.is_absolute() else self.root / stored
        path = self.root / version / "model.joblib"
        if not path.exists():
            raise FileNotFoundError(f"Model version {version!r} is not registered")
        return path

    def load(self, version: str = "latest") -> ModelBundle:
        artifact = joblib.load(self.resolve(version))
        if not isinstance(artifact, ModelBundle):
            raise TypeError("Artifact is not an IntelliChurn ModelBundle")
        return artifact
