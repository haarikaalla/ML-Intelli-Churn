from __future__ import annotations

from intelli_churn.artifacts import ModelRegistry
from intelli_churn.config import TrainingConfig
from intelli_churn.contracts import CustomerFeatures
from intelli_churn.dashboard_runtime import DashboardRuntime
from intelli_churn.demo_data import generate_customers
from intelli_churn.training import train_bundle


def _payload(rows=120) -> tuple[dict, object, dict]:
    frame = generate_customers(rows=rows, seed=17)
    bundle, report = train_bundle(
        frame,
        data_fingerprint="synthetic:dashboard-test",
        config=TrainingConfig(cv_folds=2, random_state=17),
    )
    payload = CustomerFeatures.model_validate(
        frame.drop(columns=["customerID", "Churn"]).iloc[0].to_dict()
    ).model_dump()
    return payload, bundle, report


def test_standalone_runtime_loads_model_card_and_scores(tmp_path):
    payload, bundle, report = _payload()
    ModelRegistry(tmp_path).register(bundle, report)

    runtime = DashboardRuntime.from_local_registry(tmp_path)
    prediction = runtime.predict_one(payload)
    batch = runtime.predict_many([payload, payload])

    assert runtime.source == "local-demo"
    assert runtime.model_card["model_version"] == bundle.model_version
    assert prediction["model_version"] == bundle.model_version
    assert len(batch) == 2


def test_standalone_runtime_bootstraps_when_registry_is_empty(tmp_path, monkeypatch):
    payload, bundle, report = _payload()

    monkeypatch.setattr(
        "intelli_churn.dashboard_runtime.train_bundle",
        lambda frame, data_fingerprint, config: (bundle, report),
    )
    runtime = DashboardRuntime.from_local_registry(tmp_path, demo_rows=120, demo_seed=17)

    assert runtime.source == "local-demo"
    assert (tmp_path / "latest.json").exists()
    assert runtime.predict_one(payload)["prediction"] in {"churn", "stay"}
