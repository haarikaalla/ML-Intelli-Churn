from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class TrainingConfig:
    """Reproducible training and business-decision configuration."""

    random_state: int = 42
    test_size: float = 0.20
    validation_size: float = 0.20
    cv_folds: int = 5
    false_positive_cost: float = 15.0
    false_negative_cost: float = 500.0
    true_positive_cost: float = 15.0
    min_recall: float = 0.45

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)
