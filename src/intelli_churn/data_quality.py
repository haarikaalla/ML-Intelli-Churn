from __future__ import annotations

from typing import Any

import pandas as pd

from .features import IDENTIFIER_COLUMNS, MODEL_FEATURES, TARGET


def profile_dataset(frame: pd.DataFrame) -> dict[str, Any]:
    """Return a JSON-serializable data contract and quality report."""
    available = [column for column in MODEL_FEATURES if column in frame]
    missing = sorted(set(MODEL_FEATURES + (TARGET,)) - set(frame.columns))
    duplicate_ids = 0
    if IDENTIFIER_COLUMNS[0] in frame:
        duplicate_ids = int(frame[IDENTIFIER_COLUMNS[0]].duplicated().sum())
    target = frame[TARGET] if TARGET in frame else pd.Series(dtype="object")
    if target.dtype == object:
        target = target.map({"Yes": 1, "No": 0})
    return {
        "rows": len(frame),
        "columns": len(frame.columns),
        "required_columns_missing": missing,
        "duplicate_customer_ids": duplicate_ids,
        "missing_values": {column: int(frame[column].isna().sum()) for column in available},
        "churn_rate": round(float(target.mean()), 6) if not target.empty else None,
        "quality_status": "pass" if not missing and duplicate_ids == 0 else "warning",
    }
