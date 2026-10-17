from __future__ import annotations

import numpy as np


def population_stability_index(reference, current, bins: int = 10) -> float:
    """Compute PSI using reference quantiles and numerical safeguards."""
    reference = np.asarray(reference, dtype=float)
    current = np.asarray(current, dtype=float)
    if reference.size == 0 or current.size == 0:
        raise ValueError("reference and current samples cannot be empty")
    boundaries = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    if boundaries.size < 2:
        return 0.0
    boundaries[0], boundaries[-1] = -np.inf, np.inf
    ref_counts, _ = np.histogram(reference, boundaries)
    cur_counts, _ = np.histogram(current, boundaries)
    ref_share = np.clip(ref_counts / reference.size, 1e-6, None)
    cur_share = np.clip(cur_counts / current.size, 1e-6, None)
    return float(np.sum((cur_share - ref_share) * np.log(cur_share / ref_share)))
