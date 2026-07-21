"""LAP-5 selection helpers for CoVer authority modes."""

from __future__ import annotations

import numpy as np


def highest_finite_score_index(scores: np.ndarray) -> int | None:
    """Return the lowest index among finite scores with the maximum value."""
    array = np.asarray(scores, dtype=np.float64)
    if array.ndim != 1 or array.size == 0:
        raise ValueError("scores must be a nonempty 1-D vector")
    finite = np.isfinite(array)
    if not finite.any():
        return None
    masked = np.where(finite, array, -np.inf)
    return int(np.argmax(masked))
