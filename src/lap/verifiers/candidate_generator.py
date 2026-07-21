"""Injectable Pi0.5 candidate-generation protocol for CoVer composition."""

from __future__ import annotations

from typing import Any
from typing import Protocol

import numpy as np


class CandidateGenerator(Protocol):
    """LAP-2 seam: ordered absolute UR5e candidate chunks."""

    def generate(
        self,
        *,
        observation: dict[str, Any],
        instruction: str,
        candidate_count: int,
        noise: np.ndarray | None = None,
    ) -> np.ndarray:
        """Return finite absolute candidates with shape ``[M, 10, 7]``."""
        ...
