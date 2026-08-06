"""Fake and protocol surface for LAP-4 CoVer scoring."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from lap.verifiers.cover.action_adapter import ACTION_ORDER
from lap.verifiers.cover.action_adapter import REPRESENTATION_ID


@dataclass(frozen=True)
class ScorerCompatibility:
    model_schema_version: str
    views: tuple[str, ...]
    preprocessing_contract: str
    action_dimension: int
    action_order: tuple[str, ...]
    history_length: int
    representation_id: str
    normalization_artifact_hash: str
    input_dtype: str
    output_shape_rank: int

    def validate_against_expected(self, *, expected_normalization_hash: str) -> None:
        if not self.model_schema_version:
            raise ValueError("incompatible scorer: empty model_schema_version")
        if tuple(self.views) != ("base_rgb", "wrist_rgb"):
            raise ValueError("incompatible scorer: views must be [base_rgb, wrist_rgb]")
        if not self.preprocessing_contract:
            raise ValueError("incompatible scorer: empty preprocessing_contract")
        if self.action_dimension != 7:
            raise ValueError("incompatible scorer: action_dimension must be 7")
        if tuple(self.action_order) != tuple(ACTION_ORDER):
            raise ValueError("incompatible scorer: action_order mismatch")
        if self.history_length != 10:
            raise ValueError("incompatible scorer: history_length must be 10")
        if self.representation_id != REPRESENTATION_ID:
            raise ValueError("incompatible scorer: representation_id mismatch")
        if self.normalization_artifact_hash != expected_normalization_hash:
            raise ValueError("incompatible scorer: normalization_artifact_hash mismatch")
        if self.input_dtype != "float32":
            raise ValueError("incompatible scorer: input_dtype must be float32")
        if self.output_shape_rank != 1:
            raise ValueError("incompatible scorer: output_shape_rank must be 1")


class FakeCoverScorer:
    """Deterministic score-only LAP-4 double for W4 fake integration."""

    is_fake = True

    def __init__(
        self,
        *,
        scores: np.ndarray | Sequence[float],
        compatibility: ScorerCompatibility,
    ) -> None:
        self._scores = np.asarray(scores, dtype=np.float64)
        self.compatibility = compatibility
        self.calls: list[dict[str, Any]] = []

    def score(
        self,
        *,
        base_rgb: np.ndarray,
        wrist_rgb: np.ndarray,
        instruction: str,
        action_histories: np.ndarray,
    ) -> np.ndarray:
        histories = np.asarray(action_histories)
        self.calls.append(
            {
                "base_rgb": np.asarray(base_rgb).copy(),
                "wrist_rgb": np.asarray(wrist_rgb).copy(),
                "instruction": instruction,
                "action_histories": histories.copy(),
            }
        )
        if histories.ndim != 3 or histories.shape[1:] != (10, 7):
            raise ValueError("action_histories must have shape [M, 10, 7]")
        if histories.dtype != np.float32:
            raise ValueError("action_histories must be float32")
        count = histories.shape[0]
        if self._scores.shape != (count,):
            raise ValueError(f"fake scorer configured for {self._scores.shape[0]} scores, got {count} histories")
        return np.array(self._scores, dtype=np.float64, copy=True)
