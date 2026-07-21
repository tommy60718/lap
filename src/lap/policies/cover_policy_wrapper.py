"""Composed CoVer-Pi0.5 policy wrapper (LAP-1 orchestration)."""

from __future__ import annotations

from typing import Any
from typing import Literal
from typing import Protocol

import numpy as np

from lap.verifiers.candidate_generator import CandidateGenerator

Authority = Literal["disabled", "shadow", "active"]
ExecutionContext = Literal["test", "robot"]

_REQUIRED_OBSERVATION_KEYS = (
    "base_rgb",
    "wrist_rgb",
    "eef_pos",
    "eef_rot",
    "gripper",
    "prompt",
)


class HistoryManager(Protocol):
    def clear(self) -> None: ...

    def build_histories(self, *args: Any, **kwargs: Any) -> Any: ...


class CoverScorer(Protocol):
    def score(self, *args: Any, **kwargs: Any) -> Any: ...


class CoverPolicyWrapper:
    """WebSocket-compatible policy that owns CoVer authority and call order."""

    def __init__(
        self,
        *,
        authority: Authority,
        execution_context: ExecutionContext,
        candidate_count: int,
        candidate_generator: CandidateGenerator,
        history_manager: HistoryManager | None = None,
        scorer: CoverScorer | None = None,
        diagnostics_enabled: bool = True,
    ) -> None:
        if authority not in ("disabled", "shadow", "active"):
            raise ValueError(f"unknown authority: {authority!r}")
        if execution_context not in ("test", "robot"):
            raise ValueError(f"unknown execution_context: {execution_context!r}")
        if not isinstance(candidate_count, int) or isinstance(candidate_count, bool) or candidate_count < 1:
            raise ValueError(f"candidate_count must be a positive integer, got {candidate_count!r}")

        self._authority: Authority = authority
        self._execution_context: ExecutionContext = execution_context
        self._candidate_count = candidate_count
        self._candidate_generator = candidate_generator
        self._history_manager = history_manager
        self._scorer = scorer
        self._diagnostics_enabled = diagnostics_enabled

    def infer(self, obs: dict[str, Any]) -> dict[str, Any]:
        request = dict(obs)
        self._validate_common_request(request)

        if self._authority != "disabled":
            raise NotImplementedError(f"authority {self._authority!r} is not implemented in W4-01")

        return self._infer_disabled(request)

    def _infer_disabled(self, request: dict[str, Any]) -> dict[str, Any]:
        if self._history_manager is not None:
            self._history_manager.clear()

        instruction = str(request["prompt"])
        candidates = self._candidate_generator.generate(
            observation=request,
            instruction=instruction,
            candidate_count=1,
            noise=None,
        )
        actions = self._require_valid_candidate_zero(candidates)

        response: dict[str, Any] = {"actions": actions}
        if self._diagnostics_enabled:
            response.update(
                {
                    "verifier_authority": "disabled",
                    "execution_context": self._execution_context,
                    "candidate_count": 1,
                    "returned_candidate_index": 0,
                    "hypothetical_selected_candidate_index": None,
                    "verifier_scores": None,
                    "fallback_reason": None,
                    "state_event": "disabled",
                }
            )
        return response

    def _validate_common_request(self, request: dict[str, Any]) -> None:
        missing = [key for key in _REQUIRED_OBSERVATION_KEYS if key not in request]
        if missing:
            raise KeyError(f"missing required observation fields: {missing}")
        prompt = request["prompt"]
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be a nonempty string")

    def _require_valid_candidate_zero(self, candidates: np.ndarray) -> np.ndarray:
        if candidates is None:
            raise ValueError("candidate generator returned no actions")
        array = np.asarray(candidates, dtype=np.float64)
        if array.ndim == 2:
            array = array[np.newaxis, ...]
        if array.ndim != 3 or array.shape[1:] != (10, 7) or array.shape[0] < 1:
            raise ValueError("candidate generator must return shape [M, 10, 7] with M >= 1")
        if not np.isfinite(array).all():
            raise ValueError("candidate actions must be finite")
        gripper = array[..., 6]
        if np.any((gripper < 0.0) | (gripper > 1.0)):
            raise ValueError("candidate gripper must be within [0, 1]")
        return np.array(array[0], dtype=np.float64, copy=True)
