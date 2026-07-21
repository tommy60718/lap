"""Composed CoVer-Pi0.5 policy wrapper (LAP-1 orchestration)."""

from __future__ import annotations

from typing import Any
from typing import Literal
from typing import Protocol

import numpy as np

from lap.verifiers.candidate_generator import CandidateGenerator
from lap.verifiers.selection import highest_finite_score_index

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
    artifact_identity: str

    def clear(self) -> None: ...

    def prepare_request(self, *, episode_id: str, timestep: int, instruction: str) -> str: ...

    def build_histories(
        self,
        *,
        reference_position: np.ndarray,
        reference_rotation_vector: np.ndarray,
        candidate_chunks: np.ndarray,
    ) -> Any: ...

    def store_pending_from_history(self, *, candidate_index: int) -> None: ...


class CoverScorer(Protocol):
    compatibility: Any

    def score(
        self,
        *,
        base_rgb: np.ndarray,
        wrist_rgb: np.ndarray,
        instruction: str,
        action_histories: np.ndarray,
    ) -> np.ndarray: ...


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
        if authority in ("shadow", "active") and history_manager is None:
            raise ValueError(f"{authority} authority requires a history_manager")
        if authority in ("shadow", "active") and scorer is None:
            raise ValueError(f"{authority} authority requires a scorer")

        self._authority: Authority = authority
        self._execution_context: ExecutionContext = execution_context
        self._candidate_count = candidate_count
        self._candidate_generator = candidate_generator
        self._history_manager = history_manager
        self._scorer = scorer
        self._diagnostics_enabled = diagnostics_enabled
        self._scorer_validated = False

    def infer(self, obs: dict[str, Any], *, noise: np.ndarray | None = None) -> dict[str, Any]:
        request = dict(obs)
        self._validate_common_request(request)

        if self._authority == "disabled":
            return self._infer_disabled(request)
        if self._authority == "shadow":
            return self._infer_shadow(request, noise=noise)
        raise NotImplementedError(f"authority {self._authority!r} is not implemented yet")

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
        actions = self._require_valid_candidates(candidates)[0]

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

    def _infer_shadow(self, request: dict[str, Any], *, noise: np.ndarray | None) -> dict[str, Any]:
        if self._history_manager is None or self._scorer is None:
            raise ValueError("shadow authority requires history_manager and scorer")

        episode_id, timestep = self._require_cover_metadata(request)
        instruction = str(request["prompt"])
        state_event = self._history_manager.prepare_request(
            episode_id=episode_id,
            timestep=timestep,
            instruction=instruction,
        )

        noise_copy = None if noise is None else np.array(noise, copy=True)
        candidates = self._candidate_generator.generate(
            observation=request,
            instruction=instruction,
            candidate_count=self._candidate_count,
            noise=noise_copy,
        )
        candidate_batch = self._require_valid_candidates(candidates)
        if candidate_batch.shape[0] != self._candidate_count:
            raise ValueError(
                f"candidate generator returned {candidate_batch.shape[0]} candidates, expected {self._candidate_count}"
            )

        history_batch = self._history_manager.build_histories(
            reference_position=np.asarray(request["eef_pos"], dtype=np.float64),
            reference_rotation_vector=np.asarray(request["eef_rot"], dtype=np.float64),
            candidate_chunks=candidate_batch,
        )
        histories = np.asarray(history_batch.histories)
        if histories.shape != (self._candidate_count, 10, 7) or histories.dtype != np.float32:
            raise ValueError("history manager must return float32 histories with shape [M, 10, 7]")

        self._ensure_scorer_compatible()
        scores = np.asarray(
            self._scorer.score(
                base_rgb=np.asarray(request["base_rgb"]),
                wrist_rgb=np.asarray(request["wrist_rgb"]),
                instruction=instruction,
                action_histories=histories,
            ),
            dtype=np.float64,
        )
        if scores.shape != (self._candidate_count,):
            raise ValueError("scorer must return one score per candidate")

        hypothetical = highest_finite_score_index(scores)
        returned_index = 0
        self._history_manager.store_pending_from_history(candidate_index=returned_index)
        actions = np.array(candidate_batch[returned_index], dtype=np.float64, copy=True)

        response: dict[str, Any] = {"actions": actions}
        if self._diagnostics_enabled:
            response.update(
                {
                    "verifier_authority": "shadow",
                    "execution_context": self._execution_context,
                    "candidate_count": self._candidate_count,
                    "returned_candidate_index": returned_index,
                    "hypothetical_selected_candidate_index": hypothetical,
                    "verifier_scores": scores.tolist(),
                    "fallback_reason": None,
                    "state_event": state_event,
                }
            )
        return response

    def _ensure_scorer_compatible(self) -> None:
        if self._history_manager is None or self._scorer is None:
            raise ValueError("scorer compatibility requires history_manager and scorer")
        if self._scorer_validated:
            return
        compatibility = self._scorer.compatibility
        compatibility.validate_against_expected(
            expected_normalization_hash=self._history_manager.artifact_identity
        )
        self._scorer_validated = True

    def _require_cover_metadata(self, request: dict[str, Any]) -> tuple[str, int]:
        if "episode_id" not in request or "timestep" not in request:
            raise ValueError("shadow/active requests require episode_id and timestep")
        episode_id = request["episode_id"]
        timestep = request["timestep"]
        if not isinstance(episode_id, str) or not episode_id.strip():
            raise ValueError("episode_id must be a nonempty string")
        if not isinstance(timestep, int) or isinstance(timestep, bool) or timestep < 0:
            raise ValueError("timestep must be a nonnegative integer")
        return episode_id, timestep

    def _validate_common_request(self, request: dict[str, Any]) -> None:
        missing = [key for key in _REQUIRED_OBSERVATION_KEYS if key not in request]
        if missing:
            raise KeyError(f"missing required observation fields: {missing}")
        prompt = request["prompt"]
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be a nonempty string")

    def _require_valid_candidates(self, candidates: np.ndarray) -> np.ndarray:
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
        return np.array(array, dtype=np.float64, copy=True)
