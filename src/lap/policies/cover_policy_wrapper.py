"""Composed CoVer-Pi0.5 policy wrapper (LAP-1 orchestration)."""

from __future__ import annotations

import json
import logging
from typing import Any
from typing import Literal
from typing import Protocol

import numpy as np

from lap.verifiers.candidate_generator import CandidateGenerator
from lap.verifiers.cover.faults import CoverStateError
from lap.verifiers.cover.faults import CoverVerifierError
from lap.verifiers.cover.faults import FallbackReason
from lap.verifiers.cover.history import PrepareResult
from lap.verifiers.selection import highest_finite_score_index

Authority = Literal["disabled", "shadow", "active"]
ExecutionContext = Literal["test", "robot"]
logger = logging.getLogger(__name__)

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

    def clear_pending(self) -> None: ...

    def prepare_request(
        self,
        *,
        episode_id: str,
        timestep: int,
        instruction: str,
        expected_artifact_identity: str | None = None,
        execution_context: str = "test",
    ) -> PrepareResult: ...

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
        allow_fake_scorer: bool = False,
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
        if (
            execution_context == "robot"
            and authority in ("shadow", "active")
            and scorer is not None
            and getattr(scorer, "is_fake", False)
            and not allow_fake_scorer
        ):
            raise ValueError("deployable robot shadow/active configuration rejects a fake scorer")

        self._authority: Authority = authority
        self._execution_context: ExecutionContext = execution_context
        self._candidate_count = candidate_count
        self._candidate_generator = candidate_generator
        self._history_manager = history_manager
        self._scorer = scorer
        self._diagnostics_enabled = diagnostics_enabled
        self._scorer_validated = False
        self._last_diagnostic_fault: str | None = None

    @property
    def last_diagnostic_fault(self) -> str | None:
        return self._last_diagnostic_fault

    def infer(self, obs: dict[str, Any], *, noise: np.ndarray | None = None) -> dict[str, Any]:
        request = dict(obs)
        self._validate_common_request(request)

        if self._authority == "disabled":
            return self._infer_disabled(request)
        if self._authority in ("shadow", "active"):
            return self._infer_cover(request, noise=noise)
        raise NotImplementedError(f"authority {self._authority!r} is not implemented yet")

    def reset_session(self) -> None:
        """Discard cover episode history when the owning WebSocket client disconnects."""
        if self._history_manager is not None:
            self._history_manager.clear()

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
        return self._assemble_response(
            actions=actions,
            returned_index=0,
            hypothetical=None,
            scores=None,
            state_event="disabled",
            fallback_reason=None,
            candidate_count=1,
        )

    def _infer_cover(self, request: dict[str, Any], *, noise: np.ndarray | None) -> dict[str, Any]:
        if self._history_manager is None or self._scorer is None:
            raise ValueError(f"{self._authority} authority requires history_manager and scorer")

        instruction = str(request["prompt"])
        metadata_fallback: str | None = None
        prepare = PrepareResult("advanced")
        try:
            episode_id, timestep = self._require_cover_metadata(request)
        except CoverStateError as error:
            if self._execution_context != "robot":
                raise
            metadata_fallback = error.fallback_reason
            self._history_manager.clear()
        else:
            prepare = self._history_manager.prepare_request(
                episode_id=episode_id,
                timestep=timestep,
                instruction=instruction,
                expected_artifact_identity=self._scorer.compatibility.normalization_artifact_hash,
                execution_context=self._execution_context,
            )

        # Pre-candidate failures must propagate unchanged (no broad catch).
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

        continuity_fallback = metadata_fallback or prepare.fallback_reason
        if continuity_fallback is not None:
            self._history_manager.clear()
            state_event = "metadata_fallback" if metadata_fallback else "discontinuity_reset"
            return self._assemble_response(
                actions=np.array(candidate_batch[0], dtype=np.float64, copy=True),
                returned_index=0,
                hypothetical=None,
                scores=None,
                state_event=state_event,
                fallback_reason=continuity_fallback,
            )

        try:
            return self._score_and_select(
                request=request,
                instruction=instruction,
                candidate_batch=candidate_batch,
                state_event=prepare.state_event,
            )
        except CoverVerifierError as error:
            if self._execution_context != "robot":
                raise
            return self._robot_verifier_fallback(
                candidate_batch=candidate_batch,
                state_event=prepare.state_event,
                fallback_reason=error.fallback_reason,
                store_pending=error.store_pending,
            )

    def _score_and_select(
        self,
        *,
        request: dict[str, Any],
        instruction: str,
        candidate_batch: np.ndarray,
        state_event: str,
    ) -> dict[str, Any]:
        assert self._history_manager is not None
        assert self._scorer is not None
        try:
            history_batch = self._history_manager.build_histories(
                reference_position=np.asarray(request["eef_pos"], dtype=np.float64),
                reference_rotation_vector=np.asarray(request["eef_rot"], dtype=np.float64),
                candidate_chunks=candidate_batch,
            )
        except Exception as error:  # noqa: BLE001 - classify as history fault
            raise CoverVerifierError(
                str(error),
                fallback_reason=FallbackReason.HISTORY_INVALID,
                store_pending=False,
            ) from error

        histories = np.asarray(history_batch.histories)
        if histories.shape != (self._candidate_count, 10, 7) or histories.dtype != np.float32:
            raise CoverVerifierError(
                "history manager must return float32 histories with shape [M, 10, 7]",
                fallback_reason=FallbackReason.HISTORY_INVALID,
                store_pending=False,
            )

        if self._is_missing_or_zero_wrist(request.get("wrist_rgb")):
            raise CoverVerifierError(
                "missing or all-zero wrist view",
                fallback_reason=FallbackReason.MISSING_OR_ZERO_WRIST_VIEW,
                store_pending=True,
            )

        try:
            self._ensure_scorer_compatible()
        except ValueError as error:
            raise CoverVerifierError(
                str(error),
                fallback_reason=FallbackReason.INCOMPATIBLE_VERIFIER_ASSETS,
                store_pending=True,
            ) from error

        try:
            raw_scores = self._scorer.score(
                base_rgb=np.asarray(request["base_rgb"]),
                wrist_rgb=np.asarray(request["wrist_rgb"]),
                instruction=instruction,
                action_histories=histories,
            )
        except Exception as error:  # noqa: BLE001 - unavailable scorer
            raise CoverVerifierError(
                str(error),
                fallback_reason=FallbackReason.VERIFIER_UNAVAILABLE,
                store_pending=True,
            ) from error

        scores = np.asarray(raw_scores, dtype=np.float64)
        if scores.ndim != 1 or scores.shape[0] != self._candidate_count:
            raise CoverVerifierError(
                "scorer must return one score per candidate",
                fallback_reason=FallbackReason.SCORES_INVALID,
                store_pending=True,
            )

        selected = highest_finite_score_index(scores)
        if selected is None:
            raise CoverVerifierError(
                "no finite verifier scores",
                fallback_reason=FallbackReason.SCORES_INVALID,
                store_pending=True,
            )
        if self._authority == "shadow":
            returned_index = 0
            hypothetical = selected
        else:
            returned_index = selected
            hypothetical = None

        self._history_manager.store_pending_from_history(candidate_index=returned_index)
        actions = np.array(candidate_batch[returned_index], dtype=np.float64, copy=True)
        return self._assemble_response(
            actions=actions,
            returned_index=returned_index,
            hypothetical=hypothetical,
            scores=scores,
            state_event=state_event,
            fallback_reason=None,
        )

    def _robot_verifier_fallback(
        self,
        *,
        candidate_batch: np.ndarray,
        state_event: str,
        fallback_reason: str,
        store_pending: bool,
    ) -> dict[str, Any]:
        assert self._history_manager is not None
        returned_index = 0
        if store_pending:
            self._history_manager.store_pending_from_history(candidate_index=returned_index)
        else:
            self._history_manager.clear_pending()
        return self._assemble_response(
            actions=np.array(candidate_batch[0], dtype=np.float64, copy=True),
            returned_index=returned_index,
            hypothetical=None,
            scores=None,
            state_event=state_event,
            fallback_reason=fallback_reason,
        )

    def _assemble_response(
        self,
        *,
        actions: np.ndarray,
        returned_index: int,
        hypothetical: int | None,
        scores: np.ndarray | None,
        state_event: str,
        fallback_reason: str | None,
        candidate_count: int | None = None,
    ) -> dict[str, Any]:
        response: dict[str, Any] = {"actions": actions}
        if not self._diagnostics_enabled:
            return response
        try:
            diagnostics = {
                "verifier_authority": self._authority,
                "execution_context": self._execution_context,
                "candidate_count": self._candidate_count if candidate_count is None else candidate_count,
                "returned_candidate_index": returned_index,
                "hypothetical_selected_candidate_index": hypothetical,
                "verifier_scores": None if scores is None else scores.tolist(),
                "fallback_reason": fallback_reason,
                "state_event": state_event,
            }
            json.dumps(diagnostics)
            response.update(diagnostics)
        except Exception as error:  # noqa: BLE001 - diagnostic serialization fault
            if self._execution_context != "robot":
                raise
            self._last_diagnostic_fault = FallbackReason.DIAGNOSTIC_SERIALIZATION_FAILED
            logger.exception("diagnostic serialization failed: %s", error)
            return {"actions": actions}
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
            raise CoverStateError(
                "shadow/active requests require episode_id and timestep",
                fallback_reason=FallbackReason.MISSING_COVER_METADATA,
            )
        episode_id = request["episode_id"]
        timestep = request["timestep"]
        if not isinstance(episode_id, str) or not episode_id.strip():
            raise CoverStateError(
                "episode_id must be a nonempty string",
                fallback_reason=FallbackReason.MISSING_COVER_METADATA,
            )
        if not isinstance(timestep, int) or isinstance(timestep, bool) or timestep < 0:
            raise CoverStateError(
                "timestep must be a nonnegative integer",
                fallback_reason=FallbackReason.MISSING_COVER_METADATA,
            )
        return episode_id, timestep

    def _is_missing_or_zero_wrist(self, wrist_rgb: Any) -> bool:
        if wrist_rgb is None:
            return True
        array = np.asarray(wrist_rgb)
        return array.size == 0 or not np.any(array)

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
