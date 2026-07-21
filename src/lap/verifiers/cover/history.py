"""Stateful LAP-3 episode history manager."""

from __future__ import annotations

import numpy as np

from lap.verifiers.cover.action_adapter import ActionHistoryBatch
from lap.verifiers.cover.action_adapter import NormalizationArtifact
from lap.verifiers.cover.action_adapter import build_action_histories


class EpisodeHistoryManager:
    """Owns committed/pending verifier rows and calls the shared adapter."""

    def __init__(
        self,
        *,
        normalization: NormalizationArtifact,
        artifact_identity: str,
    ) -> None:
        if not artifact_identity:
            raise ValueError("artifact_identity must be nonempty")
        self._normalization = normalization
        self._artifact_identity = artifact_identity
        self._active_episode_id: str | None = None
        self._fixed_instruction: str | None = None
        self._last_accepted_timestep: int | None = None
        self._committed: list[np.ndarray] = []
        self._pending: np.ndarray | None = None
        self._last_histories: np.ndarray | None = None

    @property
    def artifact_identity(self) -> str:
        return self._artifact_identity

    @property
    def active_episode_id(self) -> str | None:
        return self._active_episode_id

    @property
    def fixed_instruction(self) -> str | None:
        return self._fixed_instruction

    @property
    def last_accepted_timestep(self) -> int | None:
        return self._last_accepted_timestep

    @property
    def committed_rows(self) -> np.ndarray:
        if not self._committed:
            return np.empty((0, 7), dtype=np.float64)
        return np.stack(self._committed, axis=0)

    @property
    def pending_row(self) -> np.ndarray | None:
        return None if self._pending is None else np.array(self._pending, copy=True)

    @property
    def last_histories(self) -> np.ndarray:
        if self._last_histories is None:
            raise RuntimeError("no histories have been built yet")
        return self._last_histories

    def clear(self) -> None:
        self._active_episode_id = None
        self._fixed_instruction = None
        self._last_accepted_timestep = None
        self._committed.clear()
        self._pending = None
        self._last_histories = None

    def install_committed_past(self, rows: np.ndarray) -> None:
        """Install already-processed past rows for fixture/runtime parity checks."""
        past = np.asarray(rows, dtype=np.float64)
        if past.ndim != 2 or past.shape[1] != 7:
            raise ValueError("committed past must have shape [P, 7]")
        if past.shape[0] > 6:
            raise ValueError("committed past must contain at most six actions")
        if not np.isfinite(past).all():
            raise ValueError("committed past must be finite")
        self._committed = [np.array(row, dtype=np.float64, copy=True) for row in past]
        self._pending = None
        self._last_histories = None

    def prepare_request(
        self,
        *,
        episode_id: str,
        timestep: int,
        instruction: str,
        expected_artifact_identity: str | None = None,
    ) -> str:
        """Validate/advance episode state for a scoring request. Returns state_event."""
        if not isinstance(episode_id, str) or not episode_id.strip():
            raise ValueError("episode_id must be a nonempty string")
        if not isinstance(timestep, int) or isinstance(timestep, bool) or timestep < 0:
            raise ValueError("timestep must be a nonnegative integer")
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError("instruction must be a nonempty string")

        has_rows = bool(self._committed) or self._pending is not None
        if (
            has_rows
            and expected_artifact_identity is not None
            and expected_artifact_identity != self._artifact_identity
        ):
            raise ValueError("normalization artifact identity mismatch")

        if timestep == 0:
            self.clear()
            self._active_episode_id = episode_id
            self._fixed_instruction = instruction
            self._last_accepted_timestep = 0
            return "new_episode"

        if self._active_episode_id is None or self._last_accepted_timestep is None:
            raise ValueError("state discontinuity: nonzero timestep without an active episode")
        if episode_id != self._active_episode_id:
            raise ValueError("state discontinuity: episode_id changed at nonzero timestep")
        if instruction != self._fixed_instruction:
            raise ValueError("instruction changed under active episode")

        expected_timestep = self._last_accepted_timestep + 1
        if timestep == expected_timestep:
            if self._pending is not None:
                self._committed.append(np.array(self._pending, dtype=np.float64, copy=True))
                if len(self._committed) > 6:
                    self._committed = self._committed[-6:]
                self._pending = None
            self._last_accepted_timestep = timestep
            return "advanced"
        if timestep == self._last_accepted_timestep:
            raise ValueError("duplicate timestep")
        if timestep < self._last_accepted_timestep:
            raise ValueError("stale timestep")
        raise ValueError("skipped timestep")

    def build_histories(
        self,
        *,
        reference_position: np.ndarray,
        reference_rotation_vector: np.ndarray,
        candidate_chunks: np.ndarray,
    ) -> ActionHistoryBatch:
        candidates = np.asarray(candidate_chunks, dtype=np.float64)
        batch = build_action_histories(
            reference_position=reference_position,
            reference_rotation_vector=reference_rotation_vector,
            candidate_chunks=candidates,
            processed_past=self.committed_rows,
            normalization=self._normalization,
        )
        if batch.first_future_index != 6:
            raise ValueError("adapter first_future_index must be 6")
        self._last_histories = np.array(batch.histories, copy=True)
        return ActionHistoryBatch(histories=self._last_histories, first_future_index=6)

    def store_pending_from_history(self, *, candidate_index: int) -> None:
        if self._last_histories is None:
            raise RuntimeError("cannot store pending without built histories")
        if candidate_index < 0 or candidate_index >= self._last_histories.shape[0]:
            raise ValueError("candidate_index out of range for pending history")
        self._pending = np.array(self._last_histories[candidate_index, 6], copy=True)
