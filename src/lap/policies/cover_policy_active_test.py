"""W4-04 behavior tests for active CoVer candidate selection."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from lap.policies.cover_policy_wrapper import CoverPolicyWrapper
from lap.verifiers.cover.action_adapter import ACTION_ORDER
from lap.verifiers.cover.action_adapter import REPRESENTATION_ID
from lap.verifiers.cover.action_adapter import NormalizationArtifact
from lap.verifiers.cover.history import EpisodeHistoryManager
from lap.verifiers.cover.scorer import FakeCoverScorer
from lap.verifiers.cover.scorer import ScorerCompatibility


def _finite_chunk(*, seed: int, offset: float = 0.0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    chunk = rng.normal(loc=offset, scale=0.04, size=(10, 7)).astype(np.float64)
    chunk[:, 6] = np.clip(np.linspace(0.2, 0.8, 10) + 0.01 * (seed % 5), 0.0, 1.0)
    return chunk


def _candidate_batch(count: int, *, seed: int) -> np.ndarray:
    return np.stack([_finite_chunk(seed=seed + i, offset=0.05 * i) for i in range(count)], axis=0)


def _normalization() -> NormalizationArtifact:
    return NormalizationArtifact(q01=np.zeros(6, dtype=np.float64), q99=np.ones(6, dtype=np.float64))


def _compatibility(*, artifact_hash: str) -> ScorerCompatibility:
    return ScorerCompatibility(
        model_schema_version="fake_cover_v0",
        views=("base_rgb", "wrist_rgb"),
        preprocessing_contract="ur5e_ws1_uint8_224",
        action_dimension=7,
        action_order=ACTION_ORDER,
        history_length=10,
        representation_id=REPRESENTATION_ID,
        normalization_artifact_hash=artifact_hash,
        input_dtype="float32",
        output_shape_rank=1,
    )


def _request(*, episode_id: str = "ep-1", timestep: int = 0, prompt: str = "reach the peg") -> dict[str, Any]:
    return {
        "base_rgb": np.zeros((224, 224, 3), dtype=np.uint8),
        "wrist_rgb": np.full((224, 224, 3), 4, dtype=np.uint8),
        "eef_pos": np.asarray([0.1, 0.2, 0.3], dtype=np.float64),
        "eef_rot": np.asarray([0.0, 0.1, -0.1], dtype=np.float64),
        "gripper": np.asarray([0.7], dtype=np.float64),
        "prompt": prompt,
        "episode_id": episode_id,
        "timestep": timestep,
    }


class RecordingCandidateGenerator:
    def __init__(self, candidates: np.ndarray, *, vary_by_timestep: bool = False) -> None:
        self._candidates = np.asarray(candidates, dtype=np.float64)
        self._vary_by_timestep = vary_by_timestep
        self.calls: list[dict[str, Any]] = []

    def generate(
        self,
        *,
        observation: dict[str, Any],
        instruction: str,
        candidate_count: int,
        noise: np.ndarray | None = None,
    ) -> np.ndarray:
        self.calls.append({"candidate_count": candidate_count, "timestep": observation.get("timestep")})
        batch = self._candidates[:candidate_count].copy()
        if self._vary_by_timestep:
            timestep = int(observation.get("timestep", 0))
            batch = batch + np.asarray([0.02 * timestep, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
            batch[..., 6] = np.clip(batch[..., 6], 0.0, 1.0)
        return batch


class SequenceFakeScorer:
    """Returns a configured score vector per call without selecting actions."""

    def __init__(self, score_sequence: list[np.ndarray], *, compatibility: ScorerCompatibility) -> None:
        self._score_sequence = [np.asarray(scores, dtype=np.float64) for scores in score_sequence]
        self.compatibility = compatibility
        self.calls: list[dict[str, Any]] = []
        self._index = 0

    def score(
        self,
        *,
        base_rgb: np.ndarray,
        wrist_rgb: np.ndarray,
        instruction: str,
        action_histories: np.ndarray,
    ) -> np.ndarray:
        histories = np.asarray(action_histories)
        self.calls.append({"action_histories": histories.copy(), "instruction": instruction})
        if self._index >= len(self._score_sequence):
            raise RuntimeError("no more configured score vectors")
        scores = np.array(self._score_sequence[self._index], dtype=np.float64, copy=True)
        self._index += 1
        return scores


def _active_policy(
    *,
    scores: np.ndarray | list[np.ndarray],
    candidate_count: int = 3,
    seed: int = 0,
    artifact_hash: str = "a" * 64,
    vary_by_timestep: bool = False,
    candidates: np.ndarray | None = None,
) -> tuple[CoverPolicyWrapper, EpisodeHistoryManager, Any, RecordingCandidateGenerator]:
    batch = _candidate_batch(candidate_count, seed=seed) if candidates is None else candidates
    generator = RecordingCandidateGenerator(batch, vary_by_timestep=vary_by_timestep)
    history = EpisodeHistoryManager(normalization=_normalization(), artifact_identity=artifact_hash)
    compatibility = _compatibility(artifact_hash=artifact_hash)
    if isinstance(scores, list):
        scorer: Any = SequenceFakeScorer(scores, compatibility=compatibility)
    else:
        scorer = FakeCoverScorer(scores=scores, compatibility=compatibility)
    policy = CoverPolicyWrapper(
        authority="active",
        execution_context="test",
        candidate_count=candidate_count,
        candidate_generator=generator,
        history_manager=history,
        scorer=scorer,
    )
    return policy, history, scorer, generator


def test_active_returns_highest_finite_scoring_candidate_chunk() -> None:
    candidates = _candidate_batch(3, seed=3)
    policy, history, scorer, _ = _active_policy(
        scores=np.asarray([0.2, 0.9, 0.4], dtype=np.float64),
        candidates=candidates,
        candidate_count=3,
    )

    response = policy.infer(_request(timestep=0))

    assert response["verifier_authority"] == "active"
    assert response["returned_candidate_index"] == 1
    assert response["hypothetical_selected_candidate_index"] is None
    np.testing.assert_array_equal(np.asarray(response["actions"]), candidates[1])
    assert np.asarray(response["actions"]).shape == (10, 7)
    np.testing.assert_array_equal(history.pending_row, history.last_histories[1, 6])
    np.testing.assert_array_equal(scorer.calls[0]["action_histories"].shape, (3, 10, 7))


def test_active_tie_breaks_to_lowest_index() -> None:
    candidates = _candidate_batch(3, seed=7)
    policy, history, _, _ = _active_policy(
        scores=np.asarray([0.5, 0.8, 0.8], dtype=np.float64),
        candidates=candidates,
    )

    response = policy.infer(_request(timestep=0))

    assert response["returned_candidate_index"] == 1
    np.testing.assert_array_equal(np.asarray(response["actions"]), candidates[1])
    np.testing.assert_array_equal(history.pending_row, history.last_histories[1, 6])


def test_active_skips_nonfinite_scores_when_finite_exists() -> None:
    candidates = _candidate_batch(3, seed=9)
    policy, history, _, _ = _active_policy(
        scores=np.asarray([np.nan, 0.3, np.inf], dtype=np.float64),
        candidates=candidates,
    )

    response = policy.infer(_request(timestep=0))

    assert response["returned_candidate_index"] == 1
    np.testing.assert_array_equal(np.asarray(response["actions"]), candidates[1])
    np.testing.assert_array_equal(history.pending_row, history.last_histories[1, 6])
    # Diagnostics must not emit NaN/Inf tokens.
    assert response["verifier_scores"] is None


def test_active_raises_when_no_finite_scores_in_test_context() -> None:
    policy, history, scorer, generator = _active_policy(
        scores=np.asarray([np.nan, np.inf, -np.inf], dtype=np.float64),
        candidate_count=3,
        seed=11,
    )

    with pytest.raises(ValueError, match="no finite verifier scores"):
        policy.infer(_request(timestep=0))

    assert history.pending_row is None
    assert len(generator.calls) == 1
    assert len(scorer.calls) == 1


def test_active_raises_on_malformed_score_vector_in_test_context() -> None:
    class WrongShapeScorer:
        def __init__(self, compatibility: ScorerCompatibility) -> None:
            self.compatibility = compatibility
            self.calls = 0

        def score(self, **kwargs: Any) -> np.ndarray:
            self.calls += 1
            return np.asarray([0.1, 0.2], dtype=np.float64)

    artifact_hash = "a" * 64
    candidates = _candidate_batch(3, seed=13)
    generator = RecordingCandidateGenerator(candidates)
    history = EpisodeHistoryManager(normalization=_normalization(), artifact_identity=artifact_hash)
    scorer = WrongShapeScorer(_compatibility(artifact_hash=artifact_hash))
    policy = CoverPolicyWrapper(
        authority="active",
        execution_context="test",
        candidate_count=3,
        candidate_generator=generator,
        history_manager=history,
        scorer=scorer,
    )

    with pytest.raises(ValueError, match="one score per candidate"):
        policy.infer(_request(timestep=0))

    assert history.pending_row is None
    assert len(generator.calls) == 1
    assert scorer.calls == 1


def test_active_sequential_commit_follows_returned_selection() -> None:
    candidates = _candidate_batch(3, seed=15)
    policy, history, scorer, _ = _active_policy(
        scores=[
            np.asarray([0.1, 0.9, 0.2], dtype=np.float64),
            np.asarray([0.4, 0.3, 0.8], dtype=np.float64),
        ],
        candidates=candidates,
        vary_by_timestep=True,
    )

    first = policy.infer(_request(timestep=0))
    pending = history.pending_row
    assert first["returned_candidate_index"] == 1
    np.testing.assert_array_equal(pending, history.last_histories[1, 6])

    second = policy.infer(_request(timestep=1))
    assert second["state_event"] == "advanced"
    assert second["returned_candidate_index"] == 2
    np.testing.assert_array_equal(history.committed_rows[0], pending)
    # Shared past prefix exposes the previously returned active selection.
    past = scorer.calls[-1]["action_histories"][:, 5]
    for index in range(past.shape[0]):
        np.testing.assert_array_equal(past[index], pending.astype(np.float32))
    np.testing.assert_array_equal(history.pending_row, history.last_histories[2, 6])


def test_active_preserves_candidate_order_and_full_ten_row_response() -> None:
    candidates = _candidate_batch(2, seed=17)
    # Differ only after the first four future rows; CoVer histories use rows 0-3 only.
    candidates[1, :4] = candidates[0, :4]
    candidates[1, 4:] = candidates[0, 4:] + 0.25
    candidates[1, 4:, 6] = np.clip(candidates[1, 4:, 6], 0.0, 1.0)

    original = candidates.copy()
    policy, _, scorer, _ = _active_policy(
        scores=np.asarray([0.1, 0.9], dtype=np.float64),
        candidates=candidates,
        candidate_count=2,
    )

    response = policy.infer(_request(timestep=0))

    np.testing.assert_array_equal(candidates, original)
    np.testing.assert_array_equal(np.asarray(response["actions"]), original[1])
    assert np.asarray(response["actions"]).shape == (10, 7)
    # Histories futures match because only the first four candidate rows matter for scoring.
    np.testing.assert_array_equal(
        scorer.calls[0]["action_histories"][0, 6:10],
        scorer.calls[0]["action_histories"][1, 6:10],
    )
    # But the returned absolute chunk still includes the distinct later rows.
    assert not np.array_equal(original[0, 4:], original[1, 4:])


def test_active_is_deterministic_for_equal_inputs() -> None:
    def run_once() -> tuple[np.ndarray, dict[str, Any]]:
        policy, _, _, _ = _active_policy(
            scores=np.asarray([0.2, 0.7, 0.7], dtype=np.float64),
            candidate_count=3,
            seed=19,
            artifact_hash="f" * 64,
        )
        response = policy.infer(_request(timestep=0))
        return np.asarray(response["actions"]).copy(), {
            "returned_candidate_index": response["returned_candidate_index"],
            "verifier_scores": response["verifier_scores"],
            "state_event": response["state_event"],
        }

    actions_a, diag_a = run_once()
    actions_b, diag_b = run_once()
    np.testing.assert_array_equal(actions_a, actions_b)
    assert diag_a == diag_b
