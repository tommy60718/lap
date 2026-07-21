"""W4-03 behavior tests for sequential episode history commit."""

from __future__ import annotations

import json
from pathlib import Path
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

CONSUMER_FIXTURE = (
    Path(__file__).resolve().parents[1]
    / "verifiers"
    / "cover"
    / "testdata"
    / "ur5e_cover_action_adapter_v1.json"
)


def _finite_chunk(*, seed: int, offset: float = 0.0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    chunk = rng.normal(loc=offset, scale=0.04, size=(10, 7)).astype(np.float64)
    chunk[:, 6] = np.clip(np.linspace(0.15, 0.85, 10) + 0.01 * (seed % 7), 0.0, 1.0)
    return chunk


def _candidate_batch(count: int, *, seed: int) -> np.ndarray:
    return np.stack([_finite_chunk(seed=seed + i, offset=0.03 * i) for i in range(count)], axis=0)


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


def _request(
    *,
    prompt: str = "reach the peg",
    episode_id: str = "ep-1",
    timestep: int = 0,
    eef_pos: np.ndarray | None = None,
    eef_rot: np.ndarray | None = None,
) -> dict[str, Any]:
    return {
        "base_rgb": np.zeros((224, 224, 3), dtype=np.uint8),
        "wrist_rgb": np.full((224, 224, 3), 3, dtype=np.uint8),
        "eef_pos": np.asarray([0.1, 0.2, 0.3] if eef_pos is None else eef_pos, dtype=np.float64),
        "eef_rot": np.asarray([0.0, 0.1, -0.1] if eef_rot is None else eef_rot, dtype=np.float64),
        "gripper": np.asarray([0.8], dtype=np.float64),
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
        self.calls.append(
            {
                "instruction": instruction,
                "candidate_count": candidate_count,
                "timestep": observation.get("timestep"),
            }
        )
        batch = self._candidates[:candidate_count].copy()
        if self._vary_by_timestep:
            timestep = int(observation.get("timestep", 0))
            batch = batch + np.asarray([0.01 * timestep, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
            batch[..., 6] = np.clip(batch[..., 6], 0.0, 1.0)
        return batch


class FixturePastHistoryManager(EpisodeHistoryManager):
    """Reinstalls fixture past after timestep-0 prepare clears state."""

    def __init__(self, *, fixture_past: np.ndarray, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._fixture_past = np.asarray(fixture_past, dtype=np.float64)

    def prepare_request(self, **kwargs: Any) -> str:
        event = super().prepare_request(**kwargs)
        if kwargs.get("timestep") == 0:
            self.install_committed_past(self._fixture_past)
        return event


def _shadow_policy(
    *,
    candidate_count: int = 2,
    seed: int = 0,
    artifact_hash: str = "a" * 64,
    scores: np.ndarray | None = None,
    history: EpisodeHistoryManager | None = None,
    candidates: np.ndarray | None = None,
    vary_by_timestep: bool = False,
) -> tuple[CoverPolicyWrapper, EpisodeHistoryManager, FakeCoverScorer, RecordingCandidateGenerator]:
    batch = _candidate_batch(candidate_count, seed=seed) if candidates is None else candidates
    generator = RecordingCandidateGenerator(batch, vary_by_timestep=vary_by_timestep)
    history_manager = history or EpisodeHistoryManager(
        normalization=_normalization(),
        artifact_identity=artifact_hash,
    )
    if scores is None:
        scores = np.linspace(0.1, 0.9, candidate_count, dtype=np.float64)
    scorer = FakeCoverScorer(scores=scores, compatibility=_compatibility(artifact_hash=artifact_hash))
    policy = CoverPolicyWrapper(
        authority="shadow",
        execution_context="test",
        candidate_count=candidate_count,
        candidate_generator=generator,
        history_manager=history_manager,
        scorer=scorer,
    )
    return policy, history_manager, scorer, generator


def _snapshot_state(history: EpisodeHistoryManager) -> dict[str, Any]:
    return {
        "episode": history.active_episode_id,
        "instruction": history.fixed_instruction,
        "timestep": history.last_accepted_timestep,
        "committed": None if history.committed_rows.size == 0 else history.committed_rows.copy(),
        "pending": None if history.pending_row is None else history.pending_row.copy(),
    }


def test_sequential_shadow_commits_candidate_zero_pending_not_hypothetical() -> None:
    policy, history, scorer, _ = _shadow_policy(
        candidate_count=2,
        seed=11,
        scores=np.asarray([0.1, 0.9], dtype=np.float64),
    )

    first = policy.infer(_request(timestep=0))
    pending_after_first = history.pending_row
    assert pending_after_first is not None
    np.testing.assert_array_equal(pending_after_first, history.last_histories[0, 6])
    assert not np.array_equal(pending_after_first, history.last_histories[1, 6])
    assert history.committed_rows.shape == (0, 7)
    assert first["hypothetical_selected_candidate_index"] == 1
    assert first["returned_candidate_index"] == 0
    assert history.active_episode_id == "ep-1"
    assert history.fixed_instruction == "reach the peg"
    assert history.last_accepted_timestep == 0
    assert history.artifact_identity == "a" * 64

    second = policy.infer(_request(timestep=1))
    assert second["state_event"] == "advanced"
    assert history.committed_rows.shape == (1, 7)
    np.testing.assert_array_equal(history.committed_rows[0], pending_after_first)
    past = scorer.calls[-1]["action_histories"][:, 5]
    np.testing.assert_array_equal(past[0], pending_after_first.astype(np.float32))
    np.testing.assert_array_equal(past[1], pending_after_first.astype(np.float32))


def test_sequential_requests_cover_past_lengths_zero_through_six() -> None:
    policy, history, scorer, _ = _shadow_policy(candidate_count=2, seed=21, vary_by_timestep=True)
    first_committed_row: np.ndarray | None = None

    for timestep in range(0, 8):
        response = policy.infer(_request(timestep=timestep))
        assert response["returned_candidate_index"] == 0
        if timestep == 0:
            assert history.committed_rows.shape == (0, 7)
            assert response["state_event"] == "new_episode"
        else:
            assert response["state_event"] == "advanced"
            expected_len = min(timestep, 6)
            assert history.committed_rows.shape == (expected_len, 7)
            if timestep == 1:
                first_committed_row = history.committed_rows[0].copy()

    assert history.committed_rows.shape == (6, 7)
    assert first_committed_row is not None
    assert not np.array_equal(history.committed_rows[0], first_committed_row)
    histories = scorer.calls[-1]["action_histories"]
    for index in range(1, histories.shape[0]):
        np.testing.assert_array_equal(histories[index, :6], histories[0, :6])


def test_duplicate_skipped_and_stale_timesteps_raise_without_mutation() -> None:
    policy, history, _, generator = _shadow_policy(candidate_count=2, seed=31)
    policy.infer(_request(timestep=0))
    policy.infer(_request(timestep=1))
    before = _snapshot_state(history)
    calls_before = len(generator.calls)

    with pytest.raises(ValueError, match="duplicate timestep"):
        policy.infer(_request(timestep=1))
    assert _snapshot_state(history)["timestep"] == before["timestep"]
    np.testing.assert_array_equal(_snapshot_state(history)["committed"], before["committed"])
    assert len(generator.calls) == calls_before

    with pytest.raises(ValueError, match="skipped timestep"):
        policy.infer(_request(timestep=3))
    assert _snapshot_state(history)["timestep"] == before["timestep"]
    assert len(generator.calls) == calls_before

    policy.infer(_request(timestep=2))
    after = _snapshot_state(history)
    calls_after = len(generator.calls)
    with pytest.raises(ValueError, match="stale timestep"):
        policy.infer(_request(timestep=1))
    assert _snapshot_state(history)["timestep"] == after["timestep"]
    np.testing.assert_array_equal(_snapshot_state(history)["committed"], after["committed"])
    assert len(generator.calls) == calls_after

    with pytest.raises(ValueError, match="nonnegative"):
        policy.infer(_request(timestep=-1))
    assert len(generator.calls) == calls_after


def test_instruction_change_raises_before_generation() -> None:
    policy, history, _, generator = _shadow_policy(candidate_count=2, seed=41)
    policy.infer(_request(timestep=0, prompt="reach the peg"))
    before = _snapshot_state(history)
    calls_before = len(generator.calls)

    with pytest.raises(ValueError, match="instruction changed"):
        policy.infer(_request(timestep=1, prompt="insert the peg"))

    assert _snapshot_state(history)["episode"] == before["episode"]
    assert _snapshot_state(history)["instruction"] == before["instruction"]
    assert _snapshot_state(history)["timestep"] == before["timestep"]
    assert len(generator.calls) == calls_before


def test_episode_id_change_at_zero_resets_state() -> None:
    policy, history, _, _ = _shadow_policy(candidate_count=2, seed=51)
    policy.infer(_request(episode_id="ep-a", timestep=0))
    policy.infer(_request(episode_id="ep-a", timestep=1))
    assert history.committed_rows.shape == (1, 7)

    response = policy.infer(_request(episode_id="ep-b", timestep=0))
    assert response["state_event"] == "new_episode"
    assert history.active_episode_id == "ep-b"
    assert history.committed_rows.shape == (0, 7)
    assert history.pending_row is not None


def test_episode_id_change_at_nonzero_is_discontinuity() -> None:
    policy, history, _, generator = _shadow_policy(candidate_count=2, seed=61)
    policy.infer(_request(episode_id="ep-a", timestep=0))
    before = _snapshot_state(history)
    calls_before = len(generator.calls)

    with pytest.raises(ValueError, match="episode_id changed"):
        policy.infer(_request(episode_id="ep-b", timestep=1))

    assert _snapshot_state(history)["episode"] == before["episode"]
    assert _snapshot_state(history)["timestep"] == before["timestep"]
    assert len(generator.calls) == calls_before


def test_missing_cover_metadata_raises_in_test_shadow() -> None:
    policy, _, _, generator = _shadow_policy(candidate_count=2, seed=71)
    request = _request(timestep=0)
    del request["episode_id"]
    with pytest.raises(ValueError, match="episode_id and timestep"):
        policy.infer(request)
    assert generator.calls == []


def test_artifact_identity_mismatch_raises_and_preserves_rows() -> None:
    policy, history, _, generator = _shadow_policy(
        candidate_count=2,
        seed=81,
        artifact_hash="b" * 64,
    )
    policy.infer(_request(timestep=0))
    policy.infer(_request(timestep=1))
    before = _snapshot_state(history)
    calls_before = len(generator.calls)

    with pytest.raises(ValueError, match="normalization artifact identity mismatch"):
        history.prepare_request(
            episode_id="ep-1",
            timestep=2,
            instruction="reach the peg",
            expected_artifact_identity="c" * 64,
        )

    assert _snapshot_state(history)["timestep"] == before["timestep"]
    np.testing.assert_array_equal(_snapshot_state(history)["committed"], before["committed"])
    assert len(generator.calls) == calls_before


def test_equal_sequences_are_byte_identical() -> None:
    def run_sequence() -> tuple[list[np.ndarray], list[dict[str, Any]]]:
        policy, _, scorer, _ = _shadow_policy(candidate_count=2, seed=91, artifact_hash="d" * 64)
        diagnostics = []
        histories = []
        for timestep in range(3):
            response = policy.infer(_request(timestep=timestep))
            diagnostics.append(
                {
                    "state_event": response["state_event"],
                    "returned_candidate_index": response["returned_candidate_index"],
                    "hypothetical_selected_candidate_index": response["hypothetical_selected_candidate_index"],
                    "verifier_scores": response["verifier_scores"],
                }
            )
            histories.append(scorer.calls[-1]["action_histories"].copy())
        return histories, diagnostics

    histories_a, diagnostics_a = run_sequence()
    histories_b, diagnostics_b = run_sequence()
    assert diagnostics_a == diagnostics_b
    for left, right in zip(histories_a, histories_b, strict=True):
        np.testing.assert_array_equal(left, right)


def test_composed_runtime_reproduces_frozen_lap3_consumer_fixture() -> None:
    fixture = json.loads(CONSUMER_FIXTURE.read_text(encoding="utf-8"))
    payload = fixture["input"]
    expected = np.asarray(fixture["expected"]["histories"], dtype=np.float32)
    artifact_hash = "e" * 64
    raw_candidates = np.asarray(payload["candidate_chunks"], dtype=np.float64)
    # Fixture chunks are [M,5,7]; runtime contract is [M,10,7]. Pad unused tail rows.
    candidates = np.zeros((raw_candidates.shape[0], 10, 7), dtype=np.float64)
    candidates[:, : raw_candidates.shape[1], :] = raw_candidates
    candidates[:, raw_candidates.shape[1] :, 6] = raw_candidates[:, -1, 6:7]
    past = np.asarray(payload["processed_past"], dtype=np.float64)
    history = FixturePastHistoryManager(
        fixture_past=past,
        normalization=NormalizationArtifact(
            q01=np.asarray(payload["q01"], dtype=np.float64),
            q99=np.asarray(payload["q99"], dtype=np.float64),
        ),
        artifact_identity=artifact_hash,
    )
    policy, _, scorer, _ = _shadow_policy(
        candidate_count=2,
        artifact_hash=artifact_hash,
        history=history,
        candidates=candidates,
        scores=np.asarray([0.2, 0.8], dtype=np.float64),
    )
    request = _request(
        episode_id="fixture",
        timestep=0,
        eef_pos=np.asarray(payload["reference_position"], dtype=np.float64),
        eef_rot=np.asarray(payload["reference_rotation_vector"], dtype=np.float64),
    )

    policy.infer(request)

    np.testing.assert_array_equal(scorer.calls[-1]["action_histories"], expected)
    assert history.last_histories.dtype == np.float32
    np.testing.assert_array_equal(history.pending_row, expected[0, 6])


def test_clear_discards_state_like_restart() -> None:
    policy, history, _, _ = _shadow_policy(candidate_count=2, seed=101)
    policy.infer(_request(timestep=0))
    policy.infer(_request(timestep=1))
    history.clear()
    assert history.active_episode_id is None
    assert history.fixed_instruction is None
    assert history.last_accepted_timestep is None
    assert history.committed_rows.shape == (0, 7)
    assert history.pending_row is None

    response = policy.infer(_request(timestep=0))
    assert response["state_event"] == "new_episode"
    assert history.committed_rows.shape == (0, 7)
