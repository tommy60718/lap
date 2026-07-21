"""W4-02 behavior tests for shadow candidates, scoring, and cand-0 parity."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from lap.policies.cover_policy_wrapper import CoverPolicyWrapper
from lap.verifiers.cover.action_adapter import ACTION_ORDER
from lap.verifiers.cover.action_adapter import REPRESENTATION_ID
from lap.verifiers.cover.action_adapter import NormalizationArtifact
from lap.verifiers.cover.action_adapter import build_action_histories
from lap.verifiers.cover.history import EpisodeHistoryManager
from lap.verifiers.cover.scorer import FakeCoverScorer
from lap.verifiers.cover.scorer import ScorerCompatibility


def _finite_chunk(*, seed: int, offset: float = 0.0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    chunk = rng.normal(loc=offset, scale=0.05, size=(10, 7)).astype(np.float64)
    chunk[:, 6] = np.clip(np.linspace(0.1, 0.9, 10) + 0.01 * seed, 0.0, 1.0)
    return chunk


def _candidate_batch(count: int, *, seed: int = 0) -> np.ndarray:
    return np.stack([_finite_chunk(seed=seed + i, offset=0.02 * i) for i in range(count)], axis=0)


def _normalization() -> NormalizationArtifact:
    return NormalizationArtifact(
        q01=np.zeros(6, dtype=np.float64),
        q99=np.ones(6, dtype=np.float64),
    )


def _compatibility(*, artifact_hash: str = "a" * 64) -> ScorerCompatibility:
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


def _shadow_request(*, prompt: str = "reach the peg", episode_id: str = "ep-1", timestep: int = 0) -> dict[str, Any]:
    return {
        "base_rgb": np.zeros((224, 224, 3), dtype=np.uint8),
        "wrist_rgb": np.full((224, 224, 3), 7, dtype=np.uint8),
        "eef_pos": np.asarray([0.1, 0.2, 0.3], dtype=np.float64),
        "eef_rot": np.asarray([0.0, 0.1, -0.1], dtype=np.float64),
        "gripper": np.asarray([0.8], dtype=np.float64),
        "prompt": prompt,
        "episode_id": episode_id,
        "timestep": timestep,
    }


class RecordingCandidateGenerator:
    def __init__(self, candidates: np.ndarray) -> None:
        self._candidates = np.asarray(candidates, dtype=np.float64)
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
                "observation": observation,
                "instruction": instruction,
                "candidate_count": candidate_count,
                "noise": None if noise is None else np.asarray(noise).copy(),
            }
        )
        return self._candidates[:candidate_count].copy()


class NoiseMappedCandidateGenerator:
    """Maps each explicit noise row to a distinct absolute candidate chunk."""

    def __init__(self, catalog: np.ndarray) -> None:
        self._catalog = np.asarray(catalog, dtype=np.float64)
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
                "noise": None if noise is None else np.asarray(noise).copy(),
            }
        )
        if noise is None:
            return self._catalog[:candidate_count].copy()
        noise_arr = np.asarray(noise, dtype=np.float64)
        if noise_arr.ndim == 2:
            noise_arr = noise_arr[None, ...]
        indices = [int(round(float(sample.reshape(-1)[0]))) % len(self._catalog) for sample in noise_arr]
        return self._catalog[indices].copy()


def test_shadow_returns_candidate_zero_with_hypothetical_best() -> None:
    candidates = _candidate_batch(3, seed=20)
    generator = RecordingCandidateGenerator(candidates)
    normalization = _normalization()
    history = EpisodeHistoryManager(normalization=normalization, artifact_identity="b" * 64)
    scorer = FakeCoverScorer(
        scores=np.asarray([0.1, 0.9, 0.5], dtype=np.float64),
        compatibility=_compatibility(artifact_hash="b" * 64),
    )
    policy = CoverPolicyWrapper(
        authority="shadow",
        execution_context="test",
        candidate_count=3,
        candidate_generator=generator,
        history_manager=history,
        scorer=scorer,
    )

    response = policy.infer(_shadow_request())

    np.testing.assert_array_equal(np.asarray(response["actions"]), candidates[0])
    assert response["returned_candidate_index"] == 0
    assert response["hypothetical_selected_candidate_index"] == 1
    assert response["verifier_authority"] == "shadow"
    assert response["candidate_count"] == 3
    assert response["fallback_reason"] is None
    np.testing.assert_allclose(response["verifier_scores"], [0.1, 0.9, 0.5])
    np.testing.assert_array_equal(history.pending_row, history.last_histories[0, 6])


@pytest.mark.parametrize("candidate_count", [1, 2, 4, 5])
def test_shadow_covers_candidate_counts_with_empty_past_histories(candidate_count: int) -> None:
    candidates = _candidate_batch(candidate_count, seed=30)
    generator = RecordingCandidateGenerator(candidates)
    artifact_hash = "c" * 64
    history = EpisodeHistoryManager(normalization=_normalization(), artifact_identity=artifact_hash)
    scores = np.linspace(0.0, 1.0, candidate_count, dtype=np.float64)
    scorer = FakeCoverScorer(scores=scores, compatibility=_compatibility(artifact_hash=artifact_hash))
    policy = CoverPolicyWrapper(
        authority="shadow",
        execution_context="test",
        candidate_count=candidate_count,
        candidate_generator=generator,
        history_manager=history,
        scorer=scorer,
    )
    request = _shadow_request()

    response = policy.infer(request)

    assert len(generator.calls) == 1
    assert generator.calls[0]["candidate_count"] == candidate_count
    assert len(scorer.calls) == 1
    scored_histories = scorer.calls[0]["action_histories"]
    assert scored_histories.shape == (candidate_count, 10, 7)
    assert scored_histories.dtype == np.float32
    assert np.all(scored_histories[:, :6] == -5.0)
    # Shared past prefix across candidates.
    for index in range(1, candidate_count):
        np.testing.assert_array_equal(scored_histories[index, :6], scored_histories[0, :6])
    np.testing.assert_array_equal(np.asarray(response["actions"]), candidates[0])
    assert response["returned_candidate_index"] == 0
    assert response["hypothetical_selected_candidate_index"] == int(np.argmax(scores))


def test_shadow_exact_score_tie_uses_lowest_index_for_hypothetical() -> None:
    candidates = _candidate_batch(3, seed=40)
    artifact_hash = "d" * 64
    history = EpisodeHistoryManager(normalization=_normalization(), artifact_identity=artifact_hash)
    scorer = FakeCoverScorer(
        scores=np.asarray([0.7, 0.7, 0.2], dtype=np.float64),
        compatibility=_compatibility(artifact_hash=artifact_hash),
    )
    policy = CoverPolicyWrapper(
        authority="shadow",
        execution_context="test",
        candidate_count=3,
        candidate_generator=RecordingCandidateGenerator(candidates),
        history_manager=history,
        scorer=scorer,
    )

    response = policy.infer(_shadow_request())

    assert response["returned_candidate_index"] == 0
    assert response["hypothetical_selected_candidate_index"] == 0
    np.testing.assert_array_equal(history.pending_row, history.last_histories[0, 6])
    assert not np.array_equal(history.pending_row, history.last_histories[1, 6])


def test_shadow_does_not_mutate_candidates_or_noise() -> None:
    candidates = _candidate_batch(2, seed=50)
    original = candidates.copy()
    noise = np.arange(2 * 10 * 7, dtype=np.float64).reshape(2, 10, 7)
    noise_original = noise.copy()
    artifact_hash = "e" * 64
    history = EpisodeHistoryManager(normalization=_normalization(), artifact_identity=artifact_hash)
    scorer = FakeCoverScorer(
        scores=np.asarray([0.2, 0.8], dtype=np.float64),
        compatibility=_compatibility(artifact_hash=artifact_hash),
    )
    generator = RecordingCandidateGenerator(candidates)
    policy = CoverPolicyWrapper(
        authority="shadow",
        execution_context="test",
        candidate_count=2,
        candidate_generator=generator,
        history_manager=history,
        scorer=scorer,
    )

    policy.infer(_shadow_request(), noise=noise)

    np.testing.assert_array_equal(candidates, original)
    np.testing.assert_array_equal(noise, noise_original)
    np.testing.assert_array_equal(generator.calls[0]["noise"], noise_original)


def test_incompatible_scorer_metadata_fails_before_scoring_affects_response() -> None:
    candidates = _candidate_batch(2, seed=60)
    history = EpisodeHistoryManager(normalization=_normalization(), artifact_identity="f" * 64)
    bad = _compatibility(artifact_hash="0" * 64)
    scorer = FakeCoverScorer(scores=np.asarray([0.1, 0.9], dtype=np.float64), compatibility=bad)
    policy = CoverPolicyWrapper(
        authority="shadow",
        execution_context="test",
        candidate_count=2,
        candidate_generator=RecordingCandidateGenerator(candidates),
        history_manager=history,
        scorer=scorer,
    )

    with pytest.raises(ValueError, match="normalization_artifact_hash"):
        policy.infer(_shadow_request())
    assert scorer.calls == []
    assert history.pending_row is None


def test_candidate_zero_parity_with_explicit_first_noise_sample() -> None:
    catalog = _candidate_batch(4, seed=70)
    artifact_hash = "1" * 64
    generator = NoiseMappedCandidateGenerator(catalog)
    history = EpisodeHistoryManager(normalization=_normalization(), artifact_identity=artifact_hash)
    scorer = FakeCoverScorer(
        scores=np.asarray([0.1, 0.2, 0.3, 0.4], dtype=np.float64),
        compatibility=_compatibility(artifact_hash=artifact_hash),
    )
    policy = CoverPolicyWrapper(
        authority="shadow",
        execution_context="test",
        candidate_count=4,
        candidate_generator=generator,
        history_manager=history,
        scorer=scorer,
    )
    request = _shadow_request()
    noise = np.zeros((4, 10, 7), dtype=np.float64)
    noise[:, 0, 0] = [0, 1, 2, 3]

    plain = NoiseMappedCandidateGenerator(catalog).generate(
        observation=request,
        instruction=request["prompt"],
        candidate_count=1,
        noise=noise[0:1],
    )[0]
    response = policy.infer(request, noise=noise)

    np.testing.assert_array_equal(np.asarray(response["actions"]), plain)
    np.testing.assert_array_equal(np.asarray(response["actions"]), catalog[0])


def test_reordered_explicit_noise_reorders_candidates() -> None:
    catalog = _candidate_batch(3, seed=80)
    artifact_hash = "2" * 64
    generator = NoiseMappedCandidateGenerator(catalog)
    history = EpisodeHistoryManager(normalization=_normalization(), artifact_identity=artifact_hash)
    scorer = FakeCoverScorer(
        scores=np.asarray([0.5, 0.4, 0.3], dtype=np.float64),
        compatibility=_compatibility(artifact_hash=artifact_hash),
    )
    policy = CoverPolicyWrapper(
        authority="shadow",
        execution_context="test",
        candidate_count=3,
        candidate_generator=generator,
        history_manager=history,
        scorer=scorer,
    )
    noise = np.zeros((3, 10, 7), dtype=np.float64)
    noise[:, 0, 0] = [2, 0, 1]

    response = policy.infer(_shadow_request(), noise=noise)

    np.testing.assert_array_equal(generator.calls[0]["noise"][:, 0, 0], [2, 0, 1])
    np.testing.assert_array_equal(np.asarray(response["actions"]), catalog[2])
    # Scorer saw histories built from the reordered candidate absolute chunks.
    expected = build_action_histories(
        reference_position=np.asarray([0.1, 0.2, 0.3], dtype=np.float64),
        reference_rotation_vector=np.asarray([0.0, 0.1, -0.1], dtype=np.float64),
        candidate_chunks=catalog[[2, 0, 1]],
        processed_past=np.empty((0, 7), dtype=np.float64),
        normalization=_normalization(),
    ).histories
    np.testing.assert_array_equal(scorer.calls[0]["action_histories"], expected)


def test_shadow_startup_requires_history_and_scorer() -> None:
    with pytest.raises(ValueError, match="history_manager"):
        CoverPolicyWrapper(
            authority="shadow",
            execution_context="test",
            candidate_count=2,
            candidate_generator=RecordingCandidateGenerator(_candidate_batch(2)),
            scorer=FakeCoverScorer(scores=[0.1, 0.2], compatibility=_compatibility()),
        )
    with pytest.raises(ValueError, match="scorer"):
        CoverPolicyWrapper(
            authority="shadow",
            execution_context="test",
            candidate_count=2,
            candidate_generator=RecordingCandidateGenerator(_candidate_batch(2)),
            history_manager=EpisodeHistoryManager(normalization=_normalization(), artifact_identity="a" * 64),
        )


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("views", ("base_rgb",), "views"),
        ("history_length", 8, "history_length"),
        ("action_dimension", 8, "action_dimension"),
        ("action_order", ("dx", "dy", "dz", "rx", "ry", "rz", "gripper"), "action_order"),
        ("representation_id", "wrong", "representation_id"),
        ("input_dtype", "float64", "input_dtype"),
        ("output_shape_rank", 2, "output_shape_rank"),
    ],
)
def test_scorer_compatibility_rejects_contract_mismatches(field: str, value: Any, match: str) -> None:
    kwargs = _compatibility(artifact_hash="3" * 64).__dict__.copy()
    kwargs[field] = value
    compatibility = ScorerCompatibility(**kwargs)
    history = EpisodeHistoryManager(normalization=_normalization(), artifact_identity="3" * 64)
    policy = CoverPolicyWrapper(
        authority="shadow",
        execution_context="test",
        candidate_count=2,
        candidate_generator=RecordingCandidateGenerator(_candidate_batch(2, seed=90)),
        history_manager=history,
        scorer=FakeCoverScorer(scores=[0.1, 0.2], compatibility=compatibility),
    )
    with pytest.raises(ValueError, match=match):
        policy.infer(_shadow_request())
