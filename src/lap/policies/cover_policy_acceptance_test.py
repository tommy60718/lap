"""W4-07 fake-backed acceptance matrix across authority and context."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from lap.policies.cover_policy_wrapper import CoverPolicyWrapper
from lap.verifiers.cover.action_adapter import ACTION_ORDER
from lap.verifiers.cover.action_adapter import REPRESENTATION_ID
from lap.verifiers.cover.action_adapter import NormalizationArtifact
from lap.verifiers.cover.faults import FallbackReason
from lap.verifiers.cover.history import EpisodeHistoryManager
from lap.verifiers.cover.scorer import FakeCoverScorer
from lap.verifiers.cover.scorer import ScorerCompatibility
from lap.verifiers.selection import highest_finite_score_index


def _finite_chunk(*, seed: int, offset: float = 0.0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    chunk = rng.normal(loc=offset, scale=0.04, size=(10, 7)).astype(np.float64)
    chunk[:, 6] = np.clip(np.linspace(0.15, 0.85, 10) + 0.01 * seed, 0.0, 1.0)
    return chunk


def _candidate_batch(count: int, *, seed: int) -> np.ndarray:
    return np.stack([_finite_chunk(seed=seed + i, offset=0.03 * i) for i in range(count)], axis=0)


def _normalization() -> NormalizationArtifact:
    return NormalizationArtifact(q01=np.zeros(6, dtype=np.float64), q99=np.ones(6, dtype=np.float64))


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


def _request(*, episode_id: str = "ep-acc", timestep: int = 0) -> dict[str, Any]:
    return {
        "base_rgb": np.zeros((224, 224, 3), dtype=np.uint8),
        "wrist_rgb": np.full((224, 224, 3), 5, dtype=np.uint8),
        "eef_pos": np.asarray([0.11, 0.22, 0.33], dtype=np.float64),
        "eef_rot": np.asarray([0.01, -0.02, 0.03], dtype=np.float64),
        "gripper": np.asarray([0.6], dtype=np.float64),
        "prompt": "reach the peg",
        "episode_id": episode_id,
        "timestep": timestep,
    }


class RecordingCandidateGenerator:
    def __init__(self, candidates: np.ndarray) -> None:
        self._candidates = np.asarray(candidates, dtype=np.float64)
        self.calls: list[dict[str, Any]] = []

    def generate(self, **kwargs: Any) -> np.ndarray:
        self.calls.append({"candidate_count": kwargs["candidate_count"]})
        return self._candidates[: kwargs["candidate_count"]].copy()


class RecordingHistoryManager:
    def __init__(self) -> None:
        self.clear_calls = 0
        self.build_calls = 0

    def clear(self) -> None:
        self.clear_calls += 1

    def clear_pending(self) -> None:
        return None

    def build_histories(self, **kwargs: Any) -> Any:
        self.build_calls += 1
        raise AssertionError("disabled authority must not build histories")


class RecordingScorer:
    def __init__(self) -> None:
        self.score_calls = 0
        self.compatibility = _compatibility()

    def score(self, **kwargs: Any) -> Any:
        self.score_calls += 1
        raise AssertionError("disabled authority must not score")


def _policy(
    *,
    authority: str,
    execution_context: str,
    candidate_count: int,
    seed: int,
) -> tuple[CoverPolicyWrapper, Any, Any, RecordingCandidateGenerator]:
    batch = _candidate_batch(candidate_count, seed=seed)
    generator = RecordingCandidateGenerator(batch)
    if authority == "disabled":
        history: Any = RecordingHistoryManager()
        scorer: Any = RecordingScorer()
        policy = CoverPolicyWrapper(
            authority="disabled",
            execution_context=execution_context,  # type: ignore[arg-type]
            candidate_count=candidate_count,
            candidate_generator=generator,
            history_manager=history,
            scorer=scorer,
        )
        return policy, history, scorer, generator

    artifact = "a" * 64
    history = EpisodeHistoryManager(normalization=_normalization(), artifact_identity=artifact)
    scores = np.linspace(0.1, 0.9, candidate_count, dtype=np.float64)
    scorer = FakeCoverScorer(scores=scores, compatibility=_compatibility(artifact_hash=artifact))
    policy = CoverPolicyWrapper(
        authority=authority,  # type: ignore[arg-type]
        execution_context=execution_context,  # type: ignore[arg-type]
        candidate_count=candidate_count,
        candidate_generator=generator,
        history_manager=history,
        scorer=scorer,
        allow_fake_scorer=execution_context == "robot",
    )
    return policy, history, scorer, generator


@pytest.mark.parametrize("authority", ["disabled", "shadow", "active"])
@pytest.mark.parametrize("execution_context", ["test", "robot"])
@pytest.mark.parametrize("candidate_count", [1, 2, 4, 5])
def test_authority_context_candidate_matrix(
    authority: str,
    execution_context: str,
    candidate_count: int,
) -> None:
    policy, history, scorer, generator = _policy(
        authority=authority,
        execution_context=execution_context,
        candidate_count=candidate_count,
        seed=1000 + candidate_count,
    )
    request = _request()
    if authority == "disabled":
        request = {k: v for k, v in request.items() if k not in ("episode_id", "timestep")}

    response = policy.infer(request)
    actions = np.asarray(response["actions"], dtype=np.float64)
    assert actions.shape == (10, 7)
    assert np.isfinite(actions).all()
    assert response["verifier_authority"] == authority
    assert response["execution_context"] == execution_context
    assert response["fallback_reason"] is None

    if authority == "disabled":
        assert generator.calls[0]["candidate_count"] == 1
        assert response["candidate_count"] == 1
        assert response["returned_candidate_index"] == 0
        assert response["hypothetical_selected_candidate_index"] is None
        assert history.build_calls == 0
        assert scorer.score_calls == 0
        return

    assert generator.calls[0]["candidate_count"] == candidate_count
    assert response["candidate_count"] == candidate_count
    expected_selected = highest_finite_score_index(np.linspace(0.1, 0.9, candidate_count))
    assert expected_selected is not None
    if authority == "shadow":
        assert response["returned_candidate_index"] == 0
        assert response["hypothetical_selected_candidate_index"] == expected_selected
        np.testing.assert_array_equal(actions, generator._candidates[0])
        np.testing.assert_array_equal(history.pending_row, history.last_histories[0, 6])
    else:
        assert response["returned_candidate_index"] == expected_selected
        assert response["hypothetical_selected_candidate_index"] is None
        np.testing.assert_array_equal(actions, generator._candidates[expected_selected])
        np.testing.assert_array_equal(
            history.pending_row,
            history.last_histories[expected_selected, 6],
        )


def test_fallback_reason_codes_are_stable_external_contracts() -> None:
    codes = {
        FallbackReason.MISSING_COVER_METADATA,
        FallbackReason.STATE_DISCONTINUITY,
        FallbackReason.INSTRUCTION_CHANGED,
        FallbackReason.MISSING_OR_ZERO_WRIST_VIEW,
        FallbackReason.INCOMPATIBLE_VERIFIER_ASSETS,
        FallbackReason.VERIFIER_UNAVAILABLE,
        FallbackReason.HISTORY_INVALID,
        FallbackReason.SCORES_INVALID,
        FallbackReason.DIAGNOSTIC_SERIALIZATION_FAILED,
    }
    assert codes == {
        "missing_cover_metadata",
        "state_discontinuity",
        "instruction_changed",
        "missing_or_zero_wrist_view",
        "incompatible_verifier_assets",
        "verifier_unavailable",
        "history_invalid",
        "scores_invalid",
        "diagnostic_serialization_failed",
    }


def test_deployable_robot_rejects_fake_scorer_at_startup() -> None:
    with pytest.raises(ValueError, match="fake scorer"):
        CoverPolicyWrapper(
            authority="active",
            execution_context="robot",
            candidate_count=2,
            candidate_generator=RecordingCandidateGenerator(_candidate_batch(2, seed=7)),
            history_manager=EpisodeHistoryManager(
                normalization=_normalization(),
                artifact_identity="a" * 64,
            ),
            scorer=FakeCoverScorer(scores=[0.1, 0.2], compatibility=_compatibility()),
            allow_fake_scorer=False,
        )


def test_active_lowest_index_tie_break() -> None:
    candidates = _candidate_batch(4, seed=42)
    scores = np.asarray([0.5, 0.9, 0.9, 0.1], dtype=np.float64)
    history = EpisodeHistoryManager(normalization=_normalization(), artifact_identity="a" * 64)
    policy = CoverPolicyWrapper(
        authority="active",
        execution_context="test",
        candidate_count=4,
        candidate_generator=RecordingCandidateGenerator(candidates),
        history_manager=history,
        scorer=FakeCoverScorer(scores=scores, compatibility=_compatibility()),
    )
    response = policy.infer(_request())
    assert response["returned_candidate_index"] == 1
    np.testing.assert_array_equal(response["actions"], candidates[1])
