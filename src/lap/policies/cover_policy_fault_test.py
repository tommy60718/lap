"""W4-05 fault and fallback boundary tests for composed CoverPolicyWrapper."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pytest

from lap.policies.cover_policy_wrapper import CoverPolicyWrapper
from lap.verifiers.cover.action_adapter import ACTION_ORDER
from lap.verifiers.cover.action_adapter import REPRESENTATION_ID
from lap.verifiers.cover.action_adapter import NormalizationArtifact
from lap.verifiers.cover.faults import CoverVerifierError
from lap.verifiers.cover.faults import FallbackReason
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


def _request(*, episode_id: str = "ep-1", timestep: int = 0, prompt: str = "reach the peg") -> dict[str, Any]:
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
                "candidate_count": candidate_count,
                "timestep": observation.get("timestep"),
                "noise": None if noise is None else np.asarray(noise).copy(),
            }
        )
        return self._candidates[:candidate_count].copy()


class FaultingCandidateGenerator:
    def __init__(self, error: Exception) -> None:
        self._error = error
        self.calls = 0

    def generate(self, **kwargs: Any) -> np.ndarray:
        self.calls += 1
        raise self._error


class FaultingScorer:
    is_fake = True

    def __init__(
        self, *, error: Exception | None = None, scores: np.ndarray | None = None, compatibility: ScorerCompatibility
    ) -> None:
        self._error = error
        self._scores = None if scores is None else np.asarray(scores, dtype=np.float64)
        self.compatibility = compatibility
        self.calls = 0

    def score(self, **kwargs: Any) -> np.ndarray:
        self.calls += 1
        if self._error is not None:
            raise self._error
        assert self._scores is not None
        return np.array(self._scores, dtype=np.float64, copy=True)


class RealScorerStub:
    """Non-fake scorer stand-in for deployable robot configuration checks."""

    def __init__(self, *, scores: np.ndarray, compatibility: ScorerCompatibility) -> None:
        self._scores = np.asarray(scores, dtype=np.float64)
        self.compatibility = compatibility

    def score(self, **kwargs: Any) -> np.ndarray:
        return np.array(self._scores, dtype=np.float64, copy=True)


class FaultingHistoryManager(EpisodeHistoryManager):
    def build_histories(self, **kwargs: Any):  # type: ignore[override]
        raise RuntimeError("synthetic history construction failure")


def _policy(
    *,
    authority: str,
    execution_context: str,
    candidate_count: int = 2,
    seed: int = 0,
    scores: np.ndarray | None = None,
    scorer: Any | None = None,
    history: EpisodeHistoryManager | None = None,
    generator: Any | None = None,
    allow_fake_scorer: bool | None = None,
    artifact_hash: str = "a" * 64,
) -> tuple[CoverPolicyWrapper, EpisodeHistoryManager, Any, Any]:
    batch = _candidate_batch(candidate_count, seed=seed)
    gen = generator if generator is not None else RecordingCandidateGenerator(batch)
    hist = history or EpisodeHistoryManager(normalization=_normalization(), artifact_identity=artifact_hash)
    if scorer is None:
        score_vec = np.linspace(0.1, 0.9, candidate_count, dtype=np.float64) if scores is None else scores
        scorer = FakeCoverScorer(scores=score_vec, compatibility=_compatibility(artifact_hash=artifact_hash))
    if allow_fake_scorer is None:
        allow_fake_scorer = execution_context == "robot" and getattr(scorer, "is_fake", False)
    policy = CoverPolicyWrapper(
        authority=authority,  # type: ignore[arg-type]
        execution_context=execution_context,  # type: ignore[arg-type]
        candidate_count=candidate_count,
        candidate_generator=gen,
        history_manager=hist,
        scorer=scorer,
        allow_fake_scorer=allow_fake_scorer,
    )
    return policy, hist, scorer, gen


@pytest.mark.parametrize("authority", ["shadow", "active"])
def test_zero_wrist_raises_in_test_after_candidates(authority: str) -> None:
    policy, history, _, generator = _policy(authority=authority, execution_context="test", seed=11)
    request = _request()
    request["wrist_rgb"] = np.zeros((224, 224, 3), dtype=np.uint8)

    with pytest.raises(CoverVerifierError, match="wrist") as caught:
        policy.infer(request)

    assert caught.value.fallback_reason == FallbackReason.MISSING_OR_ZERO_WRIST_VIEW
    assert len(generator.calls) == 1
    assert history.pending_row is None


@pytest.mark.parametrize("authority", ["shadow", "active"])
def test_zero_wrist_robot_returns_candidate_zero_with_pending(authority: str) -> None:
    candidates = _candidate_batch(2, seed=12)
    policy, history, _, generator = _policy(
        authority=authority,
        execution_context="robot",
        seed=12,
        generator=RecordingCandidateGenerator(candidates),
        scores=np.asarray([0.2, 0.9], dtype=np.float64),
    )
    request = _request()
    request["wrist_rgb"] = np.zeros((224, 224, 3), dtype=np.uint8)

    response = policy.infer(request)

    np.testing.assert_array_equal(response["actions"], candidates[0])
    assert response["returned_candidate_index"] == 0
    assert response["fallback_reason"] == FallbackReason.MISSING_OR_ZERO_WRIST_VIEW
    assert response["verifier_authority"] == authority
    assert response["hypothetical_selected_candidate_index"] is None
    assert history.pending_row is not None
    np.testing.assert_array_equal(history.pending_row, history.last_histories[0, 6])
    assert len(generator.calls) == 1


@pytest.mark.parametrize("authority", ["shadow", "active"])
def test_unavailable_scorer_test_raises_robot_falls_back(authority: str) -> None:
    candidates = _candidate_batch(2, seed=13)
    for execution_context, expect_raise in (("test", True), ("robot", False)):
        history = EpisodeHistoryManager(normalization=_normalization(), artifact_identity="a" * 64)
        generator = RecordingCandidateGenerator(candidates)
        scorer = FaultingScorer(error=RuntimeError("scorer offline"), compatibility=_compatibility())
        policy = CoverPolicyWrapper(
            authority=authority,  # type: ignore[arg-type]
            execution_context=execution_context,  # type: ignore[arg-type]
            candidate_count=2,
            candidate_generator=generator,
            history_manager=history,
            scorer=scorer,
            allow_fake_scorer=True,
        )
        if expect_raise:
            with pytest.raises(CoverVerifierError) as caught:
                policy.infer(_request())
            assert caught.value.fallback_reason == FallbackReason.VERIFIER_UNAVAILABLE
            assert history.pending_row is None
        else:
            response = policy.infer(_request())
            np.testing.assert_array_equal(response["actions"], candidates[0])
            assert response["fallback_reason"] == FallbackReason.VERIFIER_UNAVAILABLE
            np.testing.assert_array_equal(history.pending_row, history.last_histories[0, 6])


@pytest.mark.parametrize("authority", ["shadow", "active"])
def test_malformed_scores_test_raises_robot_falls_back(authority: str) -> None:
    candidates = _candidate_batch(2, seed=14)
    scorer = FaultingScorer(scores=np.asarray([0.1, 0.2, 0.3], dtype=np.float64), compatibility=_compatibility())
    policy_test, history_test, _, _ = _policy(
        authority=authority,
        execution_context="test",
        generator=RecordingCandidateGenerator(candidates),
        scorer=scorer,
    )
    with pytest.raises(CoverVerifierError) as caught:
        policy_test.infer(_request())
    assert caught.value.fallback_reason == FallbackReason.SCORES_INVALID
    assert history_test.pending_row is None

    policy_robot, history_robot, _, _ = _policy(
        authority=authority,
        execution_context="robot",
        generator=RecordingCandidateGenerator(candidates),
        scorer=FaultingScorer(scores=np.asarray([0.1, 0.2, 0.3], dtype=np.float64), compatibility=_compatibility()),
    )
    response = policy_robot.infer(_request())
    np.testing.assert_array_equal(response["actions"], candidates[0])
    assert response["fallback_reason"] == FallbackReason.SCORES_INVALID
    np.testing.assert_array_equal(history_robot.pending_row, history_robot.last_histories[0, 6])


def test_active_no_finite_scores_robot_falls_back_to_candidate_zero() -> None:
    candidates = _candidate_batch(2, seed=15)
    scores = np.asarray([np.nan, np.inf], dtype=np.float64)
    policy, history, _, _ = _policy(
        authority="active",
        execution_context="robot",
        generator=RecordingCandidateGenerator(candidates),
        scores=scores,
    )
    response = policy.infer(_request())
    np.testing.assert_array_equal(response["actions"], candidates[0])
    assert response["fallback_reason"] == FallbackReason.SCORES_INVALID
    np.testing.assert_array_equal(history.pending_row, history.last_histories[0, 6])


@pytest.mark.parametrize("authority", ["shadow", "active"])
def test_no_finite_scores_raises_in_test(authority: str) -> None:
    policy, history, _, _ = _policy(
        authority=authority,
        execution_context="test",
        scores=np.asarray([np.nan, np.nan], dtype=np.float64),
        seed=23,
    )
    with pytest.raises(CoverVerifierError) as caught:
        policy.infer(_request())
    assert caught.value.fallback_reason == FallbackReason.SCORES_INVALID
    assert history.pending_row is None


def test_shadow_no_finite_scores_robot_falls_back() -> None:
    candidates = _candidate_batch(2, seed=24)
    policy, history, _, _ = _policy(
        authority="shadow",
        execution_context="robot",
        generator=RecordingCandidateGenerator(candidates),
        scores=np.asarray([np.nan, -np.inf], dtype=np.float64),
    )
    response = policy.infer(_request())
    np.testing.assert_array_equal(response["actions"], candidates[0])
    assert response["fallback_reason"] == FallbackReason.SCORES_INVALID
    assert response["hypothetical_selected_candidate_index"] is None
    np.testing.assert_array_equal(history.pending_row, history.last_histories[0, 6])


@pytest.mark.parametrize("authority", ["shadow", "active"])
def test_history_invalid_robot_clears_pending_only(authority: str) -> None:
    candidates = _candidate_batch(2, seed=16)
    history = FaultingHistoryManager(normalization=_normalization(), artifact_identity="a" * 64)
    policy, _, _, generator = _policy(
        authority=authority,
        execution_context="robot",
        history=history,
        generator=RecordingCandidateGenerator(candidates),
    )
    response = policy.infer(_request())
    np.testing.assert_array_equal(response["actions"], candidates[0])
    assert response["fallback_reason"] == FallbackReason.HISTORY_INVALID
    assert history.pending_row is None
    assert len(generator.calls) == 1


@pytest.mark.parametrize("authority", ["shadow", "active"])
def test_artifact_mismatch_robot_falls_back_without_pending(authority: str) -> None:
    candidates = _candidate_batch(2, seed=25)
    artifact = "a" * 64
    history = EpisodeHistoryManager(normalization=_normalization(), artifact_identity=artifact)
    scorer = FakeCoverScorer(scores=[0.1, 0.9], compatibility=_compatibility(artifact_hash=artifact))
    policy = CoverPolicyWrapper(
        authority=authority,  # type: ignore[arg-type]
        execution_context="robot",
        candidate_count=2,
        candidate_generator=RecordingCandidateGenerator(candidates),
        history_manager=history,
        scorer=scorer,
        allow_fake_scorer=True,
    )
    first = policy.infer(_request(timestep=0))
    assert first["fallback_reason"] is None
    assert history.pending_row is not None

    # Force provenance mismatch on the next request while rows still exist.
    object.__setattr__(scorer.compatibility, "normalization_artifact_hash", "b" * 64)
    response = policy.infer(_request(timestep=1))
    np.testing.assert_array_equal(response["actions"], candidates[0])
    assert response["fallback_reason"] == FallbackReason.INCOMPATIBLE_VERIFIER_ASSETS
    assert response["state_event"] == "discontinuity_reset"
    assert history.pending_row is None
    assert history.active_episode_id is None

    candidates = _candidate_batch(2, seed=17)
    bad = _compatibility(artifact_hash="b" * 64)
    policy, history, _, _ = _policy(
        authority=authority,
        execution_context="robot",
        artifact_hash="a" * 64,
        generator=RecordingCandidateGenerator(candidates),
        scorer=FakeCoverScorer(scores=[0.1, 0.2], compatibility=bad),
    )
    response = policy.infer(_request())
    np.testing.assert_array_equal(response["actions"], candidates[0])
    assert response["fallback_reason"] == FallbackReason.INCOMPATIBLE_VERIFIER_ASSETS
    np.testing.assert_array_equal(history.pending_row, history.last_histories[0, 6])


@pytest.mark.parametrize("authority", ["shadow", "active"])
def test_duplicate_episode_start_robot_falls_back_without_pending(authority: str) -> None:
    candidates = _candidate_batch(2, seed=26)
    policy, history, _, generator = _policy(
        authority=authority,
        execution_context="robot",
        generator=RecordingCandidateGenerator(candidates),
        scores=np.asarray([0.2, 0.7], dtype=np.float64),
    )
    policy.infer(_request(episode_id="ep-1", timestep=0))
    assert history.pending_row is not None
    calls_after_first = len(generator.calls)

    response = policy.infer(_request(episode_id="ep-1", timestep=0))
    np.testing.assert_array_equal(response["actions"], candidates[0])
    assert response["fallback_reason"] == FallbackReason.STATE_DISCONTINUITY
    assert response["state_event"] == "discontinuity_reset"
    assert history.pending_row is None
    assert history.active_episode_id is None
    assert len(generator.calls) == calls_after_first + 1

    ok = policy.infer(_request(episode_id="ep-2", timestep=0))
    assert ok["fallback_reason"] is None
    assert ok["state_event"] == "new_episode"


@pytest.mark.parametrize("authority", ["shadow", "active"])
def test_missing_metadata_robot_generates_then_falls_back_without_pending(authority: str) -> None:
    candidates = _candidate_batch(2, seed=18)
    policy, history, _, generator = _policy(
        authority=authority,
        execution_context="robot",
        generator=RecordingCandidateGenerator(candidates),
        scores=np.asarray([0.1, 0.9], dtype=np.float64),
    )
    request = _request()
    del request["episode_id"]

    response = policy.infer(request)

    np.testing.assert_array_equal(response["actions"], candidates[0])
    assert response["fallback_reason"] == FallbackReason.MISSING_COVER_METADATA
    assert response["state_event"] == "metadata_fallback"
    assert history.pending_row is None
    assert history.active_episode_id is None
    assert len(generator.calls) == 1


@pytest.mark.parametrize("authority", ["shadow", "active"])
def test_timestep_discontinuity_robot_clears_and_requires_new_episode(authority: str) -> None:
    candidates = _candidate_batch(2, seed=19)
    policy, history, _, generator = _policy(
        authority=authority,
        execution_context="robot",
        generator=RecordingCandidateGenerator(candidates),
        scores=np.asarray([0.4, 0.8], dtype=np.float64),
    )
    policy.infer(_request(timestep=0))
    assert history.pending_row is not None

    response = policy.infer(_request(timestep=3))
    np.testing.assert_array_equal(response["actions"], candidates[0])
    assert response["fallback_reason"] == FallbackReason.STATE_DISCONTINUITY
    assert response["state_event"] == "discontinuity_reset"
    assert history.pending_row is None
    assert history.active_episode_id is None
    assert len(generator.calls) == 2

    # Scoring resumes only after a fresh episode at timestep 0.
    still_blocked = policy.infer(_request(episode_id="ep-2", timestep=1))
    assert still_blocked["fallback_reason"] == FallbackReason.STATE_DISCONTINUITY

    ok = policy.infer(_request(episode_id="ep-2", timestep=0))
    assert ok["fallback_reason"] is None
    assert ok["state_event"] == "new_episode"
    assert history.active_episode_id == "ep-2"


@pytest.mark.parametrize("authority", ["shadow", "active"])
def test_instruction_change_robot_falls_back_without_pending(authority: str) -> None:
    candidates = _candidate_batch(2, seed=20)
    policy, history, _, _ = _policy(
        authority=authority,
        execution_context="robot",
        generator=RecordingCandidateGenerator(candidates),
        scores=np.asarray([0.3, 0.7], dtype=np.float64),
    )
    policy.infer(_request(timestep=0, prompt="reach the peg"))
    response = policy.infer(_request(timestep=1, prompt="open the drawer"))
    assert response["fallback_reason"] == FallbackReason.INSTRUCTION_CHANGED
    assert response["state_event"] == "discontinuity_reset"
    assert history.pending_row is None
    assert history.active_episode_id is None


def test_base_policy_failure_propagates_in_robot_without_fallback() -> None:
    policy, history, _, _ = _policy(
        authority="shadow",
        execution_context="robot",
        generator=FaultingCandidateGenerator(RuntimeError("base policy exploded")),
    )
    with pytest.raises(RuntimeError, match="base policy exploded"):
        policy.infer(_request())
    assert history.pending_row is None


def test_malformed_candidate_batch_propagates_in_robot() -> None:
    class BadShapeGenerator:
        calls = 0

        def generate(self, **kwargs: Any) -> np.ndarray:
            self.calls += 1
            return np.zeros((2, 5, 7), dtype=np.float64)

    policy, _, _, _ = _policy(
        authority="active",
        execution_context="robot",
        generator=BadShapeGenerator(),
    )
    with pytest.raises(ValueError, match="\\[M, 10, 7\\]"):
        policy.infer(_request())


def test_deployable_robot_rejects_fake_scorer_unless_allow_flag() -> None:
    history = EpisodeHistoryManager(normalization=_normalization(), artifact_identity="a" * 64)
    fake = FakeCoverScorer(scores=[0.1, 0.2], compatibility=_compatibility())
    with pytest.raises(ValueError, match="fake scorer"):
        CoverPolicyWrapper(
            authority="shadow",
            execution_context="robot",
            candidate_count=2,
            candidate_generator=RecordingCandidateGenerator(_candidate_batch(2)),
            history_manager=history,
            scorer=fake,
            allow_fake_scorer=False,
        )

    real = RealScorerStub(scores=np.asarray([0.1, 0.2]), compatibility=_compatibility())
    CoverPolicyWrapper(
        authority="shadow",
        execution_context="robot",
        candidate_count=2,
        candidate_generator=RecordingCandidateGenerator(_candidate_batch(2)),
        history_manager=history,
        scorer=real,
        allow_fake_scorer=False,
    )


def test_offline_authority_still_rejected() -> None:
    with pytest.raises(ValueError, match="unknown authority"):
        CoverPolicyWrapper(
            authority="offline",  # type: ignore[arg-type]
            execution_context="test",
            candidate_count=2,
            candidate_generator=RecordingCandidateGenerator(_candidate_batch(2)),
        )


def test_diagnostic_serialization_fault_robot_returns_actions_only(monkeypatch: pytest.MonkeyPatch) -> None:
    candidates = _candidate_batch(2, seed=21)
    policy, _, _, _ = _policy(
        authority="shadow",
        execution_context="robot",
        generator=RecordingCandidateGenerator(candidates),
        scores=np.asarray([0.2, 0.8], dtype=np.float64),
    )

    def boom(*args: Any, **kwargs: Any) -> str:
        raise TypeError("cannot serialize diagnostics")

    monkeypatch.setattr("lap.policies.cover_policy_wrapper.json.dumps", boom)
    response = policy.infer(_request())
    assert set(response.keys()) == {"actions"}
    np.testing.assert_array_equal(response["actions"], candidates[0])
    assert policy.last_diagnostic_fault == FallbackReason.DIAGNOSTIC_SERIALIZATION_FAILED


def test_diagnostic_serialization_fault_raises_in_test(monkeypatch: pytest.MonkeyPatch) -> None:
    policy, _, _, _ = _policy(authority="shadow", execution_context="test", seed=22)

    def boom(*args: Any, **kwargs: Any) -> str:
        raise TypeError("cannot serialize diagnostics")

    monkeypatch.setattr("lap.policies.cover_policy_wrapper.json.dumps", boom)
    with pytest.raises(TypeError, match="cannot serialize"):
        policy.infer(_request())


@pytest.mark.parametrize("authority", ["shadow", "active"])
@pytest.mark.parametrize(
    "bad_scores",
    [
        "not-a-vector",
        ["a", "b"],
        np.array([{"x": 1}, {"y": 2}], dtype=object),
        None,
    ],
)
def test_nonnumeric_scores_raise_in_test(authority: str, bad_scores: Any) -> None:
    class NonNumericScorer:
        is_fake = True

        def __init__(self) -> None:
            self.compatibility = _compatibility()

        def score(self, **kwargs: Any) -> Any:
            return bad_scores

    policy, history, _, generator = _policy(
        authority=authority,
        execution_context="test",
        seed=40,
        scorer=NonNumericScorer(),
    )
    with pytest.raises(CoverVerifierError) as caught:
        policy.infer(_request())
    assert caught.value.fallback_reason == FallbackReason.SCORES_INVALID
    assert history.pending_row is None
    assert len(generator.calls) == 1


@pytest.mark.parametrize("authority", ["shadow", "active"])
@pytest.mark.parametrize(
    "bad_scores",
    [
        ["0.2", "0.8"],  # numeric strings must not coerce into valid scores
        np.asarray([True, False]),
        np.asarray([1 + 2j, 3 + 4j]),
        np.asarray(["0.1", "0.9"], dtype=object),
    ],
)
def test_coercible_but_nonnumeric_raw_dtype_raises_in_test(authority: str, bad_scores: Any) -> None:
    class CoercibleScorer:
        is_fake = True

        def __init__(self) -> None:
            self.compatibility = _compatibility()

        def score(self, **kwargs: Any) -> Any:
            return bad_scores

    policy, history, _, generator = _policy(
        authority=authority,
        execution_context="test",
        seed=44,
        scorer=CoercibleScorer(),
    )
    with pytest.raises(CoverVerifierError) as caught:
        policy.infer(_request())
    assert caught.value.fallback_reason == FallbackReason.SCORES_INVALID
    assert history.pending_row is None
    assert len(generator.calls) == 1


@pytest.mark.parametrize("authority", ["shadow", "active"])
@pytest.mark.parametrize(
    "bad_scores",
    [
        ["0.2", "0.8"],  # numeric strings must not coerce into valid scores
        np.asarray([True, False]),
        np.asarray([1 + 2j, 3 + 4j]),
        np.asarray(["0.1", "0.9"], dtype=object),
    ],
)
def test_coercible_but_nonnumeric_raw_dtype_robot_falls_back_with_pending(authority: str, bad_scores: Any) -> None:
    candidates = _candidate_batch(2, seed=45)

    class CoercibleScorer:
        is_fake = True

        def __init__(self) -> None:
            self.compatibility = _compatibility()

        def score(self, **kwargs: Any) -> Any:
            return bad_scores

    policy, history, _, _ = _policy(
        authority=authority,
        execution_context="robot",
        generator=RecordingCandidateGenerator(candidates),
        scorer=CoercibleScorer(),
    )
    response = policy.infer(_request())
    np.testing.assert_array_equal(response["actions"], candidates[0])
    assert response["fallback_reason"] == FallbackReason.SCORES_INVALID
    assert response["verifier_scores"] is None
    np.testing.assert_array_equal(history.pending_row, history.last_histories[0, 6])


@pytest.mark.parametrize("authority", ["shadow", "active"])
def test_nonnumeric_scores_robot_falls_back_with_pending(authority: str) -> None:
    candidates = _candidate_batch(2, seed=41)

    class NonNumericScorer:
        is_fake = True

        def __init__(self) -> None:
            self.compatibility = _compatibility()

        def score(self, **kwargs: Any) -> Any:
            return ["x", "y"]

    policy, history, _, _ = _policy(
        authority=authority,
        execution_context="robot",
        generator=RecordingCandidateGenerator(candidates),
        scorer=NonNumericScorer(),
    )
    response = policy.infer(_request())
    np.testing.assert_array_equal(response["actions"], candidates[0])
    assert response["fallback_reason"] == FallbackReason.SCORES_INVALID
    np.testing.assert_array_equal(history.pending_row, history.last_histories[0, 6])


@pytest.mark.parametrize("authority", ["shadow", "active"])
def test_partial_nonfinite_scores_select_but_diagnostics_are_null(authority: str) -> None:
    candidates = _candidate_batch(3, seed=42)
    scores = np.asarray([np.nan, 0.4, np.inf], dtype=np.float64)
    policy, history, _, _ = _policy(
        authority=authority,
        execution_context="test",
        candidate_count=3,
        generator=RecordingCandidateGenerator(candidates),
        scores=scores,
    )
    response = policy.infer(_request())
    assert response["fallback_reason"] is None
    assert response["verifier_scores"] is None
    if authority == "shadow":
        assert response["returned_candidate_index"] == 0
        assert response["hypothetical_selected_candidate_index"] == 1
        np.testing.assert_array_equal(response["actions"], candidates[0])
        np.testing.assert_array_equal(history.pending_row, history.last_histories[0, 6])
    else:
        assert response["returned_candidate_index"] == 1
        assert response["hypothetical_selected_candidate_index"] is None
        np.testing.assert_array_equal(response["actions"], candidates[1])
        np.testing.assert_array_equal(history.pending_row, history.last_histories[1, 6])
    # Strict JSON must accept the assembled diagnostics.
    json.dumps(
        {k: v for k, v in response.items() if k != "actions"},
        allow_nan=False,
    )


def test_fully_finite_scores_emit_finite_list() -> None:
    policy, _, _, _ = _policy(
        authority="active",
        execution_context="test",
        candidate_count=2,
        seed=43,
        scores=np.asarray([0.2, 0.8], dtype=np.float64),
    )
    response = policy.infer(_request())
    assert response["verifier_scores"] == [0.2, 0.8]
