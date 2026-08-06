"""W7-01 public seam: production WS-1 robot shadow serve → candidate-zero response."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from lap.policies.cover_policy_wrapper import CoverPolicyWrapper
from lap.policies.w5_real_cover_shadow import build_real_cover_shadow_policy_from_packages
from lap.policies.w5_real_cover_shadow import load_w2_normalization_artifact
from lap.policies.w5_real_cover_shadow import recorded_two_view_request
from lap.policies.w7_real_cover_robot_shadow import W7_CANDIDATE_COUNT
from lap.policies.w7_real_cover_robot_shadow import W7_SERVER_TYPE
from lap.policies.w7_real_cover_robot_shadow import build_production_real_cover_robot_shadow_policy
from lap.policies.w7_real_cover_robot_shadow import build_production_real_cover_robot_shadow_policy_from_packages
from lap.policies.w7_real_cover_robot_shadow import production_server_metadata
from lap.verifiers.cover.accepted_w3_scorer import load_accepted_w3_deployment_scorer
from lap.verifiers.cover.action_adapter import ACTION_ORDER
from lap.verifiers.cover.action_adapter import REPRESENTATION_ID
from lap.verifiers.cover.faults import FallbackReason
from lap.verifiers.cover.history import EpisodeHistoryManager
from lap.verifiers.cover.scorer import FakeCoverScorer
from lap.verifiers.cover.scorer import ScorerCompatibility
from lap.verifiers.pi05_horizon import EXPOSED_HORIZON
from lap.verifiers.pi05_horizon import FULL_HORIZON
from lap.verifiers.pi05_horizon import LOCAL_RUNTIME_MANIFEST_SHA256
from lap.verifiers.pi05_horizon import SELECTED_CONFIG
from lap.verifiers.pi05_horizon import Pi05HorizonCandidateGenerator
from lap.verifiers.pi05_horizon import require_pi05_runtime_identity

_REPO_ROOT = Path(__file__).resolve().parents[3]
_W3_PACKAGE = _REPO_ROOT / "artifacts/w3/canonical_acceptance_v1"
_W2_NORM = Path("/home/yangsen/osx_ur/catkin_ws/src/osx_vla/.w2_canonical_export_v3/normalization_artifact.json")
_PI05_CHECKPOINT = Path("/home/yangsen/checkpoints_datasets/checkpoints_dual_verifier/peg_in_hole_lora_v2/10000")


class _ScriptedPi05Policy:
    def __init__(self, catalog: np.ndarray) -> None:
        self._catalog = np.asarray(catalog, dtype=np.float64)

    def infer(self, obs: dict[str, Any], *, noise: np.ndarray | None = None) -> dict[str, Any]:
        index = 0
        if noise is not None:
            noise_array = np.asarray(noise)
            index = int(noise_array.reshape(-1)[0]) if noise_array.size else 0
        del obs
        return {"actions": self._catalog[index].copy()}


def _full_horizon_catalog(count: int = 2, *, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    chunks = []
    for offset in range(count):
        chunk = rng.normal(loc=0.05 * offset, scale=0.02, size=(FULL_HORIZON, 7)).astype(np.float64)
        chunk[:, 6] = np.clip(np.linspace(0.2, 0.8, FULL_HORIZON) + 0.01 * offset, 0.0, 1.0)
        chunks.append(chunk)
    return np.stack(chunks, axis=0)


def _request(**overrides: Any) -> dict[str, Any]:
    payload = recorded_two_view_request(
        base_rgb=np.full((224, 224, 3), 40, dtype=np.uint8),
        wrist_rgb=np.full((224, 224, 3), 80, dtype=np.uint8),
        eef_pos=np.asarray([0.4, 0.1, 0.2], dtype=np.float64),
        eef_rot=np.asarray([0.0, 0.1, -0.1], dtype=np.float64),
        gripper=np.asarray([0.5], dtype=np.float64),
        prompt="insert the circular peg",
        episode_id="w7-ep-0",
        timestep=0,
    )
    payload.update(overrides)
    return payload


@pytest.fixture(scope="module")
def w3_scorer_and_norm():
    if not _W3_PACKAGE.is_dir() or not _W2_NORM.is_file():
        pytest.skip("accepted W3 package or W2 normalization artifact unavailable")
    normalization, identity = load_w2_normalization_artifact(_W2_NORM)
    scorer = load_accepted_w3_deployment_scorer(
        _W3_PACKAGE,
        expected_normalization_hash=identity,
    )
    return scorer, normalization, identity


def test_production_robot_shadow_rejects_fake_scorer(w3_scorer_and_norm) -> None:
    _, normalization, identity = w3_scorer_and_norm
    catalog = _full_horizon_catalog(2, seed=7)
    generator = Pi05HorizonCandidateGenerator(
        policy=_ScriptedPi05Policy(catalog),
        content_identity=LOCAL_RUNTIME_MANIFEST_SHA256,
    )
    history = EpisodeHistoryManager(normalization=normalization, artifact_identity=identity)
    fake = FakeCoverScorer(
        scores=np.asarray([0.2, 0.8], dtype=np.float64),
        compatibility=ScorerCompatibility(
            model_schema_version="fake_cover_v0",
            views=("base_rgb", "wrist_rgb"),
            preprocessing_contract="ur5e_ws1_uint8_224",
            action_dimension=7,
            action_order=ACTION_ORDER,
            history_length=10,
            representation_id=REPRESENTATION_ID,
            normalization_artifact_hash=identity,
            input_dtype="float32",
            output_shape_rank=1,
        ),
    )
    with pytest.raises(ValueError, match="rejects a fake scorer"):
        build_production_real_cover_robot_shadow_policy(
            candidate_generator=generator,
            history_manager=history,
            scorer=fake,
        )


def test_production_ws1_robot_shadow_serve_to_candidate_zero_response(w3_scorer_and_norm) -> None:
    scorer, normalization, identity = w3_scorer_and_norm
    catalog = _full_horizon_catalog(2, seed=11)
    generator = Pi05HorizonCandidateGenerator(
        policy=_ScriptedPi05Policy(catalog),
        config_name=SELECTED_CONFIG,
        content_identity=LOCAL_RUNTIME_MANIFEST_SHA256,
    )
    history = EpisodeHistoryManager(normalization=normalization, artifact_identity=identity)
    policy = build_production_real_cover_robot_shadow_policy(
        candidate_generator=generator,
        history_manager=history,
        scorer=scorer,
    )
    assert isinstance(policy, CoverPolicyWrapper)
    assert scorer.is_fake is False

    noise = np.zeros((2, FULL_HORIZON, 7), dtype=np.float64)
    noise[1, 0, 0] = 1
    response = policy.infer(_request(), noise=noise)

    actions = np.asarray(response["actions"], dtype=np.float64)
    assert actions.shape == (EXPOSED_HORIZON, 7)
    np.testing.assert_array_equal(actions, catalog[0, :EXPOSED_HORIZON])
    assert response["returned_candidate_index"] == 0
    assert response["verifier_authority"] == "shadow"
    assert response["execution_context"] == "robot"
    assert response["candidate_count"] == W7_CANDIDATE_COUNT
    scores = response["verifier_scores"]
    assert scores is not None
    assert len(scores) == 2
    assert all(isinstance(value, (int, float)) and np.isfinite(value) for value in scores)
    assert history.pending_row is not None
    np.testing.assert_array_equal(history.pending_row, history.last_histories[0, 6])


def test_robot_wrist_fault_returns_candidate_zero(w3_scorer_and_norm) -> None:
    scorer, normalization, identity = w3_scorer_and_norm
    catalog = _full_horizon_catalog(2, seed=21)
    generator = Pi05HorizonCandidateGenerator(
        policy=_ScriptedPi05Policy(catalog),
        content_identity=LOCAL_RUNTIME_MANIFEST_SHA256,
    )
    history = EpisodeHistoryManager(normalization=normalization, artifact_identity=identity)
    policy = build_production_real_cover_robot_shadow_policy(
        candidate_generator=generator,
        history_manager=history,
        scorer=scorer,
    )
    response = policy.infer(_request(wrist_rgb=np.zeros((224, 224, 3), dtype=np.uint8)))
    actions = np.asarray(response["actions"], dtype=np.float64)
    assert actions.shape == (EXPOSED_HORIZON, 7)
    np.testing.assert_array_equal(actions, catalog[0, :EXPOSED_HORIZON])
    assert response["returned_candidate_index"] == 0
    assert response["fallback_reason"] == FallbackReason.MISSING_OR_ZERO_WRIST_VIEW


def test_package_composition_records_pi05_content_identity() -> None:
    if not _PI05_CHECKPOINT.is_dir() or not _W3_PACKAGE.is_dir() or not _W2_NORM.is_file():
        pytest.skip("local packages unavailable")
    identity = require_pi05_runtime_identity(_PI05_CHECKPOINT)
    assert identity == LOCAL_RUNTIME_MANIFEST_SHA256
    catalog = _full_horizon_catalog(2, seed=3)
    policy = build_production_real_cover_robot_shadow_policy_from_packages(
        pi05_checkpoint_dir=_PI05_CHECKPOINT,
        w3_package_root=_W3_PACKAGE,
        normalization_artifact_path=_W2_NORM,
        pi05_policy=_ScriptedPi05Policy(catalog),
        pi05_content_identity=identity,
    )
    response = policy.infer(_request())
    assert response["execution_context"] == "robot"
    assert response["verifier_authority"] == "shadow"
    assert response["returned_candidate_index"] == 0


def test_w5_helper_default_remains_test_context(w3_scorer_and_norm) -> None:
    catalog = _full_horizon_catalog(2, seed=1)
    policy = build_real_cover_shadow_policy_from_packages(
        pi05_checkpoint_dir=_PI05_CHECKPOINT,
        w3_package_root=_W3_PACKAGE,
        normalization_artifact_path=_W2_NORM,
        pi05_policy=_ScriptedPi05Policy(catalog),
        pi05_content_identity=LOCAL_RUNTIME_MANIFEST_SHA256,
    )
    response = policy.infer(_request())
    assert response["execution_context"] == "test"


def test_production_server_metadata_binds_package_gate() -> None:
    metadata = production_server_metadata(pi05_content_identity=LOCAL_RUNTIME_MANIFEST_SHA256)
    assert metadata["server_type"] == W7_SERVER_TYPE
    assert metadata["verifier_authority"] == "shadow"
    assert metadata["execution_context"] == "robot"
    assert metadata["candidate_count"] == W7_CANDIDATE_COUNT
    assert metadata["pi05_content_identity"] == LOCAL_RUNTIME_MANIFEST_SHA256
    assert metadata["allow_fake_scorer"] is False
