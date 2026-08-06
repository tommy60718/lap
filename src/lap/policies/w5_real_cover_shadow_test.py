"""W5-01 public seam: recorded two-view request → real-CoVer shadow response."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from lap.policies.cover_policy_wrapper import CoverPolicyWrapper
from lap.policies.w5_real_cover_shadow import W5_CANDIDATE_COUNT
from lap.policies.w5_real_cover_shadow import build_real_cover_shadow_policy_from_packages
from lap.policies.w5_real_cover_shadow import load_w2_normalization_artifact
from lap.policies.w5_real_cover_shadow import recorded_two_view_request
from lap.verifiers.cover.accepted_w3_scorer import load_accepted_w3_deployment_scorer
from lap.verifiers.cover.faults import CoverVerifierError
from lap.verifiers.cover.faults import FallbackReason
from lap.verifiers.cover.history import EpisodeHistoryManager
from lap.verifiers.pi05_horizon import EXPOSED_HORIZON
from lap.verifiers.pi05_horizon import FULL_HORIZON
from lap.verifiers.pi05_horizon import LOCAL_RUNTIME_MANIFEST_SHA256
from lap.verifiers.pi05_horizon import SELECTED_CONFIG
from lap.verifiers.pi05_horizon import Pi05HorizonCandidateGenerator
from lap.verifiers.pi05_horizon import require_pi05_runtime_identity
from lap.verifiers.pi05_horizon import validate_full_horizon_actions

_REPO_ROOT = Path(__file__).resolve().parents[3]
_W3_PACKAGE = _REPO_ROOT / "artifacts/w3/canonical_acceptance_v1"
_W2_NORM = Path("/home/yangsen/osx_ur/catkin_ws/src/osx_vla/.w2_canonical_export_v3/normalization_artifact.json")
_PI05_CHECKPOINT = Path("/home/yangsen/checkpoints_datasets/checkpoints_dual_verifier/peg_in_hole_lora_v2/10000")


class _ScriptedPi05Policy:
    """Deterministic [16,7] policy for public-seam tests without loading OpenPI weights."""

    def __init__(self, catalog: np.ndarray) -> None:
        self._catalog = np.asarray(catalog, dtype=np.float64)
        self.calls: list[dict[str, Any]] = []

    def infer(self, obs: dict[str, Any], *, noise: np.ndarray | None = None) -> dict[str, Any]:
        index = 0
        if noise is not None:
            noise_array = np.asarray(noise)
            index = int(noise_array.reshape(-1)[0]) if noise_array.size else 0
        self.calls.append({"prompt": obs.get("prompt"), "index": index})
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
        episode_id="w5-ep-0",
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


def test_pi05_runtime_identity_matches_local_manifest() -> None:
    if not _PI05_CHECKPOINT.is_dir():
        pytest.skip("local Pi0.5 checkpoint unavailable")
    identity = require_pi05_runtime_identity(_PI05_CHECKPOINT)
    assert identity == LOCAL_RUNTIME_MANIFEST_SHA256


def test_pi05_runtime_identity_rejects_stale_package(tmp_path: Path) -> None:
    checkpoint = tmp_path / "ckpt"
    (checkpoint / "params").mkdir(parents=True)
    (checkpoint / "_CHECKPOINT_METADATA").write_text("stale\n", encoding="utf-8")
    (checkpoint / "params" / "x").write_bytes(b"x")
    (checkpoint / "assets" / "yutaku" / "ur5e_peg_in_hole-v2").mkdir(parents=True)
    (checkpoint / "assets" / "yutaku" / "ur5e_peg_in_hole-v2" / "norm_stats.json").write_text(
        "{}",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match=r"Pi0\.5 runtime manifest mismatch"):
        require_pi05_runtime_identity(checkpoint)


def test_horizon_adapter_rejects_short_and_nonfinite_outputs() -> None:
    class ShortPolicy:
        def infer(self, obs: dict[str, Any], *, noise: np.ndarray | None = None) -> dict[str, Any]:
            del obs, noise
            return {"actions": np.zeros((10, 7), dtype=np.float64)}

    class NonfinitePolicy:
        def infer(self, obs: dict[str, Any], *, noise: np.ndarray | None = None) -> dict[str, Any]:
            del obs, noise
            actions = np.ones((FULL_HORIZON, 7), dtype=np.float64)
            actions[0, 0] = np.nan
            return {"actions": actions}

    short = Pi05HorizonCandidateGenerator(policy=ShortPolicy(), content_identity="x" * 64)
    with pytest.raises(ValueError, match="must have shape"):
        short.generate(observation=_request(), instruction="x", candidate_count=1)

    bad = Pi05HorizonCandidateGenerator(policy=NonfinitePolicy(), content_identity="x" * 64)
    with pytest.raises(ValueError, match="must be finite"):
        bad.generate(observation=_request(), instruction="x", candidate_count=1)


def test_horizon_adapter_exposes_first_ten_rows_only() -> None:
    catalog = _full_horizon_catalog(2, seed=3)
    generator = Pi05HorizonCandidateGenerator(
        policy=_ScriptedPi05Policy(catalog),
        content_identity=LOCAL_RUNTIME_MANIFEST_SHA256,
    )
    noise = np.zeros((2, FULL_HORIZON, 7), dtype=np.float64)
    noise[0, 0, 0] = 0
    noise[1, 0, 0] = 1
    batch = generator.generate(
        observation=_request(),
        instruction="insert the circular peg",
        candidate_count=2,
        noise=noise,
    )
    assert batch.shape == (2, EXPOSED_HORIZON, 7)
    np.testing.assert_array_equal(batch[0], catalog[0, :EXPOSED_HORIZON])
    np.testing.assert_array_equal(batch[1], catalog[1, :EXPOSED_HORIZON])


def test_w3_package_rejects_wrong_normalization_hash() -> None:
    if not _W3_PACKAGE.is_dir():
        pytest.skip("accepted W3 package unavailable")
    with pytest.raises(ValueError, match="normalization_artifact_hash"):
        load_accepted_w3_deployment_scorer(_W3_PACKAGE, expected_normalization_hash="0" * 64)


def test_recorded_two_view_request_to_real_cover_shadow_response(w3_scorer_and_norm) -> None:
    scorer, normalization, identity = w3_scorer_and_norm
    catalog = _full_horizon_catalog(2, seed=11)
    generator = Pi05HorizonCandidateGenerator(
        policy=_ScriptedPi05Policy(catalog),
        config_name=SELECTED_CONFIG,
        content_identity=LOCAL_RUNTIME_MANIFEST_SHA256,
    )
    assert scorer.compatibility.normalization_artifact_hash == identity
    assert scorer.is_fake is False
    history = EpisodeHistoryManager(normalization=normalization, artifact_identity=identity)
    policy = CoverPolicyWrapper(
        authority="shadow",
        execution_context="test",
        candidate_count=W5_CANDIDATE_COUNT,
        candidate_generator=generator,
        history_manager=history,
        scorer=scorer,
        allow_fake_scorer=False,
    )
    noise = np.zeros((2, FULL_HORIZON, 7), dtype=np.float64)
    noise[1, 0, 0] = 1
    response = policy.infer(_request(), noise=noise)

    actions = np.asarray(response["actions"], dtype=np.float64)
    assert actions.shape == (EXPOSED_HORIZON, 7)
    np.testing.assert_array_equal(actions, catalog[0, :EXPOSED_HORIZON])
    assert response["returned_candidate_index"] == 0
    assert response["verifier_authority"] == "shadow"
    assert response["candidate_count"] == 2
    assert response["hypothetical_selected_candidate_index"] in (0, 1)
    scores = response["verifier_scores"]
    assert scores is not None
    assert len(scores) == 2
    assert all(isinstance(value, (int, float)) and np.isfinite(value) for value in scores)
    assert history.pending_row is not None
    np.testing.assert_array_equal(history.pending_row, history.last_histories[0, 6])


def test_zero_wrist_rejects_in_test_context_before_scoring(w3_scorer_and_norm) -> None:
    scorer, normalization, identity = w3_scorer_and_norm
    catalog = _full_horizon_catalog(2, seed=21)
    generator = Pi05HorizonCandidateGenerator(
        policy=_ScriptedPi05Policy(catalog),
        content_identity=LOCAL_RUNTIME_MANIFEST_SHA256,
    )
    history = EpisodeHistoryManager(normalization=normalization, artifact_identity=identity)
    policy = CoverPolicyWrapper(
        authority="shadow",
        execution_context="test",
        candidate_count=2,
        candidate_generator=generator,
        history_manager=history,
        scorer=scorer,
    )
    request = _request(wrist_rgb=np.zeros((224, 224, 3), dtype=np.uint8))
    with pytest.raises(CoverVerifierError, match="wrist") as error:
        policy.infer(request)
    assert error.value.fallback_reason == FallbackReason.MISSING_OR_ZERO_WRIST_VIEW


def test_composition_helper_requires_identity_for_injected_policy() -> None:
    catalog = _full_horizon_catalog(2, seed=1)
    with pytest.raises(ValueError, match="pi05_content_identity"):
        build_real_cover_shadow_policy_from_packages(
            pi05_checkpoint_dir=_PI05_CHECKPOINT,
            w3_package_root=_W3_PACKAGE,
            normalization_artifact_path=_W2_NORM,
            pi05_policy=_ScriptedPi05Policy(catalog),
            pi05_content_identity=None,
        )


def test_validate_full_horizon_actions_helper() -> None:
    good = np.ones((FULL_HORIZON, 7), dtype=np.float64)
    np.testing.assert_array_equal(validate_full_horizon_actions(good, label="x"), good)
    with pytest.raises(ValueError, match="shape"):
        validate_full_horizon_actions(np.ones((8, 7)), label="x")
