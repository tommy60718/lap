"""W5-01 composition: real Pi0.5 horizon adapter + accepted W3 scorer + W4 shadow path."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from lap.policies.cover_policy_wrapper import CoverPolicyWrapper
from lap.verifiers.cover.accepted_w3_scorer import load_accepted_w3_deployment_scorer
from lap.verifiers.cover.action_adapter import NormalizationArtifact
from lap.verifiers.cover.history import EpisodeHistoryManager
from lap.verifiers.pi05_horizon import SELECTED_CONFIG
from lap.verifiers.pi05_horizon import Pi05HorizonCandidateGenerator
from lap.verifiers.pi05_horizon import Pi05Policy
from lap.verifiers.pi05_horizon import load_pi05_horizon_generator

W5_CANDIDATE_COUNT = 2


def load_w2_normalization_artifact(path: Path) -> tuple[NormalizationArtifact, str]:
    """Load the sealed LAP-3 normalization artifact and its content hash."""

    text = Path(path).read_text(encoding="utf-8")
    payload = json.loads(text)
    identity = str(payload["content_hash"])
    return NormalizationArtifact.from_json(text), identity


def build_real_cover_shadow_policy(
    *,
    candidate_generator: Pi05HorizonCandidateGenerator,
    normalization: NormalizationArtifact,
    artifact_identity: str,
    w3_package_root: Path,
) -> CoverPolicyWrapper:
    """Compose offline test-context shadow authority through the accepted W4 wrapper."""

    scorer = load_accepted_w3_deployment_scorer(
        Path(w3_package_root),
        expected_normalization_hash=artifact_identity,
    )
    history = EpisodeHistoryManager(
        normalization=normalization,
        artifact_identity=artifact_identity,
    )
    return CoverPolicyWrapper(
        authority="shadow",
        execution_context="test",
        candidate_count=W5_CANDIDATE_COUNT,
        candidate_generator=candidate_generator,
        history_manager=history,
        scorer=scorer,
        allow_fake_scorer=False,
    )


def build_real_cover_shadow_policy_from_packages(
    *,
    pi05_checkpoint_dir: Path,
    w3_package_root: Path,
    normalization_artifact_path: Path,
    pi05_policy: Pi05Policy | None = None,
    pi05_content_identity: str | None = None,
) -> CoverPolicyWrapper:
    """Load packages (or inject a validated Pi0.5 policy) and build the shadow composition."""

    normalization, artifact_identity = load_w2_normalization_artifact(normalization_artifact_path)
    if pi05_policy is None:
        generator = load_pi05_horizon_generator(Path(pi05_checkpoint_dir))
    else:
        if not pi05_content_identity:
            raise ValueError("pi05_content_identity is required when injecting a Pi0.5 policy")
        generator = Pi05HorizonCandidateGenerator(
            policy=pi05_policy,
            config_name=SELECTED_CONFIG,
            content_identity=pi05_content_identity,
        )
    return build_real_cover_shadow_policy(
        candidate_generator=generator,
        normalization=normalization,
        artifact_identity=artifact_identity,
        w3_package_root=Path(w3_package_root),
    )


def recorded_two_view_request(
    *,
    base_rgb: Any,
    wrist_rgb: Any,
    eef_pos: Any,
    eef_rot: Any,
    gripper: Any,
    prompt: str,
    episode_id: str,
    timestep: int,
) -> dict[str, Any]:
    """Build one WS-1 recorded request for the W5 public seam."""

    return {
        "base_rgb": base_rgb,
        "wrist_rgb": wrist_rgb,
        "eef_pos": eef_pos,
        "eef_rot": eef_rot,
        "gripper": gripper,
        "prompt": prompt,
        "episode_id": episode_id,
        "timestep": timestep,
    }
