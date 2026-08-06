"""W7-01 production real-CoVer robot-context shadow composition."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from lap.policies.cover_policy_wrapper import CoverPolicyWrapper
from lap.policies.cover_policy_wrapper import CoverScorer
from lap.policies.cover_policy_wrapper import HistoryManager
from lap.policies.w5_real_cover_shadow import W5_CANDIDATE_COUNT
from lap.policies.w5_real_cover_shadow import build_real_cover_shadow_policy_from_packages
from lap.verifiers.candidate_generator import CandidateGenerator
from lap.verifiers.pi05_horizon import Pi05Policy

W7_CANDIDATE_COUNT = W5_CANDIDATE_COUNT
W7_AUTHORITY = "shadow"
W7_EXECUTION_CONTEXT = "robot"
W7_SERVER_TYPE = "pi05_cover_real"


def production_server_metadata(*, pi05_content_identity: str) -> dict[str, Any]:
    """WS metadata advertised by the production real-CoVer robot shadow serve."""

    return {
        "server_type": W7_SERVER_TYPE,
        "verifier_authority": W7_AUTHORITY,
        "execution_context": W7_EXECUTION_CONTEXT,
        "candidate_count": W7_CANDIDATE_COUNT,
        "pi05_content_identity": pi05_content_identity,
        "allow_fake_scorer": False,
    }


def build_production_real_cover_robot_shadow_policy(
    *,
    candidate_generator: CandidateGenerator,
    history_manager: HistoryManager,
    scorer: CoverScorer,
) -> CoverPolicyWrapper:
    """Compose production robot shadow: real scorer only, candidate zero authority."""

    return CoverPolicyWrapper(
        authority=W7_AUTHORITY,
        execution_context=W7_EXECUTION_CONTEXT,
        candidate_count=W7_CANDIDATE_COUNT,
        candidate_generator=candidate_generator,
        history_manager=history_manager,
        scorer=scorer,
        allow_fake_scorer=False,
    )


def build_production_real_cover_robot_shadow_policy_from_packages(
    *,
    pi05_checkpoint_dir: Path,
    w3_package_root: Path,
    normalization_artifact_path: Path,
    pi05_policy: Pi05Policy | None = None,
    pi05_content_identity: str | None = None,
) -> CoverPolicyWrapper:
    """Load accepted packages into the production robot-context shadow path."""

    return build_real_cover_shadow_policy_from_packages(
        pi05_checkpoint_dir=pi05_checkpoint_dir,
        w3_package_root=w3_package_root,
        normalization_artifact_path=normalization_artifact_path,
        pi05_policy=pi05_policy,
        pi05_content_identity=pi05_content_identity,
        execution_context=W7_EXECUTION_CONTEXT,
    )
