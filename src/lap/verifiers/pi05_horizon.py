"""LAP-2 Pi0.5 horizon adapter: validate full [16,7] then expose first ten rows."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any, Protocol

import numpy as np

FULL_HORIZON = 16
EXPOSED_HORIZON = 10
ACTION_DIM = 7
SELECTED_CONFIG = "pi05_ur5e_peg_in_hole_lora"
# Local offline content identity from docs/yangsen/1-2-2_pi05_robot_package_handoff.md
LOCAL_RUNTIME_MANIFEST_SHA256 = "545374a8b9db33e53f3ecab50fc67d9e6b1d08d1c3db81db0a34997f432fe190"


class Pi05Policy(Protocol):
    def infer(self, obs: dict[str, Any], *, noise: np.ndarray | None = None) -> Mapping[str, Any]: ...


def _canonical_manifest_line(sha256: str, size: int, relative_path: str) -> str:
    return f"{sha256}  {size}  {relative_path}\n"


def build_pi05_runtime_manifest(checkpoint_dir: Path) -> tuple[str, str]:
    """Return (manifest_text, sha256) for runtime-required checkpoint content."""

    root = Path(checkpoint_dir).resolve()
    if not root.is_dir():
        raise ValueError(f"Pi0.5 checkpoint directory missing: {root}")
    lines: list[str] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        if relative == "train_state" or relative.startswith("train_state/"):
            continue
        if relative != "_CHECKPOINT_METADATA" and not relative.startswith(("params/", "assets/")):
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        lines.append(_canonical_manifest_line(digest, path.stat().st_size, relative))
    if not lines:
        raise ValueError(f"Pi0.5 checkpoint has no runtime-required files: {root}")
    text = "".join(lines)
    return text, hashlib.sha256(text.encode("utf-8")).hexdigest()


def require_pi05_runtime_identity(
    checkpoint_dir: Path,
    *,
    expected_manifest_sha256: str = LOCAL_RUNTIME_MANIFEST_SHA256,
) -> str:
    """Reject stale or incomplete Pi0.5 packages before inference."""

    _, actual = build_pi05_runtime_manifest(checkpoint_dir)
    if actual != expected_manifest_sha256:
        raise ValueError(f"Pi0.5 runtime manifest mismatch: expected {expected_manifest_sha256}, got {actual}")
    return actual


def validate_full_horizon_actions(actions: np.ndarray, *, label: str) -> np.ndarray:
    """Require finite ordered absolute UR5e targets with shape [16, 7]."""

    array = np.asarray(actions)
    if array.ndim != 2 or array.shape != (FULL_HORIZON, ACTION_DIM):
        raise ValueError(f"{label} must have shape [{FULL_HORIZON}, {ACTION_DIM}], got {array.shape}")
    if not np.issubdtype(array.dtype, np.floating):
        raise ValueError(f"{label} must be floating-point absolute targets")
    if not np.isfinite(array).all():
        raise ValueError(f"{label} must be finite")
    return np.asarray(array, dtype=np.float64)


@dataclass
class Pi05HorizonCandidateGenerator:
    """CandidateGenerator that validates the selected policy's full horizon."""

    policy: Pi05Policy
    config_name: str = SELECTED_CONFIG
    content_identity: str = LOCAL_RUNTIME_MANIFEST_SHA256

    def generate(
        self,
        *,
        observation: dict[str, Any],
        instruction: str,
        candidate_count: int,
        noise: np.ndarray | None = None,
    ) -> np.ndarray:
        if candidate_count < 1:
            raise ValueError("candidate_count must be positive")
        if self.config_name != SELECTED_CONFIG:
            raise ValueError(f"unsupported Pi0.5 config: {self.config_name!r}")
        chunks: list[np.ndarray] = []
        for index in range(candidate_count):
            sample_noise = None
            if noise is not None:
                noise_array = np.asarray(noise)
                if noise_array.ndim == 3:
                    sample_noise = noise_array[index]
                elif noise_array.ndim == 2 and candidate_count == 1:
                    sample_noise = noise_array
                else:
                    raise ValueError("noise must be [M, H, D] for multi-candidate generation or [H, D] for M=1")
            obs = dict(observation)
            obs["prompt"] = instruction
            result = self.policy.infer(obs, noise=sample_noise)
            actions = validate_full_horizon_actions(result["actions"], label=f"candidate[{index}]")
            chunks.append(actions[:EXPOSED_HORIZON].copy())
        return np.stack(chunks, axis=0)


def load_pi05_horizon_generator(
    checkpoint_dir: Path,
    *,
    config_name: str = SELECTED_CONFIG,
    expected_manifest_sha256: str = LOCAL_RUNTIME_MANIFEST_SHA256,
    pytorch_device: str | None = "cpu",
) -> Pi05HorizonCandidateGenerator:
    """Load the selected OpenPI policy after content-identity validation."""

    from openpi.policies import policy_config as _policy_config  # noqa: PLC0415
    from openpi.training import config as _config  # noqa: PLC0415

    identity = require_pi05_runtime_identity(
        checkpoint_dir,
        expected_manifest_sha256=expected_manifest_sha256,
    )
    train_config = _config.get_config(config_name)
    policy = _policy_config.create_trained_policy(
        train_config,
        checkpoint_dir,
        pytorch_device=pytorch_device,
    )
    return Pi05HorizonCandidateGenerator(
        policy=policy,
        config_name=config_name,
        content_identity=identity,
    )
