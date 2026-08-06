"""Serve production real-CoVer robot shadow behind the single-session WebSocket server.

W4 fake-backed serve_cover_fake.py remains acceptance evidence only; this entry
loads accepted W3 + selected Pi0.5 packages with authority=shadow and
execution_context=robot.
"""

from __future__ import annotations

import dataclasses
import logging
from pathlib import Path
import socket

import tyro

from lap.policies.w7_real_cover_robot_shadow import build_production_real_cover_robot_shadow_policy_from_packages
from lap.policies.w7_real_cover_robot_shadow import production_server_metadata
from lap.serving.single_session_websocket_server import SingleSessionWebsocketPolicyServer
from lap.verifiers.pi05_horizon import LOCAL_RUNTIME_MANIFEST_SHA256
from lap.verifiers.pi05_horizon import require_pi05_runtime_identity

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_W3 = _REPO_ROOT / "artifacts/w3/canonical_acceptance_v1"
_DEFAULT_NORM = Path("/home/yangsen/osx_ur/catkin_ws/src/osx_vla/.w2_canonical_export_v3/normalization_artifact.json")


@dataclasses.dataclass
class Args:
    port: int = 8000
    pi05_checkpoint: Path = Path("/home/malek/osx_ur/dependencies/checkpoints/pi05_ur5e_peg_in_hole_lora/10000")
    w3_package: Path = _DEFAULT_W3
    normalization_artifact: Path = _DEFAULT_NORM
    expected_pi05_manifest_sha256: str = LOCAL_RUNTIME_MANIFEST_SHA256


def main(args: Args) -> None:
    identity = require_pi05_runtime_identity(
        Path(args.pi05_checkpoint),
        expected_manifest_sha256=args.expected_pi05_manifest_sha256,
    )
    policy = build_production_real_cover_robot_shadow_policy_from_packages(
        pi05_checkpoint_dir=Path(args.pi05_checkpoint),
        w3_package_root=Path(args.w3_package),
        normalization_artifact_path=Path(args.normalization_artifact),
    )
    metadata = production_server_metadata(pi05_content_identity=identity)
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info(
        "Serving real-CoVer robot shadow (host=%s ip=%s port=%s identity=%s)",
        hostname,
        local_ip,
        args.port,
        identity,
    )
    server = SingleSessionWebsocketPolicyServer(
        policy=policy,
        host="0.0.0.0",
        port=args.port,
        metadata=metadata,
        on_session_end=policy.reset_session,
    )
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
