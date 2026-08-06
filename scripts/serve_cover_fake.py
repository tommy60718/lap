"""Serve a composed CoverPolicyWrapper behind the single-session WebSocket server.

This is the owned pi05_cover transport entry for fake-backed W4 acceptance.
Real checkpoint composition remains the existing OpenPI serve_policy path; this
script only admits one robot client and clears cover history on disconnect.
"""

from __future__ import annotations

import dataclasses
import logging
import socket
from typing import Any, Literal

import numpy as np
import tyro

from lap.policies.cover_policy_wrapper import CoverPolicyWrapper
from lap.serving.single_session_websocket_server import SingleSessionWebsocketPolicyServer
from lap.verifiers.cover.action_adapter import ACTION_ORDER
from lap.verifiers.cover.action_adapter import REPRESENTATION_ID
from lap.verifiers.cover.action_adapter import NormalizationArtifact
from lap.verifiers.cover.history import EpisodeHistoryManager
from lap.verifiers.cover.scorer import FakeCoverScorer
from lap.verifiers.cover.scorer import ScorerCompatibility

Authority = Literal["disabled", "shadow", "active"]
ExecutionContext = Literal["test", "robot"]


class FixedCandidateGenerator:
    """Deterministic absolute candidate batch for fake cover serving."""

    def __init__(self, candidates: np.ndarray) -> None:
        self._candidates = np.asarray(candidates, dtype=np.float64)

    def generate(
        self,
        *,
        observation: dict[str, Any],
        instruction: str,
        candidate_count: int,
        noise: np.ndarray | None = None,
    ) -> np.ndarray:
        del observation, instruction, noise
        return self._candidates[:candidate_count].copy()


def _finite_chunk(*, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    chunk = rng.normal(scale=0.05, size=(10, 7)).astype(np.float64)
    chunk[:, 6] = np.clip(np.linspace(0.1, 0.9, 10), 0.0, 1.0)
    return chunk


@dataclasses.dataclass
class Args:
    port: int = 8000
    authority: Authority = "shadow"
    execution_context: ExecutionContext = "test"
    candidate_count: int = 2
    artifact_hash: str = "a" * 64
    allow_fake_scorer: bool = True
    seed: int = 0


def build_fake_cover_policy(args: Args) -> CoverPolicyWrapper:
    candidates = np.stack(
        [_finite_chunk(seed=args.seed + i) for i in range(args.candidate_count)],
        axis=0,
    )
    history = EpisodeHistoryManager(
        normalization=NormalizationArtifact(q01=np.zeros(6), q99=np.ones(6)),
        artifact_identity=args.artifact_hash,
    )
    compatibility = ScorerCompatibility(
        model_schema_version="fake_cover_v0",
        views=("base_rgb", "wrist_rgb"),
        preprocessing_contract="ur5e_ws1_uint8_224",
        action_dimension=7,
        action_order=ACTION_ORDER,
        history_length=10,
        representation_id=REPRESENTATION_ID,
        normalization_artifact_hash=args.artifact_hash,
        input_dtype="float32",
        output_shape_rank=1,
    )
    scores = np.linspace(0.1, 0.9, args.candidate_count, dtype=np.float64)
    scorer = None if args.authority == "disabled" else FakeCoverScorer(scores=scores, compatibility=compatibility)
    return CoverPolicyWrapper(
        authority=args.authority,
        execution_context=args.execution_context,
        candidate_count=1 if args.authority == "disabled" else args.candidate_count,
        candidate_generator=FixedCandidateGenerator(candidates),
        history_manager=None if args.authority == "disabled" else history,
        scorer=scorer,
        allow_fake_scorer=args.allow_fake_scorer,
    )


def main(args: Args) -> None:
    policy = build_fake_cover_policy(args)
    metadata = {
        "server_type": "pi05_cover",
        "verifier_authority": args.authority,
        "execution_context": args.execution_context,
        "candidate_count": args.candidate_count,
    }
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info("Serving fake pi05_cover (host=%s ip=%s port=%s)", hostname, local_ip, args.port)
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
