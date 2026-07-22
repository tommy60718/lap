"""W4-06 single-session WebSocket ownership and cover transport parity tests."""

from __future__ import annotations

import socket
import threading
import time
from typing import Any

import numpy as np
import pytest
import websockets.sync.client

from lap.policies.cover_policy_wrapper import CoverPolicyWrapper
from lap.serving.single_session_websocket_server import SingleSessionWebsocketPolicyServer
from lap.verifiers.cover.action_adapter import ACTION_ORDER
from lap.verifiers.cover.action_adapter import REPRESENTATION_ID
from lap.verifiers.cover.action_adapter import NormalizationArtifact
from lap.verifiers.cover.history import EpisodeHistoryManager
from lap.verifiers.cover.scorer import FakeCoverScorer
from lap.verifiers.cover.scorer import ScorerCompatibility
from openpi_client import msgpack_numpy


def _finite_chunk(*, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    chunk = rng.normal(scale=0.05, size=(10, 7)).astype(np.float64)
    chunk[:, 6] = np.clip(np.linspace(0.1, 0.9, 10), 0.0, 1.0)
    return chunk


def _candidate_batch(count: int, *, seed: int) -> np.ndarray:
    return np.stack([_finite_chunk(seed=seed + i) for i in range(count)], axis=0)


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


def _cover_request(*, episode_id: str = "ep-1", timestep: int = 0) -> dict[str, Any]:
    return {
        "base_rgb": np.zeros((224, 224, 3), dtype=np.uint8),
        "wrist_rgb": np.full((224, 224, 3), 3, dtype=np.uint8),
        "eef_pos": np.asarray([0.1, 0.2, 0.3], dtype=np.float64),
        "eef_rot": np.asarray([0.0, 0.1, -0.1], dtype=np.float64),
        "gripper": np.asarray([0.5], dtype=np.float64),
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


def _shadow_policy(*, seed: int = 0) -> tuple[CoverPolicyWrapper, EpisodeHistoryManager]:
    artifact = "a" * 64
    history = EpisodeHistoryManager(normalization=_normalization(), artifact_identity=artifact)
    policy = CoverPolicyWrapper(
        authority="shadow",
        execution_context="test",
        candidate_count=2,
        candidate_generator=RecordingCandidateGenerator(_candidate_batch(2, seed=seed)),
        history_manager=history,
        scorer=FakeCoverScorer(scores=[0.2, 0.8], compatibility=_compatibility(artifact_hash=artifact)),
    )
    return policy, history


def _reserve_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _wait_connect(host: str, port: int, *, attempts: int = 40):
    packer = msgpack_numpy.Packer()
    last_error: Exception | None = None
    for _ in range(attempts):
        try:
            conn = websockets.sync.client.connect(f"ws://{host}:{port}", compression=None, max_size=None)
            metadata = msgpack_numpy.unpackb(conn.recv())
            return conn, metadata, packer
        except Exception as error:  # noqa: BLE001
            last_error = error
            time.sleep(0.05)
    raise RuntimeError(f"server did not accept connections: {last_error}")


def test_server_metadata_identifies_cover_authority() -> None:
    policy, _ = _shadow_policy(seed=1)
    port = _reserve_port()
    metadata = {
        "server_type": "pi05_cover",
        "verifier_authority": "shadow",
        "execution_context": "test",
    }
    server = SingleSessionWebsocketPolicyServer(
        policy=policy,
        host="127.0.0.1",
        port=port,
        metadata=metadata,
        on_session_end=policy.reset_session,
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    conn, wire_metadata, _ = _wait_connect("127.0.0.1", port)
    try:
        assert wire_metadata["server_type"] == "pi05_cover"
        assert wire_metadata["verifier_authority"] == "shadow"
    finally:
        conn.close()


def test_direct_and_websocket_cover_diagnostics_match() -> None:
    request = _cover_request()
    direct_policy, _ = _shadow_policy(seed=2)
    direct = direct_policy.infer(request)

    served_policy, _ = _shadow_policy(seed=2)
    port = _reserve_port()
    server = SingleSessionWebsocketPolicyServer(
        policy=served_policy,
        host="127.0.0.1",
        port=port,
        metadata={"server_type": "pi05_cover", "verifier_authority": "shadow"},
        on_session_end=served_policy.reset_session,
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    conn, _, packer = _wait_connect("127.0.0.1", port)
    try:
        conn.send(packer.pack(request))
        payload = msgpack_numpy.unpackb(conn.recv())
    finally:
        conn.close()

    np.testing.assert_array_equal(np.asarray(payload["actions"]), np.asarray(direct["actions"]))
    assert payload["verifier_authority"] == direct["verifier_authority"]
    assert payload["returned_candidate_index"] == direct["returned_candidate_index"]
    assert payload["hypothetical_selected_candidate_index"] == direct["hypothetical_selected_candidate_index"]
    assert payload["verifier_scores"] == direct["verifier_scores"]
    assert payload["state_event"] == direct["state_event"]
    assert payload["fallback_reason"] == direct["fallback_reason"]
    assert "server_timing" in payload


def test_websocket_partial_nonfinite_scores_match_direct_null_diagnostics() -> None:
    request = _cover_request()
    scores = np.asarray([np.nan, 0.8], dtype=np.float64)
    artifact = "a" * 64
    history = EpisodeHistoryManager(normalization=_normalization(), artifact_identity=artifact)
    candidates = _candidate_batch(2, seed=8)
    direct_policy = CoverPolicyWrapper(
        authority="shadow",
        execution_context="test",
        candidate_count=2,
        candidate_generator=RecordingCandidateGenerator(candidates),
        history_manager=history,
        scorer=FakeCoverScorer(scores=scores, compatibility=_compatibility(artifact_hash=artifact)),
    )
    direct = direct_policy.infer(request)
    assert direct["verifier_scores"] is None
    assert direct["hypothetical_selected_candidate_index"] == 1

    served_history = EpisodeHistoryManager(normalization=_normalization(), artifact_identity=artifact)
    served_policy = CoverPolicyWrapper(
        authority="shadow",
        execution_context="test",
        candidate_count=2,
        candidate_generator=RecordingCandidateGenerator(candidates),
        history_manager=served_history,
        scorer=FakeCoverScorer(scores=scores, compatibility=_compatibility(artifact_hash=artifact)),
    )
    port = _reserve_port()
    server = SingleSessionWebsocketPolicyServer(
        policy=served_policy,
        host="127.0.0.1",
        port=port,
        metadata={"server_type": "pi05_cover"},
        on_session_end=served_policy.reset_session,
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    conn, _, packer = _wait_connect("127.0.0.1", port)
    try:
        conn.send(packer.pack(request))
        payload = msgpack_numpy.unpackb(conn.recv())
    finally:
        conn.close()

    np.testing.assert_array_equal(np.asarray(payload["actions"]), np.asarray(direct["actions"]))
    assert payload["verifier_scores"] is None
    assert payload["hypothetical_selected_candidate_index"] == direct["hypothetical_selected_candidate_index"]
    assert payload["fallback_reason"] is None


def test_second_concurrent_client_is_rejected_without_mutating_owner_state() -> None:
    policy, history = _shadow_policy(seed=3)
    port = _reserve_port()
    server = SingleSessionWebsocketPolicyServer(
        policy=policy,
        host="127.0.0.1",
        port=port,
        metadata={"server_type": "pi05_cover"},
        on_session_end=policy.reset_session,
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    owner, _, packer = _wait_connect("127.0.0.1", port)
    owner.send(packer.pack(_cover_request(timestep=0)))
    first = msgpack_numpy.unpackb(owner.recv())
    assert first["state_event"] == "new_episode"
    assert history.active_episode_id == "ep-1"
    before = history.pending_row.copy()

    rejected = websockets.sync.client.connect(f"ws://127.0.0.1:{port}", compression=None, max_size=None)
    with pytest.raises(Exception):
        # Second client must be closed before any metadata/infer exchange completes.
        _ = rejected.recv()
    rejected.close()

    assert history.active_episode_id == "ep-1"
    np.testing.assert_array_equal(history.pending_row, before)

    owner.send(packer.pack(_cover_request(timestep=1)))
    second = msgpack_numpy.unpackb(owner.recv())
    assert second["fallback_reason"] is None
    assert history.active_episode_id == "ep-1"
    owner.close()


def test_disconnect_discards_history_before_next_client() -> None:
    policy, history = _shadow_policy(seed=4)
    port = _reserve_port()
    server = SingleSessionWebsocketPolicyServer(
        policy=policy,
        host="127.0.0.1",
        port=port,
        metadata={"server_type": "pi05_cover"},
        on_session_end=policy.reset_session,
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    first_conn, _, packer = _wait_connect("127.0.0.1", port)
    first_conn.send(packer.pack(_cover_request(episode_id="ep-old", timestep=0)))
    _ = msgpack_numpy.unpackb(first_conn.recv())
    assert history.active_episode_id == "ep-old"
    first_conn.close()
    time.sleep(0.1)
    assert history.active_episode_id is None
    assert history.pending_row is None

    second_conn, _, packer2 = _wait_connect("127.0.0.1", port)
    try:
        second_conn.send(packer2.pack(_cover_request(episode_id="ep-new", timestep=0)))
        payload = msgpack_numpy.unpackb(second_conn.recv())
        assert payload["state_event"] == "new_episode"
        assert history.active_episode_id == "ep-new"
    finally:
        second_conn.close()


def test_actions_only_client_ignores_cover_diagnostics() -> None:
    policy, _ = _shadow_policy(seed=5)
    response = policy.infer(_cover_request())
    actions_only = {"actions": response["actions"]}
    assert set(actions_only.keys()) == {"actions"}
    np.testing.assert_array_equal(np.asarray(actions_only["actions"]).shape, (10, 7))


def test_new_rollout_on_same_connection_resets_server_history() -> None:
    policy, history = _shadow_policy(seed=6)
    port = _reserve_port()
    server = SingleSessionWebsocketPolicyServer(
        policy=policy,
        host="127.0.0.1",
        port=port,
        metadata={"server_type": "pi05_cover", "verifier_authority": "shadow"},
        on_session_end=policy.reset_session,
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    conn, _, packer = _wait_connect("127.0.0.1", port)
    try:
        conn.send(packer.pack(_cover_request(episode_id="rollout-1", timestep=0)))
        first = msgpack_numpy.unpackb(conn.recv())
        assert first["state_event"] == "new_episode"
        assert history.active_episode_id == "rollout-1"

        conn.send(packer.pack(_cover_request(episode_id="rollout-1", timestep=1)))
        second = msgpack_numpy.unpackb(conn.recv())
        assert second["fallback_reason"] is None
        assert history.last_accepted_timestep == 1

        # New rollout on the same socket: new episode ID at timestep 0.
        conn.send(packer.pack(_cover_request(episode_id="rollout-2", timestep=0)))
        third = msgpack_numpy.unpackb(conn.recv())
        assert third["state_event"] == "new_episode"
        assert third["fallback_reason"] is None
        assert history.active_episode_id == "rollout-2"
        assert history.last_accepted_timestep == 0
        assert history.committed_rows.shape[0] == 0
    finally:
        conn.close()
