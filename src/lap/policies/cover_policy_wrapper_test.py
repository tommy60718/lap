"""W4-01 behavior tests for disabled CoVer policy composition."""

from __future__ import annotations

import threading
from typing import Any

import numpy as np
import pytest

from lap.policies.cover_policy_wrapper import CoverPolicyWrapper


def _finite_chunk(*, seed: int = 0) -> np.ndarray:
    """Independent absolute [10, 7] fixture with gripper in [0, 1]."""
    rng = np.random.default_rng(seed)
    chunk = rng.normal(loc=0.0, scale=0.05, size=(10, 7)).astype(np.float64)
    chunk[:, 6] = np.linspace(0.1, 0.9, 10, dtype=np.float64)
    return chunk


def _valid_ws1_request(*, prompt: str = "reach the peg") -> dict[str, Any]:
    return {
        "base_rgb": np.zeros((224, 224, 3), dtype=np.uint8),
        "wrist_rgb": np.full((224, 224, 3), 7, dtype=np.uint8),
        "eef_pos": np.asarray([0.1, 0.2, 0.3], dtype=np.float64),
        "eef_rot": np.asarray([0.0, 0.1, -0.1], dtype=np.float64),
        "gripper": np.asarray([0.8], dtype=np.float64),
        "prompt": prompt,
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
                "observation": observation,
                "instruction": instruction,
                "candidate_count": candidate_count,
                "noise": None if noise is None else np.asarray(noise).copy(),
            }
        )
        if self._candidates.ndim == 2:
            return self._candidates.copy()
        return self._candidates[:candidate_count].copy()


class FaultingCandidateGenerator:
    def __init__(self, error: Exception) -> None:
        self._error = error
        self.calls = 0

    def generate(self, **kwargs: Any) -> np.ndarray:
        self.calls += 1
        raise self._error


class MalformedCandidateGenerator:
    def __init__(self, payload: np.ndarray | None) -> None:
        self._payload = payload
        self.calls = 0

    def generate(self, **kwargs: Any) -> np.ndarray:
        self.calls += 1
        if self._payload is None:
            return None  # type: ignore[return-value]
        return np.asarray(self._payload)


class RecordingHistoryManager:
    def __init__(self) -> None:
        self.build_calls = 0
        self.clear_calls = 0
        self._committed: list[np.ndarray] = []
        self._pending: np.ndarray | None = None

    def clear(self) -> None:
        self.clear_calls += 1
        self._committed.clear()
        self._pending = None

    def build_histories(self, *args: Any, **kwargs: Any) -> Any:
        self.build_calls += 1
        raise AssertionError("history construction must not run in disabled authority")

    @property
    def committed_rows(self) -> list[np.ndarray]:
        return list(self._committed)

    @property
    def pending_row(self) -> np.ndarray | None:
        return None if self._pending is None else self._pending.copy()


class RecordingScorer:
    def __init__(self) -> None:
        self.score_calls = 0

    def score(self, *args: Any, **kwargs: Any) -> Any:
        self.score_calls += 1
        raise AssertionError("scoring must not run in disabled authority")


def _disabled_policy(
    generator: Any,
    *,
    execution_context: str = "test",
    candidate_count: int = 4,
    history: RecordingHistoryManager | None = None,
    scorer: RecordingScorer | None = None,
) -> tuple[CoverPolicyWrapper, RecordingHistoryManager, RecordingScorer]:
    history_manager = history if history is not None else RecordingHistoryManager()
    scorer_obj = scorer if scorer is not None else RecordingScorer()
    policy = CoverPolicyWrapper(
        authority="disabled",
        execution_context=execution_context,  # type: ignore[arg-type]
        candidate_count=candidate_count,
        candidate_generator=generator,
        history_manager=history_manager,
        scorer=scorer_obj,
    )
    return policy, history_manager, scorer_obj


def test_disabled_infer_returns_finite_absolute_actions_chunk() -> None:
    expected = _finite_chunk(seed=3)
    generator = RecordingCandidateGenerator(expected)
    policy, _, _ = _disabled_policy(generator)

    response = policy.infer(_valid_ws1_request())

    actions = np.asarray(response["actions"], dtype=np.float64)
    assert actions.shape == (10, 7)
    assert np.isfinite(actions).all()
    np.testing.assert_array_equal(actions, expected)


def test_disabled_bypasses_history_and_scoring_and_clears_state() -> None:
    expected = _finite_chunk(seed=5)
    generator = RecordingCandidateGenerator(np.stack([expected, expected + 0.01], axis=0))
    policy, history, scorer = _disabled_policy(generator, candidate_count=2)

    policy.infer(_valid_ws1_request())

    assert len(generator.calls) == 1
    assert generator.calls[0]["candidate_count"] == 1
    assert history.build_calls == 0
    assert history.clear_calls == 1
    assert history.committed_rows == []
    assert history.pending_row is None
    assert scorer.score_calls == 0


def test_disabled_diagnostics_and_actions_only_client() -> None:
    expected = _finite_chunk(seed=7)
    policy, _, _ = _disabled_policy(RecordingCandidateGenerator(expected), execution_context="robot")

    response = policy.infer(_valid_ws1_request())

    assert response["verifier_authority"] == "disabled"
    assert response["execution_context"] == "robot"
    assert response["candidate_count"] == 1
    assert response["returned_candidate_index"] == 0
    assert response["hypothetical_selected_candidate_index"] is None
    assert response["verifier_scores"] is None
    assert response["fallback_reason"] is None
    assert response["state_event"] == "disabled"

    actions_only = {"actions": response["actions"]}
    np.testing.assert_array_equal(np.asarray(actions_only["actions"]), expected)


@pytest.mark.parametrize("execution_context", ["test", "robot"])
def test_disabled_base_policy_exception_propagates(execution_context: str) -> None:
    generator = FaultingCandidateGenerator(RuntimeError("pi05 inference failed"))
    policy, _, _ = _disabled_policy(generator, execution_context=execution_context)

    with pytest.raises(RuntimeError, match="pi05 inference failed"):
        policy.infer(_valid_ws1_request())

    assert generator.calls == 1


@pytest.mark.parametrize("execution_context", ["test", "robot"])
@pytest.mark.parametrize(
    "payload",
    [
        np.zeros((10, 6), dtype=np.float64),
        np.zeros((0, 10, 7), dtype=np.float64),
        np.full((1, 10, 7), np.nan, dtype=np.float64),
    ],
)
def test_disabled_malformed_candidates_propagate_without_fallback(
    execution_context: str,
    payload: np.ndarray,
) -> None:
    policy, _, _ = _disabled_policy(
        MalformedCandidateGenerator(payload),
        execution_context=execution_context,
    )

    with pytest.raises(ValueError) as raised:
        policy.infer(_valid_ws1_request())
    assert "fallback" not in str(raised.value).lower()


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"authority": "offline"}, "unknown authority"),
        ({"execution_context": "offline"}, "unknown execution_context"),
        ({"candidate_count": 0}, "positive integer"),
        ({"candidate_count": -1}, "positive integer"),
        ({"candidate_count": 1.5}, "positive integer"),
    ],
)
def test_startup_rejects_invalid_configuration(kwargs: dict[str, Any], match: str) -> None:
    base = {
        "authority": "disabled",
        "execution_context": "test",
        "candidate_count": 1,
        "candidate_generator": RecordingCandidateGenerator(_finite_chunk()),
    }
    base.update(kwargs)
    with pytest.raises(ValueError, match=match):
        CoverPolicyWrapper(**base)  # type: ignore[arg-type]


def _reserve_port() -> int:
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _websocket_infer(host: str, port: int, request: dict[str, Any]) -> dict[str, Any]:
    """OpenPI-compatible client with short retries for in-process smoke tests."""
    import time

    import websockets.sync.client
    from openpi_client import msgpack_numpy

    packer = msgpack_numpy.Packer()
    uri = f"ws://{host}:{port}"
    deadline = time.time() + 5.0
    last_error: Exception | None = None
    while time.time() < deadline:
        try:
            conn = websockets.sync.client.connect(uri, compression=None, max_size=None)
            _metadata = msgpack_numpy.unpackb(conn.recv())
            conn.send(packer.pack(request))
            response = conn.recv()
            if isinstance(response, str):
                raise RuntimeError(response)
            payload = msgpack_numpy.unpackb(response)
            conn.close()
            return payload
        except Exception as error:  # noqa: BLE001 - retry until server accepts
            last_error = error
            time.sleep(0.05)
    raise RuntimeError(f"websocket server did not accept connections: {last_error}")


def test_direct_and_websocket_actions_match_for_disabled() -> None:
    from openpi.serving.websocket_policy_server import WebsocketPolicyServer

    expected = _finite_chunk(seed=11)
    request = _valid_ws1_request()
    direct_policy, _, _ = _disabled_policy(RecordingCandidateGenerator(expected))
    direct = direct_policy.infer(request)

    port = _reserve_port()
    served_policy, _, _ = _disabled_policy(RecordingCandidateGenerator(expected))
    server = WebsocketPolicyServer(
        policy=served_policy,
        host="127.0.0.1",
        port=port,
        metadata={"server_type": "pi05_cover"},
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    wire = _websocket_infer("127.0.0.1", port, request)
    np.testing.assert_array_equal(np.asarray(wire["actions"]), np.asarray(direct["actions"]))
    assert wire["verifier_authority"] == direct["verifier_authority"]
    assert wire["returned_candidate_index"] == direct["returned_candidate_index"]
    assert wire["fallback_reason"] is None
    assert "server_timing" in wire

def test_disabled_omits_diagnostics_when_disabled() -> None:
    expected = _finite_chunk(seed=13)
    policy = CoverPolicyWrapper(
        authority="disabled",
        execution_context="test",
        candidate_count=1,
        candidate_generator=RecordingCandidateGenerator(expected),
        diagnostics_enabled=False,
    )

    response = policy.infer(_valid_ws1_request())

    assert set(response.keys()) == {"actions"}
    np.testing.assert_array_equal(np.asarray(response["actions"]), expected)


@pytest.mark.parametrize("execution_context", ["test", "robot"])
def test_disabled_missing_policy_output_propagates(execution_context: str) -> None:
    policy, _, _ = _disabled_policy(
        MalformedCandidateGenerator(None),  # type: ignore[arg-type]
        execution_context=execution_context,
    )

    with pytest.raises(ValueError, match="no actions"):
        policy.infer(_valid_ws1_request())


def test_cover_modules_do_not_import_reference_cover_repositories() -> None:
    import ast
    from pathlib import Path

    roots = [
        Path(__file__).with_name("cover_policy_wrapper.py"),
        Path(__file__).resolve().parents[1] / "verifiers" / "candidate_generator.py",
    ]
    forbidden = ("references_dual_verifier", "cover-vla", "cover_vla", "cover_vla_polaris")
    for path in roots:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            for name in names:
                assert not any(token in name for token in forbidden), (path, name)
