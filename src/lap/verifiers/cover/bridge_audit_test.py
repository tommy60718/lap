from __future__ import annotations

import hashlib

import pytest
import torch

from lap.verifiers.cover.bridge_audit import AUDIT_SCHEMA
from lap.verifiers.cover.bridge_audit import audit_bridge_checkpoint
from lap.verifiers.cover.bridge_audit import canonical_json_bytes


def _artifact(tmp_path, *, components=3):
    payload = {
        "ensemble_components": [
            {
                "text_pooling.weight": torch.ones(2, 2),
                "input_projection.weight": torch.ones(2, 2),
                "text_aware_visual_extraction.pos_emb": torch.ones(3, 2),
                "action_padding_value": -5.0,
            }
            for _ in range(components)
        ]
    }
    path = tmp_path / "bridge.pt"
    torch.save(payload, path)
    return path


def test_audit_classifies_selected_and_nonselected_state(tmp_path):
    path = _artifact(tmp_path)
    target = {
        "text_pooling.weight": torch.zeros(2, 2),
        "text_aware_visual_extraction.pos_emb": torch.zeros(3, 2),
        "semantic_fusion.weight": torch.zeros(2, 2),
    }
    manifest = audit_bridge_checkpoint(
        path,
        target_state=target,
        expected_size=path.stat().st_size,
        expected_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
    )
    states = {(entry["key"], entry["state"], entry["reason"]) for entry in manifest["entries"]}
    assert ("ensemble_components[0].text_pooling.weight", "transferred", "allowlisted_exact_match") in states
    assert ("ensemble_components[0].input_projection.weight", "rejected", "forbidden_single_view_fusion") in states
    assert (
        "ensemble_components[0].text_aware_visual_extraction.pos_emb",
        "rejected",
        "regenerate_deterministic_visual_position_buffer",
    ) in states
    assert ("ensemble_components[1].text_pooling.weight", "rejected", "nonselected_ensemble_member") in states
    assert ("semantic_fusion.weight", "fresh", "fresh_w3_target") in states
    assert manifest["schema"] == AUDIT_SCHEMA


def test_audit_is_byte_deterministic(tmp_path):
    path = _artifact(tmp_path)
    target = {"text_pooling.weight": torch.zeros(2, 2)}
    kwargs = {"expected_size": path.stat().st_size, "expected_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    first = audit_bridge_checkpoint(path, target_state=target, **kwargs)
    second = audit_bridge_checkpoint(path, target_state=target, **kwargs)
    assert canonical_json_bytes(first) == canonical_json_bytes(second)
    assert first["manifest_sha256"] == second["manifest_sha256"]


def test_audit_rejects_identity_mismatch(tmp_path):
    path = _artifact(tmp_path)
    with pytest.raises(ValueError, match="identity mismatch"):
        audit_bridge_checkpoint(path, target_state={}, expected_size=1, expected_sha256="0" * 64)


def test_audit_rejects_non_ensemble_checkpoint(tmp_path):
    path = tmp_path / "unsafe.pt"
    torch.save({"model": {"weight": torch.ones(1)}}, path)
    with pytest.raises(ValueError, match="ensemble_components"):
        audit_bridge_checkpoint(
            path,
            target_state={},
            expected_size=path.stat().st_size,
            expected_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        )
