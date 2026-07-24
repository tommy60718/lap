from __future__ import annotations

import copy
import hashlib

import pytest
import torch
from torch import nn

from lap.verifiers.cover.bridge_audit import AUDIT_SCHEMA
from lap.verifiers.cover.bridge_audit import TargetInventory
from lap.verifiers.cover.bridge_audit import apply_audited_initialization
from lap.verifiers.cover.bridge_audit import audit_bridge_checkpoint
from lap.verifiers.cover.bridge_audit import build_production_target_inventory
from lap.verifiers.cover.bridge_audit import canonical_json_bytes
from lap.verifiers.cover.bridge_audit import sha256_bytes


class _ToyVerifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.text_pooling = nn.Linear(2, 2, bias=False)
        self.semantic_fusion = nn.Linear(2, 2, bias=False)


def _artifact(tmp_path, *, components=3, extra=None):
    selected = {
        "text_pooling.weight": torch.ones(2, 2),
        "input_projection.weight": torch.ones(2, 2),
        "text_aware_visual_extraction.pos_emb": torch.ones(3, 2),
        "action_padding_value": -5.0,
    }
    if extra:
        selected.update(extra)
    payload = {"ensemble_components": [copy.deepcopy(selected) for _ in range(components)]}
    path = tmp_path / "bridge.pt"
    torch.save(payload, path)
    return path


def _inventory(target_state):
    return TargetInventory(
        state=target_state,
        config={"test_config": "canonical-production-seam"},
    )


def _identity_kwargs(path):
    return {
        "expected_size": path.stat().st_size,
        "expected_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def test_audit_classifies_each_source_and_target_once(tmp_path):
    path = _artifact(tmp_path)
    target = {
        "text_pooling.weight": torch.zeros(2, 2),
        "text_aware_visual_extraction.pos_emb": torch.zeros(3, 2),
        "semantic_fusion.weight": torch.zeros(2, 2),
    }
    manifest = audit_bridge_checkpoint(path, target_inventory=_inventory(target), **_identity_kwargs(path))

    source_entries = [entry for entry in manifest["entries"] if entry["side"] == "source"]
    target_entries = [entry for entry in manifest["entries"] if entry["side"] == "target"]
    assert len({entry["key"] for entry in source_entries}) == len(source_entries)
    assert len({entry["key"] for entry in target_entries}) == len(target_entries)
    assert {entry["key"] for entry in target_entries} == set(target)
    assert (
        "ensemble_components[0].text_pooling.weight",
        "transferred",
        "allowlisted_exact_match",
    ) in {(entry["key"], entry["state"], entry["reason"]) for entry in source_entries}
    assert (
        "text_pooling.weight",
        "transferred",
        "allowlisted_exact_match",
    ) in {(entry["key"], entry["state"], entry["reason"]) for entry in target_entries}
    assert (
        "ensemble_components[0].input_projection.weight",
        "rejected",
        "forbidden_single_view_fusion",
    ) in {(entry["key"], entry["state"], entry["reason"]) for entry in source_entries}
    assert (
        "ensemble_components[0].text_aware_visual_extraction.pos_emb",
        "rejected",
        "regenerate_deterministic_visual_position_buffer",
    ) in {(entry["key"], entry["state"], entry["reason"]) for entry in source_entries}
    assert (
        "ensemble_components[1].text_pooling.weight",
        "rejected",
        "nonselected_ensemble_member",
    ) in {(entry["key"], entry["state"], entry["reason"]) for entry in source_entries}
    assert manifest["schema"] == AUDIT_SCHEMA


def test_production_target_inventory_has_no_fixture_backbone_and_stable_fingerprint():
    first = build_production_target_inventory()
    second = build_production_target_inventory()

    assert first.kind == "canonical_production_verifier"
    assert not any(key.startswith("backbone.") for key in first.state)
    assert first.snapshot_provenance["revision"] == "31b4df0bbf802888308ad91850c388b2870ef922"
    assert first.fingerprint == second.fingerprint


def test_audit_is_byte_deterministic(tmp_path):
    path = _artifact(tmp_path)
    target = {"text_pooling.weight": torch.zeros(2, 2)}
    first = audit_bridge_checkpoint(path, target_inventory=_inventory(target), **_identity_kwargs(path))
    second = audit_bridge_checkpoint(path, target_inventory=_inventory(target), **_identity_kwargs(path))

    assert canonical_json_bytes(first) == canonical_json_bytes(second)
    assert first["manifest_sha256"] == second["manifest_sha256"]


def test_audit_rejects_fixture_target(tmp_path):
    path = _artifact(tmp_path)
    with pytest.raises(ValueError, match=r"fixture|independently loaded backbone"):
        audit_bridge_checkpoint(
            path,
            target_state={"backbone.image_projection.weight": torch.zeros(2, 2)},
            **_identity_kwargs(path),
        )


def test_audit_rejects_identity_mismatch(tmp_path):
    path = _artifact(tmp_path)
    with pytest.raises(ValueError, match="identity mismatch"):
        audit_bridge_checkpoint(path, target_inventory=_inventory({}), expected_size=1, expected_sha256="0" * 64)


def test_audit_rejects_non_ensemble_checkpoint(tmp_path):
    path = tmp_path / "unsafe.pt"
    torch.save({"model": {"weight": torch.ones(1)}}, path)
    with pytest.raises(ValueError, match="ensemble_components"):
        audit_bridge_checkpoint(path, target_inventory=_inventory({}), **_identity_kwargs(path))


def test_audit_rejects_ambiguous_flattened_source_key(tmp_path):
    path = _artifact(tmp_path, extra={"text_pooling": {"weight": torch.ones(2, 2)}})
    with pytest.raises(ValueError, match="ambiguous"):
        audit_bridge_checkpoint(
            path,
            target_inventory=_inventory({"text_pooling.weight": torch.zeros(2, 2)}),
            **_identity_kwargs(path),
        )


def test_audit_rejects_dtype_mismatch_before_transfer(tmp_path):
    path = _artifact(tmp_path)
    manifest = audit_bridge_checkpoint(
        path,
        target_inventory=_inventory({"text_pooling.weight": torch.zeros(2, 2, dtype=torch.float64)}),
        **_identity_kwargs(path),
    )
    entry = next(entry for entry in manifest["entries"] if entry["key"] == "ensemble_components[0].text_pooling.weight")
    assert entry["state"] == "rejected"
    assert entry["reason"] == "dtype_mismatch"


def test_apply_rejects_changed_target_inventory_before_copy(tmp_path):
    path = _artifact(tmp_path)
    target = {"text_pooling.weight": torch.zeros(2, 2), "semantic_fusion.weight": torch.zeros(2, 2)}
    manifest = audit_bridge_checkpoint(path, target_inventory=_inventory(target), **_identity_kwargs(path))
    changed = _inventory({**target, "new_target.weight": torch.zeros(2, 2)})
    model = _ToyVerifier()
    before = {key: value.detach().clone() for key, value in model.state_dict().items()}

    with pytest.raises(ValueError, match=r"target inventory|fingerprint"):
        apply_audited_initialization(model, path, manifest, target_inventory=changed)

    assert all(torch.equal(before[key], value) for key, value in model.state_dict().items())


def test_apply_validates_all_transfers_before_copying(tmp_path):
    path = _artifact(tmp_path)
    target = {"text_pooling.weight": torch.zeros(2, 2), "semantic_fusion.weight": torch.zeros(2, 2)}
    manifest = audit_bridge_checkpoint(path, target_inventory=_inventory(target), **_identity_kwargs(path))
    tampered = copy.deepcopy(manifest)
    transfer = next(
        entry for entry in tampered["entries"] if entry.get("state") == "transferred" and entry["side"] == "source"
    )
    transfer["target_key"] = "missing.weight"
    target_transfer = next(
        entry for entry in tampered["entries"] if entry.get("state") == "transferred" and entry["side"] == "target"
    )
    target_transfer["key"] = "missing.weight"
    tampered["manifest_sha256"] = sha256_bytes(
        canonical_json_bytes({key: value for key, value in tampered.items() if key != "manifest_sha256"})
    )
    model = _ToyVerifier()
    before = {key: value.detach().clone() for key, value in model.state_dict().items()}

    with pytest.raises(ValueError, match=r"target|missing"):
        apply_audited_initialization(model, path, tampered, target_inventory=_inventory(target))

    assert all(torch.equal(before[key], value) for key, value in model.state_dict().items())
