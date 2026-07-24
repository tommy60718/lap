from __future__ import annotations

import hashlib
import json

from PIL import Image
import pytest
import torch

from lap.verifiers.cover.command import _audit_manifest
from lap.verifiers.cover.command import preflight_w3
from lap.verifiers.cover.command import staged_diagnostic
from lap.verifiers.cover.w3_contracts import canonical_bytes


def _fixture_export(tmp_path):
    root = tmp_path / "w2"
    root.mkdir()
    rows = []
    for split in ("train", "validation"):
        sample_id = f"circular_negx_demo_00:{'000000' if split == 'train' else '000001'}"
        Image.new("RGB", (8, 8), color=(10, 20, 30)).save(root / f"{split}.png")
        rows.append(
            {
                "sample_id": sample_id,
                "episode_id": f"episode_{split}",
                "split": split,
                "instruction": "reach to the hole and insert the circular peg",
                "base_image": f"{split}.png",
                "wrist_image": f"{split}.png",
                "action_history": [[-5.0] * 7] * 6 + [[0.0] * 7] * 4,
                "traceability": {"peg_shape": "circular", "approach_direction": "-x"},
            }
        )
    (root / "train_samples.json").write_text(json.dumps([rows[0]]), encoding="utf-8")
    (root / "validation_samples.json").write_text(json.dumps([rows[1]]), encoding="utf-8")
    (root / "export_manifest.json").write_text(
        json.dumps({"export_schema": "fixture", "content_hash": "fixture"}), encoding="utf-8"
    )
    return root


def _audit_fixture(tmp_path):
    artifact = tmp_path / "bridge.pt"
    torch.save({"ensemble_components": [{"text_pooling.weight": torch.ones(2, 2)}, {}, {}]}, artifact)
    unsigned = {
        "schema": "osx_cover_bridge_audit_v1",
        "artifact": {"sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(), "source_index": 0},
        "entries": [{"state": "transferred", "target_key": "x", "key": "ensemble_components[0].text_pooling.weight"}],
    }
    unsigned["manifest_sha256"] = hashlib.sha256(canonical_bytes(unsigned)).hexdigest()
    audit = tmp_path / "audit.json"
    audit.write_text(json.dumps(unsigned, sort_keys=True, separators=(",", ":")), encoding="utf-8")
    return artifact, audit


def test_preflight_validates_inputs_before_publication(tmp_path):
    w2 = _fixture_export(tmp_path)
    artifact, audit = _audit_fixture(tmp_path)
    receipt = preflight_w3(
        w2_root=w2, bridge_artifact=artifact, audit_manifest=audit, output_root=tmp_path / "absent", fixture=True
    )
    assert receipt["publication"]["accepted"] is False
    with pytest.raises(FileExistsError):
        preflight_w3(w2_root=w2, bridge_artifact=artifact, audit_manifest=audit, output_root=tmp_path, fixture=True)


def test_failed_flow_can_only_leave_explicit_diagnostic_stage(tmp_path):
    stage = staged_diagnostic({"schema": "receipt"}, output_root=tmp_path / "accepted")
    assert (stage / "preflight.json").is_file()
    assert not (tmp_path / "accepted").exists()


def test_audit_manifest_rejects_other_ensemble_member(tmp_path):
    artifact, audit = _audit_fixture(tmp_path)
    data = json.loads(audit.read_text())
    data["artifact"]["source_index"] = 1
    unsigned = dict(data)
    unsigned.pop("manifest_sha256")
    data["manifest_sha256"] = hashlib.sha256(canonical_bytes(unsigned)).hexdigest()
    audit.write_text(json.dumps(data, sort_keys=True, separators=(",", ":")), encoding="utf-8")
    with pytest.raises(ValueError, match="initialize W3"):
        _audit_manifest(audit, artifact)
