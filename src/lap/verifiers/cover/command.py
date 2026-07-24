"""Public ROS-independent W3 command/preflight boundary."""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
from pathlib import Path
import platform
import subprocess
from typing import Any

from lap.verifiers.cover.bridge_audit import AUDIT_SCHEMA
from lap.verifiers.cover.bridge_audit import audit_bridge_checkpoint
from lap.verifiers.cover.bridge_audit import build_production_target_inventory
from lap.verifiers.cover.data import W2DatasetGateway
from lap.verifiers.cover.protocol import materialize_protocol
from lap.verifiers.cover.protocol import validate_protocol_directory
from lap.verifiers.cover.w3_contracts import canonical_bytes
from lap.verifiers.cover.w3_contracts import content_hash
from lap.verifiers.cover.w3_contracts import sha256_file


def _git_revision(root: Path) -> str:
    try:
        return subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def _require_mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"initialization audit manifest {name} must be an object")
    return value


def _validate_audit_manifest_structure(manifest: Mapping[str, Any]) -> None:
    required = {"schema", "artifact", "policy", "target", "entries", "manifest_sha256"}
    optional = set()
    keys = set(manifest)
    missing = required - keys
    unexpected = keys - required - optional
    if missing:
        raise ValueError(f"initialization audit manifest missing fields: {sorted(missing)}")
    if unexpected:
        raise ValueError(f"initialization audit manifest has unexpected fields: {sorted(unexpected)}")
    if manifest.get("schema") != AUDIT_SCHEMA:
        raise ValueError("initialization audit manifest schema is unsupported")

    policy = _require_mapping(manifest["policy"], name="policy")
    if policy.get("approved_source_index") != 0 or policy.get("weights_only_loader") is not True:
        raise ValueError("initialization audit manifest policy is invalid")
    if not isinstance(policy.get("transferred_prefixes"), list) or not policy["transferred_prefixes"]:
        raise ValueError("initialization audit manifest policy is missing transfer allowlist")

    artifact = _require_mapping(manifest["artifact"], name="artifact")
    artifact_required = {
        "filename",
        "size",
        "sha256",
        "format",
        "source_index",
        "repository",
        "retrieval_method",
        "license_provenance",
    }
    missing_artifact = artifact_required - set(artifact)
    if missing_artifact:
        raise ValueError(f"initialization audit manifest artifact missing fields: {sorted(missing_artifact)}")
    if not isinstance(artifact["size"], int) or artifact["size"] <= 0:
        raise ValueError("initialization audit manifest artifact size is invalid")
    if not isinstance(artifact["sha256"], str) or len(artifact["sha256"]) != 64:
        raise ValueError("initialization audit manifest artifact sha256 is invalid")
    if not isinstance(artifact["source_index"], int):
        raise ValueError("initialization audit manifest artifact source_index is invalid")

    target = _require_mapping(manifest["target"], name="target")
    target_required = {"config", "fingerprint", "inventory_kind", "key_count", "keys", "siglip2_snapshot"}
    missing_target = target_required - set(target)
    if missing_target:
        raise ValueError(f"initialization audit manifest target missing fields: {sorted(missing_target)}")
    fingerprint = target["fingerprint"]
    target_key_specs = target["keys"]
    if target.get("inventory_kind") != "canonical_production_verifier":
        raise ValueError("initialization audit manifest target is not the canonical production inventory")
    if not isinstance(fingerprint, str) or not fingerprint or "fixture" in fingerprint.lower():
        raise ValueError("initialization audit manifest target is not a production fingerprint")
    if not isinstance(target["key_count"], int) or target["key_count"] <= 0:
        raise ValueError("initialization audit manifest target key_count is invalid")
    if (
        not isinstance(target_key_specs, Mapping)
        or len(target_key_specs) != target["key_count"]
        or any(
            not isinstance(key, str)
            or not key
            or "tinyfrozenbackbone" in key.lower()
            or not isinstance(spec, Mapping)
            or set(spec) != {"dtype", "shape"}
            for key, spec in target_key_specs.items()
        )
        or list(target_key_specs) != sorted(target_key_specs)
    ):
        raise ValueError("initialization audit manifest target inventory is invalid or fixture-derived")

    provenance = _require_mapping(target["siglip2_snapshot"], name="target.siglip2_snapshot")
    for field in ("backbone_id", "revision", "state", "bridge_transfer_excluded"):
        if field not in provenance:
            raise ValueError(f"initialization audit manifest SigLIP2 provenance missing {field}")
    if provenance["state"] != "frozen" or provenance["bridge_transfer_excluded"] is not True:
        raise ValueError("initialization audit manifest SigLIP2 provenance is not frozen/excluded")

    entries = manifest["entries"]
    if not isinstance(entries, list) or not entries:
        raise ValueError("initialization audit manifest entries must be a non-empty list")
    source_keys: set[str] = set()
    target_keys_from_entries: set[str] = set()
    transferred_target_keys: set[str] = set()
    for index, entry_value in enumerate(entries):
        entry = _require_mapping(entry_value, name=f"entries[{index}]")
        for field in ("side", "key", "state", "reason"):
            if not isinstance(entry.get(field), str) or not entry[field]:
                raise ValueError(f"initialization audit manifest entry {index} missing {field}")
        side = entry["side"]
        state = entry["state"]
        if side not in {"source", "target"}:
            raise ValueError(f"initialization audit manifest entry {index} has invalid side")
        if state not in {"transferred", "fresh", "rejected"}:
            raise ValueError(f"initialization audit manifest entry {index} has invalid state")
        if side == "source":
            if entry["key"] in source_keys:
                raise ValueError("initialization audit manifest repeats a source key")
            source_keys.add(entry["key"])
            if state == "fresh":
                raise ValueError("initialization audit manifest cannot mark a source fresh")
            if state == "transferred":
                if not isinstance(entry.get("target_key"), str) or not entry["target_key"]:
                    raise ValueError("transferred initialization entry missing target_key")
                transferred_target_keys.add(entry["target_key"])
        else:
            if entry["key"] in target_keys_from_entries:
                raise ValueError("initialization audit manifest repeats a target key")
            target_keys_from_entries.add(entry["key"])
            if state == "rejected":
                raise ValueError("initialization audit manifest cannot reject a target")
    target_keys = set(target_key_specs)
    if target_keys_from_entries != target_keys:
        raise ValueError("initialization audit manifest target inventory disagrees with entries")
    if not transferred_target_keys <= target_keys_from_entries:
        raise ValueError("initialization audit manifest transfers to an unknown target key")
    target_entries_by_key = {entry["key"]: entry for entry in entries if entry["side"] == "target"}
    for target_key in transferred_target_keys:
        if target_entries_by_key[target_key]["state"] != "transferred":
            raise ValueError("initialization audit manifest transfer disposition disagrees with target entry")
    if not any(entry.get("state") == "transferred" for entry in entries):
        raise ValueError("audit manifest cannot silently fall back to all-fresh initialization")


def _audit_manifest(path: Path, bridge_artifact: Path, *, allow_fixture: bool = False) -> dict[str, Any]:
    try:
        manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("initialization audit manifest is not valid JSON") from error
    _require_mapping(manifest, name="root")
    _validate_audit_manifest_structure(manifest)
    unsigned = dict(manifest)
    actual_hash = unsigned.pop("manifest_sha256", None)
    expected_hash = hashlib.sha256(canonical_bytes(unsigned)).hexdigest()
    if actual_hash != expected_hash:
        raise ValueError("initialization audit manifest hash mismatch")
    artifact = manifest["artifact"]
    if artifact.get("sha256") != sha256_file(bridge_artifact):
        raise ValueError("initialization audit manifest artifact mismatch")
    if artifact.get("size") != Path(bridge_artifact).stat().st_size:
        raise ValueError("initialization audit manifest artifact size mismatch")
    if artifact.get("format") != "torch_weights_only_ensemble_components_v1":
        raise ValueError("initialization audit manifest artifact format mismatch")
    if artifact.get("source_index") != 0:
        raise ValueError("only ensemble_components[0] may initialize W3")
    if not allow_fixture:
        expected = audit_bridge_checkpoint(
            Path(bridge_artifact),
            target_inventory=build_production_target_inventory(),
        )
        if canonical_bytes(expected) != canonical_bytes(dict(manifest)):
            raise ValueError("initialization audit manifest does not match the canonical production audit")
    if not any(entry.get("state") == "transferred" for entry in manifest.get("entries", [])):
        raise ValueError("audit manifest cannot silently fall back to all-fresh initialization")
    return manifest


def preflight_w3(
    *,
    w2_root: Path,
    bridge_artifact: Path,
    audit_manifest: Path,
    output_root: Path,
    validator_path: Path | None = None,
    fixture: bool = False,
    protocol_dir: Path | None = None,
) -> dict[str, Any]:
    output_root = Path(output_root)
    if output_root.exists():
        raise FileExistsError(f"W3 output root must be absent: {output_root}")
    dataset = W2DatasetGateway(Path(w2_root), validator_path=validator_path, fixture=fixture)
    audit = _audit_manifest(Path(audit_manifest), Path(bridge_artifact), allow_fixture=fixture)
    protocol_receipt = None
    if protocol_dir is not None:
        protocol_receipt = validate_protocol_directory(
            Path(protocol_dir),
            require_complete=True,
            expected_audit_manifest_sha256=audit["manifest_sha256"],
            expected_target_inventory_fingerprint=audit["target"]["fingerprint"],
        )
        recorded_train_hash = protocol_receipt["protocol"]["identities"].get("train_manifest_hash")
        if recorded_train_hash != dataset.train_manifest_hash:
            raise ValueError("protocol train manifest identity does not match the W2 package")
    receipt = {
        "schema": "osx_cover_w3_preflight_receipt_v1",
        "arguments": {
            "w2_root": str(Path(w2_root)),
            "bridge_artifact": str(Path(bridge_artifact)),
            "audit_manifest": str(Path(audit_manifest)),
            "output_root": str(output_root),
            "protocol_dir": None if protocol_dir is None else str(Path(protocol_dir)),
        },
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "lap_revision": _git_revision(Path(__file__).resolve().parents[3]),
        },
        "w2": {
            "export_schema": dataset.validation_receipt.get("export_schema"),
            "train_count": len(dataset.train),
            "validation_count": len(dataset.validation),
            "manifest": dataset.validation_receipt.get("content_hash"),
        },
        "initialization": {
            "manifest_sha256": audit["manifest_sha256"],
            "artifact_sha256": audit["artifact"]["sha256"],
            "source_index": audit["artifact"]["source_index"],
        },
        "protocol": None
        if protocol_receipt is None
        else {
            "run_protocol_hash": protocol_receipt["protocol"]["content_hash"],
            "artifact_hashes": protocol_receipt["protocol"]["artifacts"],
        },
        "publication": {"accepted": False, "output_absent_at_preflight": True},
    }
    receipt["content_hash"] = content_hash(receipt)
    return receipt


def run_protocol_mode(
    *,
    w2_root: Path,
    protocol_dir: Path,
    validator_path: Path | None = None,
    fixture: bool = False,
    per_rank_batch_size: int = 16,
    batch_probe_hash: str | None = None,
    batch_probe_receipt: Path | None = None,
) -> dict[str, str]:
    dataset = W2DatasetGateway(Path(w2_root), validator_path=validator_path, fixture=fixture)
    receipt = None
    if batch_probe_receipt is not None:
        receipt = json.loads(Path(batch_probe_receipt).read_text(encoding="utf-8"))
        batch_probe_hash = receipt.get("content_hash")
    return materialize_protocol(
        Path(protocol_dir),
        train_manifest_hash=dataset.train_manifest_hash,
        validation=[
            {"sample_id": sample.sample_id, "episode_id": sample.episode_id, "traceability": sample.condition}
            for sample in dataset.validation
        ],
        per_rank_batch_size=per_rank_batch_size,
        batch_probe_hash=batch_probe_hash,
        batch_probe_receipt=receipt,
    )


def staged_diagnostic(receipt: dict[str, Any], *, output_root: Path) -> Path:
    output_root = Path(output_root)
    if output_root.exists():
        raise FileExistsError(output_root)
    staging = output_root.parent / f".{output_root.name}.diagnostic"
    if staging.exists():
        raise FileExistsError(staging)
    staging.mkdir(parents=True)
    (staging / "preflight.json").write_bytes(canonical_bytes(receipt) + b"\n")
    return staging
