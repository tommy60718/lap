"""Public ROS-independent W3 command/preflight boundary."""

from __future__ import annotations

import json
from pathlib import Path
import platform
import subprocess
from typing import Any

from lap.verifiers.cover.data import W2DatasetGateway
from lap.verifiers.cover.protocol import materialize_protocol
from lap.verifiers.cover.w3_contracts import canonical_bytes
from lap.verifiers.cover.w3_contracts import content_hash
from lap.verifiers.cover.w3_contracts import sha256_file


def _git_revision(root: Path) -> str:
    try:
        return subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def _audit_manifest(path: Path, bridge_artifact: Path) -> dict[str, Any]:
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    unsigned = dict(manifest)
    actual_hash = unsigned.pop("manifest_sha256", None)
    expected_hash = __import__("hashlib").sha256(canonical_bytes(unsigned)).hexdigest()
    if actual_hash != expected_hash:
        raise ValueError("initialization audit manifest hash mismatch")
    if manifest.get("artifact", {}).get("sha256") != sha256_file(bridge_artifact):
        raise ValueError("initialization audit manifest artifact mismatch")
    if manifest.get("artifact", {}).get("source_index") != 0:
        raise ValueError("only ensemble_components[0] may initialize W3")
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
) -> dict[str, Any]:
    output_root = Path(output_root)
    if output_root.exists():
        raise FileExistsError(f"W3 output root must be absent: {output_root}")
    dataset = W2DatasetGateway(Path(w2_root), validator_path=validator_path, fixture=fixture)
    audit = _audit_manifest(Path(audit_manifest), Path(bridge_artifact))
    receipt = {
        "schema": "osx_cover_w3_preflight_receipt_v1",
        "arguments": {
            "w2_root": str(Path(w2_root)),
            "bridge_artifact": str(Path(bridge_artifact)),
            "audit_manifest": str(Path(audit_manifest)),
            "output_root": str(output_root),
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
) -> dict[str, str]:
    dataset = W2DatasetGateway(Path(w2_root), validator_path=validator_path, fixture=fixture)
    return materialize_protocol(
        Path(protocol_dir),
        train_manifest_hash=dataset.train_manifest_hash,
        validation=[
            {"sample_id": sample.sample_id, "episode_id": sample.episode_id, "traceability": sample.condition}
            for sample in dataset.validation
        ],
        per_rank_batch_size=per_rank_batch_size,
        batch_probe_hash=batch_probe_hash,
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
