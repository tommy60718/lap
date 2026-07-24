"""Strict, deterministic compatibility audit for the released Bridge weights."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import torch

AUDIT_SCHEMA = "osx_cover_bridge_audit_v1"
EXPECTED_BRIDGE_SIZE = 316400837
EXPECTED_BRIDGE_SHA256 = "8842ec9d055c6109a73e5f5f17256cb52d361fd8fc4f3653d873ce140bdf8b31"
APPROVED_SOURCE_INDEX = 0
TRANSFERRED_PREFIXES = (
    "text_aware_visual_extraction.temperature",
    "vision_poolings.",
    "text_pooling.",
    "trajectory_encoder.",
)
REJECTED_SOURCE_REASONS = {
    "input_projection": "forbidden_single_view_fusion",
    "single_step_action_encoder": "forbidden_bridge_action_projection",
    "action_padding_value": "forbidden_padding_contract",
}


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class TargetTensor:
    shape: tuple[int, ...]
    dtype: str


def _target_spec_from_state(state: dict[str, torch.Tensor]) -> dict[str, TargetTensor]:
    return {
        key: TargetTensor(tuple(value.shape), str(value.dtype).removeprefix("torch."))
        for key, value in sorted(state.items())
    }


def _source_spec(component: dict[str, Any]) -> dict[str, TargetTensor | str]:
    result: dict[str, TargetTensor | str] = {}

    def visit(prefix: str, value: Any) -> None:
        if isinstance(value, Mapping):
            for child_key, child_value in sorted(value.items(), key=lambda item: str(item[0])):
                child_name = f"{prefix}.{child_key}" if prefix else str(child_key)
                visit(child_name, child_value)
            return
        if torch.is_tensor(value):
            result[prefix] = TargetTensor(tuple(value.shape), str(value.dtype).removeprefix("torch."))
        elif isinstance(value, (float, int, str, bool)):
            result[prefix] = type(value).__name__
        else:
            raise ValueError(f"unsupported Bridge state value at {prefix}: {type(value).__name__}")

    visit("", component)
    return result


def _is_transfer_candidate(key: str) -> bool:
    return any(key == prefix or key.startswith(prefix) for prefix in TRANSFERRED_PREFIXES)


def _reason_for_source(key: str) -> str:
    if key in REJECTED_SOURCE_REASONS:
        return REJECTED_SOURCE_REASONS[key]
    if key == "text_aware_visual_extraction.pos_emb":
        return "regenerate_deterministic_visual_position_buffer"
    if key.startswith("input_projection"):
        return "forbidden_single_view_fusion"
    if key.startswith("single_step_action_encoder"):
        return "forbidden_bridge_action_projection"
    return "not_in_transfer_allowlist"


def _lookup_path(mapping: Mapping[str, Any], dotted_key: str) -> Any:
    if dotted_key in mapping:
        return mapping[dotted_key]
    parts = dotted_key.split(".")
    for index in range(1, len(parts) + 1):
        head = ".".join(parts[:index])
        if head not in mapping:
            continue
        value = mapping[head]
        if index == len(parts):
            return value
        if isinstance(value, Mapping):
            return _lookup_path(value, ".".join(parts[index:]))
    raise KeyError(dotted_key)


def _load_weights_only(path: Path) -> dict[str, Any]:
    try:
        value = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as error:  # pragma: no cover - exact torch error varies by version
        raise ValueError("Bridge artifact is not a safe weights-only checkpoint") from error
    if not isinstance(value, dict) or set(value) != {"ensemble_components"}:
        raise ValueError("Bridge artifact must be a weights-only dictionary with ensemble_components only")
    components = value["ensemble_components"]
    if (
        not isinstance(components, list)
        or len(components) != 3
        or not all(isinstance(item, dict) for item in components)
    ):
        raise ValueError("Bridge artifact must contain exactly three ensemble_components dictionaries")
    return value


def audit_bridge_checkpoint(
    artifact_path: Path,
    *,
    target_state: dict[str, torch.Tensor],
    expected_sha256: str = EXPECTED_BRIDGE_SHA256,
    expected_size: int = EXPECTED_BRIDGE_SIZE,
) -> dict[str, Any]:
    """Audit a Bridge artifact without applying any tensor to a model."""
    artifact_path = Path(artifact_path)
    if not artifact_path.is_file():
        raise FileNotFoundError(artifact_path)
    actual_size = artifact_path.stat().st_size
    actual_sha256 = sha256_file(artifact_path)
    if actual_size != expected_size or actual_sha256 != expected_sha256:
        raise ValueError(f"Bridge artifact identity mismatch: size={actual_size}, sha256={actual_sha256}")
    loaded = _load_weights_only(artifact_path)
    components = loaded["ensemble_components"]
    target_spec = _target_spec_from_state(target_state)
    entries: list[dict[str, Any]] = []

    source_keys = set()
    for index, component in enumerate(components):
        for key, spec in _source_spec(component).items():
            source_id = f"ensemble_components[{index}].{key}"
            source_keys.add(source_id)
            if index != APPROVED_SOURCE_INDEX:
                entries.append(
                    {"side": "source", "key": source_id, "state": "rejected", "reason": "nonselected_ensemble_member"}
                )
                continue
            if not _is_transfer_candidate(key):
                entries.append(
                    {"side": "source", "key": source_id, "state": "rejected", "reason": _reason_for_source(key)}
                )
                continue
            target = target_spec.get(key)
            if not isinstance(spec, TargetTensor) or target is None:
                entries.append(
                    {"side": "source", "key": source_id, "state": "rejected", "reason": "target_key_missing"}
                )
                continue
            if spec.shape != target.shape:
                entries.append(
                    {
                        "side": "source",
                        "key": source_id,
                        "target_key": key,
                        "state": "rejected",
                        "reason": "shape_mismatch",
                    }
                )
                continue
            if spec.dtype != target.dtype:
                entries.append(
                    {
                        "side": "source",
                        "key": source_id,
                        "target_key": key,
                        "state": "rejected",
                        "reason": "dtype_mismatch",
                    }
                )
                continue
            entries.append(
                {
                    "side": "source",
                    "key": source_id,
                    "target_key": key,
                    "state": "transferred",
                    "reason": "allowlisted_exact_match",
                }
            )

    transferred_targets = {entry["target_key"] for entry in entries if entry["state"] == "transferred"}
    for key in target_spec:
        if key in transferred_targets:
            continue
        reason = "fresh_w3_target"
        if key in source_keys:
            reason = "source_not_accepted"
        entries.append({"side": "target", "key": key, "state": "fresh", "reason": reason})

    entries.sort(key=lambda item: (item["side"], item["key"]))
    manifest = {
        "schema": AUDIT_SCHEMA,
        "artifact": {
            "path_name": artifact_path.name,
            "size": actual_size,
            "sha256": actual_sha256,
            "format": "torch_weights_only_ensemble_components_v1",
            "source_index": APPROVED_SOURCE_INDEX,
            "repository": "cover-vla/cover-vla-bridge",
            "retrieval_method": "preacquired_local_artifact_sha256_verified",
            "license_provenance": "released_CoVer_artifact_read_only_reference",
        },
        "target": {"key_count": len(target_spec)},
        "entries": entries,
    }
    manifest["manifest_sha256"] = sha256_bytes(canonical_json_bytes(manifest))
    return manifest


def apply_audited_initialization(model: torch.nn.Module, artifact_path: Path, manifest: dict[str, Any]) -> None:
    """Apply exactly the transferred entries from a previously audited manifest."""
    if manifest.get("schema") != AUDIT_SCHEMA:
        raise ValueError("unsupported Bridge audit manifest schema")
    artifact_sha256 = manifest.get("artifact", {}).get("sha256")
    if artifact_sha256 != sha256_file(Path(artifact_path)):
        raise ValueError("Bridge artifact does not match audit manifest")
    loaded = _load_weights_only(Path(artifact_path))
    component = loaded["ensemble_components"][APPROVED_SOURCE_INDEX]
    state = model.state_dict()
    for entry in manifest.get("entries", []):
        if entry.get("state") != "transferred":
            continue
        source_key = entry["key"].split("].", 1)[1]
        target_key = entry["target_key"]
        source = _lookup_path(component, source_key)
        target = state.get(target_key)
        if (
            not torch.is_tensor(source)
            or target is None
            or source.shape != target.shape
            or source.dtype != target.dtype
        ):
            raise ValueError(f"audited Bridge tensor no longer matches target: {target_key}")
        target.copy_(source)
