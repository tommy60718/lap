"""Strict, deterministic compatibility audit for the released Bridge weights."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from dataclasses import field
import hashlib
import json
from pathlib import Path
from typing import Any

import torch
from torch import nn

from lap.verifiers.cover.model import VerifierConfig
from lap.verifiers.cover.model import VerifierModel
from lap.verifiers.cover.w3_contracts import BACKBONE_ID
from lap.verifiers.cover.w3_contracts import BACKBONE_REVISION

AUDIT_SCHEMA = "osx_cover_bridge_audit_v1"
EXPECTED_BRIDGE_SIZE = 316400837
EXPECTED_BRIDGE_SHA256 = "8842ec9d055c6109a73e5f5f17256cb52d361fd8fc4f3653d873ce140bdf8b31"
APPROVED_SOURCE_INDEX = 0
BRIDGE_ARTIFACT_FORMAT = "torch_weights_only_ensemble_components_v1"
CANONICAL_TARGET_KIND = "canonical_production_verifier"
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
DEFAULT_SIGLIP2_SNAPSHOT_PROVENANCE = {
    "component": "siglip2_image_and_text_encoders",
    "backbone_id": BACKBONE_ID,
    "revision": BACKBONE_REVISION,
    "loader": "OpenCLIP",
    "state": "frozen",
    "bridge_transfer_excluded": True,
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

    def to_dict(self) -> dict[str, Any]:
        return {"dtype": self.dtype, "shape": list(self.shape)}


def _target_spec_from_state(state: Mapping[str, torch.Tensor]) -> dict[str, TargetTensor]:
    result: dict[str, TargetTensor] = {}
    for key, value in sorted(state.items()):
        if not isinstance(key, str):
            raise ValueError("production target inventory keys must be strings")
        if not torch.is_tensor(value):
            raise ValueError(f"production target value is not a tensor: {key}")
        result[key] = TargetTensor(tuple(value.shape), str(value.dtype).removeprefix("torch."))
    return result


def _spec_manifest(spec: Mapping[str, TargetTensor]) -> dict[str, dict[str, Any]]:
    return {key: value.to_dict() for key, value in sorted(spec.items())}


@dataclass(frozen=True)
class TargetInventory:
    """The verifier-owned initialization surface used by an audit."""

    state: Mapping[str, torch.Tensor]
    config: Mapping[str, Any]
    snapshot_provenance: Mapping[str, Any] = field(default_factory=lambda: dict(DEFAULT_SIGLIP2_SNAPSHOT_PROVENANCE))
    kind: str = CANONICAL_TARGET_KIND
    fingerprint: str = field(init=False)

    def __post_init__(self) -> None:
        state = dict(self.state)
        if any(key.startswith("backbone.") for key in state):
            raise ValueError("production target inventory cannot contain fixture or independently loaded backbone keys")
        spec = _target_spec_from_state(state)
        if self.kind != CANONICAL_TARGET_KIND:
            raise ValueError(f"unsupported target inventory kind: {self.kind}")
        snapshot = dict(self.snapshot_provenance)
        if snapshot != DEFAULT_SIGLIP2_SNAPSHOT_PROVENANCE:
            raise ValueError("production target must use the pinned SigLIP2 snapshot provenance")
        fingerprint_payload = {
            "config": dict(self.config),
            "kind": self.kind,
            "siglip2_snapshot": snapshot,
            "target": _spec_manifest(spec),
        }
        object.__setattr__(self, "state", state)
        object.__setattr__(self, "config", dict(self.config))
        object.__setattr__(self, "snapshot_provenance", snapshot)
        object.__setattr__(self, "fingerprint", sha256_bytes(canonical_json_bytes(fingerprint_payload)))

    @property
    def spec(self) -> dict[str, TargetTensor]:
        return _target_spec_from_state(self.state)


def build_production_target_inventory(config: Any | None = None) -> TargetInventory:
    """Build the canonical verifier-owned target surface without loading SigLIP2."""
    selected_config = config or VerifierConfig()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        verifier = VerifierModel(selected_config, nn.Module())
    return TargetInventory(
        state={key: value.detach().cpu() for key, value in verifier.state_dict().items()},
        config=selected_config.to_dict(),
    )


def _target_inventory_from_state(target_state: Mapping[str, torch.Tensor]) -> TargetInventory:
    backbone_keys = [key for key in target_state if isinstance(key, str) and key.startswith("backbone.")]
    fixture_keys = [key for key in backbone_keys if not key.startswith("backbone.model.")]
    if fixture_keys:
        raise ValueError("fixture or independently loaded backbone keys are not production target inventory")
    verifier_state = {key: value for key, value in target_state.items() if not key.startswith("backbone.")}
    return TargetInventory(state=verifier_state, config=VerifierConfig().to_dict())


def _target_inventory_from_model(model: nn.Module) -> TargetInventory:
    state = model.state_dict()
    config = getattr(model, "config", None)
    if config is None or not hasattr(config, "to_dict"):
        config = None
    if config is None:
        config = VerifierConfig()
    return TargetInventory(
        state={key: value for key, value in state.items() if not key.startswith("backbone.")},
        config=config.to_dict(),
    )


def _resolve_target_inventory(
    *,
    target_state: Mapping[str, torch.Tensor] | None,
    target_inventory: TargetInventory | None,
) -> TargetInventory:
    if target_state is not None and target_inventory is not None:
        raise ValueError("provide target_state or target_inventory, not both")
    if target_inventory is not None:
        return target_inventory
    if target_state is None:
        return build_production_target_inventory()
    return _target_inventory_from_state(target_state)


def _source_spec(component: Mapping[str, Any]) -> dict[str, TargetTensor | str]:
    result: dict[str, TargetTensor | str] = {}

    def visit(prefix: str, value: Any) -> None:
        if isinstance(value, Mapping):
            for child_key, child_value in sorted(value.items(), key=lambda item: str(item[0])):
                if not isinstance(child_key, str):
                    raise ValueError("Bridge source keys must be strings")
                child_name = f"{prefix}.{child_key}" if prefix else child_key
                visit(child_name, child_value)
            return
        if prefix in result:
            raise ValueError(f"ambiguous Bridge source key: {prefix}")
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
    if key.startswith("backbone."):
        return "forbidden_bridge_backbone"
    if key.startswith(("optimizer", "scheduler", "epoch", "global_step")):
        return "forbidden_bridge_training_state"
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
        or not all(isinstance(item, Mapping) for item in components)
    ):
        raise ValueError("Bridge artifact must contain exactly three ensemble_components dictionaries")
    return value


def _entry(
    *,
    side: str,
    key: str,
    state: str,
    reason: str,
    spec: TargetTensor | str | None = None,
    target_key: str | None = None,
    source_key: str | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {"key": key, "reason": reason, "side": side, "state": state}
    if isinstance(spec, TargetTensor):
        result["dtype"] = spec.dtype
        result["shape"] = list(spec.shape)
    elif spec is not None:
        result["value_type"] = spec
    if target_key is not None:
        result["target_key"] = target_key
    if source_key is not None:
        result["source_key"] = source_key
    return result


def audit_bridge_checkpoint(
    artifact_path: Path,
    *,
    target_state: Mapping[str, torch.Tensor] | None = None,
    target_inventory: TargetInventory | None = None,
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
    inventory = _resolve_target_inventory(target_state=target_state, target_inventory=target_inventory)
    target_spec = inventory.spec
    entries: list[dict[str, Any]] = []
    source_keys: set[str] = set()
    transferred_by_target: dict[str, str] = {}

    for index, component in enumerate(components):
        for key, spec in _source_spec(component).items():
            source_id = f"ensemble_components[{index}].{key}"
            source_keys.add(source_id)
            if index != APPROVED_SOURCE_INDEX:
                entries.append(
                    _entry(
                        side="source",
                        key=source_id,
                        state="rejected",
                        reason="nonselected_ensemble_member",
                        spec=spec,
                    )
                )
                continue
            if not _is_transfer_candidate(key):
                entries.append(
                    _entry(
                        side="source",
                        key=source_id,
                        state="rejected",
                        reason=_reason_for_source(key),
                        spec=spec,
                    )
                )
                continue
            target = target_spec.get(key)
            if not isinstance(spec, TargetTensor) or target is None:
                entries.append(
                    _entry(
                        side="source",
                        key=source_id,
                        state="rejected",
                        reason="target_key_missing" if target is None else "source_not_tensor",
                        spec=spec,
                        target_key=key if target is not None else None,
                    )
                )
                continue
            if spec.shape != target.shape:
                entries.append(
                    _entry(
                        side="source",
                        key=source_id,
                        state="rejected",
                        reason="shape_mismatch",
                        spec=spec,
                        target_key=key,
                    )
                )
                continue
            if spec.dtype != target.dtype:
                entries.append(
                    _entry(
                        side="source",
                        key=source_id,
                        state="rejected",
                        reason="dtype_mismatch",
                        spec=spec,
                        target_key=key,
                    )
                )
                continue
            if key in transferred_by_target:
                raise ValueError(f"ambiguous Bridge transfer target: {key}")
            transferred_by_target[key] = source_id
            entries.append(
                _entry(
                    side="source",
                    key=source_id,
                    state="transferred",
                    reason="allowlisted_exact_match",
                    spec=spec,
                    target_key=key,
                )
            )

    for key, spec in target_spec.items():
        source_key = transferred_by_target.get(key)
        if source_key is not None:
            entries.append(
                _entry(
                    side="target",
                    key=key,
                    state="transferred",
                    reason="allowlisted_exact_match",
                    spec=spec,
                    source_key=source_key,
                )
            )
        else:
            entries.append(
                _entry(
                    side="target",
                    key=key,
                    state="fresh",
                    reason="fresh_w3_target" if key not in source_keys else "source_not_accepted",
                    spec=spec,
                )
            )

    entries.sort(key=lambda item: (item["side"], item["key"]))
    manifest = {
        "schema": AUDIT_SCHEMA,
        "artifact": {
            "filename": artifact_path.name,
            "format": BRIDGE_ARTIFACT_FORMAT,
            "license_provenance": "released_CoVer_artifact_read_only_reference",
            "repository": "cover-vla/cover-vla-bridge",
            "retrieval_method": "preacquired_local_artifact_sha256_verified",
            "sha256": actual_sha256,
            "size": actual_size,
            "source_index": APPROVED_SOURCE_INDEX,
        },
        "policy": {
            "approved_source_index": APPROVED_SOURCE_INDEX,
            "transferred_prefixes": list(TRANSFERRED_PREFIXES),
            "weights_only_loader": True,
        },
        "target": {
            "config": dict(inventory.config),
            "fingerprint": inventory.fingerprint,
            "inventory_kind": inventory.kind,
            "key_count": len(target_spec),
            "keys": _spec_manifest(target_spec),
            "siglip2_snapshot": dict(inventory.snapshot_provenance),
        },
        "entries": entries,
    }
    manifest["manifest_sha256"] = sha256_bytes(canonical_json_bytes(manifest))
    return manifest


def _manifest_hash(manifest: Mapping[str, Any]) -> str:
    value = dict(manifest)
    actual = value.pop("manifest_sha256", None)
    if not isinstance(actual, str) or actual != sha256_bytes(canonical_json_bytes(value)):
        raise ValueError("Bridge audit manifest hash is invalid")
    return actual


def _inventory_matches_model(model: nn.Module, inventory: TargetInventory) -> None:
    model_inventory = _target_inventory_from_model(model)
    if model_inventory.fingerprint != inventory.fingerprint:
        raise ValueError("model target inventory does not match audit target inventory fingerprint")


def apply_audited_initialization(
    model: nn.Module,
    artifact_path: Path,
    manifest: Mapping[str, Any],
    *,
    target_inventory: TargetInventory | None = None,
) -> None:
    """Apply an audited manifest only after revalidating every transfer."""
    if not isinstance(manifest, Mapping) or manifest.get("schema") != AUDIT_SCHEMA:
        raise ValueError("unsupported Bridge audit manifest schema")
    _manifest_hash(manifest)
    artifact = manifest.get("artifact")
    if not isinstance(artifact, Mapping) or artifact.get("format") != BRIDGE_ARTIFACT_FORMAT:
        raise ValueError("unsupported Bridge artifact format in audit manifest")
    path = Path(artifact_path)
    if artifact.get("sha256") != sha256_file(path) or artifact.get("size") != path.stat().st_size:
        raise ValueError("Bridge artifact does not match audit manifest")
    inventory = target_inventory or _target_inventory_from_model(model)
    _inventory_matches_model(model, inventory)
    expected = audit_bridge_checkpoint(
        path,
        target_inventory=inventory,
        expected_sha256=str(artifact["sha256"]),
        expected_size=int(artifact["size"]),
    )
    if canonical_json_bytes(expected) != canonical_json_bytes(dict(manifest)):
        raise ValueError("audit manifest does not match current artifact or target inventory")

    loaded = _load_weights_only(path)
    component = loaded["ensemble_components"][APPROVED_SOURCE_INDEX]
    state = model.state_dict()
    copies: list[tuple[torch.Tensor, torch.Tensor]] = []
    for entry in manifest["entries"]:
        if entry["side"] != "source" or entry["state"] != "transferred":
            continue
        source_key = entry["key"].split("].", 1)[1]
        target_key = entry.get("target_key")
        source = _lookup_path(component, source_key)
        target = state.get(target_key)
        if (
            not isinstance(source, torch.Tensor)
            or not isinstance(target, torch.Tensor)
            or target_key not in inventory.state
            or tuple(source.shape) != tuple(target.shape)
            or source.dtype != target.dtype
        ):
            raise ValueError(f"audited Bridge tensor no longer matches target: {target_key}")
        copies.append((target, source))

    with torch.no_grad():
        for target, source in copies:
            target.copy_(source)
