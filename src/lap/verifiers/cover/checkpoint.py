"""Strict resumable and inference-only W3 checkpoint bundles."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from dataclasses import field
import json
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch

from lap.verifiers.cover.scorer import ScorerCompatibility
from lap.verifiers.cover.w3_contracts import W3_CHECKPOINT_SCHEMA
from lap.verifiers.cover.w3_contracts import W3_DEPLOYMENT_SCHEMA
from lap.verifiers.cover.w3_contracts import canonical_bytes
from lap.verifiers.cover.w3_contracts import content_hash
from lap.verifiers.cover.w3_contracts import sha256_file

ACCEPTED_DEPLOYMENT_MARKER = "ACCEPTED_W3_DEPLOYMENT"
FIXTURE_DEPLOYMENT_MARKER = "FIXTURE_ONLY_W3_BUNDLE"
RESUME_KIND = "w3_exact_resume"
REQUIRED_PROGRESS_KEYS = ("epoch", "global_step", "best_metric", "world_size")
REQUIRED_CONTRACT_KEYS = (
    "resume_kind",
    "w3_01_initialization",
    "w3_03_protocol",
    "four_state_inventory",
    "model_config",
    "w2_identities",
    "environment",
)
REQUIRED_TRAINING_PAYLOAD_KEYS = (
    "schema",
    "model_state",
    "optimizer_state",
    "scheduler_state",
    "progress",
    "contract",
    "rng_states",
    "sampler_state",
)


def build_four_state_inventory() -> dict[str, list[str]]:
    """Project-owned four-state disposition inventory bound into checkpoint metadata."""

    return {
        "frozen": [
            "siglip2_image_encoder",
            "siglip2_text_encoder",
            "tokenizer_preprocessing",
            "w2_representation_normalization",
        ],
        "warm_started": [
            "text_aware_visual_extraction.temperature",
            "trajectory_encoder.compatible_tensors",
        ],
        "fresh": [
            "visual_pooling",
            "text_pooling",
            "fusion",
            "ur5e_action_projection",
            "position_buffers",
            "logit_scale",
            "optimizer",
            "scheduler",
            "rng",
            "sampler",
            "epoch_global_step_best_metric",
        ],
        "forbidden": [
            "bridge_backbone",
            "bridge_action_projection",
            "bridge_single_view_fusion",
            "bridge_training_state",
            "ensemble_components_1_2",
            "polaris_droid_policy_router_state",
            "strict_false_fallback",
        ],
    }


def build_progress(
    *,
    epoch: int,
    global_step: int,
    best_metric: float,
    world_size: int,
) -> dict[str, Any]:
    return {
        "epoch": int(epoch),
        "global_step": int(global_step),
        "best_metric": float(best_metric),
        "world_size": int(world_size),
    }


def build_checkpoint_contract(
    *,
    audit_manifest_sha256: str,
    bridge_artifact_sha256: str,
    target_fingerprint: str,
    protocol_content_hash: str,
    protocol_version: str,
    model_config: Mapping[str, Any],
    four_state_inventory: Mapping[str, Sequence[str]],
    w2_identities: Mapping[str, Any],
    environment: Mapping[str, Any],
) -> dict[str, Any]:
    inventory = {key: list(values) for key, values in four_state_inventory.items()}
    if set(inventory) != {"frozen", "warm_started", "fresh", "forbidden"}:
        raise ValueError("four_state_inventory must contain frozen/warm_started/fresh/forbidden")
    return {
        "resume_kind": RESUME_KIND,
        "w3_01_initialization": {
            "audit_manifest_sha256": audit_manifest_sha256,
            "bridge_artifact_sha256": bridge_artifact_sha256,
            "target_fingerprint": target_fingerprint,
        },
        "w3_03_protocol": {
            "protocol_content_hash": protocol_content_hash,
            "protocol_version": protocol_version,
        },
        "four_state_inventory": inventory,
        "model_config": dict(model_config),
        "w2_identities": dict(w2_identities),
        "environment": dict(environment),
    }


def select_best_checkpoint(records: Sequence[Mapping[str, Any]]) -> Path:
    """Select lowest validation loss with earliest-epoch tie breaking."""

    if not records:
        raise ValueError("best checkpoint selection requires at least one record")
    best = min(
        records,
        key=lambda record: (float(record["validation_loss"]), int(record["epoch"]), str(record["path"])),
    )
    return Path(best["path"])


def capture_rng_state() -> dict[str, Any]:
    numpy_state = np.random.get_state()
    state = {
        "python": random.getstate(),
        "numpy": {
            "algorithm": numpy_state[0],
            "keys": numpy_state[1].tolist(),
            "position": int(numpy_state[2]),
            "has_gauss": int(numpy_state[3]),
            "cached_gaussian": float(numpy_state[4]),
        },
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    numpy_state = state["numpy"]
    np.random.set_state(
        (
            numpy_state["algorithm"],
            np.asarray(numpy_state["keys"], dtype=np.uint32),
            numpy_state["position"],
            numpy_state["has_gauss"],
            numpy_state["cached_gaussian"],
        )
    )
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])


def _model_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def _require_mapping(value: Any, *, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a dictionary")
    return value


def _validate_progress(progress: Mapping[str, Any]) -> dict[str, Any]:
    progress = dict(progress)
    missing = [key for key in REQUIRED_PROGRESS_KEYS if key not in progress]
    if missing:
        raise ValueError(f"checkpoint progress missing required keys: {missing}")
    if int(progress["world_size"]) < 1:
        raise ValueError("checkpoint progress world_size must be >= 1")
    return {
        "epoch": int(progress["epoch"]),
        "global_step": int(progress["global_step"]),
        "best_metric": float(progress["best_metric"]),
        "world_size": int(progress["world_size"]),
    }


def _validate_contract(contract: Mapping[str, Any]) -> dict[str, Any]:
    contract = dict(contract)
    missing = [key for key in REQUIRED_CONTRACT_KEYS if key not in contract]
    if missing:
        raise ValueError(f"checkpoint contract missing required keys: {missing}")
    if contract.get("resume_kind") != RESUME_KIND:
        raise ValueError("checkpoint resume_kind must be w3_exact_resume, not Bridge warm-start")
    init = _require_mapping(contract["w3_01_initialization"], name="w3_01_initialization")
    for key in ("audit_manifest_sha256", "bridge_artifact_sha256", "target_fingerprint"):
        if not init.get(key):
            raise ValueError(f"checkpoint contract missing W3-01 identity: {key}")
    protocol = _require_mapping(contract["w3_03_protocol"], name="w3_03_protocol")
    for key in ("protocol_content_hash", "protocol_version"):
        if not protocol.get(key):
            raise ValueError(f"checkpoint contract missing W3-03 identity: {key}")
    inventory = _require_mapping(contract["four_state_inventory"], name="four_state_inventory")
    if set(inventory) != {"frozen", "warm_started", "fresh", "forbidden"}:
        raise ValueError("checkpoint contract four_state_inventory is incomplete")
    return contract


def _normalize_rng_states(
    *,
    world_size: int,
    rank: int,
    rng_states: Mapping[Any, Any] | None,
    complete: bool,
) -> dict[int, dict[str, Any]]:
    if rng_states is None:
        if complete and world_size != 1:
            raise ValueError("two-rank checkpoints require explicit per-rank rng_states")
        normalized = {int(rank): capture_rng_state()}
    else:
        normalized = {}
        for key, value in rng_states.items():
            normalized[int(key)] = _require_mapping(value, name=f"rng_states[{key}]")
    if complete and set(normalized) != set(range(world_size)):
        raise ValueError("checkpoint rng_states must cover every rank in world_size")
    for rank_key, state in normalized.items():
        for required in ("python", "numpy", "torch"):
            if required not in state:
                raise ValueError(f"checkpoint rng_states[{rank_key}] missing {required}")
    return normalized


def _canonical_metadata(metadata: dict[str, Any], schema: str) -> dict[str, Any]:
    result = dict(metadata)
    result["schema"] = schema
    if schema == W3_DEPLOYMENT_SCHEMA:
        result.setdefault("deployable", True)
    result["content_hash"] = content_hash(result)
    return result


def _validate_deployment_metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
    metadata = dict(metadata)
    if not metadata.get("deployable", False):
        raise ValueError("base-only or nondeployable model cannot be published as W5 deployment")
    for key in ("w3_01_initialization", "w3_03_protocol", "four_state_inventory", "scorer_compatibility"):
        if key not in metadata:
            raise ValueError(f"deployment metadata missing required key: {key}")
    inventory = _require_mapping(metadata["four_state_inventory"], name="four_state_inventory")
    if set(inventory) != {"frozen", "warm_started", "fresh", "forbidden"}:
        raise ValueError("deployment metadata four_state_inventory is incomplete")
    _require_mapping(metadata["scorer_compatibility"], name="scorer_compatibility")
    return metadata


def save_training_checkpoint(
    path: Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    progress: dict[str, Any],
    contract: dict[str, Any],
    sampler_state: dict[str, Any],
    rank: int = 0,
    rng_states: Mapping[Any, Any] | None = None,
) -> None:
    progress = _validate_progress(progress)
    contract = _validate_contract(contract)
    sampler_state = dict(sampler_state)
    normalized_rng = _normalize_rng_states(
        world_size=progress["world_size"],
        rank=rank,
        rng_states=rng_states,
        complete=True,
    )
    payload = {
        "schema": W3_CHECKPOINT_SCHEMA,
        "model_state": _model_state(model),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        "progress": progress,
        "contract": contract,
        "rng_states": normalized_rng,
        "sampler_state": sampler_state,
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def _load_payload(path: Path) -> dict[str, Any]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as error:  # pragma: no cover - torch version-specific errors
        raise ValueError("checkpoint is not a loadable W3 payload") from error
    if not isinstance(payload, dict):
        raise ValueError("checkpoint payload must be a dictionary")
    return payload


def _validate_training_payload(
    payload: Mapping[str, Any], *, expected_contract: Mapping[str, Any], model: torch.nn.Module, rank: int
) -> dict[str, Any]:
    payload = dict(payload)
    missing = [key for key in REQUIRED_TRAINING_PAYLOAD_KEYS if key not in payload]
    if missing:
        raise ValueError(f"incomplete/weights-only checkpoint missing required state: {missing}")
    if payload.get("schema") != W3_CHECKPOINT_SCHEMA:
        raise ValueError("checkpoint schema mismatch")
    contract = _validate_contract(_require_mapping(payload["contract"], name="contract"))
    if contract != dict(expected_contract):
        raise ValueError("checkpoint contract mismatch before state restore")
    progress = _validate_progress(_require_mapping(payload["progress"], name="progress"))
    rng_states = _normalize_rng_states(
        world_size=progress["world_size"],
        rank=rank,
        rng_states=_require_mapping(payload["rng_states"], name="rng_states"),
        complete=True,
    )
    if int(rank) not in rng_states:
        raise ValueError(f"checkpoint rng_states missing rank {rank}")
    if payload.get("optimizer_state") is None:
        raise ValueError("incomplete checkpoint missing optimizer_state")
    if "scheduler_state" not in payload:
        raise ValueError("incomplete checkpoint missing scheduler_state")
    expected_keys = set(model.state_dict())
    actual_keys = set(_require_mapping(payload["model_state"], name="model_state"))
    if actual_keys != expected_keys:
        raise ValueError("checkpoint model state keys mismatch")
    unexpected = set(payload) - set(REQUIRED_TRAINING_PAYLOAD_KEYS)
    if unexpected:
        raise ValueError(f"checkpoint contains unexpected keys: {sorted(unexpected)}")
    return {
        "progress": progress,
        "contract": contract,
        "rng_states": rng_states,
        "sampler_state": dict(payload["sampler_state"]),
        "model_state": payload["model_state"],
        "optimizer_state": payload["optimizer_state"],
        "scheduler_state": payload["scheduler_state"],
    }


def load_training_checkpoint(
    path: Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    expected_contract: dict[str, Any],
    rank: int = 0,
) -> dict[str, Any]:
    payload = _load_payload(Path(path))
    validated = _validate_training_payload(
        payload,
        expected_contract=_validate_contract(expected_contract),
        model=model,
        rank=rank,
    )
    model.load_state_dict(validated["model_state"], strict=True)
    optimizer.load_state_dict(validated["optimizer_state"])
    if scheduler is not None:
        if validated["scheduler_state"] is None:
            raise ValueError("checkpoint scheduler_state is missing")
        scheduler.load_state_dict(validated["scheduler_state"])
    restore_rng_state(validated["rng_states"][int(rank)])
    return {
        "progress": validated["progress"],
        "sampler_state": validated["sampler_state"],
        "contract": validated["contract"],
    }


def publish_deployment_bundle(
    root: Path,
    *,
    model: torch.nn.Module,
    metadata: dict[str, Any],
    accepted_marker: bool = True,
) -> dict[str, Any]:
    root = Path(root)
    if root.exists():
        raise FileExistsError(f"deployment output must be absent: {root}")
    staging = root.parent / f".{root.name}.staging"
    if staging.exists():
        raise FileExistsError(f"staging output already exists: {staging}")
    staging.mkdir(parents=True)
    try:
        metadata = _validate_deployment_metadata(metadata)
        metadata = _canonical_metadata(metadata, W3_DEPLOYMENT_SCHEMA)
        torch.save({"schema": W3_DEPLOYMENT_SCHEMA, "model_state": _model_state(model)}, staging / "model.pt")
        (staging / "metadata.json").write_bytes(canonical_bytes(metadata) + b"\n")
        marker = ACCEPTED_DEPLOYMENT_MARKER if accepted_marker else FIXTURE_DEPLOYMENT_MARKER
        marker_body = (
            "recorded_data_offline_only\n"
            if accepted_marker
            else "fixture_only_not_accepted\ntest_scoped_not_canonical_w3_09\n"
        )
        (staging / marker).write_text(marker_body, encoding="utf-8")
        index = {}
        for path in sorted(staging.iterdir()):
            index[path.name] = sha256_file(path)
        index_payload = {"schema": "osx_cover_w3_content_index_v1", "files": index}
        index_payload["content_hash"] = content_hash(index_payload)
        (staging / "content_index.json").write_bytes(canonical_bytes(index_payload) + b"\n")
        staging.rename(root)
    except Exception:
        for child in staging.iterdir():
            child.unlink()
        staging.rmdir()
        raise
    return {
        "root": str(root),
        "metadata_hash": metadata["content_hash"],
        "content_index_hash": content_hash(index_payload),
    }


@dataclass
class DeploymentScorer:
    model: torch.nn.Module
    compatibility: ScorerCompatibility
    preprocessing: Callable[[Any], torch.Tensor]
    device: torch.device = field(default_factory=lambda: torch.device("cpu"))

    is_fake: bool = False

    def score(self, *, base_rgb: Any, wrist_rgb: Any, instruction: str, action_histories: np.ndarray) -> np.ndarray:
        histories = np.asarray(action_histories)
        if histories.ndim != 3 or histories.shape[1:] != (10, 7) or histories.dtype != np.float32:
            raise ValueError("action_histories must be float32[M, 10, 7]")
        base = self.preprocessing(base_rgb).unsqueeze(0).to(self.device)
        wrist = self.preprocessing(wrist_rgb).unsqueeze(0).to(self.device)
        batch = histories.shape[0]
        with torch.no_grad():
            output = self.model(
                base.expand(batch, -1, -1, -1),
                wrist.expand(batch, -1, -1, -1),
                [instruction] * batch,
                torch.from_numpy(histories).to(self.device),
            )
        scores = output["semantic_to_action_logits"][0].detach().cpu().numpy().astype(np.float64)
        if scores.shape != (batch,) or not np.isfinite(scores).all():
            raise ValueError("deployment scorer must return finite rank-1 candidate scores")
        return scores


def _scorer_compatibility_from_metadata(payload: Mapping[str, Any]) -> ScorerCompatibility:
    views = payload["views"]
    action_order = payload["action_order"]
    return ScorerCompatibility(
        model_schema_version=str(payload["model_schema_version"]),
        views=tuple(views),
        preprocessing_contract=str(payload["preprocessing_contract"]),
        action_dimension=int(payload["action_dimension"]),
        action_order=tuple(action_order),
        history_length=int(payload["history_length"]),
        representation_id=str(payload["representation_id"]),
        normalization_artifact_hash=str(payload["normalization_artifact_hash"]),
        input_dtype=str(payload["input_dtype"]),
        output_shape_rank=int(payload["output_shape_rank"]),
    )


def load_deployment_bundle(
    root: Path,
    *,
    model_factory: Callable[[], torch.nn.Module],
    expected_normalization_hash: str,
    preprocessing: Callable[[Any], torch.Tensor],
) -> DeploymentScorer:
    root = Path(root)
    if not (root / ACCEPTED_DEPLOYMENT_MARKER).is_file():
        if (root / FIXTURE_DEPLOYMENT_MARKER).is_file():
            raise ValueError("fixture ACCEPTED_W3_DEPLOYMENT marker missing; fixture bundles cannot score")
        raise ValueError("deployment bundle missing ACCEPTED_W3_DEPLOYMENT marker")
    if (root / FIXTURE_DEPLOYMENT_MARKER).exists():
        raise ValueError("deployment bundle must not mix fixture and accepted markers")

    metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    if metadata.get("schema") != W3_DEPLOYMENT_SCHEMA or not metadata.get("deployable", False):
        raise ValueError("deployment bundle is not deployable W3 schema")
    if metadata.get("content_hash") != content_hash(metadata):
        raise ValueError("deployment metadata content hash mismatch")
    _validate_deployment_metadata(metadata)
    compatibility = _scorer_compatibility_from_metadata(metadata["scorer_compatibility"])
    compatibility.validate_against_expected(expected_normalization_hash=expected_normalization_hash)

    index = json.loads((root / "content_index.json").read_text(encoding="utf-8"))
    if index.get("content_hash") != content_hash(index):
        raise ValueError("deployment content index hash mismatch")
    for name, expected in index.get("files", {}).items():
        if sha256_file(root / name) != expected:
            raise ValueError(f"deployment content hash mismatch: {name}")

    payload = _load_payload(root / "model.pt")
    if payload.get("schema") != W3_DEPLOYMENT_SCHEMA:
        raise ValueError("deployment model schema mismatch")
    if "optimizer_state" in payload:
        raise ValueError("deployment bundle must not contain optimizer_state")
    model_state = _require_mapping(payload.get("model_state"), name="model_state")
    model = model_factory()
    expected_keys = set(model.state_dict())
    actual_keys = set(model_state)
    if actual_keys != expected_keys:
        raise ValueError("deployment model state keys mismatch")
    model.load_state_dict(model_state, strict=True)
    model.eval()
    return DeploymentScorer(model=model, compatibility=compatibility, preprocessing=preprocessing)
