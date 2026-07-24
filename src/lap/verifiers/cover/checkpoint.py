"""Strict resumable and inference-only W3 checkpoint bundles."""

from __future__ import annotations

from collections.abc import Callable
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


def _canonical_metadata(metadata: dict[str, Any], schema: str) -> dict[str, Any]:
    result = dict(metadata)
    result["schema"] = schema
    if schema == W3_DEPLOYMENT_SCHEMA:
        result.setdefault("deployable", True)
    result["content_hash"] = content_hash(result)
    return result


def save_training_checkpoint(
    path: Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    progress: dict[str, Any],
    contract: dict[str, Any],
    sampler_state: dict[str, Any],
) -> None:
    payload = {
        "schema": W3_CHECKPOINT_SCHEMA,
        "model_state": _model_state(model),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        "progress": dict(progress),
        "contract": dict(contract),
        "rng_state": capture_rng_state(),
        "sampler_state": dict(sampler_state),
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def _load_payload(path: Path) -> dict[str, Any]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as error:  # pragma: no cover - torch version-specific errors
        raise ValueError("checkpoint is not a safe W3 weights-only payload") from error
    if not isinstance(payload, dict):
        raise ValueError("checkpoint payload must be a dictionary")
    return payload


def load_training_checkpoint(
    path: Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    expected_contract: dict[str, Any],
) -> dict[str, Any]:
    payload = _load_payload(Path(path))
    if payload.get("schema") != W3_CHECKPOINT_SCHEMA:
        raise ValueError("checkpoint schema mismatch")
    if payload.get("contract") != expected_contract:
        raise ValueError("checkpoint contract mismatch before state restore")
    expected_keys = set(model.state_dict())
    actual_keys = set(payload.get("model_state", {}))
    if actual_keys != expected_keys:
        raise ValueError("checkpoint model state keys mismatch")
    model.load_state_dict(payload["model_state"], strict=True)
    optimizer.load_state_dict(payload["optimizer_state"])
    if scheduler is not None:
        scheduler.load_state_dict(payload["scheduler_state"])
    restore_rng_state(payload["rng_state"])
    return {"progress": payload["progress"], "sampler_state": payload["sampler_state"], "contract": payload["contract"]}


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
        metadata = _canonical_metadata(metadata, W3_DEPLOYMENT_SCHEMA)
        if not metadata.get("deployable", False):
            raise ValueError("base-only or nondeployable model cannot be published as W5 deployment")
        torch.save({"schema": W3_DEPLOYMENT_SCHEMA, "model_state": _model_state(model)}, staging / "model.pt")
        (staging / "metadata.json").write_bytes(canonical_bytes(metadata) + b"\n")
        marker = "ACCEPTED_W3_DEPLOYMENT" if accepted_marker else "FIXTURE_ONLY_W3_BUNDLE"
        marker_body = "recorded_data_offline_only\n" if accepted_marker else "fixture_only_not_accepted\n"
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


def load_deployment_bundle(
    root: Path,
    *,
    model_factory: Callable[[], torch.nn.Module],
    expected_normalization_hash: str,
    preprocessing: Callable[[Any], torch.Tensor],
) -> DeploymentScorer:
    root = Path(root)
    metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    if metadata.get("schema") != W3_DEPLOYMENT_SCHEMA or not metadata.get("deployable", False):
        raise ValueError("deployment bundle is not deployable W3 schema")
    if metadata.get("content_hash") != content_hash(metadata):
        raise ValueError("deployment metadata content hash mismatch")
    compatibility = ScorerCompatibility(**metadata["scorer_compatibility"])
    compatibility.validate_against_expected(expected_normalization_hash=expected_normalization_hash)
    model = model_factory()
    payload = _load_payload(root / "model.pt")
    if payload.get("schema") != W3_DEPLOYMENT_SCHEMA:
        raise ValueError("deployment model schema mismatch")
    model.load_state_dict(payload["model_state"], strict=True)
    model.eval()
    index = json.loads((root / "content_index.json").read_text(encoding="utf-8"))
    if index.get("content_hash") != content_hash(index):
        raise ValueError("deployment content index hash mismatch")
    for name, expected in index.get("files", {}).items():
        if sha256_file(root / name) != expected:
            raise ValueError(f"deployment content hash mismatch: {name}")
    return DeploymentScorer(model=model, compatibility=compatibility, preprocessing=preprocessing)
