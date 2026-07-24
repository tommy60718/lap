"""Small, dependency-light W3 contract and hashing helpers."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

W3_PROTOCOL_SCHEMA = "osx_cover_w3_protocol_v1"
W3_CHECKPOINT_SCHEMA = "osx_cover_verifier_checkpoint_v1"
W3_DEPLOYMENT_SCHEMA = "osx_cover_verifier_deployment_v1"
W2_EXPORT_SCHEMA = "osx_cover_training_export_v1"
W2_TRAIN_COUNT = 8908
W2_VALIDATION_COUNT = 1118
HISTORY_SHAPE = (10, 7)
BACKBONE_ID = "hf-hub:timm/ViT-L-16-SigLIP2-384"
BACKBONE_REVISION = "31b4df0bbf802888308ad91850c388b2870ef922"
REPRESENTATION_ID = "ur5e_cover_relative_eef_v1"
ACTION_ORDER = ("dx", "dy", "dz", "rotation_x", "rotation_y", "rotation_z", "gripper")
PHRASE_TEMPLATES = (
    "reach to the hole and insert the {shape} peg",
    "move the {shape} peg to the hole and insert it",
    "guide the {shape} peg to the hole and insert it",
    "align the {shape} peg with the hole and insert it",
    "position the {shape} peg at the hole and insert it",
    "bring the {shape} peg to the hole and insert it",
    "insert the {shape} peg into the hole",
    "place the {shape} peg into the hole",
)


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def canonical_bytes(value: Any) -> bytes:
    return canonical_json(value).encode("utf-8")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def content_hash(payload: dict[str, Any]) -> str:
    without_hash = dict(payload)
    without_hash.pop("content_hash", None)
    return sha256_bytes(canonical_bytes(without_hash))


def write_canonical_json(path: Path, payload: dict[str, Any]) -> str:
    body = dict(payload)
    body["content_hash"] = content_hash(body)
    Path(path).write_bytes(canonical_bytes(body) + b"\n")
    return body["content_hash"]


def require_history(value: Any, *, name: str = "history") -> Any:
    history = np.asarray(value, dtype=np.float32)
    if history.shape != HISTORY_SHAPE:
        raise ValueError(f"{name} must have shape [10, 7]")
    if not np.isfinite(history).all():
        raise ValueError(f"{name} must be finite")
    full_padding = np.all(history == -5.0, axis=1)
    seen_real = False
    for row_is_padding in full_padding:
        if row_is_padding and seen_real:
            raise ValueError(f"{name} padding must be leading full rows")
        seen_real = seen_real or not row_is_padding
    return history
