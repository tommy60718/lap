"""Materialized W3 language, sampler, mismatch, and bootstrap protocol."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any

import numpy as np

from lap.verifiers.cover.w3_contracts import PHRASE_TEMPLATES
from lap.verifiers.cover.w3_contracts import W3_PROTOCOL_SCHEMA
from lap.verifiers.cover.w3_contracts import sha256_bytes
from lap.verifiers.cover.w3_contracts import write_canonical_json


def build_phrase_manifest() -> dict[str, Any]:
    phrases = [template.format(shape=shape) for shape in ("circular", "square") for template in PHRASE_TEMPLATES]
    normalized = [phrase.casefold() for phrase in phrases]
    if len(set(normalized)) != len(phrases):
        raise ValueError("approved W3 phrase bank contains duplicate phrases")
    return {
        "schema": "osx_cover_w3_rephrase_manifest_v1",
        "authoring": {"method": "manually_reviewed_static_templates", "review": "approved"},
        "templates": list(PHRASE_TEMPLATES),
        "phrases": phrases,
        "selection": "sha256(seed:epoch:sample_id) modulo variants_for_shape",
        "validation_language": "canonical_w2_instruction",
    }


def select_training_phrase(*, seed: int, epoch: int, sample_id: str, shape: str) -> str:
    if shape not in {"circular", "square"}:
        raise ValueError(f"unsupported peg shape: {shape}")
    digest = hashlib.sha256(f"{seed}:{epoch}:{sample_id}".encode()).digest()
    index = int.from_bytes(digest[:8], "big") % len(PHRASE_TEMPLATES)
    return PHRASE_TEMPLATES[index].format(shape=shape)


@dataclass(frozen=True)
class RunProtocol:
    seed: int = 4203
    sampler_seed: int = 42
    per_rank_batch_size: int = 16
    world_size: int = 1
    epochs: int = 50
    learning_rate: float = 1e-6
    warmup_epochs: int = 10
    gradient_clip_norm: float = 1.0
    bootstrap_replicates: int = 10000

    def to_dict(self, *, train_manifest_hash: str, phrase_manifest_hash: str) -> dict[str, Any]:
        return {
            "schema": W3_PROTOCOL_SCHEMA,
            "seed": self.seed,
            "sampler": {
                "type": "DistributedSampler",
                "shuffle": True,
                "seed": self.sampler_seed,
                "world_size": self.world_size,
            },
            "batch": {"per_rank": self.per_rank_batch_size, "probe_order": [64, 32, 16]},
            "optimization": {
                "epochs": self.epochs,
                "learning_rate": self.learning_rate,
                "warmup_epochs": self.warmup_epochs,
                "scheduler": "constant_after_linear_warmup",
                "gradient_clip_norm": self.gradient_clip_norm,
            },
            "evaluation": {
                "seed": self.seed,
                "bootstrap_replicates": self.bootstrap_replicates,
                "nearby_offset": 15,
                "shuffled_count": 1112,
            },
            "identities": {"train_manifest_hash": train_manifest_hash, "phrase_manifest_hash": phrase_manifest_hash},
        }


def build_shuffled_pairs(
    validation: Sequence[dict[str, Any]], *, seed: int = 4203
) -> tuple[list[dict[str, str]], list[str]]:
    rows = list(validation)
    circular = [row for row in rows if row.get("traceability", {}).get("peg_shape") == "circular"]
    square = [row for row in rows if row.get("traceability", {}).get("peg_shape") == "square"]
    if len(circular) != 562 or len(square) != 556:
        raise ValueError("canonical shuffled construction requires 562 circular and 556 square validation rows")

    def rank(prefix: str, row: dict[str, Any]) -> bytes:
        return hashlib.sha256(f"{seed}:{prefix}:{row['sample_id']}".encode()).digest()

    selected_circular = sorted(circular, key=lambda row: rank("shuffled:selection", row))[:556]
    excluded = sorted(row["sample_id"] for row in circular if row not in selected_circular)
    circular_order = sorted(selected_circular, key=lambda row: rank("shuffled:circular_order", row))
    square_order = sorted(square, key=lambda row: rank("shuffled:square_order", row))
    pairs = []
    for circular_row, square_row in zip(circular_order, square_order, strict=True):
        pairs.append({"semantic_sample_id": circular_row["sample_id"], "history_sample_id": square_row["sample_id"]})
        pairs.append({"semantic_sample_id": square_row["sample_id"], "history_sample_id": circular_row["sample_id"]})
    return pairs, excluded


def build_nearby_pairs(
    validation: Sequence[dict[str, Any]], *, offset: int = 15
) -> tuple[list[dict[str, str]], list[str]]:
    by_episode: dict[str, list[dict[str, Any]]] = {}
    for row in validation:
        by_episode.setdefault(row["episode_id"], []).append(row)
    pairs: list[dict[str, str]] = []
    excluded: list[str] = []
    for episode_rows in by_episode.values():
        for index, row in enumerate(episode_rows):
            partner = index + offset if index + offset < len(episode_rows) else index - offset
            if partner < 0 or partner >= len(episode_rows):
                excluded.append(row["sample_id"])
            else:
                pairs.append(
                    {"semantic_sample_id": row["sample_id"], "history_sample_id": episode_rows[partner]["sample_id"]}
                )
    return pairs, excluded


def build_bootstrap_indices(episode_ids: Sequence[str], *, seed: int = 4203, replicates: int = 10000) -> np.ndarray:
    ordered = sorted(set(episode_ids))
    if len(ordered) != 8:
        raise ValueError("W3 bootstrap protocol requires exactly eight validation episodes")
    rng = np.random.Generator(np.random.PCG64(seed))
    return rng.integers(0, len(ordered), size=(replicates, len(ordered)), endpoint=False, dtype=np.int64)


def materialize_protocol(
    output_dir: Path,
    *,
    train_manifest_hash: str,
    validation: Sequence[dict[str, Any]],
    per_rank_batch_size: int = 16,
    batch_probe_hash: str | None = None,
) -> dict[str, str]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    phrase_hash = write_canonical_json(output_dir / "rephrase_manifest.json", build_phrase_manifest())
    shuffled, excluded = build_shuffled_pairs(validation)
    nearby, nearby_excluded = build_nearby_pairs(validation)
    shuffled_hash = write_canonical_json(
        output_dir / "shuffled_pairs.json", {"schema": "osx_cover_w3_shuffled_pairs_v1", "pairs": shuffled}
    )
    exclusion_hash = write_canonical_json(
        output_dir / "shuffled_exclusions.json",
        {"schema": "osx_cover_w3_shuffled_exclusions_v1", "sample_ids": excluded},
    )
    nearby_hash = write_canonical_json(
        output_dir / "nearby_pairs.json",
        {"schema": "osx_cover_w3_nearby_pairs_v1", "pairs": nearby, "excluded": nearby_excluded},
    )
    bootstrap = build_bootstrap_indices([row["episode_id"] for row in validation])
    np.save(output_dir / "bootstrap_indices.npy", bootstrap, allow_pickle=False)
    bootstrap_hash = sha256_bytes((output_dir / "bootstrap_indices.npy").read_bytes())
    protocol = RunProtocol(per_rank_batch_size=per_rank_batch_size).to_dict(
        train_manifest_hash=train_manifest_hash, phrase_manifest_hash=phrase_hash
    )
    protocol["artifacts"] = {
        "rephrase_manifest": phrase_hash,
        "shuffled_pairs": shuffled_hash,
        "shuffled_exclusions": exclusion_hash,
        "nearby_pairs": nearby_hash,
        "bootstrap_indices": bootstrap_hash,
    }
    if batch_probe_hash is not None:
        protocol["artifacts"]["batch_probe_receipt"] = batch_probe_hash
    protocol_hash = write_canonical_json(output_dir / "run_protocol.json", protocol)
    return {**protocol["artifacts"], "run_protocol": protocol_hash}
