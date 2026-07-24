"""Materialized, hash-checked W3 accepted-run protocol.

This module owns protocol identity only.  It deliberately does not choose a
batch size from a fixture model: a canonical receipt must come from an
injectable two-rank forward/backward probe of the pinned two-view model.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any
import unicodedata

import numpy as np
import torch
from torch.utils.data.distributed import DistributedSampler

from lap.verifiers.cover.w3_contracts import BACKBONE_ID
from lap.verifiers.cover.w3_contracts import BACKBONE_REVISION
from lap.verifiers.cover.w3_contracts import PHRASE_TEMPLATES
from lap.verifiers.cover.w3_contracts import W3_PROTOCOL_SCHEMA
from lap.verifiers.cover.w3_contracts import content_hash
from lap.verifiers.cover.w3_contracts import sha256_file
from lap.verifiers.cover.w3_contracts import write_canonical_json

TRAINING_SEED = 42
EVALUATION_SEED = 4203
WORLD_SIZE = 2
PROBE_ORDER = (64, 32, 16)
SHUFFLED_COUNT = 1112
VALIDATION_COUNT = 1118
BOOTSTRAP_REPLICATES = 10000
NEARBY_OFFSET = 15

APPROVED_SHUFFLED_EXCLUSIONS = (
    "circular_posy_demo_07:000110",
    "circular_posx_demo_07:000012",
    "circular_negx_demo_07:000139",
    "circular_posy_demo_07:000016",
    "circular_posx_demo_07:000043",
    "circular_negy_demo_07:000034",
)

_EXPECTED_BACKBONE_NAMES = {BACKBONE_ID, "hf-hub:timm/ViT-L-16-SigLIP2-384"}
_JSON_ARTIFACTS = {
    "rephrase_manifest": "rephrase_manifest.json",
    "shuffled_pairs": "shuffled_pairs.json",
    "shuffled_exclusions": "shuffled_exclusions.json",
    "nearby_pairs": "nearby_pairs.json",
    "batch_probe_receipt": "batch_probe_receipt.json",
}


def _canonical_text(value: str) -> str:
    return unicodedata.normalize("NFC", value).casefold()


def _with_content_hash(payload: dict[str, Any]) -> dict[str, Any]:
    result = dict(payload)
    result["content_hash"] = content_hash(result)
    return result


def build_phrase_manifest() -> dict[str, Any]:
    """Return the reviewed, exact 16-string offline language manifest."""

    phrases = [template.format(shape=shape) for shape in ("circular", "square") for template in PHRASE_TEMPLATES]
    normalized = [_canonical_text(phrase) for phrase in phrases]
    if len(set(normalized)) != len(phrases):
        raise ValueError("approved W3 phrase bank contains duplicate phrases")
    return _with_content_hash(
        {
            "schema": "osx_cover_w3_rephrase_manifest_v1",
            "authorship": {
                "owner": "W3 verifier training",
                "method": "manually_reviewed_static_templates",
                "source": "1-4_w3_cover_verifier_training_prd.md#confirmed-offline-language-bank",
                "review_status": "approved",
            },
            "templates": list(PHRASE_TEMPLATES),
            "phrases": phrases,
            "selection": {
                "method": "sha256(seed:epoch:sample_id) modulo variants_for_shape",
                "seed": TRAINING_SEED,
            },
            "validation_language": "canonical_w2_instruction",
        }
    )


def select_training_phrase(*, seed: int, epoch: int, sample_id: str, shape: str) -> str:
    """Select one static phrase without online generation or validation feedback."""

    if shape not in {"circular", "square"}:
        raise ValueError(f"unsupported peg shape: {shape}")
    digest = hashlib.sha256(f"{seed}:{epoch}:{sample_id}".encode()).digest()
    index = int.from_bytes(digest[:8], "big") % len(PHRASE_TEMPLATES)
    return PHRASE_TEMPLATES[index].format(shape=shape)


@dataclass(frozen=True)
class RunProtocol:
    """Runtime-compatible view of the immutable W3 contract."""

    seed: int = TRAINING_SEED
    sampler_seed: int = TRAINING_SEED
    evaluation_seed: int = EVALUATION_SEED
    per_rank_batch_size: int | None = None
    world_size: int = WORLD_SIZE
    epochs: int = 50
    learning_rate: float = 1e-6
    betas: tuple[float, float] = (0.9, 0.999)
    epsilon: float = 1e-8
    weight_decay: float = 0.01
    warmup_epochs: int = 10
    gradient_clip_norm: float = 1.0
    bootstrap_replicates: int = BOOTSTRAP_REPLICATES

    def to_dict(self, *, train_manifest_hash: str, phrase_manifest_hash: str) -> dict[str, Any]:
        """Serialize the protocol, retaining compatibility aliases for W3 code."""

        if self.seed != TRAINING_SEED or self.sampler_seed != TRAINING_SEED:
            raise ValueError("W3 training and sampler seed must remain 42")
        if self.evaluation_seed != EVALUATION_SEED:
            raise ValueError("W3 evaluation/bootstrap seed must remain 4203")
        if self.world_size != WORLD_SIZE:
            raise ValueError("W3 accepted protocol requires world_size=2")
        if self.epochs != 50 or self.learning_rate != 1e-6 or self.betas != (0.9, 0.999):
            raise ValueError("W3 optimizer/epoch constants drifted from the approved Bridge baseline")
        if self.epsilon != 1e-8 or self.weight_decay != 0.01 or self.warmup_epochs != 10:
            raise ValueError("W3 optimizer/scheduler constants drifted from the approved Bridge baseline")
        if self.gradient_clip_norm != 1.0 or self.bootstrap_replicates != BOOTSTRAP_REPLICATES:
            raise ValueError("W3 evaluation/gradient constants drifted from the approved protocol")
        if self.per_rank_batch_size is not None and self.per_rank_batch_size not in PROBE_ORDER:
            raise ValueError("per-rank batch size must be selected from 64, 32, or 16")
        return {
            "schema": W3_PROTOCOL_SCHEMA,
            "protocol_version": "w3-g02-accepted-run-v1",
            # Compatibility aliases are intentionally identical to the nested
            # training contract; they are not independent configuration knobs.
            "seed": self.seed,
            "sampler": {
                "type": "DistributedSampler",
                "shuffle": True,
                "seed": self.sampler_seed,
                "world_size": self.world_size,
                "dataset": "frozen_w2_train_manifest_only",
                "set_epoch": True,
                "batch_constraints": "none",
                "collision_diagnostics": "diagnostic_only_no_batch_adaptation",
            },
            "training": {
                "seed": self.seed,
                "negative_pool": "local_per_rank",
                "world_size": self.world_size,
                "trainable_dtype": "float32",
                "frozen_backbone_dtype": "bfloat16",
                "automatic_mixed_precision": "disabled_for_trainable_modules",
            },
            "batch": {
                "per_rank": self.per_rank_batch_size,
                "probe_order": list(PROBE_ORDER),
                "selection": "first_successful_two_rank_forward_backward",
            },
            "optimization": {
                "optimizer": "AdamW",
                "epochs": self.epochs,
                "learning_rate": self.learning_rate,
                "betas": list(self.betas),
                "epsilon": self.epsilon,
                "weight_decay": self.weight_decay,
                "warmup_epochs": self.warmup_epochs,
                "scheduler": "10_epoch_linear_warmup_then_constant",
                "gradient_clip_norm": self.gradient_clip_norm,
            },
            "evaluation": {
                "seed": self.evaluation_seed,
                "bootstrap_replicates": self.bootstrap_replicates,
                "bootstrap_generator": "numpy.PCG64",
                "bootstrap_shape": [BOOTSTRAP_REPLICATES, 8],
                "nearby_offset": NEARBY_OFFSET,
                "shuffled_count": SHUFFLED_COUNT,
                "validation_count": VALIDATION_COUNT,
                "retrieval_directions": ["semantic_to_action", "action_to_semantic"],
                "top_k": [1, 5],
            },
            "identities": {
                "train_manifest_hash": train_manifest_hash,
                "phrase_manifest_hash": phrase_manifest_hash,
            },
        }


def sampler_indices(
    sample_ids: Sequence[str], *, epoch: int, rank: int = 0, world_size: int = WORLD_SIZE, seed: int = TRAINING_SEED
) -> dict[str, Any]:
    """Materialize PyTorch's approved sampler semantics for diagnostics."""

    if world_size != WORLD_SIZE or seed != TRAINING_SEED:
        raise ValueError("W3 sampler constants are fixed at world_size=2 and seed=42")
    if rank < 0 or rank >= world_size:
        raise ValueError("rank must be within the two-rank W3 world")
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError("frozen W2 train manifest contains duplicate sample IDs")

    sampler = DistributedSampler(
        list(range(len(sample_ids))), num_replicas=world_size, rank=rank, shuffle=True, seed=seed, drop_last=False
    )
    sampler.set_epoch(epoch)
    indices = list(sampler)
    total_size = len(sampler) * world_size
    return {
        "epoch": epoch,
        "rank": rank,
        "world_size": world_size,
        "seed": seed,
        "sample_ids": [sample_ids[index] for index in indices],
        "indices": indices,
        "collision_diagnostics": {
            "sampler_added_duplicate_rows": total_size - len(sample_ids),
            "repeated_instruction_pair_rate": None,
            "exact_duplicate_history_pair_rate": None,
            "same_episode_pair_rate": None,
            "adapts_batches": False,
        },
    }


def _row_shape(row: Mapping[str, Any]) -> str | None:
    return row.get("traceability", {}).get("peg_shape")


def build_shuffled_pairs(
    validation: Sequence[dict[str, Any]], *, seed: int = EVALUATION_SEED
) -> tuple[list[dict[str, str]], list[str]]:
    """Build the fixed, cross-shape 1,112-row derangement."""

    if seed != EVALUATION_SEED:
        raise ValueError("W3 shuffled-pair seed is fixed at 4203")
    rows = list(validation)
    if len(rows) != VALIDATION_COUNT or len({row.get("sample_id") for row in rows}) != VALIDATION_COUNT:
        raise ValueError("canonical shuffled construction requires 1,118 unique validation rows")
    circular = [row for row in rows if _row_shape(row) == "circular"]
    square = [row for row in rows if _row_shape(row) == "square"]
    if len(circular) != 562 or len(square) != 556:
        raise ValueError("canonical shuffled construction requires 562 circular and 556 square rows")

    def rank(prefix: str, row: Mapping[str, Any]) -> bytes:
        return hashlib.sha256(f"{seed}:{prefix}:{row['sample_id']}".encode()).digest()

    selected_circular = sorted(circular, key=lambda row: rank("shuffled:selection", row))[:556]
    selected_ids = {row["sample_id"] for row in selected_circular}
    excluded = [row["sample_id"] for row in circular if row["sample_id"] not in selected_ids]
    if set(excluded) != set(APPROVED_SHUFFLED_EXCLUSIONS):
        raise ValueError("canonical shuffled construction does not reproduce the six approved exclusions")
    excluded = list(APPROVED_SHUFFLED_EXCLUSIONS)
    circular_order = sorted(selected_circular, key=lambda row: rank("shuffled:circular_order", row))
    square_order = sorted(square, key=lambda row: rank("shuffled:square_order", row))
    by_id = {row["sample_id"]: row for row in rows}
    pairs: list[dict[str, str]] = []
    for circular_row, square_row in zip(circular_order, square_order, strict=True):
        if circular_row.get("episode_id") == square_row.get("episode_id"):
            raise ValueError("shuffled pair would cross neither episode nor shape")
        pairs.extend(
            (
                {"semantic_sample_id": circular_row["sample_id"], "history_sample_id": square_row["sample_id"]},
                {"semantic_sample_id": square_row["sample_id"], "history_sample_id": circular_row["sample_id"]},
            )
        )
    expected_ids = set(by_id) - set(APPROVED_SHUFFLED_EXCLUSIONS)
    if {pair["semantic_sample_id"] for pair in pairs} != expected_ids:
        raise ValueError("shuffled construction is not bijective")
    return pairs, excluded


def build_nearby_pairs(
    validation: Sequence[dict[str, Any]], *, offset: int = NEARBY_OFFSET
) -> tuple[list[dict[str, str]], list[str]]:
    """Build the fixed within-episode offset-15 mismatch construction."""

    if offset != NEARBY_OFFSET:
        raise ValueError("W3 nearby-pair offset is fixed at 15")
    by_episode: dict[str, list[dict[str, Any]]] = {}
    for row in validation:
        by_episode.setdefault(str(row["episode_id"]), []).append(row)
    pairs: list[dict[str, str]] = []
    excluded: list[str] = []
    for episode_rows in by_episode.values():
        for index, row in enumerate(episode_rows):
            partner = index + offset if index + offset < len(episode_rows) else index - offset
            if partner < 0 or partner >= len(episode_rows):
                excluded.append(row["sample_id"])
                continue
            pairs.append(
                {"semantic_sample_id": row["sample_id"], "history_sample_id": episode_rows[partner]["sample_id"]}
            )
    return pairs, excluded


def build_bootstrap_indices(
    episode_ids: Sequence[str], *, seed: int = EVALUATION_SEED, replicates: int = BOOTSTRAP_REPLICATES
) -> np.ndarray:
    """Create the one-call PCG64 episode-index matrix."""

    if seed != EVALUATION_SEED:
        raise ValueError("W3 bootstrap seed is fixed at 4203")
    ordered = sorted(set(episode_ids))
    if len(ordered) != 8:
        raise ValueError("W3 bootstrap protocol requires exactly eight validation episodes")
    if replicates <= 0:
        raise ValueError("bootstrap replicates must be positive")
    rng = np.random.Generator(np.random.PCG64(seed))
    return rng.integers(0, len(ordered), size=(replicates, len(ordered)), endpoint=False, dtype=np.int64)


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def snapshot_nvidia_devices() -> list[dict[str, Any]]:
    """Read both visible GPUs immediately before a batch probe."""

    if not torch.cuda.is_available() or torch.cuda.device_count() < WORLD_SIZE:
        raise RuntimeError("W3 batch probe requires two visible NVIDIA GPUs")
    try:
        physical_rows = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,name,driver_version,uuid,memory.total",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        ).splitlines()
        physical_devices = {}
        for row in physical_rows:
            if not row.strip():
                continue
            fields = [field.strip() for field in row.split(",")]
            if len(fields) != 5:
                raise ValueError("unexpected nvidia-smi GPU identity row")
            physical_devices[int(fields[0])] = {
                "name": fields[1],
                "driver_version": fields[2],
                "uuid": fields[3],
                "physical_total_memory_mib": float(fields[4]),
            }
    except (OSError, subprocess.CalledProcessError, ValueError) as error:
        raise RuntimeError("W3 batch probe could not remeasure physical GPU memory with nvidia-smi") from error
    snapshots = []
    for index in range(WORLD_SIZE):
        free_bytes, allocatable_bytes = torch.cuda.mem_get_info(index)
        properties = torch.cuda.get_device_properties(index)
        if index not in physical_devices:
            raise RuntimeError(f"nvidia-smi did not report physical memory for GPU {index}")
        physical = physical_devices[index]
        snapshots.append(
            {
                "index": index,
                "name": physical["name"] or properties.name,
                "driver_version": physical["driver_version"],
                "uuid": physical["uuid"],
                "physical_total_memory_mib": physical["physical_total_memory_mib"],
                "allocatable_total_memory_mib": allocatable_bytes / (1024**2),
                "free_memory_mib": free_bytes / (1024**2),
            }
        )
    return snapshots


def _is_oom(error: BaseException) -> bool:
    message = str(error).casefold()
    return isinstance(error, MemoryError) or "out of memory" in message or "cuda error: out of memory" in message


def _validate_gpu_snapshot(snapshot: Sequence[Mapping[str, Any]], *, require_identity: bool = False) -> None:
    if len(snapshot) != WORLD_SIZE:
        raise ValueError("batch probe must record both GPUs")
    for gpu in snapshot:
        for key in ("index", "name", "physical_total_memory_mib", "allocatable_total_memory_mib", "free_memory_mib"):
            if key not in gpu:
                raise ValueError(f"GPU probe snapshot missing {key}")
        if require_identity:
            for key in ("driver_version", "uuid"):
                if not isinstance(gpu.get(key), str) or not gpu[key]:
                    raise ValueError(f"GPU probe snapshot missing {key}")
        if gpu["physical_total_memory_mib"] < gpu["allocatable_total_memory_mib"]:
            raise ValueError("allocatable GPU memory cannot exceed physical memory")


def _validate_model_identity(identity: Mapping[str, Any], *, require_canonical: bool) -> None:
    text = json.dumps(identity, sort_keys=True).casefold()
    if require_canonical:
        if "tinyfrozenbackbone" in text or "tiny" in str(identity.get("backbone", "")).casefold():
            raise ValueError("TinyFrozenBackbone cannot provide the canonical W3 batch receipt")
        if identity.get("canonical_target") is not True:
            raise ValueError("batch receipt is not from the canonical production target")
        if identity.get("backbone") not in _EXPECTED_BACKBONE_NAMES:
            raise ValueError("batch receipt must identify the pinned SigLIP2 backbone")
        if identity.get("backbone_revision") not in {None, BACKBONE_REVISION}:
            raise ValueError("batch receipt backbone revision drifted")
        for field in ("configuration_hash", "audit_manifest_sha256", "target_inventory_fingerprint"):
            value = identity.get(field)
            if not isinstance(value, str) or len(value) != 64:
                raise ValueError(f"batch receipt model identity missing {field}")


def _validate_environment(environment: Mapping[str, Any]) -> None:
    required = (
        "python_executable",
        "python",
        "torch",
        "cuda_build",
        "lap_revision",
        "pyproject_sha256",
        "uv_lock_sha256",
        "siglip2_snapshot",
    )
    for field in required:
        if field not in environment:
            raise ValueError(f"batch receipt environment missing {field}")
    if not Path(str(environment["python_executable"])).is_absolute():
        raise ValueError("batch receipt must record an absolute Python executable")
    for field in ("pyproject_sha256", "uv_lock_sha256"):
        if not isinstance(environment[field], str) or len(environment[field]) != 64:
            raise ValueError(f"batch receipt environment has invalid {field}")
    if not isinstance(environment["lap_revision"], str) or len(environment["lap_revision"]) != 40:
        raise ValueError("batch receipt environment has invalid LAP revision")
    snapshot = environment["siglip2_snapshot"]
    if not isinstance(snapshot, Mapping) or snapshot.get("backbone_id") != BACKBONE_ID or snapshot.get("revision") != BACKBONE_REVISION:
        raise ValueError("batch receipt environment does not identify the pinned SigLIP2 snapshot")


def probe_batch_sizes(
    step_fn: Callable[[int, int, Mapping[str, Any]], Mapping[str, Any] | None],
    *,
    snapshot_fn: Callable[[], Sequence[Mapping[str, Any]]] = snapshot_nvidia_devices,
    model_identity: Mapping[str, Any] | None = None,
    candidates: Sequence[int] = PROBE_ORDER,
) -> dict[str, Any]:
    """Run an injectable two-rank, forward/backward-only batch-size probe.

    ``step_fn`` owns model construction, DDP setup, one forward/backward step,
    cleanup, and finite-result checks.  It is called once per rank.  The
    default snapshot is real NVIDIA/PyTorch state; tests inject both seams.
    """

    if tuple(candidates) != PROBE_ORDER:
        raise ValueError("W3 batch probe order is fixed at 64, 32, 16")
    gpu_snapshot = [_jsonable(gpu) for gpu in snapshot_fn()]
    _validate_gpu_snapshot(gpu_snapshot)
    identity = dict(model_identity or {"backbone": BACKBONE_ID, "backbone_revision": BACKBONE_REVISION})
    _validate_model_identity(identity, require_canonical=False)
    project_root = Path(__file__).resolve().parents[4]
    environment = {
        "python_executable": sys.executable,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_build": torch.version.cuda,
        "lap_revision": subprocess.check_output(
            ["git", "-C", str(project_root), "rev-parse", "HEAD"], text=True
        ).strip(),
        "pyproject_sha256": sha256_file(project_root / "pyproject.toml"),
        "uv_lock_sha256": sha256_file(project_root / "uv.lock"),
        "siglip2_snapshot": {"backbone_id": BACKBONE_ID, "revision": BACKBONE_REVISION},
    }
    attempts = []
    for batch_size in PROBE_ORDER:
        rank_results = []
        try:
            for rank, gpu in enumerate(gpu_snapshot):
                result = step_fn(batch_size, rank, gpu)
                result = {} if result is None else dict(result)
                if result.get("status") == "failed":
                    error = RuntimeError(str(result.get("error", "probe step failed")))
                    if not _is_oom(error):
                        raise error
                    raise error
                result.setdefault("forward_backward", "passed")
                rank_results.append(_jsonable(result))
        except BaseException as error:
            if not _is_oom(error):
                raise
            attempts.append(
                {
                    "per_rank_batch_size": batch_size,
                    "status": "failed",
                    "failure": "out_of_memory",
                    "error_type": type(error).__name__,
                    "ranks_completed": len(rank_results),
                }
            )
            continue
        attempts.append(
            {
                "per_rank_batch_size": batch_size,
                "status": "passed",
                "ranks": rank_results,
            }
        )
        return _with_content_hash(
            {
                "schema": "osx_cover_w3_batch_probe_v2",
                "status": "complete",
                "host": platform.node(),
                "world_size": WORLD_SIZE,
                "probe_order": list(PROBE_ORDER),
                "attempted_batch_sizes": [attempt["per_rank_batch_size"] for attempt in attempts],
                "selected_per_rank_batch_size": batch_size,
                "selection_rule": "first_successful_two_rank_forward_backward",
                "gpu_snapshot": gpu_snapshot,
                "environment": environment,
                "model": _jsonable(identity),
                "successful_two_rank_forward_backward": {"batch_size": batch_size, "ranks": [0, 1]},
                "attempts": attempts,
            }
        )
    raise RuntimeError("no W3 batch size fit the approved two-rank probe order")


def validate_batch_probe_receipt(receipt: Mapping[str, Any], *, require_canonical: bool = True) -> None:
    """Reject incomplete, drifted, fixture, or single-rank probe receipts."""

    payload = dict(receipt)
    recorded_hash = payload.pop("content_hash", None)
    if recorded_hash != content_hash(payload):
        raise ValueError("batch probe receipt content hash mismatch")
    if payload.get("schema") != "osx_cover_w3_batch_probe_v2":
        raise ValueError("batch probe receipt is incomplete or has the wrong schema")
    if payload.get("status") == "pending_real_two_rank_probe":
        if require_canonical:
            raise ValueError("batch probe receipt is incomplete or has the wrong schema")
        if payload.get("selected_per_rank_batch_size") is not None or payload.get("attempts"):
            raise ValueError("pending batch probe receipt contains partial canonical evidence")
        return
    if payload.get("status") != "complete":
        raise ValueError("batch probe receipt is incomplete or has the wrong schema")
    if payload.get("world_size") != WORLD_SIZE or payload.get("probe_order") != list(PROBE_ORDER):
        raise ValueError("batch probe receipt has drifted world-size or probe order")
    _validate_gpu_snapshot(payload.get("gpu_snapshot", []), require_identity=require_canonical)
    if require_canonical:
        _validate_environment(payload.get("environment", {}))
    _validate_model_identity(payload.get("model", {}), require_canonical=require_canonical)
    selected = payload.get("selected_per_rank_batch_size")
    if selected not in PROBE_ORDER:
        raise ValueError("batch probe receipt has no approved selected batch size")
    attempts = payload.get("attempts", [])
    if [attempt["per_rank_batch_size"] for attempt in attempts] != payload.get("attempted_batch_sizes"):
        raise ValueError("batch probe attempts are not recorded in order")
    successful = [attempt for attempt in attempts if attempt.get("status") == "passed"]
    if len(successful) != 1 or successful[0]["per_rank_batch_size"] != selected:
        raise ValueError("batch probe receipt does not identify the first successful size")
    if successful[0].get("ranks") is None or len(successful[0]["ranks"]) != WORLD_SIZE:
        raise ValueError("batch probe success did not run both ranks")
    if payload.get("successful_two_rank_forward_backward", {}).get("ranks") != [0, 1]:
        raise ValueError("batch probe receipt lacks a two-rank success result")


def _pending_batch_probe_receipt() -> dict[str, Any]:
    return _with_content_hash(
        {
            "schema": "osx_cover_w3_batch_probe_v2",
            "status": "pending_real_two_rank_probe",
            "world_size": WORLD_SIZE,
            "probe_order": list(PROBE_ORDER),
            "attempted_batch_sizes": [],
            "selected_per_rank_batch_size": None,
            "model": {"canonical_target": False, "reason": "GPU probe not run"},
            "gpu_snapshot": [],
            "attempts": [],
        }
    )


def materialize_protocol(
    output_dir: Path,
    *,
    train_manifest_hash: str,
    validation: Sequence[dict[str, Any]],
    per_rank_batch_size: int | None = None,
    batch_probe_hash: str | None = None,
    batch_probe_receipt: Mapping[str, Any] | None = None,
) -> dict[str, str]:
    """Materialize all deterministic W3 artifacts before model loading."""

    del per_rank_batch_size  # A bare integer is never evidence for canonical W3.
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    phrase_hash = write_canonical_json(output_dir / "rephrase_manifest.json", build_phrase_manifest())
    shuffled, excluded = build_shuffled_pairs(validation)
    nearby, nearby_excluded = build_nearby_pairs(validation)
    if nearby_excluded:
        raise ValueError("canonical nearby construction must cover all 1,118 validation rows")
    shuffled_hash = write_canonical_json(
        output_dir / "shuffled_pairs.json",
        {"schema": "osx_cover_w3_shuffled_pairs_v1", "pairs": shuffled},
    )
    exclusion_hash = write_canonical_json(
        output_dir / "shuffled_exclusions.json",
        {"schema": "osx_cover_w3_shuffled_exclusions_v1", "sample_ids": excluded},
    )
    nearby_hash = write_canonical_json(
        output_dir / "nearby_pairs.json",
        {"schema": "osx_cover_w3_nearby_pairs_v1", "pairs": nearby, "excluded": nearby_excluded},
    )
    ordered_episode_ids = sorted({str(row["episode_id"]) for row in validation})
    bootstrap = build_bootstrap_indices(ordered_episode_ids)
    np.save(output_dir / "bootstrap_indices.npy", bootstrap, allow_pickle=False)
    bootstrap_hash = sha256_file(output_dir / "bootstrap_indices.npy")

    if batch_probe_receipt is None:
        receipt = _pending_batch_probe_receipt()
    else:
        receipt = dict(batch_probe_receipt)
        validate_batch_probe_receipt(receipt, require_canonical=True)
    receipt_hash = write_canonical_json(output_dir / "batch_probe_receipt.json", receipt)
    protocol = RunProtocol(
        per_rank_batch_size=receipt.get("selected_per_rank_batch_size"),
    ).to_dict(train_manifest_hash=train_manifest_hash, phrase_manifest_hash=phrase_hash)
    protocol["batch"]["per_rank"] = receipt.get("selected_per_rank_batch_size")
    protocol["status"] = "complete" if receipt.get("status") == "complete" else "incomplete_pending_batch_probe"
    protocol["evaluation"]["episode_ids"] = ordered_episode_ids
    protocol["artifacts"] = {
        "rephrase_manifest": phrase_hash,
        "shuffled_pairs": shuffled_hash,
        "shuffled_exclusions": exclusion_hash,
        "nearby_pairs": nearby_hash,
        "bootstrap_indices": bootstrap_hash,
        "batch_probe_receipt": receipt_hash,
    }
    if batch_probe_hash is not None and batch_probe_hash != receipt_hash:
        raise ValueError("external batch probe hash does not match the materialized receipt")
    protocol_hash = write_canonical_json(output_dir / "run_protocol.json", protocol)
    return {**protocol["artifacts"], "run_protocol": protocol_hash}


def _read_hashed_json(path: Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    recorded_hash = payload.pop("content_hash", None)
    if recorded_hash != content_hash(payload):
        raise ValueError(f"{path.name} content hash mismatch")
    payload["content_hash"] = recorded_hash
    return payload


def validate_protocol_directory(output_dir: Path, *, require_complete: bool = True) -> dict[str, Any]:
    """Verify every recorded protocol/artifact hash and fixed W3 constant."""

    output_dir = Path(output_dir)
    payload = _read_hashed_json(output_dir / "run_protocol.json")
    if payload.get("schema") != W3_PROTOCOL_SCHEMA:
        raise ValueError("wrong W3 protocol schema")
    if payload.get("seed") != TRAINING_SEED or payload.get("sampler", {}).get("seed") != TRAINING_SEED:
        raise ValueError("training/sampler seed drifted")
    if payload.get("evaluation", {}).get("seed") != EVALUATION_SEED:
        raise ValueError("evaluation seed drifted")
    if payload.get("sampler", {}).get("world_size") != WORLD_SIZE:
        raise ValueError("world_size drifted")
    if payload.get("sampler", {}).get("type") != "DistributedSampler":
        raise ValueError("sampler type drifted")
    if payload.get("batch", {}).get("probe_order") != list(PROBE_ORDER):
        raise ValueError("batch probe order drifted")
    if payload.get("optimization", {}).get("optimizer") != "AdamW":
        raise ValueError("optimizer drifted")
    if payload.get("optimization", {}).get("betas") != [0.9, 0.999]:
        raise ValueError("optimizer betas drifted")
    if payload.get("optimization", {}).get("epsilon") != 1e-8:
        raise ValueError("optimizer epsilon drifted")
    if payload.get("optimization", {}).get("weight_decay") != 0.01:
        raise ValueError("optimizer weight decay drifted")
    if payload.get("training", {}).get("negative_pool") != "local_per_rank":
        raise ValueError("negative-pool contract drifted")
    if payload.get("training", {}).get("trainable_dtype") != "float32":
        raise ValueError("trainable dtype drifted")
    if payload.get("training", {}).get("frozen_backbone_dtype") != "bfloat16":
        raise ValueError("frozen dtype drifted")

    artifacts = payload.get("artifacts", {})
    materialized: dict[str, Any] = {"protocol": payload}
    for key, filename in _JSON_ARTIFACTS.items():
        artifact = _read_hashed_json(output_dir / filename)
        if artifacts.get(key) != artifact["content_hash"]:
            raise ValueError(f"{filename} hash is not recorded by run_protocol.json")
        materialized[key] = artifact
    bootstrap_path = output_dir / "bootstrap_indices.npy"
    bootstrap_hash = sha256_file(bootstrap_path)
    if artifacts.get("bootstrap_indices") != bootstrap_hash:
        raise ValueError("bootstrap index hash mismatch")
    bootstrap = np.load(bootstrap_path, allow_pickle=False)
    if bootstrap.shape != (BOOTSTRAP_REPLICATES, 8) or bootstrap.dtype != np.int64:
        raise ValueError("bootstrap matrix shape or dtype drifted")
    materialized["bootstrap_indices"] = bootstrap

    if materialized["rephrase_manifest"].get("phrases") != build_phrase_manifest().get("phrases"):
        raise ValueError("phrase manifest drifted")
    if materialized["shuffled_exclusions"].get("sample_ids") != list(APPROVED_SHUFFLED_EXCLUSIONS):
        raise ValueError("approved shuffled exclusions drifted")
    if len(materialized["shuffled_pairs"].get("pairs", [])) != SHUFFLED_COUNT:
        raise ValueError("shuffled pair count drifted")
    if len(materialized["nearby_pairs"].get("pairs", [])) != VALIDATION_COUNT:
        raise ValueError("nearby pair count drifted")
    validate_batch_probe_receipt(materialized["batch_probe_receipt"], require_canonical=require_complete)
    if require_complete and payload.get("status") != "complete":
        raise ValueError("W3 protocol is incomplete pending the real two-rank batch probe")
    selected = payload.get("batch", {}).get("per_rank")
    if require_complete and selected not in PROBE_ORDER:
        raise ValueError("W3 protocol has no selected canonical per-rank batch size")
    return materialized
