"""Canonical two-rank NVIDIA optimizer-step and capacity evidence for W3-05."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import subprocess
import sys
import tempfile
from typing import Any

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel

from lap.verifiers.cover.bridge_audit import apply_audited_initialization
from lap.verifiers.cover.data import TwoViewDataset
from lap.verifiers.cover.data import W2DatasetGateway
from lap.verifiers.cover.data import collate_two_view_batch
from lap.verifiers.cover.data import make_sampler
from lap.verifiers.cover.model import OpenClipSigLIP2Backbone
from lap.verifiers.cover.model import VerifierConfig
from lap.verifiers.cover.model import VerifierModel
from lap.verifiers.cover.protocol import PROBE_ORDER
from lap.verifiers.cover.protocol import RunProtocol
from lap.verifiers.cover.protocol import snapshot_nvidia_devices
from lap.verifiers.cover.training import create_optimizer
from lap.verifiers.cover.w3_contracts import BACKBONE_ID
from lap.verifiers.cover.w3_contracts import BACKBONE_REVISION
from lap.verifiers.cover.w3_contracts import content_hash
from lap.verifiers.cover.w3_contracts import sha256_file
from lap.verifiers.cover.w3_contracts import write_canonical_json

WORLD_SIZE = 2
_BASELINE = {
    "date": "2026-07-23",
    "gpu_model": "NVIDIA RTX 6000 Ada Generation",
    "physical_total_memory_mib": [49140.0, 49140.0],
    "allocatable_total_memory_mib": [48502.69, 48510.94],
    "free_memory_mib": [47882.56, 48061.75],
}


def validate_successful_rank_records(ranks: Sequence[Mapping[str, Any]], *, batch_size: int) -> None:
    """Require exact two-rank evidence for a synchronized finite optimizer step."""

    records = [dict(rank) for rank in ranks]
    if sorted(record.get("rank") for record in records) != [0, 1]:
        raise ValueError("successful W3 probe must contain exactly ranks 0 and 1")
    for record in records:
        finite_scalars = (record.get("loss"), record.get("gradient_norm"))
        if (
            record.get("status") != "passed"
            or record.get("forward") != "passed"
            or record.get("backward") != "passed"
            or record.get("optimizer_step") != "passed"
            or not all(isinstance(value, (int, float)) and math.isfinite(value) for value in finite_scalars)
            or record.get("gradients_finite") is not True
            or record.get("parameters_finite") is not True
            or record.get("trainable_state_changed") is not True
            or record.get("frozen_state_unchanged") is not True
            or record.get("local_negative_pool_size") != batch_size
            or record.get("embedding_all_gather") is not False
            or record.get("frozen_backbone_dtype") != "bfloat16"
            or record.get("trainable_dtype") != "float32"
            or record.get("logits_dtype") != "float32"
        ):
            raise ValueError("each successful rank must record a full finite optimizer step")
    fingerprints = [record.get("gradient_fingerprint") for record in records]
    if (
        any(not isinstance(fingerprint, str) or len(fingerprint) != 64 for fingerprint in fingerprints)
        or len(set(fingerprints)) != 1
    ):
        raise ValueError("successful DDP ranks must record one synchronized gradient fingerprint")


def compare_memory_to_baseline(snapshot: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Record current memory and explain drift from the PRD observation."""

    current = [dict(gpu) for gpu in snapshot]
    if len(current) != WORLD_SIZE:
        raise ValueError("W3 memory drift requires exactly two GPUs")
    drift = []
    for index, gpu in enumerate(current):
        drift.append(
            {
                "index": index,
                "physical_total_memory": {
                    "baseline": _BASELINE["physical_total_memory_mib"][index],
                    "observed": gpu["physical_total_memory_mib"],
                    "delta": round(gpu["physical_total_memory_mib"] - _BASELINE["physical_total_memory_mib"][index], 2),
                },
                "allocatable_total_memory": {
                    "baseline": _BASELINE["allocatable_total_memory_mib"][index],
                    "observed": gpu["allocatable_total_memory_mib"],
                    "delta": round(
                        gpu["allocatable_total_memory_mib"] - _BASELINE["allocatable_total_memory_mib"][index], 2
                    ),
                },
                "free_memory": {
                    "baseline": _BASELINE["free_memory_mib"][index],
                    "observed": gpu["free_memory_mib"],
                    "delta": round(gpu["free_memory_mib"] - _BASELINE["free_memory_mib"][index], 2),
                },
            }
        )
    return {
        "baseline_date": _BASELINE["date"],
        "baseline_gpu_model": _BASELINE["gpu_model"],
        "baseline": _BASELINE,
        "observed": current,
        "drift_mib": drift,
        "explanation": (
            "Physical and allocatable totals are stable hardware/runtime facts; free memory is transient and its "
            "drift reflects concurrent allocations and driver state immediately before this probe."
        ),
    }


def probe_batch_sizes(
    step_fn: Callable[[int, int, Mapping[str, Any]], Mapping[str, Any] | None],
    *,
    snapshot_fn: Callable[[], Sequence[Mapping[str, Any]]] = snapshot_nvidia_devices,
    model_identity: Mapping[str, Any] | None = None,
    candidates: Sequence[int] = PROBE_ORDER,
) -> dict[str, Any]:
    """Run an injectable two-rank optimizer-step probe through the canonical owner."""

    if tuple(candidates) != PROBE_ORDER:
        raise ValueError("W3 batch probe order is fixed at 64, 32, 16")
    snapshot = [dict(gpu) for gpu in snapshot_fn()]
    memory_drift = compare_memory_to_baseline(snapshot)
    attempts = []
    for batch_size in PROBE_ORDER:
        rank_results = []
        try:
            for rank, gpu in enumerate(snapshot):
                result = dict(step_fn(batch_size, rank, gpu) or {})
                if result.get("status") == "failed":
                    raise RuntimeError(str(result.get("error", "probe step failed")))
                rank_results.append(result)
            validate_successful_rank_records(rank_results, batch_size=batch_size)
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
        attempts.append({"per_rank_batch_size": batch_size, "status": "passed", "ranks": rank_results})
        return build_probe_receipt(
            snapshot=snapshot,
            attempts=attempts,
            selected_batch_size=batch_size,
            memory_drift=memory_drift,
            evidence=model_identity,
        )
    raise RuntimeError("no W3 batch size fit the approved two-rank probe order")


def build_probe_receipt(
    *,
    snapshot: Sequence[Mapping[str, Any]],
    attempts: Sequence[Mapping[str, Any]],
    selected_batch_size: int,
    memory_drift: Mapping[str, Any],
    host: str | None = None,
    evidence: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the hashable canonical receipt after a successful DDP probe."""

    if selected_batch_size not in PROBE_ORDER:
        raise ValueError("selected batch size is outside the approved W3 probe order")
    successful = [
        attempt
        for attempt in attempts
        if attempt.get("status") == "passed" and attempt.get("per_rank_batch_size") == selected_batch_size
    ]
    if len(successful) != 1:
        raise ValueError("successful W3 probe must contain exactly ranks 0 and 1")
    validate_successful_rank_records(successful[0].get("ranks", []), batch_size=selected_batch_size)
    selected_evidence = dict(evidence or {})
    environment = {
        "python_executable": selected_evidence.get("python_executable", sys.executable),
        "python": selected_evidence.get("python", platform.python_version()),
        "torch": selected_evidence.get("torch", torch.__version__),
        "cuda_build": selected_evidence.get("cuda_build", torch.version.cuda),
        "lap_revision": selected_evidence.get("lap_revision", _git_revision()),
        "pyproject_sha256": selected_evidence.get("pyproject_sha256", _project_file_hash("pyproject.toml")),
        "uv_lock_sha256": selected_evidence.get("uv_lock_sha256", _project_file_hash("uv.lock")),
        "siglip2_snapshot": selected_evidence.get(
            "siglip2_snapshot",
            {"backbone_id": BACKBONE_ID, "revision": BACKBONE_REVISION},
        ),
    }
    model_provenance = {
        "canonical_target": selected_evidence.get("canonical_target", True),
        "backbone": selected_evidence.get("backbone", BACKBONE_ID),
        "backbone_revision": selected_evidence.get("backbone_revision", BACKBONE_REVISION),
        "configuration": selected_evidence.get("configuration", VerifierConfig().to_dict()),
        "configuration_hash": selected_evidence.get("configuration_hash", content_hash(VerifierConfig().to_dict())),
        "audit_manifest_sha256": selected_evidence.get("audit_manifest_sha256", "0" * 64),
        "target_inventory_fingerprint": selected_evidence.get("target_inventory_fingerprint", "0" * 64),
        "preprocessing_fingerprint": selected_evidence.get("preprocessing_fingerprint", "0" * 64),
        "tokenizer_fingerprint": selected_evidence.get("tokenizer_fingerprint", "0" * 64),
        "two_view": True,
        "negative_pool": "rank_local",
        "gradient_synchronization": "two_rank_ddp",
        "trainable_dtype": "float32",
        "frozen_backbone_dtype": "bfloat16",
        "logits_dtype": "float32",
        "frozen_encoder_autocast": "cuda_bfloat16",
    }
    payload = {
        "schema": "osx_cover_w3_batch_probe_v2",
        "status": "complete",
        "host": host or platform.node(),
        "environment": environment,
        "world_size": WORLD_SIZE,
        "probe_order": list(PROBE_ORDER),
        "attempted_batch_sizes": [attempt["per_rank_batch_size"] for attempt in attempts],
        "selected_per_rank_batch_size": selected_batch_size,
        "selection_rule": "first_successful_two_rank_optimizer_step",
        "gpu_snapshot": [dict(gpu) for gpu in snapshot],
        "memory_drift": dict(memory_drift),
        "model": model_provenance,
        "attempts": [dict(attempt) for attempt in attempts],
        "successful_two_rank_optimizer_step": {"batch_size": selected_batch_size, "ranks": [0, 1]},
    }
    payload["content_hash"] = content_hash(payload)
    return payload


def _git_revision() -> str:
    root = Path(__file__).resolve().parents[4]
    try:
        return subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def _project_file_hash(filename: str) -> str:
    root = Path(__file__).resolve().parents[4]
    path = root / filename
    return sha256_file(path) if path.is_file() else "unavailable"


def _canonical_probe_evidence(*, model_dir: Path, audit_manifest: Path) -> dict[str, Any]:
    manifest = json.loads(Path(audit_manifest).read_text(encoding="utf-8"))
    configuration = VerifierConfig().to_dict()
    return {
        "python_executable": sys.executable,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_build": torch.version.cuda,
        "lap_revision": _git_revision(),
        "pyproject_sha256": _project_file_hash("pyproject.toml"),
        "uv_lock_sha256": _project_file_hash("uv.lock"),
        "siglip2_snapshot": {
            "backbone_id": BACKBONE_ID,
            "revision": BACKBONE_REVISION,
            "local_snapshot": str(Path(model_dir).resolve()),
        },
        "configuration": configuration,
        "configuration_hash": content_hash(configuration),
        "audit_manifest_sha256": manifest["manifest_sha256"],
        "target_inventory_fingerprint": manifest["target"]["fingerprint"],
    }


def _is_oom(error: BaseException | str) -> bool:
    text = str(error).casefold()
    return isinstance(error, MemoryError) or "out of memory" in text or "cuda error: out of memory" in text


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _tensor_fingerprint(named_tensors: Sequence[tuple[str, torch.Tensor]]) -> str:
    digest = hashlib.sha256()
    for name, tensor in named_tensors:
        digest.update(name.encode("utf-8"))
        value = tensor.detach().contiguous().cpu()
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _parameter_fingerprint(model: torch.nn.Module, *, trainable: bool) -> str:
    return _tensor_fingerprint(
        [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad is trainable]
    )


def _gradient_fingerprint(model: torch.nn.Module) -> str:
    gradients = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.grad is None:
            raise FloatingPointError(f"W3 trainable parameter has no gradient: {name}")
        if not torch.isfinite(parameter.grad).all():
            raise FloatingPointError(f"W3 trainable parameter has nonfinite gradient: {name}")
        gradients.append((name, parameter.grad))
    return _tensor_fingerprint(gradients)


def _resolve_pinned_snapshot() -> Path:
    from huggingface_hub import snapshot_download  # noqa: PLC0415

    return Path(
        snapshot_download(
            BACKBONE_ID.removeprefix("hf-hub:"),
            revision=BACKBONE_REVISION,
            local_files_only=True,
        )
    )


def _build_model(*, device: torch.device, w2_root: Path, bridge_artifact: Path, audit_manifest: Path, model_dir: Path):
    dataset = W2DatasetGateway(w2_root)
    backbone = OpenClipSigLIP2Backbone(model_name=f"local-dir:{model_dir}")
    model = VerifierModel(VerifierConfig(), backbone)
    audit = json.loads(audit_manifest.read_text(encoding="utf-8"))
    apply_audited_initialization(model, bridge_artifact, audit)
    model.to(device)
    return model, dataset, backbone


def run_probe_worker(
    *,
    mode: str,
    batch_size: int,
    w2_root: Path,
    bridge_artifact: Path,
    audit_manifest: Path,
    model_dir: Path,
    result_dir: Path,
) -> None:
    """Run one rank of either an independent fit check or the successful DDP step."""

    rank = int(os.environ["LOCAL_RANK"])
    device = torch.device("cuda", rank)
    result: dict[str, Any] = {"rank": rank, "per_rank_batch_size": batch_size}
    process_group = False
    try:
        _seed_everything(42)
        if mode == "ddp":
            torch.distributed.init_process_group("nccl", rank=rank, world_size=WORLD_SIZE)
            process_group = True
        model, dataset, backbone = _build_model(
            device=device,
            w2_root=w2_root,
            bridge_artifact=bridge_artifact,
            audit_manifest=audit_manifest,
            model_dir=model_dir,
        )
        backbone.model.to(dtype=torch.bfloat16)
        if backbone.asset_fingerprints is None:
            raise ValueError("W3 canonical probe did not load pinned preprocessing and tokenizer assets")
        frozen_before = _parameter_fingerprint(model, trainable=False)
        trainable_before = _parameter_fingerprint(model, trainable=True)
        train_dataset = TwoViewDataset(dataset.train, seed=42, training=True, preprocess=backbone.preprocess)
        sampler = make_sampler(train_dataset, seed=42, world_size=WORLD_SIZE, rank=rank)
        loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=sampler,
            collate_fn=collate_two_view_batch,
            num_workers=0,
        )
        batch = next(iter(loader))
        if mode == "ddp":
            model = DistributedDataParallel(model, device_ids=[rank], output_device=rank, broadcast_buffers=False)
        base_model = model.module if isinstance(model, DistributedDataParallel) else model
        model.train()
        optimizer, _ = create_optimizer(base_model, RunProtocol(per_rank_batch_size=batch_size))
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.reset_peak_memory_stats(device)
        output = model(
            batch["base_rgb"].to(device),
            batch["wrist_rgb"].to(device),
            batch["instructions"],
            batch["action_histories"].to(device),
        )
        loss, _ = base_model.contrastive_loss(output)
        if any(not torch.isfinite(value).all() for value in output.values()) or not torch.isfinite(loss):
            raise FloatingPointError("W3 probe output or loss is nonfinite")
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            [parameter for parameter in base_model.parameters() if parameter.requires_grad],
            RunProtocol().gradient_clip_norm,
        )
        if not torch.isfinite(torch.as_tensor(gradient_norm)):
            raise FloatingPointError("W3 probe gradient norm is nonfinite")
        gradient_fingerprint = _gradient_fingerprint(base_model)
        optimizer.step()
        with torch.no_grad():
            base_model.logit_scale.clamp_(
                0.0,
                torch.log(torch.tensor(100.0, device=base_model.logit_scale.device)),
            )
        if not optimizer.state:
            raise RuntimeError("W3 probe optimizer step did not initialize optimizer state")
        parameters_finite = all(
            torch.isfinite(parameter).all() for parameter in base_model.parameters() if parameter.requires_grad
        )
        trainable_changed = trainable_before != _parameter_fingerprint(base_model, trainable=True)
        frozen_unchanged = frozen_before == _parameter_fingerprint(base_model, trainable=False)
        frozen_dtypes = {str(parameter.dtype).removeprefix("torch.") for parameter in backbone.parameters()}
        trainable_dtypes = {
            str(parameter.dtype).removeprefix("torch.")
            for parameter in base_model.parameters()
            if parameter.requires_grad
        }
        if (
            not parameters_finite
            or not trainable_changed
            or not frozen_unchanged
            or frozen_dtypes != {"bfloat16"}
            or trainable_dtypes != {"float32"}
        ):
            raise FloatingPointError("W3 probe state failed finite trainable/frozen invariants")
        if mode == "ddp":
            torch.distributed.barrier()
        torch.cuda.synchronize(device)
        free_bytes, _ = torch.cuda.mem_get_info(device)
        result.update(
            {
                "status": "passed",
                "forward": "passed",
                "backward": "passed",
                "optimizer_step": "passed",
                "loss": float(loss.detach().cpu()),
                "gradient_norm": float(gradient_norm),
                "gradients_finite": True,
                "parameters_finite": True,
                "trainable_state_changed": trainable_changed,
                "frozen_state_unchanged": frozen_unchanged,
                "gradient_fingerprint": gradient_fingerprint,
                "local_negative_pool_size": batch_size,
                "embedding_all_gather": False,
                "frozen_backbone_dtype": frozen_dtypes.pop(),
                "trainable_dtype": trainable_dtypes.pop(),
                "logits_dtype": str(output["semantic_to_action_logits"].dtype).removeprefix("torch."),
                **backbone.asset_fingerprints,
                "max_memory_allocated_mib": round(torch.cuda.max_memory_allocated(device) / (1024**2), 2),
                "max_memory_reserved_mib": round(torch.cuda.max_memory_reserved(device) / (1024**2), 2),
                "free_memory_after_mib": round(free_bytes / (1024**2), 2),
            }
        )
    except BaseException as error:  # the parent classifies OOM and continues the fixed probe order
        result.update(
            {
                "status": "failed",
                "failure": "out_of_memory" if _is_oom(error) else "error",
                "error_type": type(error).__name__,
                "error": str(error),
            }
        )
    finally:
        result_dir.mkdir(parents=True, exist_ok=True)
        (result_dir / f"rank-{rank}.json").write_text(json.dumps(result, sort_keys=True), encoding="utf-8")
        if process_group:
            # NCCL teardown can wait after the one-step receipt is durable.
            # The torchrun worker is disposable; exit after writing its result
            # so the parent can continue the fixed probe order without a
            # cleanup-induced false timeout.
            os._exit(0)


def _run_worker_process(
    *,
    worker_script: Path,
    mode: str,
    batch_size: int,
    w2_root: Path,
    bridge_artifact: Path,
    audit_manifest: Path,
    model_dir: Path,
    result_dir: Path,
    timeout_seconds: int,
) -> list[dict[str, Any]]:
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc_per_node=2",
        str(worker_script),
        "--worker",
        "--worker-mode",
        mode,
        "--batch-size",
        str(batch_size),
        "--w2-root",
        str(w2_root),
        "--bridge-artifact",
        str(bridge_artifact),
        "--audit-manifest",
        str(audit_manifest),
        "--model-dir",
        str(model_dir),
        "--result-dir",
        str(result_dir),
    ]
    try:
        completed = subprocess.run(command, check=False, capture_output=True, text=True, timeout=timeout_seconds)
    except subprocess.TimeoutExpired as error:
        raise RuntimeError(f"W3 {mode} probe timed out at per-rank batch size {batch_size}") from error
    results = []
    for rank in range(WORLD_SIZE):
        path = result_dir / f"rank-{rank}.json"
        if path.is_file():
            results.append(json.loads(path.read_text(encoding="utf-8")))
    if len(results) != WORLD_SIZE:
        raise RuntimeError(
            f"W3 {mode} probe did not produce both rank receipts (exit={completed.returncode}): "
            f"{completed.stderr[-2000:]}"
        )
    non_oom_failures = [result for result in results if result.get("failure") not in {None, "out_of_memory"}]
    if non_oom_failures:
        raise RuntimeError(f"W3 {mode} probe failed: {non_oom_failures}")
    return results


def run_real_batch_probe(
    *,
    worker_script: Path,
    w2_root: Path,
    bridge_artifact: Path,
    audit_manifest: Path,
    output_path: Path,
    timeout_seconds: int = 1800,
) -> dict[str, Any]:
    """Run the pinned two-rank model in the approved 64/32/16 order."""

    snapshot = snapshot_nvidia_devices()
    memory_drift = compare_memory_to_baseline(snapshot)
    model_dir = _resolve_pinned_snapshot()
    evidence = _canonical_probe_evidence(model_dir=model_dir, audit_manifest=audit_manifest)
    attempts: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="osx-cover-w3-probe-") as temporary:
        for batch_size in PROBE_ORDER:
            independent_dir = Path(temporary) / f"independent-{batch_size}"
            independent = _run_worker_process(
                worker_script=worker_script,
                mode="independent",
                batch_size=batch_size,
                w2_root=w2_root,
                bridge_artifact=bridge_artifact,
                audit_manifest=audit_manifest,
                model_dir=model_dir,
                result_dir=independent_dir,
                timeout_seconds=timeout_seconds,
            )
            if any(result.get("failure") == "out_of_memory" for result in independent):
                attempts.append(
                    {
                        "per_rank_batch_size": batch_size,
                        "status": "failed",
                        "failure": "out_of_memory",
                        "ranks": independent,
                    }
                )
                continue

            ddp_dir = Path(temporary) / f"ddp-{batch_size}"
            ddp = _run_worker_process(
                worker_script=worker_script,
                mode="ddp",
                batch_size=batch_size,
                w2_root=w2_root,
                bridge_artifact=bridge_artifact,
                audit_manifest=audit_manifest,
                model_dir=model_dir,
                result_dir=ddp_dir,
                timeout_seconds=timeout_seconds,
            )
            if any(result.get("failure") == "out_of_memory" for result in ddp):
                attempts.append(
                    {
                        "per_rank_batch_size": batch_size,
                        "status": "failed",
                        "failure": "out_of_memory",
                        "ranks": ddp,
                    }
                )
                continue
            validate_successful_rank_records(ddp, batch_size=batch_size)
            preprocessing_fingerprints = {result["preprocessing_fingerprint"] for result in ddp}
            tokenizer_fingerprints = {result["tokenizer_fingerprint"] for result in ddp}
            if len(preprocessing_fingerprints) != 1 or len(tokenizer_fingerprints) != 1:
                raise ValueError("W3 ranks loaded different preprocessing or tokenizer identities")
            attempts.append({"per_rank_batch_size": batch_size, "status": "passed", "ranks": ddp})
            receipt = build_probe_receipt(
                snapshot=snapshot,
                attempts=attempts,
                selected_batch_size=batch_size,
                memory_drift=memory_drift,
                evidence={
                    **evidence,
                    "preprocessing_fingerprint": preprocessing_fingerprints.pop(),
                    "tokenizer_fingerprint": tokenizer_fingerprints.pop(),
                },
            )
            write_canonical_json(Path(output_path), receipt)
            return receipt
    raise RuntimeError("no W3 batch size fit the approved two-rank probe order")
