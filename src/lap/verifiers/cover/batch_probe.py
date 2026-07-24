"""Canonical two-rank NVIDIA batch-size preflight for W3-03."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
from typing import Any

import torch
from torch.nn.parallel import DistributedDataParallel

from lap.verifiers.cover.bridge_audit import apply_audited_initialization
from lap.verifiers.cover.data import TwoViewDataset
from lap.verifiers.cover.data import W2DatasetGateway
from lap.verifiers.cover.data import make_sampler
from lap.verifiers.cover.model import OpenClipSigLIP2Backbone
from lap.verifiers.cover.model import VerifierConfig
from lap.verifiers.cover.model import VerifierModel
from lap.verifiers.cover.pipeline import _collate
from lap.verifiers.cover.protocol import PROBE_ORDER
from lap.verifiers.cover.protocol import snapshot_nvidia_devices
from lap.verifiers.cover.w3_contracts import BACKBONE_ID
from lap.verifiers.cover.w3_contracts import BACKBONE_REVISION
from lap.verifiers.cover.w3_contracts import content_hash
from lap.verifiers.cover.w3_contracts import write_canonical_json

WORLD_SIZE = 2
_BASELINE = {
    "date": "2026-07-23",
    "gpu_model": "NVIDIA RTX 6000 Ada Generation",
    "physical_total_memory_mib": [49140.0, 49140.0],
    "allocatable_total_memory_mib": [48502.69, 48510.94],
    "free_memory_mib": [47882.56, 48061.75],
}


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


def build_probe_receipt(
    *,
    snapshot: Sequence[Mapping[str, Any]],
    attempts: Sequence[Mapping[str, Any]],
    selected_batch_size: int,
    memory_drift: Mapping[str, Any],
    host: str | None = None,
) -> dict[str, Any]:
    """Build the hashable canonical receipt after a successful DDP probe."""

    if selected_batch_size not in PROBE_ORDER:
        raise ValueError("selected batch size is outside the approved W3 probe order")
    payload = {
        "schema": "osx_cover_w3_batch_probe_v2",
        "status": "complete",
        "host": host or platform.node(),
        "world_size": WORLD_SIZE,
        "probe_order": list(PROBE_ORDER),
        "attempted_batch_sizes": [attempt["per_rank_batch_size"] for attempt in attempts],
        "selected_per_rank_batch_size": selected_batch_size,
        "selection_rule": "first_successful_two_rank_forward_backward",
        "gpu_snapshot": [dict(gpu) for gpu in snapshot],
        "memory_drift": dict(memory_drift),
        "model": {
            "canonical_target": True,
            "backbone": BACKBONE_ID,
            "backbone_revision": BACKBONE_REVISION,
            "two_view": True,
            "trainable_dtype": "float32",
            "frozen_backbone_dtype": "bfloat16",
            "automatic_mixed_precision": False,
        },
        "attempts": [dict(attempt) for attempt in attempts],
        "successful_two_rank_forward_backward": {"batch_size": selected_batch_size, "ranks": [0, 1]},
    }
    payload["content_hash"] = content_hash(payload)
    return payload


def _is_oom(error: BaseException | str) -> bool:
    text = str(error).casefold()
    return isinstance(error, MemoryError) or "out of memory" in text or "cuda error: out of memory" in text


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
        train_dataset = TwoViewDataset(dataset.train, seed=42, training=True, preprocess=backbone.preprocess)
        sampler = make_sampler(train_dataset, seed=42, world_size=WORLD_SIZE, rank=rank)
        loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=sampler,
            collate_fn=_collate,
            num_workers=0,
        )
        batch = next(iter(loader))
        if mode == "ddp":
            model = DistributedDataParallel(model, device_ids=[rank], output_device=rank, broadcast_buffers=False)
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-6, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.reset_peak_memory_stats(device)
        output = model(
            batch["base_rgb"].to(device),
            batch["wrist_rgb"].to(device),
            batch["instructions"],
            batch["action_histories"].to(device),
        )
        base_model = model.module if isinstance(model, DistributedDataParallel) else model
        loss, _ = base_model.contrastive_loss(output)
        if not torch.isfinite(loss):
            raise FloatingPointError("W3 probe loss is nonfinite")
        loss.backward()
        if mode == "ddp":
            torch.distributed.barrier()
        torch.cuda.synchronize(device)
        free_bytes, _ = torch.cuda.mem_get_info(device)
        result.update(
            {
                "status": "passed",
                "forward_backward": "passed",
                "loss": float(loss.detach().cpu()),
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
    if completed.returncode != 0 and non_oom_failures:
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
            attempts.append({"per_rank_batch_size": batch_size, "status": "passed", "ranks": ddp})
            receipt = build_probe_receipt(
                snapshot=snapshot,
                attempts=attempts,
                selected_batch_size=batch_size,
                memory_drift=memory_drift,
            )
            write_canonical_json(Path(output_path), receipt)
            return receipt
    raise RuntimeError("no W3 batch size fit the approved two-rank probe order")
