#!/usr/bin/env python3
"""Two-rank DDP training worker for canonical W3-09 acceptance."""

# ruff: noqa: E402,PLC0415,PLW0603
from __future__ import annotations

import os


def _pin_local_cuda_device() -> None:
    """Isolate one physical GPU per rank before any CUDA context is created."""

    local = os.environ.get("LOCAL_RANK")
    if local is None:
        return
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is None or visible.strip() == "":
        os.environ["CUDA_VISIBLE_DEVICES"] = str(local)
        return
    devices = [part.strip() for part in visible.split(",") if part.strip() != ""]
    index = int(local)
    if 0 <= index < len(devices):
        os.environ["CUDA_VISIBLE_DEVICES"] = devices[index]


_pin_local_cuda_device()

import argparse
import json
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader

from lap.verifiers.cover.bridge_audit import apply_audited_initialization
from lap.verifiers.cover.canonical import next_batch_after_resume
from lap.verifiers.cover.canonical import resolve_initialization_manifest_for_model
from lap.verifiers.cover.checkpoint import build_checkpoint_contract
from lap.verifiers.cover.checkpoint import build_four_state_inventory
from lap.verifiers.cover.checkpoint import build_progress
from lap.verifiers.cover.checkpoint import capture_rng_state
from lap.verifiers.cover.checkpoint import load_training_checkpoint
from lap.verifiers.cover.checkpoint import recorded_cuda_rng_device_count
from lap.verifiers.cover.checkpoint import save_training_checkpoint
from lap.verifiers.cover.data import TwoViewDataset
from lap.verifiers.cover.data import W2DatasetGateway
from lap.verifiers.cover.data import build_epoch_collision_report
from lap.verifiers.cover.data import collate_two_view_batch
from lap.verifiers.cover.data import make_sampler
from lap.verifiers.cover.model import OpenClipSigLIP2Backbone
from lap.verifiers.cover.model import VerifierConfig
from lap.verifiers.cover.model import VerifierModel
from lap.verifiers.cover.protocol import WORLD_SIZE
from lap.verifiers.cover.protocol import RunProtocol
from lap.verifiers.cover.training import create_optimizer
from lap.verifiers.cover.training import make_base_only_config
from lap.verifiers.cover.w3_contracts import canonical_bytes
from lap.verifiers.cover.w3_contracts import content_hash

_GLOO_GROUP = None


def parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--worker-mode", choices=("train", "preaccept"))
    parser.add_argument("--w2-root", type=Path)
    parser.add_argument("--bridge-artifact", type=Path)
    parser.add_argument("--audit-manifest", type=Path)
    parser.add_argument("--protocol-dir", type=Path)
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument("--result-path", type=Path)
    parser.add_argument("--validator", type=Path, default=None)
    parser.add_argument("--use-wrist", choices=("true", "false"), default="true")
    parser.add_argument("--epochs", type=int, default=None)
    return parser


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _resolve_pinned_snapshot() -> Path:
    from huggingface_hub import snapshot_download

    from lap.verifiers.cover.w3_contracts import BACKBONE_ID
    from lap.verifiers.cover.w3_contracts import BACKBONE_REVISION

    return Path(
        snapshot_download(
            BACKBONE_ID.removeprefix("hf-hub:"),
            revision=BACKBONE_REVISION,
            local_files_only=True,
        )
    )


def _build_model(
    *,
    device: torch.device,
    w2_root: Path,
    bridge_artifact: Path,
    audit_manifest: Path,
    model_dir: Path,
    use_wrist: bool,
    validator_path: Path | None = None,
) -> tuple[VerifierModel, W2DatasetGateway, Any, dict[str, Any]]:
    dataset = W2DatasetGateway(w2_root, validator_path=validator_path)
    backbone = OpenClipSigLIP2Backbone(model_name=f"local-dir:{model_dir}")
    config = VerifierConfig() if use_wrist else make_base_only_config(VerifierConfig())
    model = VerifierModel(config, backbone)
    production_audit = json.loads(audit_manifest.read_text(encoding="utf-8"))
    init_manifest = resolve_initialization_manifest_for_model(
        model,
        bridge_artifact=bridge_artifact,
        production_manifest=production_audit,
    )
    apply_audited_initialization(model, bridge_artifact, init_manifest)
    backbone.model.to(dtype=torch.bfloat16)
    model.to(device)
    return model, dataset, backbone, init_manifest


def _load_protocol(protocol_dir: Path) -> tuple[RunProtocol, dict[str, Any]]:
    payload = json.loads((Path(protocol_dir) / "run_protocol.json").read_text(encoding="utf-8"))
    protocol = RunProtocol(
        seed=int(payload["training"]["seed"]),
        sampler_seed=int(payload["sampler"]["seed"]),
        evaluation_seed=int(payload["evaluation"]["seed"]),
        per_rank_batch_size=int(payload["batch"]["per_rank"]),
        world_size=int(payload["sampler"]["world_size"]),
        epochs=int(payload["optimization"]["epochs"]),
        learning_rate=float(payload["optimization"]["learning_rate"]),
        warmup_epochs=int(payload["optimization"]["warmup_epochs"]),
        gradient_clip_norm=float(payload["optimization"]["gradient_clip_norm"]),
        bootstrap_replicates=int(payload["evaluation"]["bootstrap_replicates"]),
    )
    if protocol.world_size != WORLD_SIZE:
        raise ValueError("canonical worker requires world_size=2")
    if protocol.per_rank_batch_size != 64:
        raise ValueError("canonical worker requires per_rank_batch_size=64")
    return protocol, payload


def _build_checkpoint_contract(
    *,
    audit: dict[str, Any],
    protocol_payload: dict[str, Any],
    model: VerifierModel,
    dataset: W2DatasetGateway,
    use_wrist: bool,
) -> dict[str, Any]:
    environment: dict[str, Any] = {
        "torch": torch.__version__,
        "backbone_revision": getattr(model.backbone, "backbone_revision", None),
        "preprocessing_contract": "openclip_model_eval_transform_v1",
        "views": ["base_rgb", "wrist_rgb"] if use_wrist else ["base_rgb"],
        "cuda_rng_device_count": recorded_cuda_rng_device_count(),
        "gradient_synchronization": "two_rank_ddp",
    }
    if not use_wrist:
        environment["variant"] = "base_only"
        environment["deployable"] = False
    w2_identities: dict[str, Any] = {
        "train_manifest_hash": protocol_payload["identities"]["train_manifest_hash"],
        "phrase_manifest_hash": protocol_payload["identities"]["phrase_manifest_hash"],
        "normalization_artifact_hash": dataset.validation_receipt.get(
            "normalization_artifact_hash", dataset.validation_receipt.get("content_hash", "")
        ),
        "w2_validation_receipt": dataset.validation_receipt,
    }
    if not use_wrist:
        w2_identities["variant"] = "base_only"
        w2_identities["deployable"] = False
    return build_checkpoint_contract(
        audit_manifest_sha256=audit["manifest_sha256"],
        bridge_artifact_sha256=audit["artifact"]["sha256"],
        target_fingerprint=audit["target"]["fingerprint"],
        protocol_content_hash=protocol_payload["content_hash"],
        protocol_version=protocol_payload["protocol_version"],
        model_config=model.config.to_dict(),
        four_state_inventory=build_four_state_inventory(),
        w2_identities=w2_identities,
        environment=environment,
    )


def _gather_rng_states(*, cuda_rng_device_count: int) -> dict[int, dict[str, Any]]:
    import torch.distributed as dist

    global _GLOO_GROUP
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local = capture_rng_state(cuda_rng_device_count=cuda_rng_device_count)
    if _GLOO_GROUP is None:
        _GLOO_GROUP = dist.new_group(backend="gloo")
    gathered: list[dict[str, Any] | None] = [None] * world_size
    dist.gather_object(local, gathered if rank == 0 else None, dst=0, group=_GLOO_GROUP)
    if rank != 0:
        return {}
    return {index: state for index, state in enumerate(gathered) if state is not None}


def _configure_nccl_timeouts() -> None:
    """Allow long rank-0 checkpoint I/O without NCCL watchdog false kills."""

    os.environ.setdefault("TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC", "86400")
    os.environ.setdefault("NCCL_TIMEOUT", "86400")


def _local_cuda_device() -> torch.device:
    """After CUDA_VISIBLE_DEVICES pinning, each rank owns local cuda:0."""

    if not torch.cuda.is_available():
        raise RuntimeError("canonical W3 worker requires CUDA")
    torch.cuda.set_device(0)
    return torch.device("cuda", 0)


def _init_ddp() -> int:
    _configure_nccl_timeouts()
    torch.distributed.init_process_group("nccl")
    rank = int(os.environ["LOCAL_RANK"])
    _local_cuda_device()
    return rank


def _ddp_train_step(
    ddp_model: DistributedDataParallel,
    base_model: VerifierModel,
    batch: dict[str, Any],
    optimizer: torch.optim.Optimizer,
    *,
    clip_norm: float,
) -> dict[str, float]:
    ddp_model.train()
    optimizer.zero_grad(set_to_none=True)
    output = ddp_model(
        batch["base_rgb"],
        batch.get("wrist_rgb"),
        batch["instructions"],
        batch["action_histories"],
    )
    loss, metrics = base_model.contrastive_loss(output)
    if not torch.isfinite(loss):
        raise FloatingPointError("W3 canonical DDP training loss is nonfinite")
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(
        [parameter for parameter in base_model.parameters() if parameter.requires_grad],
        clip_norm,
    )
    if not torch.isfinite(torch.as_tensor(grad_norm)):
        raise FloatingPointError("W3 canonical DDP gradient norm is nonfinite")
    optimizer.step()
    with torch.no_grad():
        base_model.logit_scale.clamp_(0.0, torch.log(torch.tensor(100.0, device=base_model.logit_scale.device)))
    metrics["gradient_norm"] = float(grad_norm)
    return metrics


def _validation_loss(
    model: VerifierModel,
    validation_dataset: TwoViewDataset,
    *,
    device: torch.device,
    batch_size: int,
) -> float:
    loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_two_view_batch)
    losses = []
    model.eval()
    with torch.no_grad():
        for validation_batch in loader:
            output = model(
                validation_batch["base_rgb"].to(device),
                validation_batch["wrist_rgb"].to(device),
                validation_batch["instructions"],
                validation_batch["action_histories"].to(device),
            )
            losses.append(float(model.contrastive_loss(output)[0]))
    return float(np.mean(losses)) if losses else float("inf")


def _configure_nccl_timeouts() -> None:
    """Allow long rank-0 checkpoint I/O without NCCL watchdog false kills."""

    os.environ.setdefault("TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC", "86400")
    os.environ.setdefault("NCCL_TIMEOUT", "86400")


def _destroy_process_group_quiet() -> None:
    try:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
    except Exception:
        pass


def _write_worker_error(path: Path, error: BaseException) -> None:
    import traceback

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(traceback.format_exception(type(error), error, error.__traceback__)), encoding="utf-8")


def _exit_worker(*, exit_code: int, process_group: bool) -> None:
    """Exit without an NCCL barrier so rank-0 I/O cannot hang peers."""

    if process_group:
        _destroy_process_group_quiet()
    os._exit(exit_code)


def _gather_objects(local: Any) -> list[Any]:
    import torch.distributed as dist

    global _GLOO_GROUP
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if _GLOO_GROUP is None:
        _GLOO_GROUP = dist.new_group(backend="gloo")
    gathered: list[Any | None] = [None] * world_size
    dist.gather_object(local, gathered if rank == 0 else None, dst=0, group=_GLOO_GROUP)
    if rank != 0:
        return []
    return [item for item in gathered if item is not None]


def run_preaccept_worker(
    *,
    w2_root: Path,
    bridge_artifact: Path,
    audit_manifest: Path,
    protocol_dir: Path,
    result_path: Path,
    validator_path: Path | None,
) -> None:
    """Fresh two-rank step, then both-rank W3-06 exact resume with continuation equivalence."""

    from lap.verifiers.cover.batch_probe import _parameter_fingerprint

    rank = int(os.environ["LOCAL_RANK"])
    process_group = False
    exit_code = 0
    error_path = Path(result_path).parent / f"preaccept_worker_error_rank{rank}.txt"
    try:
        _seed_everything(42)
        _init_ddp()
        process_group = True
        device = _local_cuda_device()
        protocol, protocol_payload = _load_protocol(protocol_dir)
        model_dir = _resolve_pinned_snapshot()
        use_wrist = True
        model, dataset, backbone, init_manifest = _build_model(
            device=device,
            w2_root=w2_root,
            bridge_artifact=bridge_artifact,
            audit_manifest=audit_manifest,
            model_dir=model_dir,
            use_wrist=use_wrist,
            validator_path=validator_path,
        )
        if backbone.asset_fingerprints is None:
            raise ValueError("canonical preaccept did not load pinned preprocessing and tokenizer assets")
        ddp_model = DistributedDataParallel(model, device_ids=[0], output_device=0, broadcast_buffers=False)
        base_model = ddp_model.module
        train_dataset = TwoViewDataset(dataset.train, seed=protocol.seed, training=True, preprocess=backbone.preprocess)
        sampler = make_sampler(train_dataset, seed=protocol.sampler_seed, world_size=WORLD_SIZE, rank=rank)
        loader = DataLoader(
            train_dataset,
            batch_size=protocol.per_rank_batch_size,
            sampler=sampler,
            collate_fn=collate_two_view_batch,
            num_workers=0,
        )
        iterator = iter(loader)
        batch1 = next(iterator)
        device_batch1 = {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch1.items()}
        optimizer, scheduler = create_optimizer(base_model, protocol)
        _ddp_train_step(
            ddp_model,
            base_model,
            device_batch1,
            optimizer,
            clip_norm=protocol.gradient_clip_norm,
        )
        torch.distributed.barrier()
        contract = _build_checkpoint_contract(
            audit=init_manifest,
            protocol_payload=protocol_payload,
            model=base_model,
            dataset=dataset,
            use_wrist=use_wrist,
        )
        cuda_rng_device_count = contract["environment"]["cuda_rng_device_count"]
        torch.distributed.barrier()
        rng_states = _gather_rng_states(cuda_rng_device_count=cuda_rng_device_count)
        checkpoint_path = Path(result_path).parent / "preaccept_checkpoint.pt"
        local_sampler_state = {"epoch": 0, "rank": rank, "world_size": WORLD_SIZE}
        if rank == 0:
            progress = build_progress(
                epoch=1,
                global_step=1,
                best_metric=0.0,
                world_size=WORLD_SIZE,
            )
            # Shared W3-06 sampler_state carries epoch/world_size; per-rank rank is restored locally.
            save_training_checkpoint(
                checkpoint_path,
                model=base_model,
                optimizer=optimizer,
                scheduler=scheduler,
                progress=progress,
                contract=contract,
                sampler_state={"epoch": 0, "rank": 0, "world_size": WORLD_SIZE},
                rank=0,
                rng_states=rng_states,
            )
        torch.distributed.barrier()

        batch2 = next(iterator)
        device_batch2 = {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch2.items()}
        uninterrupted_sample_ids = list(batch2["sample_ids"])
        metrics_uninterrupted = _ddp_train_step(
            ddp_model,
            base_model,
            device_batch2,
            optimizer,
            clip_norm=protocol.gradient_clip_norm,
        )
        fingerprint_uninterrupted = _parameter_fingerprint(base_model, trainable=True)
        torch.distributed.barrier()

        loaded = load_training_checkpoint(
            checkpoint_path,
            model=base_model,
            optimizer=optimizer,
            scheduler=scheduler,
            expected_contract=contract,
            rank=rank,
        )
        batches_already_consumed = int(loaded["progress"]["global_step"])
        restored_epoch = int(loaded["sampler_state"]["epoch"])
        resumed_batch = next_batch_after_resume(
            dataset=train_dataset,
            seed=protocol.sampler_seed,
            world_size=WORLD_SIZE,
            rank=rank,
            epoch=restored_epoch,
            batches_already_consumed=batches_already_consumed,
            batch_size=protocol.per_rank_batch_size,
            collate_fn=collate_two_view_batch,
        )
        device_batch_resumed = {
            key: value.to(device) if torch.is_tensor(value) else value for key, value in resumed_batch.items()
        }
        resumed_sample_ids = list(resumed_batch["sample_ids"])
        metrics_resumed = _ddp_train_step(
            ddp_model,
            base_model,
            device_batch_resumed,
            optimizer,
            clip_norm=protocol.gradient_clip_norm,
        )
        fingerprint_resumed = _parameter_fingerprint(base_model, trainable=True)
        local_evidence = {
            "rank": rank,
            "loss_uninterrupted": float(metrics_uninterrupted["loss"]),
            "loss_resumed": float(metrics_resumed["loss"]),
            "loss_match": float(metrics_uninterrupted["loss"]) == float(metrics_resumed["loss"]),
            "gradient_norm_uninterrupted": float(metrics_uninterrupted["gradient_norm"]),
            "gradient_norm_resumed": float(metrics_resumed["gradient_norm"]),
            "parameter_fingerprint_uninterrupted": fingerprint_uninterrupted,
            "parameter_fingerprint_resumed": fingerprint_resumed,
            "parameter_fingerprint_match": fingerprint_uninterrupted == fingerprint_resumed,
            "sampler_state_local": local_sampler_state,
            "sampler_epoch_restored": restored_epoch,
            "sampler_batches_skipped": batches_already_consumed,
            "uninterrupted_sample_ids": uninterrupted_sample_ids,
            "resumed_sample_ids": resumed_sample_ids,
            "next_batch_sample_ids_match": uninterrupted_sample_ids == resumed_sample_ids,
        }
        gathered = _gather_objects(local_evidence)
        torch.distributed.barrier()
        _destroy_process_group_quiet()
        process_group = False
        if rank == 0:
            by_rank = {int(item["rank"]): item for item in gathered}
            if set(by_rank) != {0, 1}:
                raise RuntimeError(f"preaccept resume did not restore both ranks: {sorted(by_rank)}")
            loss_matches = {rank_id: bool(item["loss_match"]) for rank_id, item in by_rank.items()}
            fingerprint_matches = {
                rank_id: bool(item["parameter_fingerprint_match"]) for rank_id, item in by_rank.items()
            }
            sample_id_matches = {
                rank_id: bool(item["next_batch_sample_ids_match"]) for rank_id, item in by_rank.items()
            }
            if (
                not all(loss_matches.values())
                or not all(fingerprint_matches.values())
                or not all(sample_id_matches.values())
            ):
                raise RuntimeError(
                    "preaccept uninterrupted-versus-resumed mismatch: "
                    f"loss={loss_matches} fingerprints={fingerprint_matches} sample_ids={sample_id_matches}"
                )
            receipt = {
                "schema": "osx_cover_w3_preaccept_checkpoint_v1",
                "two_rank_step": "passed",
                "exact_resume": "passed",
                "uninterrupted_versus_resumed": "passed",
                "ranks_restored": [0, 1],
                "resume_equivalence": {
                    "rank0_loss_match": loss_matches[0],
                    "rank1_loss_match": loss_matches[1],
                    "gradient_fingerprint_match": all(fingerprint_matches.values()),
                    "next_batch_sample_ids_match": all(sample_id_matches.values()),
                },
                "rank_evidence": by_rank,
                "checkpoint": str(checkpoint_path),
                "world_size": WORLD_SIZE,
                "per_rank_batch_size": protocol.per_rank_batch_size,
                "gradient_synchronization": "two_rank_ddp",
            }
            receipt["content_hash"] = content_hash(receipt)
            result_path.parent.mkdir(parents=True, exist_ok=True)
            result_path.write_bytes(canonical_bytes(receipt) + b"\n")
    except Exception as error:
        exit_code = 1
        _write_worker_error(error_path, error)
    _exit_worker(exit_code=exit_code, process_group=process_group)


def run_train_worker(
    *,
    w2_root: Path,
    bridge_artifact: Path,
    audit_manifest: Path,
    protocol_dir: Path,
    checkpoint_dir: Path,
    validator_path: Path | None,
    use_wrist: bool,
    epochs: int | None,
) -> None:
    rank = int(os.environ["LOCAL_RANK"])
    process_group = False
    exit_code = 0
    try:
        _seed_everything(42)
        _init_ddp()
        process_group = True
        device = _local_cuda_device()
        protocol, protocol_payload = _load_protocol(protocol_dir)
        total_epochs = protocol.epochs if epochs is None else epochs
        model_dir = _resolve_pinned_snapshot()
        model, dataset, backbone, init_manifest = _build_model(
            device=device,
            w2_root=w2_root,
            bridge_artifact=bridge_artifact,
            audit_manifest=audit_manifest,
            model_dir=model_dir,
            use_wrist=use_wrist,
            validator_path=validator_path,
        )
        ddp_model = DistributedDataParallel(model, device_ids=[0], output_device=0, broadcast_buffers=False)
        base_model = ddp_model.module
        optimizer, scheduler = create_optimizer(base_model, protocol)
        train_dataset = TwoViewDataset(dataset.train, seed=protocol.seed, training=True, preprocess=backbone.preprocess)
        validation_dataset = TwoViewDataset(dataset.validation, training=False, preprocess=backbone.preprocess)
        train_sampler = make_sampler(train_dataset, seed=protocol.sampler_seed, world_size=WORLD_SIZE, rank=rank)
        train_loader = DataLoader(
            train_dataset,
            batch_size=protocol.per_rank_batch_size,
            sampler=train_sampler,
            collate_fn=collate_two_view_batch,
            num_workers=0,
        )
        contract = _build_checkpoint_contract(
            audit=init_manifest,
            protocol_payload=protocol_payload,
            model=base_model,
            dataset=dataset,
            use_wrist=use_wrist,
        )
        cuda_rng_device_count = contract["environment"]["cuda_rng_device_count"]
        history: list[dict[str, Any]] = []
        best_validation_loss = float("inf")
        global_step = 0
        checkpoint_dir = Path(checkpoint_dir)
        if rank == 0:
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
        for epoch in range(total_epochs):
            train_dataset.set_epoch(epoch)
            train_sampler.set_epoch(epoch)
            epoch_metrics: list[dict[str, float]] = []
            epoch_rows: list[dict[str, Any]] = []
            for raw_batch in train_loader:
                if rank == 0:
                    epoch_rows.extend(
                        {
                            "sample_id": sample_id,
                            "episode_id": episode_id,
                            "instruction": instruction,
                            "history": action_history.detach().cpu().numpy(),
                        }
                        for sample_id, episode_id, instruction, action_history in zip(
                            raw_batch["sample_ids"],
                            raw_batch["episode_ids"],
                            raw_batch["instructions"],
                            raw_batch["action_histories"],
                            strict=True,
                        )
                    )
                device_batch = {
                    key: value.to(device) if torch.is_tensor(value) else value for key, value in raw_batch.items()
                }
                epoch_metrics.append(
                    _ddp_train_step(
                        ddp_model,
                        base_model,
                        device_batch,
                        optimizer,
                        clip_norm=protocol.gradient_clip_norm,
                    )
                )
                global_step += 1
            scheduler.step()
            validation_loss = None
            if rank == 0:
                validation_loss = _validation_loss(
                    base_model,
                    validation_dataset,
                    device=device,
                    batch_size=protocol.per_rank_batch_size,
                )
            epoch_record = {
                "epoch": epoch,
                "loss": float(np.mean([item["loss"] for item in epoch_metrics])) if epoch_metrics else float("nan"),
                "validation_loss": validation_loss,
            }
            # All ranks must participate in gather_object before rank-0 checkpointing.
            rng_states = _gather_rng_states(cuda_rng_device_count=cuda_rng_device_count)
            if rank == 0:
                epoch_record["sampler_diagnostics"] = build_epoch_collision_report(epoch_rows)
                history.append(epoch_record)
                selection_loss = validation_loss if validation_loss is not None else epoch_record["loss"]
                improved = selection_loss < best_validation_loss
                if improved:
                    best_validation_loss = selection_loss
                progress = build_progress(
                    epoch=epoch + 1,
                    global_step=global_step,
                    best_metric=best_validation_loss,
                    world_size=WORLD_SIZE,
                )
                sampler_state = {"epoch": epoch, "rank": 0, "world_size": WORLD_SIZE}
                save_training_checkpoint(
                    checkpoint_dir / "latest.pt",
                    model=base_model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    progress=progress,
                    contract=contract,
                    sampler_state=sampler_state,
                    rank=0,
                    rng_states=rng_states,
                )
                if improved:
                    save_training_checkpoint(
                        checkpoint_dir / "best.pt",
                        model=base_model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        progress=progress,
                        contract=contract,
                        sampler_state=sampler_state,
                        rank=0,
                        rng_states=rng_states,
                    )
            torch.distributed.barrier()
        if rank == 0:
            receipt = {
                "schema": "osx_cover_w3_canonical_ddp_train_v1",
                "history": history,
                "progress": build_progress(
                    epoch=len(history),
                    global_step=global_step,
                    best_metric=best_validation_loss if best_validation_loss < float("inf") else float("nan"),
                    world_size=WORLD_SIZE,
                ),
                "world_size": WORLD_SIZE,
                "gradient_synchronization": "two_rank_ddp",
                "model_config": base_model.config.to_dict(),
                "backbone_revision": getattr(base_model.backbone, "backbone_revision", None),
                "resumable_checkpoints": ["latest.pt", "best.pt"],
            }
            receipt["content_hash"] = content_hash(receipt)
            (checkpoint_dir / "train_result.json").write_bytes(canonical_bytes(receipt) + b"\n")
        torch.distributed.barrier()
    except Exception as error:
        exit_code = 1
        _write_worker_error(Path(checkpoint_dir) / f"train_worker_error_rank{rank}.txt", error)
    _exit_worker(exit_code=exit_code, process_group=process_group)


def main() -> None:
    args = parser().parse_args()
    if not args.worker:
        raise ValueError("invoke via torch.distributed.run with --worker")
    if args.worker_mode == "preaccept":
        if args.result_path is None:
            raise ValueError("preaccept worker requires --result-path")
        run_preaccept_worker(
            w2_root=args.w2_root,
            bridge_artifact=args.bridge_artifact,
            audit_manifest=args.audit_manifest,
            protocol_dir=args.protocol_dir,
            result_path=args.result_path,
            validator_path=args.validator,
        )
        return
    if args.worker_mode == "train":
        if args.checkpoint_dir is None:
            raise ValueError("train worker requires --checkpoint-dir")
        run_train_worker(
            w2_root=args.w2_root,
            bridge_artifact=args.bridge_artifact,
            audit_manifest=args.audit_manifest,
            protocol_dir=args.protocol_dir,
            checkpoint_dir=args.checkpoint_dir,
            validator_path=args.validator,
            use_wrist=args.use_wrist == "true",
            epochs=args.epochs,
        )
        return
    raise ValueError(f"unsupported worker mode: {args.worker_mode}")


if __name__ == "__main__":
    main()
