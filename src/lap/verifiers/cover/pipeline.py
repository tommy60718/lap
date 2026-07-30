"""End-to-end W3 composition used by the public script and acceptance tests."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from lap.verifiers.cover.bridge_audit import apply_audited_initialization
from lap.verifiers.cover.checkpoint import build_checkpoint_contract
from lap.verifiers.cover.checkpoint import build_four_state_inventory
from lap.verifiers.cover.checkpoint import build_progress
from lap.verifiers.cover.checkpoint import capture_rng_state
from lap.verifiers.cover.checkpoint import publish_deployment_bundle
from lap.verifiers.cover.checkpoint import recorded_cuda_rng_device_count
from lap.verifiers.cover.checkpoint import save_training_checkpoint
from lap.verifiers.cover.checkpoint import validate_best_checkpoint_for_evaluation
from lap.verifiers.cover.command import preflight_w3
from lap.verifiers.cover.data import TwoViewDataset
from lap.verifiers.cover.data import W2DatasetGateway
from lap.verifiers.cover.data import build_epoch_collision_report
from lap.verifiers.cover.data import collate_two_view_batch
from lap.verifiers.cover.data import make_sampler
from lap.verifiers.cover.evaluator import compare_ablation
from lap.verifiers.cover.evaluator import evaluate_embeddings
from lap.verifiers.cover.evaluator import evaluate_explicit_best_checkpoint
from lap.verifiers.cover.evaluator import load_fixed_best_checkpoint_logit_scale
from lap.verifiers.cover.evaluator import require_explicit_best_checkpoint
from lap.verifiers.cover.evaluator import require_matched_ablation_identities
from lap.verifiers.cover.model import OpenClipSigLIP2Backbone
from lap.verifiers.cover.model import TinyFrozenBackbone
from lap.verifiers.cover.model import VerifierConfig
from lap.verifiers.cover.model import VerifierModel
from lap.verifiers.cover.protocol import RunProtocol
from lap.verifiers.cover.protocol import validate_protocol_directory
from lap.verifiers.cover.training import base_only_config_delta
from lap.verifiers.cover.training import create_optimizer
from lap.verifiers.cover.training import make_base_only_config
from lap.verifiers.cover.training import require_acceptance_metrics
from lap.verifiers.cover.training import require_permitted_config_delta
from lap.verifiers.cover.training import reset_approved_seed
from lap.verifiers.cover.training import train_one_batch
from lap.verifiers.cover.w3_contracts import canonical_bytes
from lap.verifiers.cover.w3_contracts import content_hash
from lap.verifiers.cover.w3_contracts import sha256_file


def _verifier_config_from_dict(payload: dict[str, Any]) -> VerifierConfig:
    allowed = {
        "backbone_width",
        "embedding_width",
        "visual_tokens",
        "num_heads",
        "pooling_layers",
        "trajectory_layers",
        "feed_forward_width",
        "history_length",
        "action_width",
        "use_wrist",
        "text_aware_extraction_contract",
        "trajectory_activation",
        "trajectory_position_contract",
        "attention_pooling_contract",
    }
    return VerifierConfig(**{key: payload[key] for key in allowed if key in payload})


def train_model(
    model: VerifierModel,
    dataset: TwoViewDataset,
    *,
    protocol: RunProtocol,
    device: torch.device,
    epochs: int | None = None,
    validation_dataset: TwoViewDataset | None = None,
    checkpoint_dir: Path | None = None,
    checkpoint_contract: dict[str, Any] | None = None,
) -> dict[str, Any]:
    model.to(device)
    optimizer, scheduler = create_optimizer(model, protocol)
    sampler = make_sampler(dataset, seed=protocol.sampler_seed, world_size=protocol.world_size)
    loader = DataLoader(
        dataset,
        batch_size=protocol.per_rank_batch_size,
        sampler=sampler,
        collate_fn=collate_two_view_batch,
    )
    history = []
    best_validation_loss = float("inf")
    global_step = 0
    if checkpoint_dir is not None:
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    for epoch in range(epochs if epochs is not None else protocol.epochs):
        dataset.set_epoch(epoch)
        sampler.set_epoch(epoch)
        epoch_metrics = []
        epoch_rows: list[dict[str, Any]] = []
        for raw_batch in loader:
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
            epoch_metrics.append(train_one_batch(model, device_batch, optimizer))
            global_step += 1
        scheduler.step()
        validation_loss = None
        if validation_dataset is not None:
            model.eval()
            validation_loader = DataLoader(
                validation_dataset,
                batch_size=protocol.per_rank_batch_size,
                shuffle=False,
                collate_fn=collate_two_view_batch,
            )
            validation_losses = []
            with torch.no_grad():
                for validation_batch in validation_loader:
                    output = model(
                        validation_batch["base_rgb"].to(device),
                        validation_batch["wrist_rgb"].to(device),
                        validation_batch["instructions"],
                        validation_batch["action_histories"].to(device),
                    )
                    validation_losses.append(float(model.contrastive_loss(output)[0]))
            validation_loss = float(np.mean(validation_losses)) if validation_losses else float("inf")
        epoch_record = {
            "epoch": epoch,
            "loss": float(np.mean([item["loss"] for item in epoch_metrics])) if epoch_metrics else float("nan"),
            "validation_loss": validation_loss,
            "sampler_diagnostics": build_epoch_collision_report(epoch_rows),
        }
        history.append(epoch_record)
        if checkpoint_dir is not None and checkpoint_contract is not None:
            selection_loss = validation_loss if validation_loss is not None else epoch_record["loss"]
            improved = selection_loss < best_validation_loss
            if improved:
                best_validation_loss = selection_loss
            progress = build_progress(
                epoch=epoch + 1,
                global_step=global_step,
                best_metric=best_validation_loss,
                world_size=protocol.world_size,
            )
            sampler_state = {"epoch": epoch, "rank": 0, "world_size": protocol.world_size}
            cuda_rng_device_count = checkpoint_contract["environment"]["cuda_rng_device_count"]
            rng_states = {
                rank: capture_rng_state(cuda_rng_device_count=cuda_rng_device_count)
                for rank in range(protocol.world_size)
            }
            save_training_checkpoint(
                Path(checkpoint_dir) / "latest.pt",
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                progress=progress,
                contract=checkpoint_contract,
                sampler_state=sampler_state,
                rank=0,
                rng_states=rng_states,
            )
            if improved:
                save_training_checkpoint(
                    Path(checkpoint_dir) / "best.pt",
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    progress=progress,
                    contract=checkpoint_contract,
                    sampler_state=sampler_state,
                    rank=0,
                    rng_states=rng_states,
                )
    return {
        "history": history,
        "progress": build_progress(
            epoch=len(history),
            global_step=global_step,
            best_metric=best_validation_loss if best_validation_loss < float("inf") else float("nan"),
            world_size=protocol.world_size,
        ),
        "optimizer": optimizer,
        "scheduler": scheduler,
    }


def collect_embeddings(
    model: VerifierModel, dataset: TwoViewDataset, *, device: torch.device, batch_size: int
) -> dict[str, Any]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_two_view_batch)
    semantic = []
    action = []
    sample_ids = []
    episodes = []
    conditions = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            output = model(
                batch["base_rgb"].to(device),
                batch["wrist_rgb"].to(device),
                batch["instructions"],
                batch["action_histories"].to(device),
            )
            semantic.append(output["semantic_embedding"].cpu().numpy())
            action.append(output["action_embedding"].cpu().numpy())
            sample_ids.extend(batch["sample_ids"])
            episodes.extend(batch["episode_ids"])
            conditions.extend(batch["conditions"])
    return {
        "semantic": np.concatenate(semantic),
        "action": np.concatenate(action),
        "sample_ids": sample_ids,
        "episode_ids": episodes,
        "conditions": conditions,
    }


def build_matched_ablation_identity(
    *,
    protocol: RunProtocol,
    sample_ids: Sequence[str] | list[str],
    phrase_manifest_hash: str,
    protocol_content_hash: str,
    evaluation_artifact_hashes: Mapping[str, str],
) -> dict[str, Any]:
    """Identity receipt proving matched rows/phrases/sampler/optimizer/selection/eval."""

    return {
        "seed": int(protocol.seed),
        "sampler_seed": int(protocol.sampler_seed),
        "phrase_manifest_hash": phrase_manifest_hash,
        "optimizer": {
            "name": "AdamW",
            "learning_rate": float(protocol.learning_rate),
            "betas": [float(protocol.betas[0]), float(protocol.betas[1])],
            "epsilon": float(protocol.epsilon),
            "weight_decay": float(protocol.weight_decay),
            "warmup_epochs": int(protocol.warmup_epochs),
            "epochs": int(protocol.epochs),
        },
        "checkpoint_selection": "lowest_validation_loss_earliest_epoch",
        "evaluation": {
            "shuffled_pairs_hash": evaluation_artifact_hashes["shuffled_pairs_hash"],
            "nearby_pairs_hash": evaluation_artifact_hashes["nearby_pairs_hash"],
            "bootstrap_indices_hash": evaluation_artifact_hashes["bootstrap_indices_hash"],
        },
        "sample_ids": list(sample_ids),
        "protocol_content_hash": protocol_content_hash,
    }


def bootstrap_indices_content_hash(bootstrap_indices: np.ndarray | list[Any]) -> str:
    """Content identity for bootstrap arrays bound into matched ablation receipts."""

    return content_hash({"bootstrap_indices": np.asarray(bootstrap_indices).tolist()})


def require_bootstrap_matches_matched_identity(bootstrap_indices: np.ndarray, identity: Mapping[str, Any]) -> None:
    """Reject substituted/mutated bootstrap content before comparison or output."""

    evaluation = identity.get("evaluation")
    if not isinstance(evaluation, Mapping) or "bootstrap_indices_hash" not in evaluation:
        raise ValueError("matched ablation identity missing bootstrap_indices_hash")
    actual = bootstrap_indices_content_hash(bootstrap_indices)
    if actual != evaluation["bootstrap_indices_hash"]:
        raise ValueError("bootstrap indices drifted from matched evaluation identity")


def _evaluation_artifact_hashes(
    *,
    shuffled_pairs: list[dict[str, str]],
    nearby_pairs: list[dict[str, str]],
    bootstrap_indices: np.ndarray,
) -> dict[str, str]:
    return {
        "shuffled_pairs_hash": content_hash({"pairs": shuffled_pairs}),
        "nearby_pairs_hash": content_hash({"pairs": nearby_pairs}),
        "bootstrap_indices_hash": bootstrap_indices_content_hash(bootstrap_indices),
    }


def run_matched_base_only(
    *,
    two_view_model: VerifierModel,
    bridge_artifact: Path,
    audit_manifest: dict[str, Any],
    dataset: W2DatasetGateway,
    protocol: RunProtocol,
    shuffled_pairs: list[dict[str, str]],
    nearby_pairs: list[dict[str, str]],
    bootstrap_indices: np.ndarray,
    device: torch.device,
    two_view_identity: Mapping[str, Any] | None = None,
    phrase_manifest_hash: str = "",
    protocol_content_hash: str = "",
) -> dict[str, Any]:
    """Train/evaluate the nondeployable wrist-omitting control on shared artifacts."""
    reset_approved_seed(protocol.seed)
    base_config = make_base_only_config(two_view_model.config)
    config_delta = base_only_config_delta(two_view_model.config, base_config)
    base_model = VerifierModel(base_config, two_view_model.backbone)
    apply_audited_initialization(base_model, Path(bridge_artifact), audit_manifest)
    preprocess = getattr(two_view_model.backbone, "preprocess", None)
    dataset_kwargs = {} if preprocess is None else {"preprocess": preprocess}
    train_dataset = TwoViewDataset(dataset.train, seed=protocol.seed, training=True, **dataset_kwargs)
    training = train_model(base_model, train_dataset, protocol=protocol, device=device)
    validation_dataset = TwoViewDataset(dataset.validation, training=False, **dataset_kwargs)
    embeddings = collect_embeddings(
        base_model, validation_dataset, device=device, batch_size=protocol.per_rank_batch_size
    )
    report = evaluate_embeddings(
        embeddings["semantic"],
        embeddings["action"],
        sample_ids=embeddings["sample_ids"],
        episode_ids=embeddings["episode_ids"],
        conditions=embeddings["conditions"],
        shuffled_pairs=shuffled_pairs,
        nearby_pairs=nearby_pairs,
        bootstrap_indices=bootstrap_indices,
        checkpoint_logit_scale=float(base_model.logit_scale.detach().clamp(0.0, np.log(100.0)).exp()),
        strict_protocol=True,
    )
    report["variant"] = "base_only"
    report["deployable"] = False
    report["training"] = training["history"]
    report["content_hash"] = content_hash({key: value for key, value in report.items() if key != "content_hash"})
    identity = build_matched_ablation_identity(
        protocol=protocol,
        sample_ids=embeddings["sample_ids"],
        phrase_manifest_hash=phrase_manifest_hash
        or str(
            getattr(dataset, "validation_receipt", {}).get("phrase_manifest_hash")
            or (two_view_identity or {}).get("phrase_manifest_hash", "")
        ),
        protocol_content_hash=protocol_content_hash or str((two_view_identity or {}).get("protocol_content_hash", "")),
        evaluation_artifact_hashes=_evaluation_artifact_hashes(
            shuffled_pairs=shuffled_pairs,
            nearby_pairs=nearby_pairs,
            bootstrap_indices=bootstrap_indices,
        ),
    )
    if two_view_identity is not None:
        require_matched_ablation_identities(two_view_identity, identity)
    return {
        "model": base_model,
        "training": training,
        "report": report,
        "identity": identity,
        "config_delta": config_delta,
    }


def matched_base_only_run_to_paired_ablation_report(
    *,
    two_view_report: Mapping[str, Any],
    two_view_identity: Mapping[str, Any],
    base_only_result: Mapping[str, Any],
    bootstrap_indices: np.ndarray,
    output_root: Path,
) -> dict[str, Any]:
    """Public seam: matched base-only evidence → paired ablation report (never deployable)."""

    require_matched_ablation_identities(two_view_identity, base_only_result["identity"])
    require_bootstrap_matches_matched_identity(bootstrap_indices, two_view_identity)
    require_bootstrap_matches_matched_identity(bootstrap_indices, base_only_result["identity"])
    base_report = base_only_result["report"]
    if base_report.get("deployable") is not False or base_report.get("variant") != "base_only":
        raise ValueError("base-only report must be explicitly nondeployable evidence")
    if "config_delta" not in base_only_result:
        raise ValueError("matched base-only result must include the controlled config delta")
    config_delta = require_permitted_config_delta(base_only_result["config_delta"])
    ablation = compare_ablation(dict(two_view_report), dict(base_report), bootstrap_indices=bootstrap_indices)
    payload = {
        "schema": "osx_cover_w3_paired_ablation_report_v1",
        "deployable": False,
        "w5_eligible": False,
        "accepted": False,
        "variant": "base_only_evidence",
        "config_delta": config_delta,
        "matched_identity": base_only_result["identity"],
        "ablation": ablation,
        "two_view_report_hash": two_view_report.get("content_hash"),
        "base_only_report_hash": base_report.get("content_hash"),
    }
    payload["content_hash"] = content_hash(payload)
    root = Path(output_root)
    if root.exists():
        raise FileExistsError(root)
    root.mkdir(parents=True)
    (root / "paired_ablation.json").write_bytes(canonical_bytes(payload) + b"\n")
    (root / "EVIDENCE_ONLY_NONDEPLOYABLE").write_text(
        "W3-08 matched base-only ablation evidence; not a W5 deployment candidate.\n",
        encoding="utf-8",
    )
    return payload


def run_fixture_end_to_end(*, output_root: Path) -> dict[str, Any]:
    """Produce a tiny deterministic package used by command-level acceptance tests."""
    output_root = Path(output_root)
    if output_root.exists():
        raise FileExistsError(output_root)
    model = VerifierModel(
        VerifierConfig(
            backbone_width=32,
            embedding_width=16,
            visual_tokens=8,
            num_heads=4,
            pooling_layers=1,
            trajectory_layers=1,
            feed_forward_width=32,
        ),
        TinyFrozenBackbone(width=32, tokens=8),
    )
    # The fixture executes one direct diagnostic step, but its serialized
    # contract must still use the immutable W3 constants rather than inventing
    # a second protocol variant.
    protocol = RunProtocol(per_rank_batch_size=16)
    root = output_root.parent / f".{output_root.name}.staging"
    root.mkdir(parents=True)
    try:
        optimizer, scheduler = create_optimizer(model, protocol)
        images = torch.zeros(2, 3, 384, 384)
        histories = torch.zeros(2, 10, 7)
        metrics = train_one_batch(
            model,
            {
                "base_rgb": images,
                "wrist_rgb": images,
                "instructions": ["insert peg", "insert peg"],
                "action_histories": histories,
            },
            optimizer,
        )
        save_training_checkpoint(
            root / "latest.pt",
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            progress=build_progress(epoch=1, global_step=1, best_metric=float(metrics["loss"]), world_size=1),
            contract=build_checkpoint_contract(
                audit_manifest_sha256="0" * 64,
                bridge_artifact_sha256="0" * 64,
                target_fingerprint="0" * 64,
                protocol_content_hash="0" * 64,
                protocol_version="w3-g02-accepted-run-v1",
                model_config=model.config.to_dict(),
                four_state_inventory=build_four_state_inventory(),
                w2_identities={
                    "train_manifest_hash": "fixture",
                    "phrase_manifest_hash": "fixture",
                    "normalization_artifact_hash": "fixture",
                },
                environment={
                    "torch": torch.__version__,
                    "fixture": True,
                    "cuda_rng_device_count": recorded_cuda_rng_device_count(),
                },
            ),
            sampler_state={"epoch": 0, "rank": 0, "world_size": 1},
            rank=0,
        )
        metadata = {
            "deployable": True,
            "model_config": model.config.to_dict(),
            "w3_01_initialization": {
                "audit_manifest_sha256": "0" * 64,
                "bridge_artifact_sha256": "0" * 64,
                "target_fingerprint": "0" * 64,
            },
            "w3_03_protocol": {
                "protocol_content_hash": "0" * 64,
                "protocol_version": "w3-g02-accepted-run-v1",
            },
            "four_state_inventory": build_four_state_inventory(),
            "package_scope": "test_scoped_fixture_not_canonical_w3_09",
            "scorer_compatibility": {
                "model_schema_version": "osx_cover_verifier_checkpoint_v1",
                "views": ["base_rgb", "wrist_rgb"],
                "preprocessing_contract": "fixture_384_center_crop_v1",
                "action_dimension": 7,
                "action_order": ["dx", "dy", "dz", "rotation_x", "rotation_y", "rotation_z", "gripper"],
                "history_length": 10,
                "representation_id": "ur5e_cover_relative_eef_v1",
                "normalization_artifact_hash": "fixture",
                "input_dtype": "float32",
                "output_shape_rank": 1,
            },
        }
        deployment_root = root / "deployment"
        publish_deployment_bundle(deployment_root, model=model, metadata=metadata, accepted_marker=False)
        report = {"schema": "osx_cover_w3_fixture_acceptance_v1", "training": metrics, "deployment": "deployment"}
        (root / "acceptance.json").write_bytes(
            canonical_bytes({**report, "content_hash": content_hash(report)}) + b"\n"
        )
        root.rename(output_root)
    except Exception:
        for path in sorted(root.rglob("*"), reverse=True):
            if path.is_file():
                path.unlink()
            elif path.is_dir():
                path.rmdir()
        root.rmdir()
        raise
    return report


def run_train_mode(
    *,
    w2_root: Path,
    bridge_artifact: Path,
    audit_manifest: Path,
    protocol_dir: Path,
    output_root: Path,
    validator_path: Path | None = None,
    device: torch.device | None = None,
) -> dict[str, Any]:
    """Run only the resumable training stage; never evaluate or publish W5."""

    receipt = preflight_w3(
        w2_root=w2_root,
        bridge_artifact=bridge_artifact,
        audit_manifest=audit_manifest,
        output_root=output_root,
        validator_path=validator_path,
        protocol_dir=protocol_dir,
    )
    protocol_evidence = validate_protocol_directory(Path(protocol_dir), require_complete=True)
    protocol_payload = protocol_evidence["protocol"]
    protocol = RunProtocol(
        seed=int(protocol_payload["training"]["seed"]),
        sampler_seed=int(protocol_payload["sampler"]["seed"]),
        evaluation_seed=int(protocol_payload["evaluation"]["seed"]),
        per_rank_batch_size=int(protocol_payload["batch"]["per_rank"]),
        world_size=int(protocol_payload["sampler"]["world_size"]),
        epochs=int(protocol_payload["optimization"]["epochs"]),
        learning_rate=float(protocol_payload["optimization"]["learning_rate"]),
        warmup_epochs=int(protocol_payload["optimization"]["warmup_epochs"]),
        gradient_clip_norm=float(protocol_payload["optimization"]["gradient_clip_norm"]),
        bootstrap_replicates=int(protocol_payload["evaluation"]["bootstrap_replicates"]),
    )
    dataset = W2DatasetGateway(Path(w2_root), validator_path=validator_path)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VerifierModel(VerifierConfig(), OpenClipSigLIP2Backbone(pretrained="hf-hub:timm/ViT-L-16-SigLIP2-384"))
    audit = json.loads(Path(audit_manifest).read_text(encoding="utf-8"))
    apply_audited_initialization(model, Path(bridge_artifact), audit)
    preprocess = getattr(model.backbone, "preprocess", None)
    dataset_kwargs = {} if preprocess is None else {"preprocess": preprocess}
    train_dataset = TwoViewDataset(dataset.train, seed=protocol.seed, training=True, **dataset_kwargs)
    validation_dataset = TwoViewDataset(dataset.validation, training=False, **dataset_kwargs)
    staging = Path(output_root).parent / f".{Path(output_root).name}.staging"
    if staging.exists():
        raise FileExistsError(staging)
    staging.mkdir(parents=True)
    try:
        training = train_model(
            model,
            train_dataset,
            protocol=protocol,
            device=device,
            validation_dataset=validation_dataset,
            checkpoint_dir=staging,
            checkpoint_contract=build_checkpoint_contract(
                audit_manifest_sha256=audit["manifest_sha256"],
                bridge_artifact_sha256=audit["artifact"]["sha256"],
                target_fingerprint=audit["target"]["fingerprint"],
                protocol_content_hash=protocol_payload["content_hash"],
                protocol_version=protocol_payload["protocol_version"],
                model_config=model.config.to_dict(),
                four_state_inventory=build_four_state_inventory(),
                w2_identities={
                    "train_manifest_hash": protocol_payload["identities"]["train_manifest_hash"],
                    "phrase_manifest_hash": protocol_payload["identities"]["phrase_manifest_hash"],
                    "normalization_artifact_hash": dataset.validation_receipt.get(
                        "normalization_artifact_hash", dataset.validation_receipt.get("content_hash", "")
                    ),
                    "w2_validation_receipt": dataset.validation_receipt,
                },
                environment={
                    "torch": torch.__version__,
                    "backbone_revision": getattr(model.backbone, "backbone_revision", None),
                    "views": ["base_rgb", "wrist_rgb"],
                    "cuda_rng_device_count": recorded_cuda_rng_device_count(),
                },
            ),
        )
        train_receipt = {
            "schema": "osx_cover_w3_train_receipt_v1",
            "mode": "train",
            "preflight": receipt,
            "protocol": protocol_evidence,
            "progress": training["progress"],
            "resumable_checkpoints": ["latest.pt", "best.pt"],
            "accepted": False,
        }
        (staging / "train_receipt.json").write_bytes(canonical_bytes(train_receipt) + b"\n")
        staging.rename(output_root)
    except Exception:
        for path in sorted(staging.rglob("*"), reverse=True):
            if path.is_file():
                path.unlink()
            elif path.is_dir():
                path.rmdir()
        staging.rmdir()
        raise
    return train_receipt


def run_evaluate_mode(
    *,
    checkpoint: Path,
    output_root: Path,
    w2_root: Path,
    protocol_dir: Path,
    validator_path: Path | None = None,
    device: torch.device | None = None,
    model_factory: Any | None = None,
) -> dict[str, Any]:
    """Reload an explicit best checkpoint and emit canonical plus readable reports."""

    checkpoint = require_explicit_best_checkpoint(Path(checkpoint))
    output_root = Path(output_root)
    if output_root.exists():
        raise FileExistsError(output_root)
    protocol = validate_protocol_directory(Path(protocol_dir), require_complete=True)
    protocol_payload = protocol["protocol"]
    validate_best_checkpoint_for_evaluation(
        checkpoint,
        expected_protocol_content_hash=str(protocol_payload["content_hash"]),
        expected_train_manifest_hash=str(protocol_payload["identities"]["train_manifest_hash"]),
        expected_phrase_manifest_hash=str(protocol_payload["identities"]["phrase_manifest_hash"]),
    )
    dataset = W2DatasetGateway(Path(w2_root), validator_path=validator_path)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or "contract" not in payload:
        raise ValueError("evaluation requires an explicit W3 training checkpoint")
    contract = payload["contract"]
    config = _verifier_config_from_dict(contract["model_config"])
    if model_factory is not None:
        model = model_factory(config)
    elif contract.get("environment", {}).get("fixture"):
        model = VerifierModel(
            config,
            TinyFrozenBackbone(width=config.backbone_width, tokens=config.visual_tokens),
        )
    else:
        model = VerifierModel(config, OpenClipSigLIP2Backbone(pretrained="hf-hub:timm/ViT-L-16-SigLIP2-384"))
    load_fixed_best_checkpoint_logit_scale(
        checkpoint,
        model,
        expected_protocol_content_hash=str(protocol_payload["content_hash"]),
        expected_train_manifest_hash=str(protocol_payload["identities"]["train_manifest_hash"]),
        expected_phrase_manifest_hash=str(protocol_payload["identities"]["phrase_manifest_hash"]),
    )
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    preprocess = getattr(model.backbone, "preprocess", None)
    dataset_kwargs = {} if preprocess is None else {"preprocess": preprocess}
    validation_dataset = TwoViewDataset(dataset.validation, training=False, **dataset_kwargs)
    batch_size = int(protocol["protocol"]["batch"]["per_rank"])
    embeddings = collect_embeddings(model, validation_dataset, device=device, batch_size=batch_size)
    return evaluate_explicit_best_checkpoint(
        checkpoint=checkpoint,
        protocol_dir=Path(protocol_dir),
        output_root=output_root,
        semantic_embeddings=embeddings["semantic"],
        action_embeddings=embeddings["action"],
        sample_ids=embeddings["sample_ids"],
        episode_ids=embeddings["episode_ids"],
        conditions=embeddings["conditions"],
        model=model,
    )


def run_package_mode(*, evidence_root: Path, output_root: Path) -> dict[str, Any]:
    """Stage package input evidence without granting accepted deployment status."""

    evidence_root = Path(evidence_root)
    evaluation = evidence_root / "evaluation.json"
    if not evaluation.is_file():
        raise FileNotFoundError(f"package requires passed evaluation evidence: {evaluation}")
    output_root = Path(output_root)
    if output_root.exists():
        raise FileExistsError(output_root)
    staging = output_root.parent / f".{output_root.name}.staging"
    if staging.exists():
        raise FileExistsError(staging)
    staging.mkdir(parents=True)
    try:
        package_receipt = {
            "schema": "osx_cover_w3_package_input_v1",
            "mode": "package",
            "evidence_root": str(evidence_root),
            "evaluation_sha256": sha256_file(evaluation),
            "accepted": False,
        }
        (staging / "package_input.json").write_bytes(canonical_bytes(package_receipt) + b"\n")
        staging.rename(output_root)
    except Exception:
        for path in sorted(staging.rglob("*"), reverse=True):
            if path.is_file():
                path.unlink()
            elif path.is_dir():
                path.rmdir()
        staging.rmdir()
        raise
    return package_receipt


def run_canonical_acceptance(
    *,
    w2_root: Path,
    bridge_artifact: Path,
    audit_manifest: Path,
    protocol_dir: Path,
    output_root: Path,
    validator_path: Path | None = None,
    device: torch.device | None = None,
) -> dict[str, Any]:
    """Run the complete canonical path; publication occurs only after acceptance."""
    receipt = preflight_w3(
        w2_root=w2_root,
        bridge_artifact=bridge_artifact,
        audit_manifest=audit_manifest,
        output_root=output_root,
        validator_path=validator_path,
        protocol_dir=protocol_dir,
    )
    dataset = W2DatasetGateway(Path(w2_root), validator_path=validator_path)
    protocol_payload = json.loads((Path(protocol_dir) / "run_protocol.json").read_text(encoding="utf-8"))
    protocol = RunProtocol(
        seed=int(protocol_payload["seed"]),
        sampler_seed=int(protocol_payload["sampler"]["seed"]),
        per_rank_batch_size=int(protocol_payload["batch"]["per_rank"]),
        world_size=int(protocol_payload["sampler"]["world_size"]),
        epochs=int(protocol_payload["optimization"]["epochs"]),
        learning_rate=float(protocol_payload["optimization"]["learning_rate"]),
        warmup_epochs=int(protocol_payload["optimization"]["warmup_epochs"]),
        gradient_clip_norm=float(protocol_payload["optimization"]["gradient_clip_norm"]),
        bootstrap_replicates=int(protocol_payload["evaluation"]["bootstrap_replicates"]),
    )
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VerifierModel(VerifierConfig(), OpenClipSigLIP2Backbone(pretrained="hf-hub:timm/ViT-L-16-SigLIP2-384"))
    audit = json.loads(Path(audit_manifest).read_text(encoding="utf-8"))
    apply_audited_initialization(model, Path(bridge_artifact), audit)
    train_preprocess = getattr(model.backbone, "preprocess", None)
    dataset_kwargs = {} if train_preprocess is None else {"preprocess": train_preprocess}
    train_dataset = TwoViewDataset(dataset.train, seed=protocol.seed, training=True, **dataset_kwargs)
    staging = Path(output_root).parent / f".{Path(output_root).name}.staging"
    if staging.exists():
        raise FileExistsError(staging)
    staging.mkdir(parents=True)
    validation_dataset = TwoViewDataset(dataset.validation, training=False, **dataset_kwargs)
    checkpoint_contract = build_checkpoint_contract(
        audit_manifest_sha256=audit["manifest_sha256"],
        bridge_artifact_sha256=audit["artifact"]["sha256"],
        target_fingerprint=audit["target"]["fingerprint"],
        protocol_content_hash=protocol_payload["content_hash"],
        protocol_version=protocol_payload["protocol_version"],
        model_config=model.config.to_dict(),
        four_state_inventory=build_four_state_inventory(),
        w2_identities={
            "train_manifest_hash": protocol_payload["identities"]["train_manifest_hash"],
            "phrase_manifest_hash": protocol_payload["identities"]["phrase_manifest_hash"],
            "normalization_artifact_hash": dataset.validation_receipt.get(
                "normalization_artifact_hash", dataset.validation_receipt.get("content_hash", "")
            ),
            "w2_validation_receipt": dataset.validation_receipt,
        },
        environment={
            "torch": torch.__version__,
            "backbone_revision": getattr(model.backbone, "backbone_revision", None),
            "preprocessing_contract": "openclip_model_eval_transform_v1",
            "views": ["base_rgb", "wrist_rgb"],
            "cuda_rng_device_count": recorded_cuda_rng_device_count(),
        },
    )
    training = train_model(
        model,
        train_dataset,
        protocol=protocol,
        device=device,
        validation_dataset=validation_dataset,
        checkpoint_dir=staging,
        checkpoint_contract=checkpoint_contract,
    )
    try:
        embeddings = collect_embeddings(
            model, validation_dataset, device=device, batch_size=protocol.per_rank_batch_size
        )
        shuffled = json.loads((Path(protocol_dir) / "shuffled_pairs.json").read_text(encoding="utf-8"))["pairs"]
        nearby = json.loads((Path(protocol_dir) / "nearby_pairs.json").read_text(encoding="utf-8"))["pairs"]
        bootstrap = np.load(Path(protocol_dir) / "bootstrap_indices.npy")
        report = evaluate_embeddings(
            embeddings["semantic"],
            embeddings["action"],
            sample_ids=embeddings["sample_ids"],
            episode_ids=embeddings["episode_ids"],
            conditions=embeddings["conditions"],
            shuffled_pairs=shuffled,
            nearby_pairs=nearby,
            bootstrap_indices=bootstrap,
            checkpoint_logit_scale=float(model.logit_scale.detach().clamp(0.0, np.log(100.0)).exp()),
            strict_protocol=True,
        )
        require_acceptance_metrics(report)
        evaluation_hashes = _evaluation_artifact_hashes(
            shuffled_pairs=shuffled, nearby_pairs=nearby, bootstrap_indices=bootstrap
        )
        two_view_identity = build_matched_ablation_identity(
            protocol=protocol,
            sample_ids=embeddings["sample_ids"],
            phrase_manifest_hash=protocol_payload["identities"]["phrase_manifest_hash"],
            protocol_content_hash=protocol_payload["content_hash"],
            evaluation_artifact_hashes=evaluation_hashes,
        )
        base_only = run_matched_base_only(
            two_view_model=model,
            bridge_artifact=Path(bridge_artifact),
            audit_manifest=audit,
            dataset=dataset,
            protocol=protocol,
            shuffled_pairs=shuffled,
            nearby_pairs=nearby,
            bootstrap_indices=bootstrap,
            device=device,
            two_view_identity=two_view_identity,
            phrase_manifest_hash=protocol_payload["identities"]["phrase_manifest_hash"],
            protocol_content_hash=protocol_payload["content_hash"],
        )
        (staging / "base_only").mkdir()
        ablation_payload = matched_base_only_run_to_paired_ablation_report(
            two_view_report=report,
            two_view_identity=two_view_identity,
            base_only_result=base_only,
            bootstrap_indices=bootstrap,
            output_root=staging / "base_only" / "paired_ablation",
        )
        ablation = ablation_payload["ablation"]
        save_training_checkpoint(
            staging / "base_only" / "latest.pt",
            model=base_only["model"],
            optimizer=base_only["training"]["optimizer"],
            scheduler=base_only["training"]["scheduler"],
            progress=base_only["training"]["progress"],
            contract=build_checkpoint_contract(
                audit_manifest_sha256=audit["manifest_sha256"],
                bridge_artifact_sha256=audit["artifact"]["sha256"],
                target_fingerprint=audit["target"]["fingerprint"],
                protocol_content_hash=protocol_payload["content_hash"],
                protocol_version=protocol_payload["protocol_version"],
                model_config=base_only["model"].config.to_dict(),
                four_state_inventory=build_four_state_inventory(),
                w2_identities={
                    "train_manifest_hash": protocol_payload["identities"]["train_manifest_hash"],
                    "phrase_manifest_hash": protocol_payload["identities"]["phrase_manifest_hash"],
                    "normalization_artifact_hash": dataset.validation_receipt.get(
                        "normalization_artifact_hash", dataset.validation_receipt.get("content_hash", "")
                    ),
                    "variant": "base_only",
                    "deployable": False,
                },
                environment={
                    "torch": torch.__version__,
                    "variant": "base_only",
                    "deployable": False,
                    "cuda_rng_device_count": recorded_cuda_rng_device_count(),
                },
            ),
            sampler_state={
                "epoch": base_only["training"]["progress"]["epoch"],
                "rank": 0,
                "world_size": base_only["training"]["progress"]["world_size"],
            },
            rank=0,
            rng_states={
                rank: capture_rng_state(cuda_rng_device_count=recorded_cuda_rng_device_count())
                for rank in range(int(base_only["training"]["progress"]["world_size"]))
            },
        )
        (staging / "base_only" / "metadata.json").write_bytes(
            canonical_bytes(
                {
                    "schema": "osx_cover_w3_base_only_evidence_v1",
                    "deployable": False,
                    "model_config": base_only["model"].config.to_dict(),
                    "evaluation_report": base_only["report"],
                    "ablation_report": ablation,
                }
            )
            + b"\n"
        )
        metadata = {
            "deployable": True,
            "model_config": model.config.to_dict(),
            "backbone_revision": getattr(model.backbone, "backbone_revision", None),
            "preprocessing_contract": "openclip_model_eval_transform_v1",
            "tokenizer_id": "open_clip.get_tokenizer(hf-hub:timm/ViT-L-16-SigLIP2-384)",
            "evaluation_report": report,
            "base_only_ablation": ablation,
            "w3_01_initialization": {
                "audit_manifest_sha256": audit["manifest_sha256"],
                "bridge_artifact_sha256": audit["artifact"]["sha256"],
                "target_fingerprint": audit["target"]["fingerprint"],
            },
            "w3_03_protocol": {
                "protocol_content_hash": protocol_payload["content_hash"],
                "protocol_version": protocol_payload["protocol_version"],
            },
            "four_state_inventory": build_four_state_inventory(),
            "package_scope": "pipeline_acceptance_not_canonical_w3_09",
            "scorer_compatibility": {
                "model_schema_version": "osx_cover_verifier_checkpoint_v1",
                "views": ["base_rgb", "wrist_rgb"],
                "preprocessing_contract": "openclip_siglip2_384_center_crop_v1",
                "action_dimension": 7,
                "action_order": ["dx", "dy", "dz", "rotation_x", "rotation_y", "rotation_z", "gripper"],
                "history_length": 10,
                "representation_id": "ur5e_cover_relative_eef_v1",
                "normalization_artifact_hash": dataset.validation_receipt["normalization_artifact_hash"],
                "input_dtype": "float32",
                "output_shape_rank": 1,
            },
        }
        publish_deployment_bundle(staging / "deployment", model=model, metadata=metadata)
        final_report = {
            "schema": "osx_cover_w3_acceptance_v1",
            "preflight": receipt,
            "training": training["history"],
            "evaluation": report,
            "deployment": "deployment",
            "authority": "recorded_data_offline_integration_only",
        }
        final_report["content_hash"] = content_hash(final_report)
        (staging / "acceptance.json").write_bytes(canonical_bytes(final_report) + b"\n")
        staging.rename(output_root)
    except Exception:
        for path in sorted(staging.rglob("*"), reverse=True):
            if path.is_file():
                path.unlink()
            elif path.is_dir():
                path.rmdir()
        staging.rmdir()
        raise
    return final_report
