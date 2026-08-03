"""W3-09 canonical two-rank DDP training and acceptance helpers."""

# ruff: noqa: PLC0415
from __future__ import annotations

from collections.abc import Mapping
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any

import numpy as np
import torch

from lap.verifiers.cover.batch_probe import validate_successful_rank_records
from lap.verifiers.cover.bridge_audit import _target_inventory_from_model
from lap.verifiers.cover.bridge_audit import audit_bridge_checkpoint
from lap.verifiers.cover.checkpoint import load_deployment_bundle
from lap.verifiers.cover.data import TwoViewDataset
from lap.verifiers.cover.data import W2DatasetGateway
from lap.verifiers.cover.evaluator import evaluate_embeddings
from lap.verifiers.cover.evaluator import evaluate_explicit_best_checkpoint
from lap.verifiers.cover.evaluator import load_fixed_best_checkpoint_logit_scale
from lap.verifiers.cover.evaluator import require_explicit_best_checkpoint
from lap.verifiers.cover.model import OpenClipSigLIP2Backbone
from lap.verifiers.cover.model import VerifierConfig
from lap.verifiers.cover.model import VerifierModel
from lap.verifiers.cover.protocol import WORLD_SIZE
from lap.verifiers.cover.protocol import RunProtocol
from lap.verifiers.cover.protocol import snapshot_nvidia_devices
from lap.verifiers.cover.protocol import validate_protocol_directory
from lap.verifiers.cover.training import base_only_config_delta
from lap.verifiers.cover.training import make_base_only_config
from lap.verifiers.cover.training import require_acceptance_metrics
from lap.verifiers.cover.training import reset_approved_seed
from lap.verifiers.cover.w3_contracts import canonical_bytes
from lap.verifiers.cover.w3_contracts import content_hash

CANONICAL_PACKAGE_SCOPE = "canonical_w3_09"
REQUIRED_GPU_SUBSTRING = "RTX 6000 Ada"
_CANONICAL_PER_RANK_BATCH = 64


def resolve_initialization_manifest_for_model(
    model: VerifierModel,
    *,
    bridge_artifact: Path,
    production_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Select the audited init document for this model inventory.

    Two-view models must use the locked production audit. The matched base-only
    control re-audits the same Bridge artifact against the wrist-omitted
    inventory so transferable tensors still apply without rewriting W3-01.
    """

    inventory = _target_inventory_from_model(model)
    target = production_manifest.get("target")
    if not isinstance(target, Mapping):
        raise ValueError("production audit missing target inventory")
    if inventory.fingerprint == target.get("fingerprint"):
        return dict(production_manifest)
    if getattr(model.config, "use_wrist", True):
        raise ValueError("model target inventory drifted from the locked production audit")
    artifact = production_manifest.get("artifact")
    if not isinstance(artifact, Mapping):
        raise ValueError("production audit missing artifact identity")
    return audit_bridge_checkpoint(
        Path(bridge_artifact),
        target_inventory=inventory,
        expected_sha256=str(artifact["sha256"]),
        expected_size=int(artifact["size"]),
    )


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _train_worker_script() -> Path:
    return _repo_root() / "scripts" / "w3_canonical_ddp_train.py"


def _batch_probe_worker_script() -> Path:
    return _repo_root() / "scripts" / "w3_batch_probe.py"


def protocol_from_validated(protocol_payload: dict[str, Any]) -> RunProtocol:
    """Build the runtime protocol view from a validated W3-03 payload."""

    return RunProtocol(
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


def _require_canonical_protocol(protocol: RunProtocol) -> None:
    if protocol.world_size != WORLD_SIZE:
        raise ValueError("canonical W3 execution requires world_size=2")
    if protocol.per_rank_batch_size != _CANONICAL_PER_RANK_BATCH:
        raise ValueError("canonical W3 execution requires per_rank_batch_size=64")
    if protocol.epochs != 50 or protocol.learning_rate != 1e-6:
        raise ValueError("canonical W3 execution requires the accepted 50-epoch AdamW 1e-6 contract")


def _default_w2_validator() -> Path:
    """Resolve the outer W2 validator even when LAP is checked out in a pinned worktree."""

    candidates = [
        Path("/home/yangsen/osx_ur/catkin_ws/src/osx_vla/scripts/export_cover_training.py"),
        Path(__file__).resolve().parents[6] / "scripts" / "export_cover_training.py",
    ]
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError("W2 validator export_cover_training.py not found for canonical W3 execution")


def require_canonical_training_host() -> dict[str, Any]:
    """Require two RTX 6000 Ada GPUs and a visible NVIDIA control device."""

    nvidia_ctl = Path("/dev/nvidiactl")
    if not nvidia_ctl.exists():
        raise RuntimeError("canonical training requires /dev/nvidiactl")
    snapshot = snapshot_nvidia_devices()
    if len(snapshot) != WORLD_SIZE:
        raise RuntimeError("canonical training requires exactly two visible NVIDIA GPUs")
    names = []
    for gpu in snapshot:
        name = str(gpu["name"])
        if REQUIRED_GPU_SUBSTRING not in name:
            raise ValueError(f"canonical training requires {REQUIRED_GPU_SUBSTRING}; observed {name}")
        names.append(name)
    return {
        "nvidia_ctl": str(nvidia_ctl),
        "gpus": names,
        "world_size": WORLD_SIZE,
        "gpu_snapshot": snapshot,
    }


def _evaluation_artifact_hashes(
    *,
    shuffled_pairs: list[dict[str, str]],
    nearby_pairs: list[dict[str, str]],
    bootstrap_indices: np.ndarray,
) -> dict[str, str]:
    from lap.verifiers.cover.pipeline import bootstrap_indices_content_hash

    return {
        "shuffled_pairs_hash": content_hash({"pairs": shuffled_pairs}),
        "nearby_pairs_hash": content_hash({"pairs": nearby_pairs}),
        "bootstrap_indices_hash": bootstrap_indices_content_hash(bootstrap_indices),
    }


def _run_torchrun_worker(
    *,
    worker_script: Path,
    worker_mode: str,
    extra_args: list[str],
    timeout_seconds: int = 172800,
) -> subprocess.CompletedProcess[str]:
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc_per_node={WORLD_SIZE}",
        str(worker_script),
        "--worker",
        "--worker-mode",
        worker_mode,
        *extra_args,
    ]
    env = os.environ.copy()
    src = str(_repo_root() / "src")
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = src if not existing else f"{src}:{existing}"
    try:
        return subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            env=env,
        )
    except subprocess.TimeoutExpired as error:
        raise RuntimeError(f"canonical {worker_mode} worker timed out") from error


def _run_fresh_two_rank_ddp_step(
    *,
    w2_root: Path,
    bridge_artifact: Path,
    audit_manifest: Path,
    work_dir: Path,
    timeout_seconds: int = 1800,
) -> list[dict[str, Any]]:
    from lap.verifiers.cover.batch_probe import _resolve_pinned_snapshot
    from lap.verifiers.cover.batch_probe import _run_worker_process

    model_dir = _resolve_pinned_snapshot()
    result_dir = work_dir / "fresh_ddp_step"
    if result_dir.exists():
        shutil.rmtree(result_dir)
    results = _run_worker_process(
        worker_script=_batch_probe_worker_script(),
        mode="ddp",
        batch_size=_CANONICAL_PER_RANK_BATCH,
        w2_root=w2_root,
        bridge_artifact=bridge_artifact,
        audit_manifest=audit_manifest,
        model_dir=model_dir,
        result_dir=result_dir,
        timeout_seconds=timeout_seconds,
    )
    validate_successful_rank_records(results, batch_size=_CANONICAL_PER_RANK_BATCH)
    return results


def _run_preaccept_checkpoint_resume(
    *,
    w2_root: Path,
    bridge_artifact: Path,
    audit_manifest: Path,
    protocol_dir: Path,
    work_dir: Path,
    validator_path: Path | None,
    timeout_seconds: int = 1800,
) -> dict[str, Any]:
    result_path = work_dir / "preaccept_checkpoint_result.json"
    if result_path.exists():
        result_path.unlink()
    extra = [
        "--w2-root",
        str(w2_root),
        "--bridge-artifact",
        str(bridge_artifact),
        "--audit-manifest",
        str(audit_manifest),
        "--protocol-dir",
        str(protocol_dir),
        "--result-path",
        str(result_path),
    ]
    if validator_path is not None:
        extra.extend(["--validator", str(validator_path)])
    completed = _run_torchrun_worker(
        worker_script=_train_worker_script(),
        worker_mode="preaccept",
        extra_args=extra,
        timeout_seconds=timeout_seconds,
    )
    if not result_path.is_file():
        error_bits = [
            f"{path.name}:\n{path.read_text(encoding='utf-8')[-4000:]}"
            for path in sorted(Path(work_dir).glob("preaccept_worker_error_rank*.txt"))
        ]
        detail = "\n".join(error_bits) if error_bits else completed.stderr[-2000:]
        raise RuntimeError(
            f"canonical preaccept checkpoint worker did not emit a result (exit={completed.returncode}): {detail}"
        )
    receipt = json.loads(result_path.read_text(encoding="utf-8"))
    return require_preaccept_exact_resume_receipt(receipt)


def run_preaccept_two_rank_step_and_resume(
    *,
    w2_root: Path,
    bridge_artifact: Path,
    audit_manifest: Path,
    protocol_dir: Path,
    work_dir: Path,
    validator_path: Path | None = None,
) -> dict[str, Any]:
    """Fresh two-rank batch-64 step plus W3-06 exact resume before full training."""

    host = require_canonical_training_host()
    protocol_evidence = validate_protocol_directory(Path(protocol_dir), require_complete=True)
    protocol = protocol_from_validated(protocol_evidence["protocol"])
    _require_canonical_protocol(protocol)
    work = Path(work_dir)
    work.mkdir(parents=True, exist_ok=True)
    resolved_validator = Path(validator_path) if validator_path is not None else _default_w2_validator()
    ddp_ranks = _run_fresh_two_rank_ddp_step(
        w2_root=Path(w2_root),
        bridge_artifact=Path(bridge_artifact),
        audit_manifest=Path(audit_manifest),
        work_dir=work,
    )
    checkpoint_receipt = _run_preaccept_checkpoint_resume(
        w2_root=Path(w2_root),
        bridge_artifact=Path(bridge_artifact),
        audit_manifest=Path(audit_manifest),
        protocol_dir=Path(protocol_dir),
        work_dir=work,
        validator_path=resolved_validator,
    )
    return require_preaccept_exact_resume_receipt(
        {
            "two_rank_step": "passed",
            "exact_resume": "passed",
            "uninterrupted_versus_resumed": checkpoint_receipt.get("uninterrupted_versus_resumed"),
            "ranks_restored": checkpoint_receipt.get("ranks_restored"),
            "resume_equivalence": checkpoint_receipt.get("resume_equivalence"),
            "host": host,
            "per_rank_batch_size": _CANONICAL_PER_RANK_BATCH,
            "world_size": WORLD_SIZE,
            "ddp_step_ranks": ddp_ranks,
            "checkpoint_resume": checkpoint_receipt,
        }
    )


def train_two_rank_ddp(
    *,
    w2_root: Path,
    bridge_artifact: Path,
    audit_manifest: Path,
    protocol_dir: Path,
    checkpoint_dir: Path,
    validator_path: Path | None = None,
    use_wrist: bool = True,
    epochs: int | None = None,
) -> dict[str, Any]:
    """Launch the canonical world-size-2 DDP training worker and return its receipt."""

    require_canonical_training_host()
    protocol_evidence = validate_protocol_directory(Path(protocol_dir), require_complete=True)
    protocol = protocol_from_validated(protocol_evidence["protocol"])
    _require_canonical_protocol(protocol)
    if epochs is not None and int(epochs) != int(protocol.epochs):
        raise ValueError("canonical training cannot override the accepted epoch count")
    checkpoint = Path(checkpoint_dir)
    checkpoint.mkdir(parents=True, exist_ok=True)
    result_path = checkpoint / "train_result.json"
    if result_path.exists():
        result_path.unlink()
    resolved_validator = Path(validator_path) if validator_path is not None else _default_w2_validator()
    extra = [
        "--w2-root",
        str(w2_root),
        "--bridge-artifact",
        str(bridge_artifact),
        "--audit-manifest",
        str(audit_manifest),
        "--protocol-dir",
        str(protocol_dir),
        "--checkpoint-dir",
        str(checkpoint),
        "--use-wrist",
        "true" if use_wrist else "false",
        "--validator",
        str(resolved_validator),
    ]
    if epochs is not None:
        extra.extend(["--epochs", str(epochs)])
    completed = _run_torchrun_worker(
        worker_script=_train_worker_script(),
        worker_mode="train",
        extra_args=extra,
        timeout_seconds=172800,
    )
    if not result_path.is_file():
        error_bits = [
            f"{path.name}:\n{path.read_text(encoding='utf-8')[-4000:]}"
            for path in sorted(Path(checkpoint).glob("train_worker_error_rank*.txt"))
        ]
        detail = "\n".join(error_bits) if error_bits else completed.stderr[-2000:]
        raise RuntimeError(
            f"canonical DDP training worker did not emit train_result.json (exit={completed.returncode}): {detail}"
        )
    receipt = json.loads(result_path.read_text(encoding="utf-8"))
    if receipt.get("gradient_synchronization") != "two_rank_ddp":
        raise ValueError("canonical training receipt missing two_rank_ddp synchronization evidence")
    if int(receipt.get("world_size", 0)) != WORLD_SIZE:
        raise ValueError("canonical training receipt must record world_size=2")
    return receipt


def run_matched_base_only_two_rank_ddp(
    *,
    w2_root: Path,
    bridge_artifact: Path,
    audit_manifest: Path,
    protocol_dir: Path,
    checkpoint_dir: Path,
    validator_path: Path | None = None,
    two_view_model: VerifierModel | None = None,
    shuffled_pairs: list[dict[str, str]] | None = None,
    nearby_pairs: list[dict[str, str]] | None = None,
    bootstrap_indices: np.ndarray | None = None,
    two_view_identity: dict[str, Any] | None = None,
    phrase_manifest_hash: str = "",
    protocol_content_hash: str = "",
) -> dict[str, Any]:
    """Train/evaluate the wrist-omitting control with the canonical two-rank worker."""

    from lap.verifiers.cover.pipeline import build_matched_ablation_identity
    from lap.verifiers.cover.pipeline import collect_embeddings
    from lap.verifiers.cover.pipeline import require_matched_ablation_identities

    protocol_evidence = validate_protocol_directory(Path(protocol_dir), require_complete=True)
    protocol_payload = protocol_evidence["protocol"]
    protocol = protocol_from_validated(protocol_payload)
    reset_approved_seed(protocol.seed)
    shuffled = shuffled_pairs or protocol_evidence["shuffled_pairs"]["pairs"]
    nearby = nearby_pairs or protocol_evidence["nearby_pairs"]["pairs"]
    bootstrap = bootstrap_indices if bootstrap_indices is not None else protocol_evidence["bootstrap_indices"]
    training = train_two_rank_ddp(
        w2_root=w2_root,
        bridge_artifact=bridge_artifact,
        audit_manifest=audit_manifest,
        protocol_dir=protocol_dir,
        checkpoint_dir=checkpoint_dir,
        validator_path=validator_path,
        use_wrist=False,
    )
    dataset = W2DatasetGateway(Path(w2_root), validator_path=validator_path)
    if two_view_model is None:
        backbone = OpenClipSigLIP2Backbone(pretrained="hf-hub:timm/ViT-L-16-SigLIP2-384")
    else:
        backbone = two_view_model.backbone
    base_config = make_base_only_config(two_view_model.config if two_view_model is not None else VerifierConfig())
    base_model = VerifierModel(base_config, backbone)
    # Trained base-only weights come from the DDP worker checkpoint; do not
    # re-apply the two-view production audit onto the wrist-omitted inventory.
    checkpoint = require_explicit_best_checkpoint(Path(checkpoint_dir) / "best.pt")
    load_fixed_best_checkpoint_logit_scale(
        checkpoint,
        base_model,
        expected_protocol_content_hash=str(protocol_payload["content_hash"]),
        expected_train_manifest_hash=str(protocol_payload["identities"]["train_manifest_hash"]),
        expected_phrase_manifest_hash=str(protocol_payload["identities"]["phrase_manifest_hash"]),
    )
    device = torch.device("cuda", 0)
    base_model.to(device)
    preprocess = getattr(base_model.backbone, "preprocess", None)
    dataset_kwargs = {} if preprocess is None else {"preprocess": preprocess}
    validation_dataset = TwoViewDataset(dataset.validation, training=False, **dataset_kwargs)
    embeddings = collect_embeddings(
        base_model,
        validation_dataset,
        device=device,
        batch_size=protocol.per_rank_batch_size,
    )
    report = evaluate_embeddings(
        embeddings["semantic"],
        embeddings["action"],
        sample_ids=embeddings["sample_ids"],
        episode_ids=embeddings["episode_ids"],
        conditions=embeddings["conditions"],
        shuffled_pairs=shuffled,
        nearby_pairs=nearby,
        bootstrap_indices=bootstrap,
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
            dataset.validation_receipt.get("phrase_manifest_hash")
            or (two_view_identity or {}).get("phrase_manifest_hash", "")
        ),
        protocol_content_hash=protocol_content_hash or str((two_view_identity or {}).get("protocol_content_hash", "")),
        evaluation_artifact_hashes=_evaluation_artifact_hashes(
            shuffled_pairs=shuffled,
            nearby_pairs=nearby,
            bootstrap_indices=bootstrap,
        ),
    )
    if two_view_identity is not None:
        require_matched_ablation_identities(two_view_identity, identity)
    config_delta = base_only_config_delta(
        two_view_model.config if two_view_model is not None else VerifierConfig(),
        base_model.config,
    )
    return {
        "model": base_model,
        "training": training,
        "report": report,
        "identity": identity,
        "config_delta": config_delta,
        "model_config": base_model.config.to_dict(),
        "history": training["history"],
        "progress": training["progress"],
        "checkpoint_dir": str(checkpoint_dir),
    }


def _evaluate_best_checkpoint(
    *,
    checkpoint: Path,
    w2_root: Path,
    protocol_dir: Path,
    output_root: Path,
    validator_path: Path | None,
) -> dict[str, Any]:
    from lap.verifiers.cover.pipeline import collect_embeddings

    checkpoint = require_explicit_best_checkpoint(Path(checkpoint))
    output_root = Path(output_root)
    if output_root.exists():
        raise FileExistsError(output_root)
    protocol_evidence = validate_protocol_directory(Path(protocol_dir), require_complete=True)
    protocol_payload = protocol_evidence["protocol"]
    protocol = protocol_from_validated(protocol_payload)
    dataset = W2DatasetGateway(Path(w2_root), validator_path=validator_path)
    config = VerifierConfig()
    model = VerifierModel(config, OpenClipSigLIP2Backbone(pretrained="hf-hub:timm/ViT-L-16-SigLIP2-384"))
    load_fixed_best_checkpoint_logit_scale(
        checkpoint,
        model,
        expected_protocol_content_hash=str(protocol_payload["content_hash"]),
        expected_train_manifest_hash=str(protocol_payload["identities"]["train_manifest_hash"]),
        expected_phrase_manifest_hash=str(protocol_payload["identities"]["phrase_manifest_hash"]),
    )
    device = torch.device("cuda", 0)
    model.to(device)
    preprocess = getattr(model.backbone, "preprocess", None)
    dataset_kwargs = {} if preprocess is None else {"preprocess": preprocess}
    validation_dataset = TwoViewDataset(dataset.validation, training=False, **dataset_kwargs)
    embeddings = collect_embeddings(
        model,
        validation_dataset,
        device=device,
        batch_size=protocol.per_rank_batch_size,
    )
    metrics = evaluate_embeddings(
        embeddings["semantic"],
        embeddings["action"],
        sample_ids=embeddings["sample_ids"],
        episode_ids=embeddings["episode_ids"],
        conditions=embeddings["conditions"],
        shuffled_pairs=protocol_evidence["shuffled_pairs"]["pairs"],
        nearby_pairs=protocol_evidence["nearby_pairs"]["pairs"],
        bootstrap_indices=protocol_evidence["bootstrap_indices"],
        checkpoint_logit_scale=float(model.logit_scale.detach().clamp(0.0, np.log(100.0)).exp()),
        strict_protocol=True,
    )
    report = evaluate_explicit_best_checkpoint(
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
    return {"report": report, "metrics": metrics}


def run_repeated_fixed_best_evaluation(
    *,
    checkpoint: Path,
    w2_root: Path,
    protocol_dir: Path,
    output_root: Path,
    validator_path: Path | None = None,
) -> dict[str, Any]:
    """Evaluate explicit best.pt twice and require byte-identical canonical JSON."""

    output_root = Path(output_root)
    parent = output_root.parent
    first = parent / f".{output_root.name}.repeat-a"
    second = parent / f".{output_root.name}.repeat-b"
    for path in (first, second):
        if path.exists():
            shutil.rmtree(path)
    try:
        first_eval = _evaluate_best_checkpoint(
            checkpoint=checkpoint,
            w2_root=w2_root,
            protocol_dir=protocol_dir,
            output_root=first,
            validator_path=validator_path,
        )
        second_eval = _evaluate_best_checkpoint(
            checkpoint=checkpoint,
            w2_root=w2_root,
            protocol_dir=protocol_dir,
            output_root=second,
            validator_path=validator_path,
        )
        report_a = first_eval["report"]
        report_b = second_eval["report"]
        metrics_a = first_eval["metrics"]
        evaluation_bytes = (first / "evaluation.json").read_bytes()
        repeated_bytes = (second / "evaluation.json").read_bytes()
        if evaluation_bytes != repeated_bytes:
            raise ValueError("repeated fixed-best evaluation is not byte-identical")
        require_acceptance_metrics(report_a)
        if report_a != report_b:
            raise ValueError("repeated fixed-best evaluation reports diverged")
        if metrics_a.get("row_metrics", {}).get("sample_ids") is None:
            raise ValueError("canonical evaluation metrics must retain row_metrics for ablation")
        if output_root.exists():
            raise FileExistsError(output_root)
        shutil.move(str(first), str(output_root))
        return {
            "report": report_a,
            "metrics": metrics_a,
            "evaluation_bytes": evaluation_bytes,
            "repeated_byte_identical": True,
        }
    finally:
        if second.exists():
            shutil.rmtree(second)
        if first.exists():
            shutil.rmtree(first)


def load_deployment_bundle_for_acceptance(
    root: Path,
    *,
    expected_normalization_hash: str,
    model_factory: Any,
    preprocessing: Any,
) -> dict[str, Any]:
    """Load an accepted deployment bundle and prove scorer construction succeeds."""

    scorer = load_deployment_bundle(
        Path(root),
        model_factory=model_factory,
        expected_normalization_hash=expected_normalization_hash,
        preprocessing=preprocessing,
    )
    probe = scorer.score(
        base_rgb=np.zeros((384, 384, 3), dtype=np.uint8),
        wrist_rgb=np.zeros((384, 384, 3), dtype=np.uint8),
        instruction="insert the circular peg",
        action_histories=np.zeros((1, 10, 7), dtype=np.float32),
    )
    if probe.shape != (1,) or not np.isfinite(probe).all():
        raise ValueError("deployment scorer did not return finite rank-1 scores")
    return {
        "deployment_root": str(root),
        "scores_finite": True,
        "scorer": scorer,
    }


def require_preaccept_exact_resume_receipt(receipt: Mapping[str, Any]) -> dict[str, Any]:
    """Require W3-06-compatible both-rank uninterrupted-versus-resumed proof."""

    payload = dict(receipt)
    if payload.get("exact_resume") != "passed":
        raise ValueError(f"canonical preaccept exact resume failed: {payload}")
    if payload.get("uninterrupted_versus_resumed") != "passed":
        raise ValueError(
            "canonical preaccept missing uninterrupted_versus_resumed=passed "
            f"(both-rank continuation proof required): {payload}"
        )
    ranks = payload.get("ranks_restored")
    if list(ranks or []) != [0, 1]:
        raise ValueError(f"canonical preaccept must restore ranks [0, 1]: {payload}")
    equivalence = payload.get("resume_equivalence")
    if not isinstance(equivalence, Mapping):
        raise ValueError("canonical preaccept missing resume_equivalence evidence")
    for key in (
        "rank0_loss_match",
        "rank1_loss_match",
        "gradient_fingerprint_match",
        "next_batch_sample_ids_match",
    ):
        if equivalence.get(key) is not True:
            raise ValueError(f"canonical preaccept resume equivalence failed at {key}: {payload}")
    return payload


def next_batch_after_resume(
    *,
    dataset: Any,
    seed: int,
    world_size: int,
    rank: int,
    epoch: int,
    batches_already_consumed: int,
    batch_size: int,
    collate_fn: Any,
) -> dict[str, Any]:
    """Recreate the per-rank sampler and return the next batch after the saved cursor."""

    from torch.utils.data import DataLoader

    from lap.verifiers.cover.data import make_sampler

    if batches_already_consumed < 0:
        raise ValueError("batches_already_consumed must be >= 0")
    sampler = make_sampler(dataset, seed=seed, world_size=world_size, rank=rank)
    sampler.set_epoch(int(epoch))
    loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, collate_fn=collate_fn, num_workers=0)
    iterator = iter(loader)
    for _ in range(int(batches_already_consumed)):
        next(iterator)
    return next(iterator)


def next_batch_sample_ids_after_resume(
    *,
    dataset: Any,
    seed: int,
    world_size: int,
    rank: int,
    epoch: int,
    batches_already_consumed: int,
    batch_size: int,
    collate_fn: Any,
) -> list[Any]:
    """Public helper: sample ids of the first batch after restored sampler cursor."""

    batch = next_batch_after_resume(
        dataset=dataset,
        seed=seed,
        world_size=world_size,
        rank=rank,
        epoch=epoch,
        batches_already_consumed=batches_already_consumed,
        batch_size=batch_size,
        collate_fn=collate_fn,
    )
    sample_ids = batch.get("sample_ids")
    if not isinstance(sample_ids, list):
        raise ValueError("restored sampler batch missing sample_ids")
    return list(sample_ids)


def require_published_deployment_payloads(root: Path) -> dict[str, Any]:
    """Fail closed when an accepted deployment is missing indexed on-disk payloads."""

    from lap.verifiers.cover.checkpoint import _validate_deployment_content_index

    root = Path(root)
    index_path = root / "content_index.json"
    if not index_path.is_file():
        raise ValueError(f"deployment content index missing: {index_path}")
    index = json.loads(index_path.read_text(encoding="utf-8"))
    _validate_deployment_content_index(root, index)
    if not (root / "model.pt").is_file():
        raise ValueError("deployment content index missing file on disk: model.pt")
    return {"root": str(root.resolve()), "payloads_present": True}


PACKAGE_IDENTITY_SCHEMA = "osx_cover_w3_09_package_identity_v1"
_PACKAGE_IDENTITY_NAME = "PACKAGE_IDENTITY.json"


def _package_relative_file_map(package_root: Path) -> dict[str, Path]:
    package_root = Path(package_root).resolve()
    files: dict[str, Path] = {}
    for path in sorted(package_root.rglob("*")):
        if not path.is_file():
            continue
        if path.name == _PACKAGE_IDENTITY_NAME and path.parent == package_root:
            continue
        rel = path.relative_to(package_root).as_posix()
        files[rel] = path
    return files


def validate_package_identity(package_root: Path) -> dict[str, Any]:
    """Independently re-hash every indexed path; reject missing or stale entries."""

    from lap.verifiers.cover.w3_contracts import sha256_file

    package_root = Path(package_root).resolve()
    identity_path = package_root / _PACKAGE_IDENTITY_NAME
    if not identity_path.is_file():
        raise ValueError(f"PACKAGE_IDENTITY missing: {identity_path}")
    payload = json.loads(identity_path.read_text(encoding="utf-8"))
    if payload.get("schema") != PACKAGE_IDENTITY_SCHEMA:
        raise ValueError(f"PACKAGE_IDENTITY schema mismatch: {payload.get('schema')}")
    files = payload.get("files")
    if not isinstance(files, Mapping) or not files:
        raise ValueError("PACKAGE_IDENTITY files map is empty or invalid")
    for rel, meta in files.items():
        if not isinstance(meta, Mapping):
            raise ValueError(f"PACKAGE_IDENTITY entry invalid: {rel}")
        path = (package_root / rel).resolve()
        try:
            path.relative_to(package_root)
        except ValueError as error:
            raise ValueError(f"PACKAGE_IDENTITY path escapes package root: {rel}") from error
        if not path.is_file():
            raise ValueError(f"PACKAGE_IDENTITY missing file on disk: {rel}")
        actual_size = path.stat().st_size
        expected_size = meta.get("size")
        if expected_size != actual_size:
            raise ValueError(f"PACKAGE_IDENTITY stale size for {rel}: expected {expected_size}, actual {actual_size}")
        actual_sha = sha256_file(path)
        expected_sha = meta.get("sha256")
        if expected_sha != actual_sha:
            raise ValueError(f"PACKAGE_IDENTITY sha256 mismatch (stale) for {rel}")
    expected_hash = payload.get("content_hash")
    recomputed = content_hash(dict(payload))
    if expected_hash != recomputed:
        raise ValueError("PACKAGE_IDENTITY content_hash mismatch")
    return payload


def write_and_validate_package_identity(
    *,
    package_root: Path,
    host_run: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Generate PACKAGE_IDENTITY only after the package tree is final, then revalidate."""

    from lap.verifiers.cover.w3_contracts import sha256_file

    package_root = Path(package_root).resolve()
    if not package_root.is_dir():
        raise ValueError(f"package root missing: {package_root}")
    identity_path = package_root / _PACKAGE_IDENTITY_NAME
    if identity_path.exists():
        identity_path.unlink()
    file_map = _package_relative_file_map(package_root)
    if not file_map:
        raise ValueError(f"package root has no sealable files: {package_root}")
    files = {rel: {"sha256": sha256_file(path), "size": path.stat().st_size} for rel, path in file_map.items()}
    payload: dict[str, Any] = {
        "schema": PACKAGE_IDENTITY_SCHEMA,
        "package_root": str(package_root),
        "files": files,
        "host_run": dict(host_run or {}),
    }
    payload["content_hash"] = content_hash(payload)
    identity_path.write_bytes(canonical_bytes(payload) + b"\n")
    return validate_package_identity(package_root)


def write_w5_handoff_payload(
    *,
    output_root: Path,
    acceptance_report: dict[str, Any],
    deployment_root: Path,
) -> dict[str, Any]:
    """Write the compact W3→W5 handoff consumed by downstream integration."""

    output_root = Path(output_root).resolve()
    deployment_root = Path(deployment_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    if "staging" in deployment_root.parts:
        raise ValueError(
            "w5 handoff deployment_root must name the final published deployment, "
            f"not a staging path: {deployment_root}"
        )
    if not deployment_root.is_dir():
        raise ValueError(f"w5 handoff deployment_root does not exist: {deployment_root}")
    try:
        deployment_root.relative_to(output_root)
    except ValueError as error:
        raise ValueError(
            "w5 handoff deployment_root must resolve under the published package root: "
            f"{deployment_root} not in {output_root}"
        ) from error
    payload = {
        "schema": "osx_cover_w3_to_w5_handoff_v1",
        "handoff_status": "READY_FOR_W5",
        "authority": "recorded_data_offline_integration_only",
        "package_scope": CANONICAL_PACKAGE_SCOPE,
        "acceptance_schema": acceptance_report.get("schema"),
        "acceptance_content_hash": acceptance_report.get("content_hash"),
        "deployment_root": str(deployment_root),
    }
    payload["content_hash"] = content_hash(payload)
    (output_root / "w5_handoff.json").write_bytes(canonical_bytes(payload) + b"\n")
    return payload
