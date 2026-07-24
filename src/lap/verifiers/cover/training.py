"""W3 training step, bounded GPU probe, and controlled base-only construction."""

from __future__ import annotations

from collections.abc import Iterable
import platform
from typing import Any

import torch

from lap.verifiers.cover.model import VerifierConfig
from lap.verifiers.cover.model import VerifierModel
from lap.verifiers.cover.protocol import RunProtocol


def train_one_batch(
    model: VerifierModel, batch: dict[str, Any], optimizer: torch.optim.Optimizer, *, clip_norm: float = 1.0
) -> dict[str, float]:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    output = model(batch["base_rgb"], batch.get("wrist_rgb"), batch["instructions"], batch["action_histories"])
    loss, metrics = model.contrastive_loss(output)
    if not torch.isfinite(loss):
        raise FloatingPointError("W3 training loss is nonfinite")
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
    if not torch.isfinite(torch.as_tensor(grad_norm)):
        raise FloatingPointError("W3 gradient norm is nonfinite")
    optimizer.step()
    with torch.no_grad():
        model.logit_scale.clamp_(0.0, torch.log(torch.tensor(100.0, device=model.logit_scale.device)))
    metrics["gradient_norm"] = float(grad_norm)
    return metrics


def create_optimizer(
    model: VerifierModel, protocol: RunProtocol
) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=protocol.learning_rate)

    def schedule(epoch: int) -> float:
        if epoch < protocol.warmup_epochs:
            return float(epoch + 1) / float(protocol.warmup_epochs)
        return 1.0

    return optimizer, torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)


def _gpu_snapshot() -> list[dict[str, Any]]:
    if not torch.cuda.is_available():
        return []
    snapshots = []
    for index in range(torch.cuda.device_count()):
        free, total = torch.cuda.mem_get_info(index)
        properties = torch.cuda.get_device_properties(index)
        snapshots.append(
            {
                "index": index,
                "name": properties.name,
                "physical_bytes": properties.total_memory,
                "free_bytes": free,
                "total_visible_bytes": total,
            }
        )
    return snapshots


def probe_batch_sizes(
    model_factory, batch_factory, *, candidates: Iterable[int] = (64, 32, 16), device: torch.device | None = None
) -> dict[str, Any]:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    attempts = []
    before = _gpu_snapshot()
    for batch_size in candidates:
        model = model_factory().to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-6)
        try:
            metrics = train_one_batch(model, batch_factory(batch_size, device), optimizer)
            attempts.append({"batch_size": batch_size, "status": "passed", "metrics": metrics})
            return {
                "schema": "osx_cover_w3_batch_probe_v1",
                "host": platform.node(),
                "backbone": model.backbone.__class__.__name__,
                "gpu_before": before,
                "attempts": attempts,
                "selected_per_rank_batch_size": batch_size,
            }
        except (RuntimeError, FloatingPointError) as error:
            message = str(error)
            if isinstance(error, RuntimeError) and "out of memory" not in message.lower():
                raise
            attempts.append({"batch_size": batch_size, "status": "failed", "error": type(error).__name__})
            if device.type == "cuda":
                torch.cuda.empty_cache()
    raise RuntimeError("no W3 batch size fit the approved probe order")


def make_base_only_config(config: VerifierConfig) -> VerifierConfig:
    return VerifierConfig(
        backbone_width=config.backbone_width,
        embedding_width=config.embedding_width,
        visual_tokens=config.visual_tokens,
        num_heads=config.num_heads,
        pooling_layers=config.pooling_layers,
        trajectory_layers=config.trajectory_layers,
        feed_forward_width=config.feed_forward_width,
        history_length=config.history_length,
        action_width=config.action_width,
        use_wrist=False,
    )


def require_acceptance_metrics(report: dict[str, Any]) -> None:
    retrieval = report["retrieval"]
    chance = report["pool"]["top1_chance"]
    if retrieval["semantic_to_action_top1_ci95"][0] <= chance or retrieval["action_to_semantic_top1_ci95"][0] <= chance:
        raise ValueError("W3 retrieval lower confidence bounds do not exceed chance")
    for key in ("aligned_minus_shuffled", "aligned_minus_nearby"):
        if report["margins"][key]["ci95"][0] <= 0:
            raise ValueError(f"W3 {key} confidence interval does not exceed zero")
