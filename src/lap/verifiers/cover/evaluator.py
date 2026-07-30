"""Deterministic fixed-checkpoint evaluation and clustered confidence intervals."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

from lap.verifiers.cover.protocol import SHUFFLED_COUNT
from lap.verifiers.cover.protocol import VALIDATION_COUNT
from lap.verifiers.cover.protocol import validate_protocol_directory
from lap.verifiers.cover.w3_contracts import W3_CHECKPOINT_SCHEMA
from lap.verifiers.cover.w3_contracts import canonical_bytes
from lap.verifiers.cover.w3_contracts import content_hash
from lap.verifiers.cover.w3_contracts import sha256_file

REQUIRED_CONDITION_KEYS = ("circular", "square")
REQUIRED_DIRECTIONS = ("-x", "+x", "-y", "+y")
CANONICAL_TIMING_FORBIDDEN_KEYS = frozenset(
    {"timestamp", "timestamps", "timing", "seconds", "elapsed", "started_at", "finished_at"}
)


def _topk(logits: np.ndarray, k: int) -> float:
    labels = np.arange(logits.shape[0])
    return float(np.mean(np.any(np.argsort(-logits, axis=1)[:, :k] == labels[:, None], axis=1)))


def _summary(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "std": float(np.std(values)),
        "q05": float(np.quantile(values, 0.05)),
        "q95": float(np.quantile(values, 0.95)),
        "fraction_gt_zero": float(np.mean(values > 0)),
    }


def clustered_percentile_interval(
    values: np.ndarray, episodes: Sequence[str], bootstrap_indices: np.ndarray
) -> list[float]:
    values = np.asarray(values, dtype=np.float64)
    episodes = np.asarray(episodes)
    ordered = sorted(set(episodes.tolist()))
    by_episode = [values[episodes == episode] for episode in ordered]
    estimates = []
    for row in np.asarray(bootstrap_indices):
        pieces = [by_episode[int(index)] for index in row]
        estimates.append(float(np.concatenate(pieces).mean()))
    return [
        float(np.quantile(estimates, 0.025, method="linear")),
        float(np.quantile(estimates, 0.975, method="linear")),
    ]


def require_explicit_best_checkpoint(checkpoint: Path) -> Path:
    """Reject latest, unmarked, or non-best checkpoint paths before any scoring."""

    checkpoint = Path(checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if checkpoint.name != "best.pt":
        raise ValueError("evaluation requires an explicitly named best.pt checkpoint")
    return checkpoint


def load_fixed_best_checkpoint_logit_scale(checkpoint: Path, model: torch.nn.Module) -> float:
    """Strictly load model weights from an explicit best checkpoint and return its logit scale."""

    checkpoint = require_explicit_best_checkpoint(checkpoint)
    try:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    except Exception as error:  # pragma: no cover - torch-version-specific errors
        raise ValueError("evaluation checkpoint is not a loadable W3 payload") from error
    if not isinstance(payload, dict) or payload.get("schema") != W3_CHECKPOINT_SCHEMA:
        raise ValueError("evaluation requires an explicit W3 training checkpoint")
    model_state = payload.get("model_state")
    if not isinstance(model_state, dict):
        raise ValueError("evaluation checkpoint missing model_state")
    expected_keys = set(model.state_dict())
    actual_keys = set(model_state)
    if actual_keys != expected_keys:
        raise ValueError("evaluation checkpoint model state keys mismatch")
    model.load_state_dict(model_state, strict=True)
    model.eval()
    return float(model.logit_scale.detach().clamp(0.0, np.log(100.0)).exp())


def evaluate_embeddings(
    semantic_embeddings: np.ndarray,
    action_embeddings: np.ndarray,
    *,
    sample_ids: Sequence[str],
    episode_ids: Sequence[str],
    conditions: Sequence[dict[str, str]],
    shuffled_pairs: Sequence[dict[str, str]],
    nearby_pairs: Sequence[dict[str, str]],
    bootstrap_indices: np.ndarray,
    checkpoint_logit_scale: float = 1.0,
    strict_protocol: bool = False,
    expected_sample_ids: Sequence[str] | None = None,
    expected_shuffled_pairs: Sequence[dict[str, str]] | None = None,
    expected_nearby_pairs: Sequence[dict[str, str]] | None = None,
    expected_bootstrap_indices: np.ndarray | None = None,
) -> dict[str, Any]:
    semantic = np.asarray(semantic_embeddings, dtype=np.float64)
    action = np.asarray(action_embeddings, dtype=np.float64)
    if semantic.ndim != 2 or action.shape != semantic.shape or len(sample_ids) != semantic.shape[0]:
        raise ValueError("embedding and sample-id shapes are incompatible")
    if len(sample_ids) != len(episode_ids) or len(sample_ids) != len(conditions):
        raise ValueError("sample, episode, and condition pools must align")
    if len(sample_ids) == 0:
        raise ValueError("evaluation pool cannot be empty")
    pool_count = len(sample_ids)
    if expected_sample_ids is not None and list(sample_ids) != list(expected_sample_ids):
        raise ValueError("validation manifest order drifted from the accepted sample_id order")
    if expected_shuffled_pairs is not None and list(shuffled_pairs) != list(expected_shuffled_pairs):
        raise ValueError("shuffled pair artifact drifted from the accepted pair order")
    if expected_nearby_pairs is not None and list(nearby_pairs) != list(expected_nearby_pairs):
        raise ValueError("nearby pair artifact drifted from the accepted pair order")
    if expected_bootstrap_indices is not None and not np.array_equal(
        np.asarray(bootstrap_indices), np.asarray(expected_bootstrap_indices)
    ):
        raise ValueError("bootstrap artifact drifted from the accepted bootstrap identity")

    if strict_protocol:
        if (
            pool_count != VALIDATION_COUNT
            or len(shuffled_pairs) != SHUFFLED_COUNT
            or len(nearby_pairs) != VALIDATION_COUNT
        ):
            raise ValueError("strict W3 evaluation requires the canonical pool and pair counts")
        expected_conditions = {
            (shape, direction) for shape in REQUIRED_CONDITION_KEYS for direction in REQUIRED_DIRECTIONS
        }
        observed_conditions = {(item["peg_shape"], item["approach_direction"]) for item in conditions}
        if observed_conditions != expected_conditions:
            raise ValueError("strict W3 evaluation requires all eight conditions")

    sample_index = {sample_id: index for index, sample_id in enumerate(sample_ids)}
    logits = checkpoint_logit_scale * semantic @ action.T
    semantic_hits = np.argmax(logits, axis=1) == np.arange(len(sample_ids))
    action_hits = np.argmax(logits.T, axis=1) == np.arange(len(sample_ids))
    metrics: dict[str, Any] = {
        "schema": "osx_cover_w3_evaluation_v1",
        "pool": {
            "count": pool_count,
            "top1_chance": 1 / pool_count,
            "top5_chance": 5 / pool_count,
        },
        "retrieval": {
            "semantic_to_action_top1": float(semantic_hits.mean()),
            "semantic_to_action_top5": _topk(logits, min(5, pool_count)),
            "action_to_semantic_top1": float(action_hits.mean()),
            "action_to_semantic_top5": _topk(logits.T, min(5, pool_count)),
            "semantic_to_action_top1_ci95": clustered_percentile_interval(
                semantic_hits.astype(np.float64), episode_ids, bootstrap_indices
            ),
            "action_to_semantic_top1_ci95": clustered_percentile_interval(
                action_hits.astype(np.float64), episode_ids, bootstrap_indices
            ),
        },
        "row_metrics": {
            "sample_ids": list(sample_ids),
            "episode_ids": list(episode_ids),
            "semantic_to_action_hit": semantic_hits.tolist(),
            "action_to_semantic_hit": action_hits.tolist(),
        },
        "conditions": {},
    }

    def pair_scores(pairs: Sequence[dict[str, str]]) -> tuple[np.ndarray, np.ndarray, list[str]]:
        scores = []
        aligned_scores = []
        pair_episodes = []
        for pair in pairs:
            if pair["semantic_sample_id"] not in sample_index or pair["history_sample_id"] not in sample_index:
                raise ValueError("pair manifest references a sample outside the evaluation pool")
            semantic_index = sample_index[pair["semantic_sample_id"]]
            history_index = sample_index[pair["history_sample_id"]]
            scores.append(logits[semantic_index, history_index])
            aligned_scores.append(logits[semantic_index, semantic_index])
            pair_episodes.append(episode_ids[semantic_index])
        return np.asarray(scores, dtype=np.float64), np.asarray(aligned_scores, dtype=np.float64), pair_episodes

    shuffled, shuffled_aligned, shuffled_episodes = pair_scores(shuffled_pairs)
    nearby, nearby_aligned, nearby_episodes = pair_scores(nearby_pairs)
    if len(shuffled) == 0 or len(nearby) == 0:
        raise ValueError("mismatch pair manifests cannot be empty")
    if strict_protocol:
        condition_by_id = dict(zip(sample_ids, conditions, strict=True))
        episode_by_id = dict(zip(sample_ids, episode_ids, strict=True))
        for pair in shuffled_pairs:
            semantic = condition_by_id[pair["semantic_sample_id"]]
            history = condition_by_id[pair["history_sample_id"]]
            if semantic == history or semantic["peg_shape"] == history["peg_shape"]:
                raise ValueError("strict shuffled pairs must cross peg shapes")
        for pair in nearby_pairs:
            if (
                pair["semantic_sample_id"] == pair["history_sample_id"]
                or episode_by_id[pair["semantic_sample_id"]] != episode_by_id[pair["history_sample_id"]]
            ):
                raise ValueError("strict nearby pairs must be non-self pairs within an episode")
    shuffled_margin = shuffled_aligned - shuffled
    nearby_margin = nearby_aligned - nearby
    metrics["margins"] = {
        "aligned_minus_shuffled": {
            **_summary(shuffled_margin),
            "ci95": clustered_percentile_interval(shuffled_margin, shuffled_episodes, bootstrap_indices),
        },
        "aligned_minus_nearby": {
            **_summary(nearby_margin),
            "ci95": clustered_percentile_interval(nearby_margin, nearby_episodes, bootstrap_indices),
        },
    }
    condition_by_id = dict(zip(sample_ids, conditions, strict=True))
    for condition in sorted({(item["peg_shape"], item["approach_direction"]) for item in conditions}):
        mask = np.asarray([(item["peg_shape"], item["approach_direction"]) == condition for item in conditions])
        shuffled_mask = np.asarray(
            [
                (
                    condition_by_id[pair["semantic_sample_id"]]["peg_shape"],
                    condition_by_id[pair["semantic_sample_id"]]["approach_direction"],
                )
                == condition
                for pair in shuffled_pairs
            ]
        )
        nearby_mask = np.asarray(
            [
                (
                    condition_by_id[pair["semantic_sample_id"]]["peg_shape"],
                    condition_by_id[pair["semantic_sample_id"]]["approach_direction"],
                )
                == condition
                for pair in nearby_pairs
            ]
        )
        metrics["conditions"][f"{condition[0]}:{condition[1]}"] = {
            "count": int(mask.sum()),
            "semantic_to_action_top1": float(semantic_hits[mask].mean()),
            "action_to_semantic_top1": float(action_hits[mask].mean()),
            "aligned_minus_shuffled_mean": float(shuffled_margin[shuffled_mask].mean())
            if shuffled_mask.any()
            else None,
            "aligned_minus_nearby_mean": float(nearby_margin[nearby_mask].mean()) if nearby_mask.any() else None,
            "failures": [sample_ids[index] for index in np.flatnonzero(mask & ~semantic_hits)[:10]],
        }
    if strict_protocol and set(metrics["conditions"]) != {
        f"{shape}:{direction}" for shape in REQUIRED_CONDITION_KEYS for direction in REQUIRED_DIRECTIONS
    }:
        raise ValueError("condition coverage drifted from the required eight shape/direction cells")
    metrics["content_hash"] = content_hash(metrics)
    return metrics


def render_readable_report(report: Mapping[str, Any], *, timing: Mapping[str, Any] | None = None) -> str:
    """Human-readable companion; timing may appear here but never in canonical JSON."""

    lines = [
        "W3 fixed-checkpoint evaluation report",
        f"schema: {report.get('schema')}",
        f"pool_count: {report.get('pool', {}).get('count')}",
        f"top1_chance: {report.get('pool', {}).get('top1_chance')}",
        f"top5_chance: {report.get('pool', {}).get('top5_chance')}",
        "retrieval:",
    ]
    retrieval = report.get("retrieval", {})
    lines.extend(
        f"  {key}: {retrieval[key]}"
        for key in (
            "semantic_to_action_top1",
            "semantic_to_action_top5",
            "action_to_semantic_top1",
            "action_to_semantic_top5",
            "semantic_to_action_top1_ci95",
            "action_to_semantic_top1_ci95",
        )
        if key in retrieval
    )
    lines.append("margins:")
    for name, payload in report.get("margins", {}).items():
        lines.append(
            f"  {name}: mean={payload.get('mean')} fraction_gt_zero={payload.get('fraction_gt_zero')} "
            f"ci95={payload.get('ci95')}"
        )
    lines.append("conditions:")
    for name, payload in sorted(report.get("conditions", {}).items()):
        lines.append(
            f"  {name}: count={payload.get('count')} "
            f"semantic_to_action_top1={payload.get('semantic_to_action_top1')} "
            f"action_to_semantic_top1={payload.get('action_to_semantic_top1')} "
            f"aligned_minus_shuffled_mean={payload.get('aligned_minus_shuffled_mean')} "
            f"aligned_minus_nearby_mean={payload.get('aligned_minus_nearby_mean')} "
            f"failures={payload.get('failures')}"
        )
    if timing:
        lines.append("timing:")
        for key, value in timing.items():
            lines.append(f"  {key}: {value}")
    lines.append(f"content_hash: {report.get('content_hash')}")
    lines.append("accepted: false")
    lines.append("usefulness_decision_owner: W3-09")
    return "\n".join(lines) + "\n"


def _assert_no_timing_fields(payload: Mapping[str, Any]) -> None:
    forbidden = CANONICAL_TIMING_FORBIDDEN_KEYS.intersection(payload)
    if forbidden:
        raise ValueError(f"canonical evaluation JSON must not include timing fields: {sorted(forbidden)}")


def evaluate_explicit_best_checkpoint(
    *,
    checkpoint: Path,
    protocol_dir: Path,
    output_root: Path,
    semantic_embeddings: np.ndarray,
    action_embeddings: np.ndarray,
    sample_ids: Sequence[str],
    episode_ids: Sequence[str],
    conditions: Sequence[dict[str, str]],
    checkpoint_logit_scale: float | None = None,
    model: torch.nn.Module | None = None,
    timing: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Public seam: explicit best checkpoint → byte-canonical JSON + readable report."""

    checkpoint = require_explicit_best_checkpoint(checkpoint)
    output_root = Path(output_root)
    if output_root.exists():
        raise FileExistsError(output_root)
    protocol = validate_protocol_directory(Path(protocol_dir), require_complete=True)
    expected_sample_ids = [row["sample_id"] for row in protocol["validation_semantics"]["rows"]]
    shuffled_pairs = protocol["shuffled_pairs"]["pairs"]
    nearby_pairs = protocol["nearby_pairs"]["pairs"]
    bootstrap_indices = protocol["bootstrap_indices"]
    if model is not None:
        scale = load_fixed_best_checkpoint_logit_scale(checkpoint, model)
    elif checkpoint_logit_scale is None:
        # Validate payload shape without requiring a live model when embeddings are supplied.
        try:
            payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        except Exception as error:  # pragma: no cover
            raise ValueError("evaluation checkpoint is not a loadable W3 payload") from error
        if not isinstance(payload, dict) or payload.get("schema") != W3_CHECKPOINT_SCHEMA:
            raise ValueError("evaluation requires an explicit W3 training checkpoint")
        if "model_state" not in payload:
            raise ValueError("evaluation checkpoint missing model_state")
        scale = 1.0
    else:
        scale = float(checkpoint_logit_scale)
        try:
            payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        except Exception as error:  # pragma: no cover
            raise ValueError("evaluation checkpoint is not a loadable W3 payload") from error
        if not isinstance(payload, dict) or payload.get("schema") != W3_CHECKPOINT_SCHEMA:
            raise ValueError("evaluation requires an explicit W3 training checkpoint")

    metrics = evaluate_embeddings(
        semantic_embeddings,
        action_embeddings,
        sample_ids=sample_ids,
        episode_ids=episode_ids,
        conditions=conditions,
        shuffled_pairs=shuffled_pairs,
        nearby_pairs=nearby_pairs,
        bootstrap_indices=bootstrap_indices,
        checkpoint_logit_scale=scale if checkpoint_logit_scale is None else float(checkpoint_logit_scale),
        strict_protocol=True,
        expected_sample_ids=expected_sample_ids,
        expected_shuffled_pairs=shuffled_pairs,
        expected_nearby_pairs=nearby_pairs,
        expected_bootstrap_indices=bootstrap_indices,
    )
    staging = output_root.parent / f".{output_root.name}.staging"
    if staging.exists():
        raise FileExistsError(staging)
    staging.mkdir(parents=True)
    try:
        canonical = {
            "schema": "osx_cover_w3_fixed_checkpoint_evaluation_v1",
            "mode": "evaluate",
            "accepted": False,
            "usefulness_decision_owner": "W3-09",
            "checkpoint": {
                "path": str(checkpoint),
                "name": checkpoint.name,
                "sha256": sha256_file(checkpoint),
                "selection": "explicit_best",
            },
            "protocol": {
                "dir": str(Path(protocol_dir)),
                "content_hash": protocol["protocol"]["content_hash"],
                "shuffled_count": len(shuffled_pairs),
                "nearby_count": len(nearby_pairs),
            },
            "pool": metrics["pool"],
            "retrieval": metrics["retrieval"],
            "margins": metrics["margins"],
            "conditions": metrics["conditions"],
            "metrics_content_hash": metrics["content_hash"],
        }
        _assert_no_timing_fields(canonical)
        canonical["content_hash"] = content_hash(canonical)
        _assert_no_timing_fields(canonical)
        (staging / "evaluation.json").write_bytes(canonical_bytes(canonical) + b"\n")
        (staging / "evaluation_report.txt").write_text(
            render_readable_report(metrics, timing=timing),
            encoding="utf-8",
        )
        staging.rename(output_root)
    except Exception:
        for path in sorted(staging.rglob("*"), reverse=True):
            if path.is_file():
                path.unlink()
            elif path.is_dir():
                path.rmdir()
        if staging.exists():
            staging.rmdir()
        raise
    return canonical


def compare_ablation(
    two_view: dict[str, Any], base_only: dict[str, Any], *, bootstrap_indices: np.ndarray | None = None
) -> dict[str, Any]:
    if two_view.get("pool") != base_only.get("pool"):
        raise ValueError("ablation reports must use the same evaluation pool")
    two_rows = two_view.get("row_metrics", {})
    base_rows = base_only.get("row_metrics", {})
    if two_rows.get("sample_ids") != base_rows.get("sample_ids"):
        raise ValueError("ablation reports must use the same sample IDs")
    if two_rows.get("episode_ids") != base_rows.get("episode_ids"):
        raise ValueError("ablation reports must use the same episode IDs")
    if bootstrap_indices is None:
        if two_view.get("bootstrap_indices") is None or base_only.get("bootstrap_indices") is None:
            raise ValueError("ablation reports must include shared bootstrap indices")
        two_bootstrap = np.asarray(two_view["bootstrap_indices"])
        base_bootstrap = np.asarray(base_only["bootstrap_indices"])
    else:
        two_bootstrap = base_bootstrap = np.asarray(bootstrap_indices)
    if not np.array_equal(two_bootstrap, base_bootstrap):
        raise ValueError("ablation reports must use the same bootstrap indices")
    keys = ("semantic_to_action_top1", "action_to_semantic_top1")
    differences = {key: two_view["retrieval"][key] - base_only["retrieval"][key] for key in keys}
    paired_ci95 = {}
    for key, row_key in (
        ("semantic_to_action_top1", "semantic_to_action_hit"),
        ("action_to_semantic_top1", "action_to_semantic_hit"),
    ):
        difference = np.asarray(two_rows[row_key], dtype=np.float64) - np.asarray(base_rows[row_key], dtype=np.float64)
        if difference.shape != (two_view["pool"]["count"],):
            raise ValueError("ablation row metrics must cover the complete evaluation pool")
        paired_ci95[key] = clustered_percentile_interval(difference, two_rows["episode_ids"], two_bootstrap)
    established = all(interval[0] > 0 for interval in paired_ci95.values())
    payload = {
        "schema": "osx_cover_w3_ablation_v1",
        "two_view_minus_base_only": differences,
        "paired_ci95": paired_ci95,
        "wrist_benefit_established": established,
    }
    payload["content_hash"] = content_hash(payload)
    return payload
