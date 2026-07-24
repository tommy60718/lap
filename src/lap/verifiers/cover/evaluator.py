"""Deterministic fixed-checkpoint evaluation and clustered confidence intervals."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from lap.verifiers.cover.w3_contracts import content_hash


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
) -> dict[str, Any]:
    semantic = np.asarray(semantic_embeddings, dtype=np.float64)
    action = np.asarray(action_embeddings, dtype=np.float64)
    if semantic.ndim != 2 or action.shape != semantic.shape or len(sample_ids) != semantic.shape[0]:
        raise ValueError("embedding and sample-id shapes are incompatible")
    sample_index = {sample_id: index for index, sample_id in enumerate(sample_ids)}
    logits = checkpoint_logit_scale * semantic @ action.T
    semantic_hits = np.argmax(logits, axis=1) == np.arange(len(sample_ids))
    action_hits = np.argmax(logits.T, axis=1) == np.arange(len(sample_ids))
    if len(sample_ids) == 0:
        raise ValueError("evaluation pool cannot be empty")
    metrics: dict[str, Any] = {
        "schema": "osx_cover_w3_evaluation_v1",
        "pool": {"count": len(sample_ids), "top1_chance": 1 / len(sample_ids), "top5_chance": 5 / len(sample_ids)},
        "retrieval": {
            "semantic_to_action_top1": float(semantic_hits.mean()),
            "semantic_to_action_top5": _topk(logits, min(5, len(sample_ids))),
            "action_to_semantic_top1": float(action_hits.mean()),
            "action_to_semantic_top5": _topk(logits.T, min(5, len(sample_ids))),
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
        expected_conditions = {
            (shape, direction)
            for shape in ("circular", "square")
            for direction in ("-x", "+x", "-y", "+y")
        }
        observed_conditions = {(item["peg_shape"], item["approach_direction"]) for item in conditions}
        if observed_conditions != expected_conditions:
            raise ValueError("strict W3 evaluation requires all eight conditions")
        if len(sample_ids) != 1118 or len(shuffled) != 1112 or len(nearby) != 1118:
            raise ValueError("strict W3 evaluation requires the canonical pool and pair counts")
        condition_by_id = dict(zip(sample_ids, conditions, strict=True))
        episode_by_id = dict(zip(sample_ids, episode_ids, strict=True))
        for pair in shuffled_pairs:
            semantic = condition_by_id[pair["semantic_sample_id"]]
            history = condition_by_id[pair["history_sample_id"]]
            if semantic == history or semantic["peg_shape"] == history["peg_shape"]:
                raise ValueError("strict shuffled pairs must cross peg shapes")
        for pair in nearby_pairs:
            if pair["semantic_sample_id"] == pair["history_sample_id"] or episode_by_id[
                pair["semantic_sample_id"]
            ] != episode_by_id[pair["history_sample_id"]]:
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
    metrics["content_hash"] = content_hash(metrics)
    return metrics


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
