from __future__ import annotations

import numpy as np
import pytest

from lap.verifiers.cover.evaluator import compare_ablation
from lap.verifiers.cover.evaluator import evaluate_embeddings
from lap.verifiers.cover.protocol import build_bootstrap_indices


def test_evaluator_reports_full_pool_retrieval_margins_conditions_and_hash():
    embeddings = np.eye(8, dtype=np.float64)
    sample_ids = [f"sample-{i}" for i in range(8)]
    episodes = [f"episode-{i}" for i in range(8)]
    conditions = [{"peg_shape": "circular" if i % 2 else "square", "approach_direction": "-x"} for i in range(8)]
    pairs = [{"semantic_sample_id": sample_ids[i], "history_sample_id": sample_ids[(i + 1) % 8]} for i in range(8)]
    bootstrap = build_bootstrap_indices(episodes, replicates=4)
    result = evaluate_embeddings(
        embeddings,
        embeddings,
        sample_ids=sample_ids,
        episode_ids=episodes,
        conditions=conditions,
        shuffled_pairs=pairs,
        nearby_pairs=pairs,
        bootstrap_indices=bootstrap,
    )
    assert result["pool"]["count"] == 8
    assert result["retrieval"]["semantic_to_action_top1"] == 1.0
    assert result["margins"]["aligned_minus_shuffled"]["ci95"][0] > 0
    assert result["content_hash"]
    assert len(result["conditions"]) == 2


def test_ablation_is_matched_and_states_wrist_benefit():
    row_metrics = {
        "sample_ids": [f"sample-{i}" for i in range(8)],
        "episode_ids": [f"episode-{i}" for i in range(8)],
        "semantic_to_action_hit": [False] * 8,
        "action_to_semantic_hit": [False] * 8,
    }
    bootstrap = build_bootstrap_indices(row_metrics["episode_ids"], replicates=4)
    base = {
        "pool": {"count": 8},
        "retrieval": {"semantic_to_action_top1": 0.0, "action_to_semantic_top1": 0.0},
        "row_metrics": row_metrics,
        "bootstrap_indices": bootstrap,
    }
    two = {
        "pool": {"count": 8},
        "retrieval": {"semantic_to_action_top1": 1.0, "action_to_semantic_top1": 1.0},
        "row_metrics": {
            **row_metrics,
            "semantic_to_action_hit": [True] * 8,
            "action_to_semantic_hit": [True] * 8,
        },
        "bootstrap_indices": bootstrap,
    }
    result = compare_ablation(two, base)
    assert result["two_view_minus_base_only"]["semantic_to_action_top1"] == 1.0
    assert result["paired_ci95"]["semantic_to_action_top1"][0] > 0
    assert result["wrist_benefit_established"] is True


def test_ablation_rejects_mismatched_rows():
    report = {
        "pool": {"count": 1},
        "retrieval": {"semantic_to_action_top1": 1.0, "action_to_semantic_top1": 1.0},
        "row_metrics": {
            "sample_ids": ["a"],
            "episode_ids": ["e"],
            "semantic_to_action_hit": [True],
            "action_to_semantic_hit": [True],
        },
        "bootstrap_indices": np.zeros((1, 1), dtype=np.int64),
    }
    mismatched = {**report, "row_metrics": {**report["row_metrics"], "sample_ids": ["b"]}}
    with pytest.raises(ValueError, match="same sample"):
        compare_ablation(report, mismatched)
