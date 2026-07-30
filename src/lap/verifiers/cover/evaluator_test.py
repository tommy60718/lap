from __future__ import annotations

import json
from pathlib import Path
import random

import numpy as np
import pytest
import torch

from lap.verifiers.cover.checkpoint import build_checkpoint_contract
from lap.verifiers.cover.checkpoint import build_four_state_inventory
from lap.verifiers.cover.checkpoint import build_progress
from lap.verifiers.cover.checkpoint import save_training_checkpoint
from lap.verifiers.cover.evaluator import clustered_percentile_interval
from lap.verifiers.cover.evaluator import compare_ablation
from lap.verifiers.cover.evaluator import evaluate_embeddings
from lap.verifiers.cover.evaluator import evaluate_explicit_best_checkpoint
from lap.verifiers.cover.evaluator import render_readable_report
from lap.verifiers.cover.model import TinyFrozenBackbone
from lap.verifiers.cover.model import VerifierConfig
from lap.verifiers.cover.model import VerifierModel
from lap.verifiers.cover.protocol import build_bootstrap_indices
from lap.verifiers.cover.w3_contracts import canonical_bytes
from lap.verifiers.cover.w3_contracts import content_hash

PROTOCOL_DIR = Path(__file__).resolve().parents[4] / "artifacts" / "w3" / "protocol"


def _tiny_model():
    return VerifierModel(
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


def _save_best(tmp_path: Path, *, name: str = "best.pt") -> Path:
    model = _tiny_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
    contract = build_checkpoint_contract(
        audit_manifest_sha256="a" * 64,
        bridge_artifact_sha256="b" * 64,
        target_fingerprint="c" * 64,
        protocol_content_hash="d" * 64,
        protocol_version="w3-g02-accepted-run-v1",
        model_config=model.config.to_dict(),
        four_state_inventory=build_four_state_inventory(),
        w2_identities={
            "train_manifest_hash": "e" * 64,
            "phrase_manifest_hash": "f" * 64,
            "normalization_artifact_hash": "1" * 64,
        },
        environment={"torch": torch.__version__, "cuda_rng_device_count": 0, "fixture": True},
    )
    path = tmp_path / name
    save_training_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        progress=build_progress(epoch=2, global_step=10, best_metric=0.1, world_size=1),
        contract=contract,
        sampler_state={"epoch": 1, "rank": 0, "world_size": 1},
        rank=0,
        rng_states={
            0: {
                "python": random.getstate(),
                "numpy": {
                    "algorithm": np.random.get_state()[0],
                    "keys": np.random.get_state()[1].tolist(),
                    "position": int(np.random.get_state()[2]),
                    "has_gauss": int(np.random.get_state()[3]),
                    "cached_gaussian": float(np.random.get_state()[4]),
                },
                "torch": torch.get_rng_state(),
            }
        },
    )
    return path


def _canonical_pool_inputs():
    semantics = json.loads((PROTOCOL_DIR / "validation_semantics.json").read_text(encoding="utf-8"))["rows"]
    sample_ids = [row["sample_id"] for row in semantics]
    episode_ids = [row["episode_id"] for row in semantics]
    conditions = []
    for sample_id in sample_ids:
        shape = "circular" if sample_id.startswith("circular") else "square"
        token = sample_id.split("_")[1]
        direction = {"negx": "-x", "posx": "+x", "negy": "-y", "posy": "+y"}[token]
        conditions.append({"peg_shape": shape, "approach_direction": direction})
    shuffled = json.loads((PROTOCOL_DIR / "shuffled_pairs.json").read_text(encoding="utf-8"))["pairs"]
    nearby = json.loads((PROTOCOL_DIR / "nearby_pairs.json").read_text(encoding="utf-8"))["pairs"]
    bootstrap = np.load(PROTOCOL_DIR / "bootstrap_indices.npy")
    embeddings = np.eye(len(sample_ids), dtype=np.float64)
    return {
        "sample_ids": sample_ids,
        "episode_ids": episode_ids,
        "conditions": conditions,
        "shuffled_pairs": shuffled,
        "nearby_pairs": nearby,
        "bootstrap_indices": bootstrap,
        "semantic_embeddings": embeddings,
        "action_embeddings": embeddings,
    }


def test_explicit_best_checkpoint_seam_rejects_latest_and_writes_canonical_and_readable(tmp_path):
    latest = _save_best(tmp_path, name="latest.pt")
    best = _save_best(tmp_path, name="best.pt")
    pool = _canonical_pool_inputs()
    out_latest = tmp_path / "out-latest"
    with pytest.raises(ValueError, match=r"best\.pt|explicit best"):
        evaluate_explicit_best_checkpoint(
            checkpoint=latest,
            protocol_dir=PROTOCOL_DIR,
            output_root=out_latest,
            semantic_embeddings=pool["semantic_embeddings"],
            action_embeddings=pool["action_embeddings"],
            sample_ids=pool["sample_ids"],
            episode_ids=pool["episode_ids"],
            conditions=pool["conditions"],
        )
    assert not out_latest.exists()

    out_a = tmp_path / "out-a"
    out_b = tmp_path / "out-b"
    first = evaluate_explicit_best_checkpoint(
        checkpoint=best,
        protocol_dir=PROTOCOL_DIR,
        output_root=out_a,
        semantic_embeddings=pool["semantic_embeddings"],
        action_embeddings=pool["action_embeddings"],
        sample_ids=pool["sample_ids"],
        episode_ids=pool["episode_ids"],
        conditions=pool["conditions"],
        timing={"seconds": 1.25},
    )
    second = evaluate_explicit_best_checkpoint(
        checkpoint=best,
        protocol_dir=PROTOCOL_DIR,
        output_root=out_b,
        semantic_embeddings=pool["semantic_embeddings"],
        action_embeddings=pool["action_embeddings"],
        sample_ids=pool["sample_ids"],
        episode_ids=pool["episode_ids"],
        conditions=pool["conditions"],
        timing={"seconds": 9.99},
    )
    canonical_a = (out_a / "evaluation.json").read_bytes()
    canonical_b = (out_b / "evaluation.json").read_bytes()
    assert canonical_a == canonical_b
    payload = json.loads(canonical_a.decode("utf-8"))
    assert "timestamp" not in payload
    assert "timing" not in payload
    assert "seconds" not in payload
    assert payload["accepted"] is False
    assert payload["checkpoint"]["path"].endswith("best.pt")
    assert payload["pool"]["count"] == 1118
    assert payload["pool"]["top1_chance"] == pytest.approx(1 / 1118)
    assert payload["pool"]["top5_chance"] == pytest.approx(5 / 1118)
    assert (out_a / "evaluation_report.txt").is_file()
    readable = (out_a / "evaluation_report.txt").read_text(encoding="utf-8")
    assert "1.25" in readable or "seconds" in readable.lower()
    assert first["content_hash"] == second["content_hash"]
    assert first["accepted"] is False


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


def test_canonical_pool_reports_exact_chance_denominators_and_consumes_accepted_pairs():
    pool = _canonical_pool_inputs()
    result = evaluate_embeddings(
        pool["semantic_embeddings"],
        pool["action_embeddings"],
        sample_ids=pool["sample_ids"],
        episode_ids=pool["episode_ids"],
        conditions=pool["conditions"],
        shuffled_pairs=pool["shuffled_pairs"],
        nearby_pairs=pool["nearby_pairs"],
        bootstrap_indices=pool["bootstrap_indices"],
        checkpoint_logit_scale=2.5,
        strict_protocol=True,
        expected_sample_ids=pool["sample_ids"],
        expected_shuffled_pairs=pool["shuffled_pairs"],
        expected_nearby_pairs=pool["nearby_pairs"],
        expected_bootstrap_indices=pool["bootstrap_indices"],
    )
    assert result["pool"] == {"count": 1118, "top1_chance": 1 / 1118, "top5_chance": 5 / 1118}
    assert result["retrieval"]["semantic_to_action_top1"] == 1.0
    assert result["retrieval"]["action_to_semantic_top1"] == 1.0
    assert result["retrieval"]["semantic_to_action_top5"] == 1.0
    assert result["retrieval"]["action_to_semantic_top5"] == 1.0
    assert len(pool["shuffled_pairs"]) == 1112
    assert len(pool["nearby_pairs"]) == 1118
    assert result["margins"]["aligned_minus_shuffled"]["mean"] == pytest.approx(2.5)
    assert set(result["conditions"]) == {
        f"{shape}:{direction}" for shape in ("circular", "square") for direction in ("-x", "+x", "-y", "+y")
    }
    for condition in result["conditions"].values():
        assert condition["count"] > 0
        assert "semantic_to_action_top1" in condition
        assert "action_to_semantic_top1" in condition
        assert "aligned_minus_shuffled_mean" in condition
        assert "aligned_minus_nearby_mean" in condition
        assert isinstance(condition["failures"], list)


def test_clustered_interval_resamples_episodes_not_rows():
    values = np.asarray([1.0, 1.0, 0.0, 0.0], dtype=np.float64)
    episodes = ["a", "a", "b", "b"]
    # Episode indices 0=a (mean 1.0) and 1=b (mean 0.0). Row-level indexing of
    # the same pattern would only see values[0]/values[1] (both 1.0).
    bootstrap = np.asarray([[0, 0], [1, 1]], dtype=np.int64)

    interval = clustered_percentile_interval(values, episodes, bootstrap)
    assert interval[0] < 0.5 < interval[1]


def test_fixed_logit_scale_scales_aligned_minus_wrong_margins():
    embeddings = np.eye(8, dtype=np.float64)
    sample_ids = [f"s{i}" for i in range(8)]
    episodes = [f"e{i}" for i in range(8)]
    conditions = [{"peg_shape": "circular", "approach_direction": "-x"} for _ in range(8)]
    pairs = [{"semantic_sample_id": sample_ids[i], "history_sample_id": sample_ids[(i + 1) % 8]} for i in range(8)]
    bootstrap = build_bootstrap_indices(episodes, replicates=4)
    low = evaluate_embeddings(
        embeddings,
        embeddings,
        sample_ids=sample_ids,
        episode_ids=episodes,
        conditions=conditions,
        shuffled_pairs=pairs,
        nearby_pairs=pairs,
        bootstrap_indices=bootstrap,
        checkpoint_logit_scale=1.0,
    )
    high = evaluate_embeddings(
        embeddings,
        embeddings,
        sample_ids=sample_ids,
        episode_ids=episodes,
        conditions=conditions,
        shuffled_pairs=pairs,
        nearby_pairs=pairs,
        bootstrap_indices=bootstrap,
        checkpoint_logit_scale=3.0,
    )
    assert high["margins"]["aligned_minus_shuffled"]["mean"] == pytest.approx(
        3.0 * low["margins"]["aligned_minus_shuffled"]["mean"]
    )


def test_failure_tests_reject_pool_order_pair_bootstrap_and_condition_drift():
    pool = _canonical_pool_inputs()
    kwargs = {
        "semantic_embeddings": pool["semantic_embeddings"],
        "action_embeddings": pool["action_embeddings"],
        "sample_ids": pool["sample_ids"],
        "episode_ids": pool["episode_ids"],
        "conditions": pool["conditions"],
        "shuffled_pairs": pool["shuffled_pairs"],
        "nearby_pairs": pool["nearby_pairs"],
        "bootstrap_indices": pool["bootstrap_indices"],
        "strict_protocol": True,
        "expected_sample_ids": pool["sample_ids"],
        "expected_shuffled_pairs": pool["shuffled_pairs"],
        "expected_nearby_pairs": pool["nearby_pairs"],
        "expected_bootstrap_indices": pool["bootstrap_indices"],
    }
    with pytest.raises(ValueError, match=r"manifest order|sample_id order"):
        evaluate_embeddings(
            **{
                **kwargs,
                "sample_ids": list(reversed(pool["sample_ids"])),
                "episode_ids": list(reversed(pool["episode_ids"])),
                "conditions": list(reversed(pool["conditions"])),
                "semantic_embeddings": pool["semantic_embeddings"][::-1],
                "action_embeddings": pool["action_embeddings"][::-1],
            }
        )
    with pytest.raises(ValueError, match="pair"):
        evaluate_embeddings(
            **{
                **kwargs,
                "shuffled_pairs": list(reversed(pool["shuffled_pairs"])),
            }
        )
    with pytest.raises(ValueError, match="bootstrap"):
        evaluate_embeddings(
            **{
                **kwargs,
                "bootstrap_indices": pool["bootstrap_indices"][::-1],
            }
        )
    bad_conditions = [{"peg_shape": "circular", "approach_direction": "-x"} for _ in pool["conditions"]]
    with pytest.raises(ValueError, match="condition"):
        evaluate_embeddings(**{**kwargs, "conditions": bad_conditions})
    with pytest.raises(ValueError, match=r"pool|1118|count"):
        evaluate_embeddings(
            pool["semantic_embeddings"][:10],
            pool["action_embeddings"][:10],
            sample_ids=pool["sample_ids"][:10],
            episode_ids=pool["episode_ids"][:10],
            conditions=pool["conditions"][:10],
            shuffled_pairs=pool["shuffled_pairs"][:10],
            nearby_pairs=pool["nearby_pairs"][:10],
            bootstrap_indices=pool["bootstrap_indices"],
            strict_protocol=True,
        )


def test_readable_report_includes_retrieval_margins_and_conditions():
    pool = _canonical_pool_inputs()
    result = evaluate_embeddings(
        pool["semantic_embeddings"],
        pool["action_embeddings"],
        sample_ids=pool["sample_ids"],
        episode_ids=pool["episode_ids"],
        conditions=pool["conditions"],
        shuffled_pairs=pool["shuffled_pairs"],
        nearby_pairs=pool["nearby_pairs"],
        bootstrap_indices=pool["bootstrap_indices"],
        strict_protocol=True,
        expected_sample_ids=pool["sample_ids"],
        expected_shuffled_pairs=pool["shuffled_pairs"],
        expected_nearby_pairs=pool["nearby_pairs"],
        expected_bootstrap_indices=pool["bootstrap_indices"],
    )
    text = render_readable_report(result, timing={"seconds": 0.5})
    assert "semantic_to_action_top1" in text
    assert "aligned_minus_shuffled" in text
    assert "circular:-x" in text
    assert "0.5" in text


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


def test_canonical_json_bytes_are_stable_helpers():
    payload = {"schema": "x", "value": 1}
    assert canonical_bytes({**payload, "content_hash": content_hash(payload)})
