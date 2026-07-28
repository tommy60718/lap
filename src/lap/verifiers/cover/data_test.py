from __future__ import annotations

import json

from PIL import Image
import pytest
import torch

from lap.verifiers.cover.data import TwoViewDataset
from lap.verifiers.cover.data import W2DatasetGateway
from lap.verifiers.cover.data import build_epoch_collision_report
from lap.verifiers.cover.data import collate_two_view_batch
from lap.verifiers.cover.data import make_sampler


def _fixture_export(tmp_path):
    root = tmp_path / "w2"
    root.mkdir()
    Image.new("RGB", (8, 8), color=(10, 20, 30)).save(root / "image.png")
    rows = []
    for index, split in enumerate(("train", "validation")):
        rows.append(
            {
                "sample_id": f"circular_negx_demo_00:00000{index}",
                "episode_id": f"episode_{split}",
                "split": split,
                "instruction": "reach to the hole and insert the circular peg",
                "base_image": "image.png",
                "wrist_image": "image.png",
                "action_history": [[-5.0] * 7] * 6 + [[0.0] * 7] * 4,
                "traceability": {"peg_shape": "circular", "approach_direction": "-x"},
            }
        )
    (root / "train_samples.json").write_text(json.dumps([rows[0]]), encoding="utf-8")
    (root / "validation_samples.json").write_text(json.dumps([rows[1]]), encoding="utf-8")
    (root / "export_manifest.json").write_text(json.dumps({"export_schema": "fixture"}), encoding="utf-8")
    return root


def test_loader_decodes_both_views_and_uses_canonical_validation_language(tmp_path):
    gateway = W2DatasetGateway(_fixture_export(tmp_path), fixture=True)
    dataset = TwoViewDataset(gateway.validation, training=False)
    row = dataset[0]
    assert row["base_rgb"].shape == (3, 384, 384)
    assert row["wrist_rgb"].shape == (3, 384, 384)
    assert row["instruction"] == row["canonical_instruction"]
    assert row["action_history"].dtype == torch.float32


def test_public_collator_preserves_two_views_and_row_provenance(tmp_path):
    gateway = W2DatasetGateway(_fixture_export(tmp_path), fixture=True)
    train_row = TwoViewDataset(gateway.train, training=True)[0]
    validation_row = TwoViewDataset(gateway.validation, training=False)[0]

    batch = collate_two_view_batch([train_row, validation_row])

    assert batch["base_rgb"].shape == (2, 3, 384, 384)
    assert batch["wrist_rgb"].shape == (2, 3, 384, 384)
    assert batch["action_histories"].shape == (2, 10, 7)
    assert batch["sample_ids"] == [train_row["sample_id"], validation_row["sample_id"]]
    assert batch["episode_ids"] == [train_row["episode_id"], validation_row["episode_id"]]
    assert batch["instructions"] == [train_row["instruction"], validation_row["instruction"]]


def test_epoch_collision_report_uses_complete_off_diagonal_population():
    zero = torch.zeros(10, 7)
    one = torch.ones(10, 7)
    rows = [
        {"sample_id": "a", "episode_id": "e1", "instruction": "x", "history": zero},
        {"sample_id": "b", "episode_id": "e1", "instruction": "x", "history": zero},
        {"sample_id": "c", "episode_id": "e1", "instruction": "y", "history": one},
        {"sample_id": "c", "episode_id": "e2", "instruction": "x", "history": zero},
    ]

    report = build_epoch_collision_report(rows)

    assert report["sampler_rows"] == 4
    assert report["sampler_added_duplicate_rows"] == 1
    assert report["off_diagonal_population"] == {
        "pair_definition": "unordered_distinct_row_positions_i_lt_j",
        "denominator": 6,
    }
    assert report["repeated_instruction_pairs"] == {"count": 3, "denominator": 6, "rate": 0.5}
    assert report["exact_duplicate_history_pairs"] == {"count": 3, "denominator": 6, "rate": 0.5}
    assert report["repeated_language_history_pairs"] == {"count": 3, "denominator": 6, "rate": 0.5}
    assert report["same_episode_pairs"] == {"count": 3, "denominator": 6, "rate": 0.5}
    assert report["normalized_history_distances"]["all_pairs"]["denominator"] == 6
    assert sum(report["normalized_history_distances"]["all_pairs"]["histogram_counts"]) == 6
    assert report["normalized_history_distances"]["same_episode_pairs"]["denominator"] == 3
    assert sum(report["normalized_history_distances"]["same_episode_pairs"]["histogram_counts"]) == 3
    assert report["adapts_batches"] is False


def test_single_row_collision_report_has_explicit_zero_denominators():
    report = build_epoch_collision_report(
        [{"sample_id": "a", "episode_id": "e1", "instruction": "x", "history": torch.zeros(10, 7)}]
    )

    assert report["off_diagonal_population"]["denominator"] == 0
    assert report["same_episode_pairs"] == {"count": 0, "denominator": 0, "rate": 0.0}
    assert report["normalized_history_distances"]["all_pairs"]["denominator"] == 0


def test_empty_collision_report_has_explicit_zero_population():
    report = build_epoch_collision_report([])

    assert report["sampler_rows"] == 0
    assert report["unique_sampler_rows"] == 0
    assert report["sampler_added_duplicate_rows"] == 0
    assert report["off_diagonal_population"]["denominator"] == 0
    assert report["repeated_instruction_pairs"] == {"count": 0, "denominator": 0, "rate": 0.0}
    assert report["exact_duplicate_history_pairs"] == {"count": 0, "denominator": 0, "rate": 0.0}
    assert report["repeated_language_history_pairs"] == {"count": 0, "denominator": 0, "rate": 0.0}
    assert report["same_episode_pairs"] == {"count": 0, "denominator": 0, "rate": 0.0}
    assert report["normalized_history_distances"]["all_pairs"]["denominator"] == 0
    assert report["normalized_history_distances"]["all_pairs"]["histogram_counts"] == [0] * 12
    assert report["normalized_history_distances"]["same_episode_pairs"]["denominator"] == 0
    assert report["normalized_history_distances"]["same_episode_pairs"]["histogram_counts"] == [0] * 12
    assert report["adapts_batches"] is False


def test_sampler_is_seeded_distributed_sampler_with_epoch_order(tmp_path):
    gateway = W2DatasetGateway(_fixture_export(tmp_path), fixture=True)
    dataset = TwoViewDataset(gateway.train)
    sampler = make_sampler(dataset, seed=42)
    first = list(iter(sampler))
    sampler.set_epoch(1)
    second = list(iter(sampler))
    assert sampler.shuffle is True
    assert first != second or len(first) < 2


def test_loader_rejects_nonleading_padding(tmp_path):
    root = _fixture_export(tmp_path)
    rows = json.loads((root / "train_samples.json").read_text())
    rows[0]["action_history"][0] = [0.0] * 7
    rows[0]["action_history"][1] = [-5.0] * 7
    (root / "train_samples.json").write_text(json.dumps(rows), encoding="utf-8")
    with pytest.raises(ValueError, match="leading"):
        W2DatasetGateway(root, fixture=True)
