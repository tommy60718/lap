from __future__ import annotations

import json

from PIL import Image
import pytest
import torch

from lap.verifiers.cover.data import TwoViewDataset
from lap.verifiers.cover.data import W2DatasetGateway
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
