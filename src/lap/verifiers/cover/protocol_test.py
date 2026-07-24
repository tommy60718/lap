from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from torch.utils.data.distributed import DistributedSampler

from lap.verifiers.cover.protocol import APPROVED_SHUFFLED_EXCLUSIONS
from lap.verifiers.cover.protocol import EVALUATION_SEED
from lap.verifiers.cover.protocol import PROBE_ORDER
from lap.verifiers.cover.protocol import TRAINING_SEED
from lap.verifiers.cover.protocol import WORLD_SIZE
from lap.verifiers.cover.protocol import build_bootstrap_indices
from lap.verifiers.cover.protocol import build_phrase_manifest
from lap.verifiers.cover.protocol import build_shuffled_pairs
from lap.verifiers.cover.protocol import materialize_protocol
from lap.verifiers.cover.protocol import probe_batch_sizes
from lap.verifiers.cover.protocol import sampler_indices
from lap.verifiers.cover.protocol import select_training_phrase
from lap.verifiers.cover.protocol import validate_protocol_directory

CANONICAL_EXPORT = Path(__file__).resolve().parents[6] / ".w2_canonical_export_v3"


def _canonical_validation_rows():
    return json.loads((CANONICAL_EXPORT / "validation_samples.json").read_text(encoding="utf-8"))


def test_phrase_manifest_has_exact_approved_bank():
    manifest = build_phrase_manifest()
    assert len(manifest["phrases"]) == 16
    assert manifest["phrases"][0] == "reach to the hole and insert the circular peg"
    assert manifest["phrases"][-1] == "place the square peg into the hole"
    assert len(set(manifest["phrases"])) == 16
    assert manifest["authorship"]["review_status"] == "approved"
    assert manifest["authorship"]["source"].endswith("#confirmed-offline-language-bank")
    assert manifest["content_hash"]


def test_language_selection_is_deterministic_and_shape_specific():
    first = select_training_phrase(seed=42, epoch=3, sample_id="sample", shape="circular")
    second = select_training_phrase(seed=42, epoch=3, sample_id="sample", shape="circular")
    assert first == second
    assert "circular" in first


def test_bootstrap_indices_use_fixed_shape_and_seed():
    indices = build_bootstrap_indices([f"episode_{i}" for i in range(8)], replicates=4)
    expected = np.array(
        [[7, 2, 0, 5, 5, 7, 7, 6], [2, 7, 3, 1, 5, 0, 5, 0], [7, 1, 1, 3, 6, 1, 1, 2], [5, 1, 4, 6, 0, 3, 5, 6]]
    )
    assert indices.shape == (4, 8)
    assert np.array_equal(indices, expected)


def test_sampler_indices_match_pytorch_distributed_sampler():
    sample_ids = [f"sample-{index}" for index in range(11)]
    actual = sampler_indices(sample_ids, epoch=3, rank=1)
    expected_sampler = DistributedSampler(
        list(range(len(sample_ids))), num_replicas=WORLD_SIZE, rank=1, shuffle=True, seed=TRAINING_SEED
    )
    expected_sampler.set_epoch(3)
    expected = [sample_ids[index] for index in list(expected_sampler)]
    assert actual["sample_ids"] == expected
    assert actual["seed"] == TRAINING_SEED
    assert actual["world_size"] == WORLD_SIZE
    assert actual["collision_diagnostics"]["adapts_batches"] is False


def test_shuffled_pairs_use_the_six_approved_exclusions():
    pairs, exclusions = build_shuffled_pairs(_canonical_validation_rows())
    assert len(pairs) == 1112
    assert exclusions == list(APPROVED_SHUFFLED_EXCLUSIONS)
    assert {pair["semantic_sample_id"] for pair in pairs} == {
        row["sample_id"] for row in _canonical_validation_rows()
    } - set(APPROVED_SHUFFLED_EXCLUSIONS)


def test_batch_probe_records_two_gpu_attempts_and_first_success():
    snapshots = [
        {
            "index": 0,
            "name": "NVIDIA RTX 6000 Ada Generation",
            "physical_total_memory_mib": 49140.0,
            "allocatable_total_memory_mib": 48502.69,
            "free_memory_mib": 47000.0,
        },
        {
            "index": 1,
            "name": "NVIDIA RTX 6000 Ada Generation",
            "physical_total_memory_mib": 49140.0,
            "allocatable_total_memory_mib": 48510.94,
            "free_memory_mib": 47100.0,
        },
    ]
    calls = []

    def step(batch_size, rank, _gpu):
        calls.append((batch_size, rank))
        if batch_size == 64:
            raise RuntimeError("CUDA out of memory")
        return {"forward_backward": "passed", "rank": rank, "batch_size": batch_size}

    receipt = probe_batch_sizes(
        step,
        snapshot_fn=lambda: snapshots,
        model_identity={"backbone": "hf-hub:timm/ViT-L-16-SigLIP2-384", "canonical_target": True},
    )
    assert receipt["attempted_batch_sizes"] == list(PROBE_ORDER[:2])
    assert receipt["selected_per_rank_batch_size"] == 32
    assert receipt["gpu_snapshot"] == snapshots
    assert receipt["successful_two_rank_forward_backward"]["ranks"] == [0, 1]
    assert calls == [(64, 0), (32, 0), (32, 1)]


def test_tiny_probe_cannot_be_materialized_as_canonical(tmp_path):
    receipt = probe_batch_sizes(
        lambda batch_size, rank, _gpu: {"forward_backward": "passed", "batch_size": batch_size, "rank": rank},
        snapshot_fn=lambda: [
            {
                "index": 0,
                "name": "fixture",
                "physical_total_memory_mib": 1,
                "allocatable_total_memory_mib": 1,
                "free_memory_mib": 1,
            },
            {
                "index": 1,
                "name": "fixture",
                "physical_total_memory_mib": 1,
                "allocatable_total_memory_mib": 1,
                "free_memory_mib": 1,
            },
        ],
        model_identity={"backbone": "TinyFrozenBackbone", "canonical_target": False},
    )
    with pytest.raises(ValueError, match=r"canonical|Tiny"):
        materialize_protocol(
            tmp_path,
            train_manifest_hash="a" * 64,
            validation=_canonical_validation_rows(),
            batch_probe_receipt=receipt,
        )


def test_materialized_protocol_is_hashable_and_rejects_drift(tmp_path):
    receipt = probe_batch_sizes(
        lambda batch_size, rank, _gpu: {"forward_backward": "passed", "batch_size": batch_size, "rank": rank},
        snapshot_fn=lambda: [
            {
                "index": 0,
                "name": "NVIDIA RTX 6000 Ada Generation",
                "physical_total_memory_mib": 49140,
                "allocatable_total_memory_mib": 48502,
                "free_memory_mib": 47000,
            },
            {
                "index": 1,
                "name": "NVIDIA RTX 6000 Ada Generation",
                "physical_total_memory_mib": 49140,
                "allocatable_total_memory_mib": 48510,
                "free_memory_mib": 47100,
            },
        ],
        model_identity={"backbone": "hf-hub:timm/ViT-L-16-SigLIP2-384", "canonical_target": True},
    )
    materialize_protocol(
        tmp_path,
        train_manifest_hash="a" * 64,
        validation=_canonical_validation_rows(),
        batch_probe_receipt=receipt,
    )
    validated = validate_protocol_directory(tmp_path)
    assert validated["protocol"]["training"]["seed"] == TRAINING_SEED
    assert validated["protocol"]["evaluation"]["seed"] == EVALUATION_SEED
    assert validated["protocol"]["sampler"]["world_size"] == WORLD_SIZE

    run_protocol = tmp_path / "run_protocol.json"
    payload = json.loads(run_protocol.read_text(encoding="utf-8"))
    payload["evaluation"]["seed"] = 7
    run_protocol.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match=r"hash|protocol|seed"):
        validate_protocol_directory(tmp_path)
