from __future__ import annotations

import json
from pathlib import Path

import pytest

from lap.verifiers.cover.batch_probe import build_probe_receipt
from lap.verifiers.cover.batch_probe import compare_memory_to_baseline
from lap.verifiers.cover.bridge_audit import build_production_target_inventory
from lap.verifiers.cover.model import VerifierConfig
from lap.verifiers.cover.model import fingerprint_siglip2_assets
from lap.verifiers.cover.protocol import validate_batch_probe_receipt
from lap.verifiers.cover.w3_contracts import BACKBONE_ID
from lap.verifiers.cover.w3_contracts import BACKBONE_REVISION
from lap.verifiers.cover.w3_contracts import content_hash


def _snapshot():
    return [
        {
            "index": 0,
            "name": "NVIDIA RTX 6000 Ada Generation",
            "driver_version": "570.124.06",
            "uuid": "GPU-test-0",
            "physical_total_memory_mib": 49140.0,
            "allocatable_total_memory_mib": 48502.69,
            "free_memory_mib": 47800.0,
        },
        {
            "index": 1,
            "name": "NVIDIA RTX 6000 Ada Generation",
            "driver_version": "570.124.06",
            "uuid": "GPU-test-1",
            "physical_total_memory_mib": 49140.0,
            "allocatable_total_memory_mib": 48510.94,
            "free_memory_mib": 48000.0,
        },
    ]


def _rank_success(rank: int, *, batch_size: int = 64, gradient_fingerprint: str = "a" * 64):
    return {
        "rank": rank,
        "status": "passed",
        "forward": "passed",
        "backward": "passed",
        "optimizer_step": "passed",
        "loss": 1.25,
        "gradient_norm": 0.5,
        "gradients_finite": True,
        "parameters_finite": True,
        "trainable_state_changed": True,
        "frozen_state_unchanged": True,
        "gradient_fingerprint": gradient_fingerprint,
        "local_negative_pool_size": batch_size,
        "embedding_all_gather": False,
        "frozen_backbone_dtype": "bfloat16",
        "trainable_dtype": "float32",
        "logits_dtype": "float32",
    }


def _asset_evidence(tmp_path: Path):
    snapshot = tmp_path / BACKBONE_REVISION
    snapshot.mkdir()
    (snapshot / "open_clip_config.json").write_text(
        json.dumps({"preprocess_cfg": {"mean": [0.5]}, "model_cfg": {"text_cfg": {"context_length": 64}}}),
        encoding="utf-8",
    )
    for filename in ("special_tokens_map.json", "tokenizer.json", "tokenizer_config.json"):
        (snapshot / filename).write_text(json.dumps({"source": filename}), encoding="utf-8")
    return {
        "siglip2_snapshot": {
            "backbone_id": BACKBONE_ID,
            "revision": BACKBONE_REVISION,
            "local_snapshot": str(snapshot),
        },
        **fingerprint_siglip2_assets(snapshot),
    }


def test_memory_drift_distinguishes_stable_and_transient_fields():
    report = compare_memory_to_baseline(_snapshot())

    assert report["baseline_date"] == "2026-07-23"
    assert report["drift_mib"][0]["physical_total_memory"]["delta"] == 0.0
    assert report["drift_mib"][0]["allocatable_total_memory"]["delta"] == 0.0
    assert report["drift_mib"][0]["free_memory"]["delta"] == -82.56
    assert "transient" in report["explanation"]


def test_probe_receipt_records_canonical_two_rank_success(tmp_path):
    configuration = VerifierConfig().to_dict()
    asset_evidence = _asset_evidence(tmp_path)
    receipt = build_probe_receipt(
        snapshot=_snapshot(),
        attempts=[
            {
                "per_rank_batch_size": 64,
                "status": "passed",
                "ranks": [_rank_success(0), _rank_success(1)],
            }
        ],
        selected_batch_size=64,
        memory_drift=compare_memory_to_baseline(_snapshot()),
        evidence={
            **asset_evidence,
            "configuration": configuration,
            "configuration_hash": content_hash(configuration),
            "audit_manifest_sha256": "1" * 64,
            "target_inventory_fingerprint": build_production_target_inventory().fingerprint,
        },
    )

    validate_batch_probe_receipt(receipt)
    assert receipt["selected_per_rank_batch_size"] == 64
    assert receipt["successful_two_rank_optimizer_step"] == {"batch_size": 64, "ranks": [0, 1]}
    assert receipt["model"]["backbone_revision"]
    assert receipt["model"]["preprocessing_fingerprint"] == asset_evidence["preprocessing_fingerprint"]
    assert receipt["model"]["tokenizer_fingerprint"] == asset_evidence["tokenizer_fingerprint"]
    assert receipt["model"]["negative_pool"] == "rank_local"
    assert receipt["memory_drift"]["baseline_date"] == "2026-07-23"


def test_probe_receipt_rejects_duplicate_or_incomplete_rank_success():
    attempts = [
        {
            "per_rank_batch_size": 64,
            "status": "passed",
            "ranks": [_rank_success(0), _rank_success(0)],
        }
    ]

    with pytest.raises(ValueError, match="exactly ranks 0 and 1"):
        build_probe_receipt(
            snapshot=_snapshot(),
            attempts=attempts,
            selected_batch_size=64,
            memory_drift=compare_memory_to_baseline(_snapshot()),
        )


def test_probe_receipt_rejects_rank_without_full_optimizer_evidence():
    incomplete = _rank_success(1)
    incomplete.pop("optimizer_step")

    with pytest.raises(ValueError, match="full finite optimizer step"):
        build_probe_receipt(
            snapshot=_snapshot(),
            attempts=[
                {
                    "per_rank_batch_size": 64,
                    "status": "passed",
                    "ranks": [_rank_success(0), incomplete],
                }
            ],
            selected_batch_size=64,
            memory_drift=compare_memory_to_baseline(_snapshot()),
        )


def test_receipt_validation_rejects_rehashed_invalid_rank_records(tmp_path):
    receipt = build_probe_receipt(
        snapshot=_snapshot(),
        attempts=[
            {
                "per_rank_batch_size": 64,
                "status": "passed",
                "ranks": [_rank_success(0), _rank_success(1)],
            }
        ],
        selected_batch_size=64,
        memory_drift=compare_memory_to_baseline(_snapshot()),
        evidence={
            **_asset_evidence(tmp_path),
            "configuration": VerifierConfig().to_dict(),
            "configuration_hash": content_hash(VerifierConfig().to_dict()),
            "audit_manifest_sha256": "1" * 64,
            "target_inventory_fingerprint": build_production_target_inventory().fingerprint,
        },
    )
    receipt["attempts"][0]["ranks"][1]["rank"] = 0
    receipt.pop("content_hash")
    receipt["content_hash"] = content_hash(receipt)

    with pytest.raises(ValueError, match="exactly ranks 0 and 1"):
        validate_batch_probe_receipt(receipt)


def test_canonical_receipt_requires_per_rank_batch_64(tmp_path):
    receipt = build_probe_receipt(
        snapshot=_snapshot(),
        attempts=[
            {
                "per_rank_batch_size": 32,
                "status": "passed",
                "ranks": [_rank_success(0, batch_size=32), _rank_success(1, batch_size=32)],
            }
        ],
        selected_batch_size=32,
        memory_drift=compare_memory_to_baseline(_snapshot()),
        evidence={
            **_asset_evidence(tmp_path),
            "configuration": VerifierConfig().to_dict(),
            "configuration_hash": content_hash(VerifierConfig().to_dict()),
            "audit_manifest_sha256": "1" * 64,
            "target_inventory_fingerprint": build_production_target_inventory().fingerprint,
        },
    )

    with pytest.raises(ValueError, match="per-rank batch size 64"):
        validate_batch_probe_receipt(receipt)
