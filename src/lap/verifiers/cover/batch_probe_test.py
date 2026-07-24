from __future__ import annotations

from lap.verifiers.cover.batch_probe import build_probe_receipt
from lap.verifiers.cover.batch_probe import compare_memory_to_baseline
from lap.verifiers.cover.protocol import validate_batch_probe_receipt


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


def test_memory_drift_distinguishes_stable_and_transient_fields():
    report = compare_memory_to_baseline(_snapshot())

    assert report["baseline_date"] == "2026-07-23"
    assert report["drift_mib"][0]["physical_total_memory"]["delta"] == 0.0
    assert report["drift_mib"][0]["allocatable_total_memory"]["delta"] == 0.0
    assert report["drift_mib"][0]["free_memory"]["delta"] == -82.56
    assert "transient" in report["explanation"]


def test_probe_receipt_records_canonical_two_rank_success():
    receipt = build_probe_receipt(
        snapshot=_snapshot(),
        attempts=[
            {
                "per_rank_batch_size": 32,
                "status": "passed",
                "ranks": [
                    {"rank": 0, "forward_backward": "passed"},
                    {"rank": 1, "forward_backward": "passed"},
                ],
            }
        ],
        selected_batch_size=32,
        memory_drift=compare_memory_to_baseline(_snapshot()),
    )

    validate_batch_probe_receipt(receipt)
    assert receipt["selected_per_rank_batch_size"] == 32
    assert receipt["successful_two_rank_forward_backward"]["ranks"] == [0, 1]
    assert receipt["model"]["backbone_revision"]
    assert receipt["memory_drift"]["baseline_date"] == "2026-07-23"
