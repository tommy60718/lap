"""Tests for read-only W3 post-acceptance analysis (public CLI seam)."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pytest

from lap.verifiers.cover.analysis import analyze_accepted_w3_package
from lap.verifiers.cover.w3_contracts import canonical_bytes
from lap.verifiers.cover.w3_contracts import write_canonical_json


def _write_minimal_package(root: Path, *, mutate: dict | None = None) -> Path:
    """Build a tiny accepted-looking package with known numeric literals."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "deployment").mkdir(parents=True, exist_ok=True)
    (root / "deployment" / "ACCEPTED_W3_DEPLOYMENT").write_text(
        "recorded_data_offline_only\n", encoding="utf-8"
    )
    acceptance = {
        "schema": "osx_cover_w3_acceptance_v1",
        "authority": "recorded_data_offline_integration_only",
        "training": [
            {"epoch": 0, "loss": 4.0, "validation_loss": 3.5},
            {"epoch": 1, "loss": 3.0, "validation_loss": 2.5},
        ],
        "evaluation": {
            "pool": {"count": 100, "top1_chance": 0.01, "top5_chance": 0.05},
            "retrieval": {
                "action_to_semantic_top1": 0.2,
                "action_to_semantic_top1_ci95": [0.1, 0.3],
                "action_to_semantic_top5": 0.4,
                "semantic_to_action_top1": 0.25,
                "semantic_to_action_top1_ci95": [0.15, 0.35],
                "semantic_to_action_top5": 0.45,
            },
            "margins": {
                "aligned_minus_shuffled": {
                    "mean": 5.0,
                    "ci95": [4.0, 6.0],
                    "fraction_gt_zero": 0.9,
                },
                "aligned_minus_nearby": {
                    "mean": 1.0,
                    "ci95": [0.5, 1.5],
                    "fraction_gt_zero": 0.7,
                },
            },
            "conditions": {
                "circular:+x": {
                    "count": 10,
                    "action_to_semantic_top1": 0.1,
                    "semantic_to_action_top1": 0.2,
                    "aligned_minus_shuffled_mean": 4.0,
                    "aligned_minus_nearby_mean": 0.5,
                    "failures": [],
                },
                "circular:+y": {
                    "count": 10,
                    "action_to_semantic_top1": 0.15,
                    "semantic_to_action_top1": 0.25,
                    "aligned_minus_shuffled_mean": 4.5,
                    "aligned_minus_nearby_mean": 0.6,
                    "failures": [],
                },
                "circular:-x": {
                    "count": 10,
                    "action_to_semantic_top1": 0.12,
                    "semantic_to_action_top1": 0.22,
                    "aligned_minus_shuffled_mean": 4.2,
                    "aligned_minus_nearby_mean": 0.55,
                    "failures": [],
                },
                "circular:-y": {
                    "count": 10,
                    "action_to_semantic_top1": 0.11,
                    "semantic_to_action_top1": 0.21,
                    "aligned_minus_shuffled_mean": 4.1,
                    "aligned_minus_nearby_mean": 0.52,
                    "failures": [],
                },
                "square:+x": {
                    "count": 10,
                    "action_to_semantic_top1": 0.13,
                    "semantic_to_action_top1": 0.23,
                    "aligned_minus_shuffled_mean": 4.3,
                    "aligned_minus_nearby_mean": 0.58,
                    "failures": [],
                },
                "square:+y": {
                    "count": 10,
                    "action_to_semantic_top1": 0.14,
                    "semantic_to_action_top1": 0.24,
                    "aligned_minus_shuffled_mean": 4.4,
                    "aligned_minus_nearby_mean": 0.59,
                    "failures": [],
                },
                "square:-x": {
                    "count": 10,
                    "action_to_semantic_top1": 0.16,
                    "semantic_to_action_top1": 0.26,
                    "aligned_minus_shuffled_mean": 4.6,
                    "aligned_minus_nearby_mean": 0.61,
                    "failures": [],
                },
                "square:-y": {
                    "count": 10,
                    "action_to_semantic_top1": 0.17,
                    "semantic_to_action_top1": 0.27,
                    "aligned_minus_shuffled_mean": 4.7,
                    "aligned_minus_nearby_mean": 0.62,
                    "failures": [],
                },
            },
        },
    }
    if mutate:
        for key, value in mutate.items():
            if value is None and key in acceptance:
                del acceptance[key]
            else:
                acceptance[key] = value
    write_canonical_json(root / "acceptance.json", acceptance)

    ablation = {
        "schema": "osx_cover_w3_paired_ablation_package_v1",
        "accepted": False,
        "deployable": False,
        "ablation": {
            "schema": "osx_cover_w3_ablation_v1",
            "wrist_benefit_established": False,
            "two_view_minus_base_only": {
                "action_to_semantic_top1": 0.01,
                "semantic_to_action_top1": 0.02,
                "aligned_minus_shuffled": 0.5,
                "aligned_minus_nearby": 0.1,
            },
            "paired_ci95": {
                "action_to_semantic_top1": [-0.01, 0.03],
                "semantic_to_action_top1": [-0.01, 0.05],
                "aligned_minus_shuffled": [0.2, 0.8],
                "aligned_minus_nearby": [-0.1, 0.3],
            },
        },
    }
    abl_dir = root / "base_only" / "paired_ablation"
    abl_dir.mkdir(parents=True, exist_ok=True)
    write_canonical_json(abl_dir / "paired_ablation.json", ablation)
    # Decoy checkpoint must never be copied into analysis output.
    (root / "best.pt").write_bytes(b"FAKE_CHECKPOINT_BYTES")
    (root / "deployment" / "model.pt").write_bytes(b"FAKE_MODEL_BYTES")
    return root


def _fingerprint_tree(root: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if path.is_file():
            rel = str(path.relative_to(root))
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            out[rel] = f"{path.stat().st_size}:{digest}"
    return out


def test_analyze_rejects_missing_acceptance_marker(tmp_path: Path) -> None:
    package = _write_minimal_package(tmp_path / "pkg")
    (package / "deployment" / "ACCEPTED_W3_DEPLOYMENT").unlink()
    with pytest.raises(ValueError, match="ACCEPTED_W3_DEPLOYMENT"):
        analyze_accepted_w3_package(
            package_root=package,
            output_root=tmp_path / "out",
        )


def test_analyze_rejects_incomplete_package(tmp_path: Path) -> None:
    package = _write_minimal_package(tmp_path / "pkg")
    (package / "acceptance.json").unlink()
    with pytest.raises(ValueError, match="acceptance.json"):
        analyze_accepted_w3_package(
            package_root=package,
            output_root=tmp_path / "out",
        )


def test_analyze_rejects_missing_ablation(tmp_path: Path) -> None:
    package = _write_minimal_package(tmp_path / "pkg")
    shutil.rmtree(package / "base_only")
    with pytest.raises(ValueError, match="paired_ablation"):
        analyze_accepted_w3_package(
            package_root=package,
            output_root=tmp_path / "out",
        )


def test_analyze_package_is_read_only(tmp_path: Path) -> None:
    package = _write_minimal_package(tmp_path / "pkg")
    before = _fingerprint_tree(package)
    analyze_accepted_w3_package(
        package_root=package,
        output_root=tmp_path / "out",
    )
    assert _fingerprint_tree(package) == before


def test_analyze_does_not_copy_checkpoints(tmp_path: Path) -> None:
    package = _write_minimal_package(tmp_path / "pkg")
    out = tmp_path / "out"
    analyze_accepted_w3_package(package_root=package, output_root=out)
    copied = [p for p in out.rglob("*.pt")]
    assert copied == []


def test_analyze_numeric_values_come_from_accepted_artifacts(tmp_path: Path) -> None:
    package = _write_minimal_package(tmp_path / "pkg")
    result = analyze_accepted_w3_package(
        package_root=package,
        output_root=tmp_path / "out",
    )
    summary = json.loads((tmp_path / "out" / "analysis_summary.json").read_text(encoding="utf-8"))
    assert summary["retrieval"]["action_to_semantic_top1"] == 0.2
    assert summary["retrieval"]["semantic_to_action_top5"] == 0.45
    assert summary["chance_baselines"]["top1"] == 0.01
    assert summary["chance_baselines"]["top5"] == 0.05
    assert summary["margins"]["aligned_minus_shuffled"]["mean"] == 5.0
    assert summary["margins"]["aligned_minus_shuffled"]["ci95"] == [4.0, 6.0]
    assert summary["margins"]["aligned_minus_nearby"]["ci95"] == [0.5, 1.5]
    assert summary["training"]["epochs"] == [0, 1]
    assert summary["training"]["loss"] == [4.0, 3.0]
    assert summary["training"]["validation_loss"] == [3.5, 2.5]
    assert summary["authority"] == "recorded_data_offline_integration_only"
    assert summary["wrist_benefit_established"] is False
    assert summary["paired_ablation"]["two_view_minus_base_only"]["aligned_minus_shuffled"] == 0.5
    assert result["output_root"] == str((tmp_path / "out").resolve())


def test_analyze_repeated_generation_is_deterministic(tmp_path: Path) -> None:
    package = _write_minimal_package(tmp_path / "pkg")
    out_a = tmp_path / "out_a"
    out_b = tmp_path / "out_b"
    analyze_accepted_w3_package(package_root=package, output_root=out_a)
    analyze_accepted_w3_package(package_root=package, output_root=out_b)

    summary_a = (out_a / "analysis_summary.json").read_bytes()
    summary_b = (out_b / "analysis_summary.json").read_bytes()
    assert summary_a == summary_b

    report_a = (out_a / "report.md").read_bytes()
    report_b = (out_b / "report.md").read_bytes()
    assert report_a == report_b

    for name in (
        "fig_loss_curves.pdf",
        "fig_retrieval_vs_chance.pdf",
        "fig_margins_ci95.pdf",
        "fig_conditions_heatmap.pdf",
        "fig_paired_ablation.pdf",
    ):
        assert (out_a / "figures" / name).read_bytes() == (out_b / "figures" / name).read_bytes()


def test_analyze_writes_required_artifacts(tmp_path: Path) -> None:
    package = _write_minimal_package(tmp_path / "pkg")
    out = tmp_path / "out"
    analyze_accepted_w3_package(package_root=package, output_root=out)
    assert (out / "analysis_summary.json").is_file()
    assert (out / "report.md").is_file()
    for stem in (
        "fig_loss_curves",
        "fig_retrieval_vs_chance",
        "fig_margins_ci95",
        "fig_conditions_heatmap",
        "fig_paired_ablation",
    ):
        assert (out / "figures" / f"{stem}.png").is_file()
        assert (out / "figures" / f"{stem}.pdf").is_file()
    report = (out / "report.md").read_text(encoding="utf-8")
    assert "offline" in report.lower()
    assert "wrist" in report.lower()


def test_analyze_refuses_output_inside_package(tmp_path: Path) -> None:
    package = _write_minimal_package(tmp_path / "pkg")
    with pytest.raises(ValueError, match="inside package"):
        analyze_accepted_w3_package(
            package_root=package,
            output_root=package / "analysis_nested",
        )
