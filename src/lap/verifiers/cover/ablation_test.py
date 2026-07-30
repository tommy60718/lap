"""W3-08 public seam: matched base-only run → paired ablation report."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from lap.verifiers.cover.checkpoint import publish_deployment_bundle
from lap.verifiers.cover.evaluator import compare_ablation
from lap.verifiers.cover.model import TinyFrozenBackbone
from lap.verifiers.cover.model import VerifierConfig
from lap.verifiers.cover.model import VerifierModel
from lap.verifiers.cover.pipeline import bootstrap_indices_content_hash
from lap.verifiers.cover.pipeline import build_matched_ablation_identity
from lap.verifiers.cover.pipeline import matched_base_only_run_to_paired_ablation_report
from lap.verifiers.cover.pipeline import run_matched_base_only
from lap.verifiers.cover.protocol import RunProtocol
from lap.verifiers.cover.protocol import build_bootstrap_indices
from lap.verifiers.cover.training import base_only_config_delta
from lap.verifiers.cover.training import make_base_only_config
from lap.verifiers.cover.training import reset_approved_seed


def _bound_configs():
    two_view = VerifierConfig(
        backbone_width=32,
        embedding_width=16,
        visual_tokens=8,
        num_heads=4,
        pooling_layers=1,
        trajectory_layers=1,
        feed_forward_width=32,
        use_wrist=True,
    )
    base_model = VerifierModel(make_base_only_config(two_view), TinyFrozenBackbone(width=32, tokens=8))
    return two_view, base_model, base_only_config_delta(two_view, base_model.config)


def _identity(bootstrap, **overrides):
    payload = {
        "seed": 42,
        "sampler_seed": 42,
        "phrase_manifest_hash": "p" * 64,
        "optimizer": {
            "name": "AdamW",
            "learning_rate": 1e-6,
            "betas": [0.9, 0.999],
            "epsilon": 1e-8,
            "weight_decay": 0.01,
            "warmup_epochs": 10,
            "epochs": 50,
        },
        "checkpoint_selection": "lowest_validation_loss_earliest_epoch",
        "evaluation": {
            "shuffled_pairs_hash": "s" * 64,
            "nearby_pairs_hash": "n" * 64,
            "bootstrap_indices_hash": bootstrap_indices_content_hash(bootstrap),
        },
        "sample_ids": [f"sample-{i}" for i in range(8)],
        "protocol_content_hash": "c" * 64,
    }
    payload.update(overrides)
    return payload


def _reports(*, two_better: bool = True, include_margins: bool = True):
    sample_ids = [f"sample-{i}" for i in range(8)]
    episode_ids = [f"episode-{i}" for i in range(8)]
    bootstrap = build_bootstrap_indices(episode_ids, replicates=4)
    base_hits = [False] * 8
    two_hits = [True] * 8 if two_better else [False] * 8
    row_metrics = {
        "sample_ids": sample_ids,
        "episode_ids": episode_ids,
        "semantic_to_action_hit": base_hits,
        "action_to_semantic_hit": base_hits,
    }
    two_rows = {
        **row_metrics,
        "semantic_to_action_hit": two_hits,
        "action_to_semantic_hit": two_hits,
    }
    base_margins = [0.0] * 8
    two_margins = [1.0] * 8 if two_better else [0.0] * 8
    pair_metrics = {
        "aligned_minus_shuffled": base_margins,
        "aligned_minus_shuffled_episode_ids": episode_ids,
        "aligned_minus_nearby": base_margins,
        "aligned_minus_nearby_episode_ids": episode_ids,
    }
    two_pair = {
        "aligned_minus_shuffled": two_margins,
        "aligned_minus_shuffled_episode_ids": episode_ids,
        "aligned_minus_nearby": two_margins,
        "aligned_minus_nearby_episode_ids": episode_ids,
    }
    base = {
        "pool": {"count": 8},
        "retrieval": {"semantic_to_action_top1": 0.0, "action_to_semantic_top1": 0.0},
        "margins": {
            "aligned_minus_shuffled": {"mean": 0.0},
            "aligned_minus_nearby": {"mean": 0.0},
        },
        "row_metrics": row_metrics,
        "pair_metrics": pair_metrics if include_margins else {},
        "bootstrap_indices": bootstrap,
        "deployable": False,
        "variant": "base_only",
        "content_hash": "basehash",
    }
    two = {
        "pool": {"count": 8},
        "retrieval": {
            "semantic_to_action_top1": 1.0 if two_better else 0.0,
            "action_to_semantic_top1": 1.0 if two_better else 0.0,
        },
        "margins": {
            "aligned_minus_shuffled": {"mean": 1.0 if two_better else 0.0},
            "aligned_minus_nearby": {"mean": 1.0 if two_better else 0.0},
        },
        "row_metrics": two_rows,
        "pair_metrics": two_pair if include_margins else {},
        "bootstrap_indices": bootstrap,
        "content_hash": "twohash",
    }
    return two, base, bootstrap


def test_base_only_config_delta_is_only_wrist_and_fusion_width():
    two_view = VerifierConfig(
        backbone_width=1024,
        embedding_width=512,
        visual_tokens=8,
        num_heads=4,
        pooling_layers=1,
        trajectory_layers=1,
        feed_forward_width=32,
        use_wrist=True,
    )
    base = make_base_only_config(two_view)
    delta = base_only_config_delta(two_view, base)
    assert delta["use_wrist"] == {"two_view": True, "base_only": False}
    assert delta["fusion_input_width"] == {"two_view": 1536, "base_only": 1024}


def test_base_only_config_delta_rejects_unrelated_drift():
    two_view = VerifierConfig(embedding_width=16, use_wrist=True)
    drifted = VerifierConfig(embedding_width=32, use_wrist=False)
    with pytest.raises(ValueError, match="drifted"):
        base_only_config_delta(two_view, drifted)


def test_reset_approved_seed_restores_torch_rng():
    torch.manual_seed(0)
    _ = torch.rand(3)
    reset_approved_seed(42)
    first = torch.rand(4).clone()
    reset_approved_seed(42)
    second = torch.rand(4)
    torch.testing.assert_close(first, second)


def test_public_seam_rejects_mismatched_run_identity(tmp_path):
    two, base, bootstrap = _reports()
    two_view, base_model, honest = _bound_configs()
    two_identity = _identity(bootstrap)
    base_identity = _identity(bootstrap, sample_ids=[f"other-{i}" for i in range(8)])
    with pytest.raises(ValueError, match="identity mismatch"):
        matched_base_only_run_to_paired_ablation_report(
            two_view_report=two,
            two_view_identity=two_identity,
            two_view_config=two_view,
            base_only_result={
                "report": base,
                "identity": base_identity,
                "config_delta": honest,
                "model": base_model,
            },
            bootstrap_indices=bootstrap,
            output_root=tmp_path / "ablation",
        )
    assert not (tmp_path / "ablation" / "paired_ablation.json").exists()


def test_public_seam_rejects_mismatched_evaluation_artifact_hash(tmp_path):
    two, base, bootstrap = _reports()
    two_view, base_model, honest = _bound_configs()
    two_identity = _identity(bootstrap)
    base_identity = _identity(bootstrap, evaluation={**two_identity["evaluation"], "bootstrap_indices_hash": "x" * 64})
    with pytest.raises(ValueError, match="identity mismatch"):
        matched_base_only_run_to_paired_ablation_report(
            two_view_report=two,
            two_view_identity=two_identity,
            two_view_config=two_view,
            base_only_result={
                "report": base,
                "identity": base_identity,
                "config_delta": honest,
                "model": base_model,
            },
            bootstrap_indices=bootstrap,
            output_root=tmp_path / "ablation",
        )
    assert not (tmp_path / "ablation" / "paired_ablation.json").exists()


def test_w3_08_r1_public_seam_rejects_forged_config_delta_without_output(tmp_path):
    two, base, bootstrap = _reports()
    two_view, base_model, honest = _bound_configs()
    identity = _identity(bootstrap)
    forged = {
        **honest,
        "learning_rate": {"two_view": 1e-6, "base_only": 1e-5},
    }
    with pytest.raises(ValueError, match="exactly wrist omission and fresh fusion width"):
        matched_base_only_run_to_paired_ablation_report(
            two_view_report=two,
            two_view_identity=identity,
            two_view_config=two_view,
            base_only_result={
                "report": base,
                "identity": identity,
                "config_delta": forged,
                "model": base_model,
            },
            bootstrap_indices=bootstrap,
            output_root=tmp_path / "ablation-r1",
        )
    assert not (tmp_path / "ablation-r1").exists()
    assert not (tmp_path / "ablation-r1" / "paired_ablation.json").exists()


def test_w3_08_r1_public_seam_rejects_inconsistent_fusion_width_without_output(tmp_path):
    two, base, bootstrap = _reports()
    two_view, base_model, _honest = _bound_configs()
    identity = _identity(bootstrap)
    inconsistent = {
        "use_wrist": {"two_view": True, "base_only": False},
        "fusion_input_width": {"two_view": 48, "base_only": 48},
    }
    with pytest.raises(ValueError, match="fusion widths"):
        matched_base_only_run_to_paired_ablation_report(
            two_view_report=two,
            two_view_identity=identity,
            two_view_config=two_view,
            base_only_result={
                "report": base,
                "identity": identity,
                "config_delta": inconsistent,
                "model": base_model,
            },
            bootstrap_indices=bootstrap,
            output_root=tmp_path / "ablation-r1-fusion",
        )
    assert not (tmp_path / "ablation-r1-fusion").exists()


def test_w3_08_r1_rejects_structurally_valid_delta_that_contradicts_model(tmp_path):
    two, base, bootstrap = _reports()
    two_view, base_model, honest = _bound_configs()
    identity = _identity(bootstrap)
    assert base_model.config.fusion_input_width == 32
    assert honest["fusion_input_width"]["base_only"] == 32
    forged = {
        "use_wrist": {"two_view": True, "base_only": False},
        "fusion_input_width": {"two_view": 300, "base_only": 200},
    }
    with pytest.raises(ValueError, match="authoritative model configuration"):
        matched_base_only_run_to_paired_ablation_report(
            two_view_report=two,
            two_view_identity=identity,
            two_view_config=two_view,
            base_only_result={
                "report": base,
                "identity": identity,
                "config_delta": forged,
                "model": base_model,
            },
            bootstrap_indices=bootstrap,
            output_root=tmp_path / "ablation-r1-authority",
        )
    assert not (tmp_path / "ablation-r1-authority").exists()
    assert not (tmp_path / "ablation-r1-authority" / "paired_ablation.json").exists()


def test_w3_08_r1_emits_exact_authoritative_wrist_and_fusion_delta(tmp_path):
    two, base, bootstrap = _reports()
    two_view, base_model, honest = _bound_configs()
    identity = _identity(bootstrap)
    payload = matched_base_only_run_to_paired_ablation_report(
        two_view_report=two,
        two_view_identity=identity,
        two_view_config=two_view,
        base_only_result={
            "report": base,
            "identity": identity,
            "config_delta": honest,
            "model": base_model,
        },
        bootstrap_indices=bootstrap,
        output_root=tmp_path / "ablation-r1-green",
    )
    assert payload["config_delta"] == honest
    assert payload["config_delta"]["use_wrist"] == {"two_view": True, "base_only": False}
    assert payload["config_delta"]["fusion_input_width"] == {"two_view": 48, "base_only": 32}


def test_w3_08_r2_public_seam_rejects_substituted_bootstrap_without_output(tmp_path):
    two, base, bootstrap = _reports()
    two_view, base_model, honest = _bound_configs()
    identity = _identity(bootstrap)
    substituted = np.zeros_like(bootstrap)
    assert not np.array_equal(substituted, bootstrap)
    with pytest.raises(ValueError, match=r"bootstrap .*matched evaluation identity"):
        matched_base_only_run_to_paired_ablation_report(
            two_view_report=two,
            two_view_identity=identity,
            two_view_config=two_view,
            base_only_result={
                "report": base,
                "identity": identity,
                "config_delta": honest,
                "model": base_model,
            },
            bootstrap_indices=substituted,
            output_root=tmp_path / "ablation-r2",
        )
    assert not (tmp_path / "ablation-r2").exists()
    assert not (tmp_path / "ablation-r2" / "paired_ablation.json").exists()


def test_compare_ablation_reports_paired_retrieval_and_margin_intervals():
    two, base, bootstrap = _reports(two_better=True)
    result = compare_ablation(two, base, bootstrap_indices=bootstrap)
    assert result["two_view_minus_base_only"]["semantic_to_action_top1"] == 1.0
    assert result["two_view_minus_base_only"]["aligned_minus_shuffled"] == 1.0
    assert result["two_view_minus_base_only"]["aligned_minus_nearby"] == 1.0
    assert result["paired_ci95"]["semantic_to_action_top1"][0] > 0
    assert result["paired_ci95"]["aligned_minus_shuffled"][0] > 0
    assert result["paired_ci95"]["aligned_minus_nearby"][0] > 0
    assert result["wrist_benefit_established"] is True


def test_compare_ablation_reports_zero_containing_interval_honestly():
    two, base, bootstrap = _reports(two_better=False)
    result = compare_ablation(two, base, bootstrap_indices=bootstrap)
    assert result["two_view_minus_base_only"]["semantic_to_action_top1"] == 0.0
    assert result["paired_ci95"]["semantic_to_action_top1"][0] <= 0
    assert result["paired_ci95"]["aligned_minus_shuffled"][0] <= 0
    assert result["wrist_benefit_established"] is False


def test_public_seam_writes_nondeployable_paired_evidence_and_w5_rejects(tmp_path):
    two, base, bootstrap = _reports()
    two_view, base_model, honest = _bound_configs()
    identity = _identity(bootstrap)
    output_root = tmp_path / "ablation"
    payload = matched_base_only_run_to_paired_ablation_report(
        two_view_report=two,
        two_view_identity=identity,
        two_view_config=two_view,
        base_only_result={
            "report": base,
            "identity": identity,
            "config_delta": honest,
            "model": base_model,
        },
        bootstrap_indices=bootstrap,
        output_root=output_root,
    )
    assert payload["schema"] == "osx_cover_w3_paired_ablation_report_v1"
    assert payload["deployable"] is False
    assert payload["w5_eligible"] is False
    assert payload["accepted"] is False
    assert payload["config_delta"] == honest
    assert (output_root / "paired_ablation.json").is_file()
    assert (output_root / "EVIDENCE_ONLY_NONDEPLOYABLE").is_file()
    assert payload["ablation"]["wrist_benefit_established"] is True
    with pytest.raises(ValueError, match="nondeployable"):
        publish_deployment_bundle(
            output_root / "deployment_attempt",
            model=base_model,
            metadata={"deployable": False},
        )


def test_run_matched_base_only_resets_seed_before_construction(monkeypatch):
    calls: list[int] = []

    def fake_reset(seed: int) -> None:
        calls.append(int(seed))

    monkeypatch.setattr("lap.verifiers.cover.pipeline.reset_approved_seed", fake_reset)

    class BoomError(Exception):
        pass

    def boom_model(*args, **kwargs):
        raise BoomError("stop after seed reset")

    monkeypatch.setattr("lap.verifiers.cover.pipeline.VerifierModel", boom_model)
    with pytest.raises(BoomError):
        run_matched_base_only(
            two_view_model=type(
                "M",
                (),
                {
                    "config": VerifierConfig(embedding_width=16, use_wrist=True),
                    "backbone": object(),
                },
            )(),
            bridge_artifact=Path("/tmp/unused"),
            audit_manifest={},
            dataset=object(),
            protocol=RunProtocol(),
            shuffled_pairs=[],
            nearby_pairs=[],
            bootstrap_indices=np.zeros((1, 1), dtype=np.int64),
            device=torch.device("cpu"),
        )
    assert calls == [42]


def test_build_matched_ablation_identity_is_stable():
    protocol = RunProtocol()
    bootstrap = np.arange(8, dtype=np.int64).reshape(1, 8)
    hashes = {
        "shuffled_pairs_hash": "s" * 64,
        "nearby_pairs_hash": "n" * 64,
        "bootstrap_indices_hash": bootstrap_indices_content_hash(bootstrap),
    }
    first = build_matched_ablation_identity(
        protocol=protocol,
        sample_ids=["a", "b"],
        phrase_manifest_hash="p" * 64,
        protocol_content_hash="c" * 64,
        evaluation_artifact_hashes=hashes,
    )
    second = build_matched_ablation_identity(
        protocol=protocol,
        sample_ids=["a", "b"],
        phrase_manifest_hash="p" * 64,
        protocol_content_hash="c" * 64,
        evaluation_artifact_hashes=hashes,
    )
    assert first == second
    assert first["checkpoint_selection"] == "lowest_validation_loss_earliest_epoch"
    assert first["evaluation"]["bootstrap_indices_hash"] == bootstrap_indices_content_hash(bootstrap)
