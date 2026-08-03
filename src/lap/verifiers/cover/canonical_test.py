"""W3-09 public-seam tests: canonical accept composition and publication gates."""

# ruff: noqa: SLF001
from __future__ import annotations

from pathlib import Path

import pytest

from lap.verifiers.cover import canonical
from lap.verifiers.cover import pipeline


def test_resolve_initialization_manifest_reaudits_base_only_inventory(monkeypatch):
    """Base-only control must re-audit the same Bridge against wrist-omitted inventory."""

    production = {
        "target": {"fingerprint": "two-view-fp"},
        "artifact": {"sha256": "a" * 64, "size": 12},
        "manifest_sha256": "m" * 64,
    }
    model = type(
        "M",
        (),
        {"config": type("C", (), {"use_wrist": False})()},
    )()
    inventory = type("I", (), {"fingerprint": "base-only-fp"})()
    monkeypatch.setattr(canonical, "_target_inventory_from_model", lambda _model: inventory)

    captured: dict = {}

    def _audit(path, *, target_inventory, expected_sha256, expected_size):
        captured["path"] = path
        captured["fingerprint"] = target_inventory.fingerprint
        captured["sha"] = expected_sha256
        captured["size"] = expected_size
        return {"schema": "re-audited", "target": {"fingerprint": "base-only-fp"}}

    monkeypatch.setattr(canonical, "audit_bridge_checkpoint", _audit)
    resolved = canonical.resolve_initialization_manifest_for_model(
        model,
        bridge_artifact=Path("/tmp/bridge.pt"),
        production_manifest=production,
    )
    assert resolved["schema"] == "re-audited"
    assert captured["fingerprint"] == "base-only-fp"
    assert captured["sha"] == "a" * 64


def test_resolve_initialization_manifest_keeps_locked_two_view_audit(monkeypatch):
    production = {
        "target": {"fingerprint": "two-view-fp"},
        "artifact": {"sha256": "a" * 64, "size": 12},
        "manifest_sha256": "m" * 64,
    }
    model = type("M", (), {"config": type("C", (), {"use_wrist": True})()})()
    inventory = type("I", (), {"fingerprint": "two-view-fp"})()
    monkeypatch.setattr(canonical, "_target_inventory_from_model", lambda _model: inventory)
    resolved = canonical.resolve_initialization_manifest_for_model(
        model,
        bridge_artifact=Path("/tmp/bridge.pt"),
        production_manifest=production,
    )
    assert resolved is not production
    assert resolved["manifest_sha256"] == "m" * 64


def test_preserve_failure_evidence_copies_worker_errors(tmp_path):
    staging = tmp_path / ".canonical_acceptance_v1.staging"
    (staging / "base_only").mkdir(parents=True)
    err = staging / "base_only" / "train_worker_error_rank1.txt"
    err.write_text("ValueError: audit mismatch\n", encoding="utf-8")
    evidence = tmp_path / "evidence"
    destination = pipeline._preserve_failure_evidence(staging, evidence)
    assert destination is not None
    copied = list(destination.rglob("train_worker_error_rank1.txt"))
    assert len(copied) == 1
    assert copied[0].read_text(encoding="utf-8") == "ValueError: audit mismatch\n"
    assert (destination / "failure_manifest.json").is_file()


def test_w5_handoff_deployment_root_is_package_local_final_path(tmp_path):
    """R3: handoff must name the published package deployment, never staging."""

    output_root = tmp_path / "canonical_acceptance_v1"
    output_root.mkdir()
    (output_root / "deployment").mkdir()
    report = {"schema": "osx_cover_w3_acceptance_v1", "content_hash": "a" * 64}
    payload = canonical.write_w5_handoff_payload(
        output_root=output_root,
        acceptance_report=report,
        deployment_root=output_root / "deployment",
    )
    root = Path(payload["deployment_root"])
    assert "staging" not in str(root)
    assert root == (output_root / "deployment").resolve()
    assert root.is_dir()
    loaded = __import__("json").loads((output_root / "w5_handoff.json").read_text(encoding="utf-8"))
    assert Path(loaded["deployment_root"]) == root


def test_preaccept_receipt_requires_both_rank_resume_equivalence():
    """R4: exact_resume=passed alone is insufficient without both-rank proof."""

    with pytest.raises(ValueError, match="uninterrupted_versus_resumed"):
        canonical.require_preaccept_exact_resume_receipt(
            {
                "exact_resume": "passed",
                "two_rank_step": "passed",
                "world_size": 2,
            }
        )
    ok = canonical.require_preaccept_exact_resume_receipt(
        {
            "exact_resume": "passed",
            "two_rank_step": "passed",
            "world_size": 2,
            "uninterrupted_versus_resumed": "passed",
            "ranks_restored": [0, 1],
            "resume_equivalence": {
                "rank0_loss_match": True,
                "rank1_loss_match": True,
                "gradient_fingerprint_match": True,
            },
        }
    )
    assert ok["uninterrupted_versus_resumed"] == "passed"


def test_finalize_published_package_rejects_missing_indexed_payloads(tmp_path):
    """R2: accepted marker cannot seal a package missing indexed .pt payloads."""

    root = tmp_path / "deployment"
    root.mkdir()
    (root / "metadata.json").write_text('{"schema":"x"}\n', encoding="utf-8")
    (root / "ACCEPTED_W3_DEPLOYMENT").write_text("recorded_data_offline_only\n", encoding="utf-8")
    (root / "content_index.json").write_text(
        '{"schema":"osx_cover_w3_content_index_v1","files":{"model.pt":"'
        + "a" * 64
        + '","metadata.json":"'
        + "b" * 64
        + '","ACCEPTED_W3_DEPLOYMENT":"'
        + "c" * 64
        + '"},"content_hash":"'
        + "d" * 64
        + '"}\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match=r"does not match bundle files|missing file on disk|model\.pt"):
        canonical.require_published_deployment_payloads(root)


def _protocol_evidence() -> dict:
    return {
        "protocol": {
            "seed": 42,
            "content_hash": "c" * 64,
            "protocol_version": "w3-g02-accepted-run-v1",
            "sampler": {"seed": 42, "world_size": 2},
            "batch": {"per_rank": 64},
            "optimization": {
                "epochs": 50,
                "learning_rate": 1e-6,
                "warmup_epochs": 10,
                "gradient_clip_norm": 1.0,
            },
            "evaluation": {"seed": 4203, "bootstrap_replicates": 1000},
            "identities": {"train_manifest_hash": "t" * 64, "phrase_manifest_hash": "h" * 64},
            "training": {"seed": 42},
        },
        "shuffled_pairs": {"pairs": [{"a": "1", "b": "2"}]},
        "nearby_pairs": {"pairs": [{"a": "1", "b": "3"}]},
        "bootstrap_indices": [[0]],
    }


def _gateway():
    sample = type("S", (), {"sample_id": "s0", "episode_id": "e0"})()
    return type(
        "G",
        (),
        {
            "train": [sample],
            "validation": [sample],
            "validation_receipt": {"normalization_artifact_hash": "n" * 64},
            "train_manifest_hash": "t" * 64,
        },
    )()


def _fake_model():
    config = type(
        "Cfg",
        (),
        {
            "to_dict": lambda self: {"use_wrist": True, "embedding_width": 512},
            "use_wrist": True,
            "embedding_width": 512,
            "fusion_input_width": 1536,
        },
    )()
    backbone = type("B", (), {"backbone_revision": "siglip2", "preprocess": lambda image: image})()
    return type("M", (), {"config": config, "backbone": backbone})()


def test_canonical_accept_requires_preaccept_gates_before_full_train(tmp_path, monkeypatch):
    """Public seam must run host + preaccept before any full two-view train."""

    calls: list[str] = []

    monkeypatch.setattr(
        pipeline,
        "preflight_w3",
        lambda **_: calls.append("preflight") or {"schema": "preflight", "content_hash": "p" * 64},
    )
    monkeypatch.setattr(
        pipeline,
        "require_canonical_training_host",
        lambda: calls.append("host") or {"gpus": ["NVIDIA RTX 6000 Ada Generation"] * 2},
    )
    monkeypatch.setattr(
        pipeline,
        "run_preaccept_two_rank_step_and_resume",
        lambda **_: (
            calls.append("preaccept")
            or {
                "two_rank_step": "passed",
                "exact_resume": "passed",
                "uninterrupted_versus_resumed": "passed",
                "ranks_restored": [0, 1],
                "resume_equivalence": {
                    "rank0_loss_match": True,
                    "rank1_loss_match": True,
                    "gradient_fingerprint_match": True,
                },
            }
        ),
    )
    monkeypatch.setattr(pipeline, "validate_protocol_directory", lambda *a, **k: _protocol_evidence())
    monkeypatch.setattr(pipeline, "W2DatasetGateway", lambda *a, **k: _gateway())
    monkeypatch.setattr(
        pipeline,
        "protocol_from_validated",
        lambda payload: pipeline.RunProtocol(per_rank_batch_size=64),
    )

    def _forbid_train(**kwargs):
        calls.append("train")
        raise AssertionError("full train must not start before preaccept returns in this test")

    monkeypatch.setattr(pipeline, "train_two_rank_ddp", _forbid_train)

    with pytest.raises(AssertionError, match="full train must not start"):
        pipeline.run_canonical_acceptance(
            w2_root=tmp_path / "w2",
            bridge_artifact=tmp_path / "bridge.pt",
            audit_manifest=tmp_path / "audit.json",
            protocol_dir=tmp_path / "protocol",
            output_root=tmp_path / "out",
        )

    assert calls[:3] == ["preflight", "host", "preaccept"]
    assert "train" in calls


def test_failed_usefulness_publishes_no_accepted_marker(tmp_path, monkeypatch):
    output = tmp_path / "out"

    monkeypatch.setattr(pipeline, "preflight_w3", lambda **_: {"ok": True})
    monkeypatch.setattr(
        pipeline, "require_canonical_training_host", lambda: {"gpus": ["NVIDIA RTX 6000 Ada Generation"] * 2}
    )
    monkeypatch.setattr(
        pipeline,
        "run_preaccept_two_rank_step_and_resume",
        lambda **_: {
            "two_rank_step": "passed",
            "exact_resume": "passed",
            "uninterrupted_versus_resumed": "passed",
            "ranks_restored": [0, 1],
            "resume_equivalence": {
                "rank0_loss_match": True,
                "rank1_loss_match": True,
                "gradient_fingerprint_match": True,
            },
        },
    )
    monkeypatch.setattr(pipeline, "validate_protocol_directory", lambda *a, **k: _protocol_evidence())
    monkeypatch.setattr(pipeline, "W2DatasetGateway", lambda *a, **k: _gateway())
    monkeypatch.setattr(
        pipeline,
        "protocol_from_validated",
        lambda payload: pipeline.RunProtocol(per_rank_batch_size=64),
    )

    def _train(**kwargs):
        root = Path(kwargs["checkpoint_dir"])
        root.mkdir(parents=True, exist_ok=True)
        (root / "best.pt").write_text("best", encoding="utf-8")
        (root / "latest.pt").write_text("latest", encoding="utf-8")
        return {
            "history": [{"epoch": 0, "loss": 1.0}],
            "progress": {"epoch": 1, "global_step": 1, "best_metric": 1.0, "world_size": 2},
            "world_size": 2,
            "gradient_synchronization": "two_rank_ddp",
        }

    monkeypatch.setattr(pipeline, "train_two_rank_ddp", _train)
    monkeypatch.setattr(
        pipeline,
        "run_repeated_fixed_best_evaluation",
        lambda **_: (_ for _ in ()).throw(ValueError("W3 retrieval lower confidence bounds do not exceed chance")),
    )
    monkeypatch.setattr(
        pipeline,
        "publish_deployment_bundle",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not publish on failed usefulness")),
    )

    with pytest.raises(ValueError, match="retrieval lower confidence bounds"):
        pipeline.run_canonical_acceptance(
            w2_root=tmp_path / "w2",
            bridge_artifact=tmp_path / "bridge.pt",
            audit_manifest=tmp_path / "audit.json",
            protocol_dir=tmp_path / "protocol",
            output_root=output,
        )

    assert not output.exists()
    assert not list(tmp_path.glob("**/ACCEPTED_W3_DEPLOYMENT"))


def test_successful_canonical_accept_sets_canonical_scope_and_offline_authority(tmp_path, monkeypatch):
    output = tmp_path / "accepted"
    published: dict[str, object] = {}
    ablation_seen: dict[str, object] = {}
    audit_path = tmp_path / "audit.json"
    audit_path.write_text(
        '{"manifest_sha256":"'
        + "a" * 64
        + '","artifact":{"sha256":"'
        + "b" * 64
        + '"},"target":{"fingerprint":"'
        + "c" * 64
        + '"}}',
        encoding="utf-8",
    )

    monkeypatch.setattr(pipeline, "preflight_w3", lambda **_: {"schema": "preflight", "content_hash": "p" * 64})
    monkeypatch.setattr(
        pipeline, "require_canonical_training_host", lambda: {"gpus": ["NVIDIA RTX 6000 Ada Generation"] * 2}
    )
    monkeypatch.setattr(
        pipeline,
        "run_preaccept_two_rank_step_and_resume",
        lambda **_: {
            "two_rank_step": "passed",
            "exact_resume": "passed",
            "uninterrupted_versus_resumed": "passed",
            "ranks_restored": [0, 1],
            "resume_equivalence": {
                "rank0_loss_match": True,
                "rank1_loss_match": True,
                "gradient_fingerprint_match": True,
            },
        },
    )
    monkeypatch.setattr(pipeline, "validate_protocol_directory", lambda *a, **k: _protocol_evidence())
    monkeypatch.setattr(pipeline, "W2DatasetGateway", lambda *a, **k: _gateway())
    monkeypatch.setattr(
        pipeline,
        "protocol_from_validated",
        lambda payload: pipeline.RunProtocol(per_rank_batch_size=64),
    )

    def _train(**kwargs):
        root = Path(kwargs["checkpoint_dir"])
        root.mkdir(parents=True, exist_ok=True)
        (root / "best.pt").write_text("best", encoding="utf-8")
        (root / "latest.pt").write_text("latest", encoding="utf-8")
        return {
            "history": [{"epoch": 0, "loss": 0.1, "sampler_diagnostics": {"ok": True}}],
            "progress": {"epoch": 1, "global_step": 10, "best_metric": 0.1, "world_size": 2},
            "world_size": 2,
            "gradient_synchronization": "two_rank_ddp",
            "model_config": {"use_wrist": True, "embedding_width": 512},
            "backbone_revision": "siglip2",
        }

    report = {
        "retrieval": {
            "semantic_to_action_top1_ci95": [0.1, 0.2],
            "action_to_semantic_top1_ci95": [0.1, 0.2],
        },
        "margins": {
            "aligned_minus_shuffled": {"ci95": [0.01, 0.02]},
            "aligned_minus_nearby": {"ci95": [0.01, 0.02]},
        },
        "pool": {"top1_chance": 1 / 1118},
        "conditions": {"all_eight": True},
        "content_hash": "e" * 64,
    }
    metrics = {
        **report,
        "row_metrics": {
            "sample_ids": ["s0"],
            "episode_ids": ["e0"],
            "semantic_to_action_hit": [1],
            "action_to_semantic_hit": [1],
        },
        "pair_metrics": {
            "aligned_minus_shuffled": [0.1],
            "aligned_minus_shuffled_episode_ids": ["e0"],
            "aligned_minus_nearby": [0.1],
            "aligned_minus_nearby_episode_ids": ["e0"],
        },
    }

    monkeypatch.setattr(pipeline, "train_two_rank_ddp", _train)
    monkeypatch.setattr(
        pipeline,
        "run_repeated_fixed_best_evaluation",
        lambda **_: {
            "report": report,
            "metrics": metrics,
            "evaluation_bytes": b'{"ok":true}\n',
            "repeated_byte_identical": True,
        },
    )
    monkeypatch.setattr(pipeline, "_load_two_view_model_from_best", lambda *a, **k: _fake_model())
    monkeypatch.setattr(
        pipeline,
        "run_matched_base_only_two_rank_ddp",
        lambda **_: {
            "model": _fake_model(),
            "report": {**metrics, "variant": "base_only", "deployable": False, "content_hash": "b" * 64},
            "identity": {"seed": 42, "sample_ids": ["s0"]},
            "config_delta": {
                "use_wrist": {"two_view": True, "base_only": False},
                "fusion_input_width": {"two_view": 1536, "base_only": 1024},
            },
            "model_config": {"use_wrist": False, "embedding_width": 512},
            "history": [{"epoch": 0, "loss": 0.2}],
            "progress": {"epoch": 1, "global_step": 10, "best_metric": 0.2, "world_size": 2},
            "checkpoint_dir": None,
        },
    )

    def _ablation(**kwargs):
        ablation_seen.update(kwargs)
        return {
            "schema": "osx_cover_w3_paired_ablation_report_v1",
            "deployable": False,
            "ablation": {"paired": True},
            "content_hash": "a" * 64,
        }

    monkeypatch.setattr(pipeline, "matched_base_only_run_to_paired_ablation_report", _ablation)

    def _publish(root, *, model, metadata, accepted_marker=True):
        published["root"] = Path(root)
        published["metadata"] = metadata
        published["accepted_marker"] = accepted_marker
        Path(root).mkdir(parents=True, exist_ok=True)
        (Path(root) / ("ACCEPTED_W3_DEPLOYMENT" if accepted_marker else "FIXTURE_ONLY_W3_BUNDLE")).write_text(
            "recorded_data_offline_only\n" if accepted_marker else "fixture\n",
            encoding="utf-8",
        )
        return {"root": str(root)}

    monkeypatch.setattr(pipeline, "publish_deployment_bundle", _publish)
    monkeypatch.setattr(
        pipeline,
        "load_deployment_bundle_for_acceptance",
        lambda root, **kwargs: {"scores_finite": True},
    )
    monkeypatch.setattr(
        pipeline,
        "require_published_deployment_payloads",
        lambda root: {"root": str(root), "payloads_present": True},
    )

    final = pipeline.run_canonical_acceptance(
        w2_root=tmp_path / "w2",
        bridge_artifact=tmp_path / "bridge.pt",
        audit_manifest=audit_path,
        protocol_dir=tmp_path / "protocol",
        output_root=output,
    )

    assert output.is_dir()
    assert (output / "deployment" / "ACCEPTED_W3_DEPLOYMENT").is_file()
    assert published["accepted_marker"] is True
    assert published["metadata"]["package_scope"] == "canonical_w3_09"
    assert final["authority"] == "recorded_data_offline_integration_only"
    assert final["schema"] == "osx_cover_w3_acceptance_v1"
    assert final["preaccept"]["exact_resume"] == "passed"
    assert final["evaluation"]["repeated_byte_identical"] is True
    assert ablation_seen["two_view_report"]["row_metrics"]["sample_ids"] == ["s0"]
    assert ablation_seen["two_view_identity"]["sample_ids"] == ["s0"]
    handoff = __import__("json").loads((output / "w5_handoff.json").read_text(encoding="utf-8"))
    assert "staging" not in handoff["deployment_root"]
    assert Path(handoff["deployment_root"]).resolve() == (output / "deployment").resolve()
