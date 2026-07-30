from __future__ import annotations

import json
from pathlib import Path

import pytest

from lap.verifiers.cover.pipeline import run_evaluate_mode
from lap.verifiers.cover.pipeline import run_fixture_end_to_end


def test_fixture_acceptance_publishes_complete_resumable_and_deployment_bundle(tmp_path):
    root = tmp_path / "w3-output"
    report = run_fixture_end_to_end(output_root=root)
    assert report["schema"] == "osx_cover_w3_fixture_acceptance_v1"
    assert (root / "latest.pt").is_file()
    assert (root / "deployment" / "FIXTURE_ONLY_W3_BUNDLE").is_file()
    payload = json.loads((root / "acceptance.json").read_text())
    assert payload["content_hash"]


def test_run_evaluate_mode_reloads_best_and_delegates_to_public_seam(tmp_path, monkeypatch):
    best = tmp_path / "best.pt"
    best.write_text("ok", encoding="utf-8")
    latest = tmp_path / "latest.pt"
    latest.write_text("ok", encoding="utf-8")
    out = tmp_path / "eval-out"
    observed: dict[str, object] = {}

    monkeypatch.setattr(
        "lap.verifiers.cover.pipeline.require_explicit_best_checkpoint",
        lambda path: (
            (_ for _ in ()).throw(ValueError("evaluation requires an explicitly named best.pt checkpoint"))
            if Path(path).name != "best.pt"
            else Path(path)
        ),
    )
    monkeypatch.setattr(
        "lap.verifiers.cover.pipeline.validate_protocol_directory",
        lambda *args, **kwargs: {
            "protocol": {"batch": {"per_rank": 16}, "content_hash": "p" * 64},
            "validation_semantics": {"rows": []},
            "shuffled_pairs": {"pairs": []},
            "nearby_pairs": {"pairs": []},
            "bootstrap_indices": None,
        },
    )
    monkeypatch.setattr(
        "lap.verifiers.cover.pipeline.W2DatasetGateway",
        lambda *args, **kwargs: type("Gateway", (), {"validation": [object()]})(),
    )
    monkeypatch.setattr(
        "lap.verifiers.cover.pipeline.torch.load",
        lambda *args, **kwargs: {
            "schema": "osx_cover_verifier_checkpoint_v1",
            "contract": {
                "model_config": {
                    "backbone_width": 32,
                    "embedding_width": 16,
                    "visual_tokens": 8,
                    "num_heads": 4,
                    "pooling_layers": 1,
                    "trajectory_layers": 1,
                    "feed_forward_width": 32,
                },
                "environment": {"fixture": True},
            },
        },
    )
    monkeypatch.setattr(
        "lap.verifiers.cover.pipeline._verifier_config_from_dict",
        lambda payload: type("Cfg", (), {"backbone_width": 32, "visual_tokens": 8})(),
    )
    fake_model = type(
        "Model",
        (),
        {
            "to": lambda self, device: self,
            "backbone": type("B", (), {})(),
        },
    )()
    monkeypatch.setattr("lap.verifiers.cover.pipeline.VerifierModel", lambda *args, **kwargs: fake_model)
    monkeypatch.setattr("lap.verifiers.cover.pipeline.TinyFrozenBackbone", lambda **kwargs: object())
    monkeypatch.setattr(
        "lap.verifiers.cover.pipeline.load_fixed_best_checkpoint_logit_scale",
        lambda checkpoint, model: 2.5,
    )
    monkeypatch.setattr(
        "lap.verifiers.cover.pipeline.TwoViewDataset",
        lambda *args, **kwargs: object(),
    )
    monkeypatch.setattr(
        "lap.verifiers.cover.pipeline.collect_embeddings",
        lambda *args, **kwargs: {
            "semantic": object(),
            "action": object(),
            "sample_ids": ["a"],
            "episode_ids": ["e"],
            "conditions": [{"peg_shape": "circular", "approach_direction": "-x"}],
        },
    )
    monkeypatch.setattr(
        "lap.verifiers.cover.pipeline.train_model",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("evaluate must not train")),
    )
    monkeypatch.setattr(
        "lap.verifiers.cover.pipeline.publish_deployment_bundle",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("evaluate must not package")),
    )

    def _seam(**kwargs):
        observed.update(kwargs)
        out.mkdir()
        (out / "evaluation.json").write_text('{"accepted":false}\n', encoding="utf-8")
        (out / "evaluation_report.txt").write_text("report\n", encoding="utf-8")
        return {"schema": "osx_cover_w3_fixed_checkpoint_evaluation_v1", "accepted": False, "mode": "evaluate"}

    monkeypatch.setattr("lap.verifiers.cover.pipeline.evaluate_explicit_best_checkpoint", _seam)

    with pytest.raises(ValueError, match=r"best\.pt|explicit best"):
        run_evaluate_mode(
            checkpoint=latest,
            output_root=tmp_path / "bad",
            w2_root=tmp_path / "w2",
            protocol_dir=tmp_path / "protocol",
        )

    report = run_evaluate_mode(
        checkpoint=best,
        output_root=out,
        w2_root=tmp_path / "w2",
        protocol_dir=tmp_path / "protocol",
    )
    assert report["accepted"] is False
    assert report["mode"] == "evaluate"
    assert Path(observed["checkpoint"]) == best
    assert observed["checkpoint_logit_scale"] == 2.5
    assert (out / "evaluation.json").is_file()
    assert (out / "evaluation_report.txt").is_file()
