from __future__ import annotations

import json
import sys
from types import SimpleNamespace

import pytest
import torch

from lap.verifiers.cover.model import OpenClipSigLIP2Backbone
from lap.verifiers.cover.model import TextAwareVisualExtraction
from lap.verifiers.cover.model import TinyFrozenBackbone
from lap.verifiers.cover.model import VerifierConfig
from lap.verifiers.cover.model import VerifierModel
from lap.verifiers.cover.w3_contracts import BACKBONE_REVISION


def _model(*, use_wrist=True):
    config = VerifierConfig(
        backbone_width=32,
        embedding_width=16,
        visual_tokens=8,
        num_heads=4,
        pooling_layers=1,
        trajectory_layers=1,
        feed_forward_width=32,
        use_wrist=use_wrist,
    )
    return VerifierModel(config, TinyFrozenBackbone(width=32, tokens=8))


def _batch(batch_size=3):
    images = torch.rand(batch_size, 3, 384, 384)
    histories = torch.rand(batch_size, 10, 7)
    histories[:, :6] = -5.0
    return images, histories, [f"insert peg {index}" for index in range(batch_size)]


def test_two_view_forward_has_normalized_embeddings_and_finite_logits():
    model = _model()
    base, histories, instructions = _batch()
    output = model(base, base.clone(), instructions, histories)
    assert output["semantic_embedding"].shape == (3, 16)
    assert output["action_embedding"].shape == (3, 16)
    assert torch.allclose(output["semantic_embedding"].norm(dim=-1), torch.ones(3), atol=1e-5)
    assert torch.allclose(output["action_embedding"].norm(dim=-1), torch.ones(3), atol=1e-5)
    assert torch.isfinite(output["semantic_to_action_logits"]).all()


def test_text_aware_extraction_matches_bridge_text_conditioning_contract():
    extractor = TextAwareVisualExtraction(width=4, tokens=3)
    visual = torch.tensor([[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]]])
    first_text = torch.tensor([[[1.0, 0.0, 0.0, 0.0]]])
    second_text = torch.tensor([[[0.0, 1.0, 0.0, 0.0]]])

    first = extractor(visual, first_text)
    second = extractor(visual, second_text)

    assert first.shape == (1, 1, 4)
    assert second.shape == (1, 1, 4)
    assert not torch.allclose(first, second)


def test_trajectory_encoder_uses_bridge_default_relu_activation():
    model = _model()
    assert model.trajectory_encoder.layers[0].activation.__name__ == "relu"


def test_verifier_config_fingerprint_includes_semantic_behavior_contracts():
    config = VerifierConfig().to_dict()

    assert config["text_aware_extraction_contract"] == "bridge_clearclip_v1"
    assert config["trajectory_activation"] == "relu"
    assert config["trajectory_position_contract"] == "sinusoidal_v1"
    assert config["attention_pooling_contract"] == "lap_fresh_attention_pool_v1"


def test_openclip_backbone_loads_preprocessing_and_tokenizer_from_one_pinned_snapshot(tmp_path, monkeypatch):
    snapshot = tmp_path / BACKBONE_REVISION
    snapshot.mkdir()
    (snapshot / "open_clip_config.json").write_text(
        json.dumps({"preprocess_cfg": {"mean": [0.5]}, "model_cfg": {"text_cfg": {"context_length": 64}}}),
        encoding="utf-8",
    )
    for filename in ("special_tokens_map.json", "tokenizer.json", "tokenizer_config.json"):
        (snapshot / filename).write_text(json.dumps({"source": filename}), encoding="utf-8")
    calls = []
    frozen_model = torch.nn.Linear(1, 1)

    def create_model_and_transforms(model_name, *, pretrained):
        calls.append(("preprocess", model_name))
        return frozen_model, None, "pinned-preprocess"

    def get_tokenizer(model_name):
        calls.append(("tokenizer", model_name))
        return SimpleNamespace()

    monkeypatch.setitem(
        sys.modules,
        "open_clip",
        SimpleNamespace(create_model_and_transforms=create_model_and_transforms, get_tokenizer=get_tokenizer),
    )
    model_name = f"local-dir:{snapshot}"

    backbone = OpenClipSigLIP2Backbone(model_name=model_name)

    assert calls == [("preprocess", model_name), ("tokenizer", model_name)]
    assert backbone.asset_fingerprints == {
        "preprocessing_fingerprint": "67c8af8a0d007115bf310b8443924703884c93abbe38bf55e3555e40d26768f3",
        "tokenizer_fingerprint": "73a4aeff34b428112bcadf6e490f211110d12c6dcdb4115d3151b2b7f1381d40",
    }


def test_model_rejects_unimplemented_semantic_contract():
    config = VerifierConfig(trajectory_activation="gelu")

    with pytest.raises(ValueError, match="trajectory activation"):
        VerifierModel(config, TinyFrozenBackbone(width=1024, tokens=576))


def test_backbone_is_frozen_but_verifier_step_has_trainable_gradients():
    model = _model()
    base, histories, instructions = _batch()
    loss, _ = model.contrastive_loss(model(base, base, instructions, histories))
    loss.backward()
    assert all(not parameter.requires_grad for parameter in model.backbone.parameters())
    assert any(parameter.grad is not None for parameter in model.semantic_fusion.parameters())
    assert all(parameter.grad is None for parameter in model.backbone.parameters())


def test_action_padding_must_be_leading_and_not_all_padding():
    model = _model()
    base, histories, instructions = _batch(batch_size=1)
    histories[:, 0] = 0.0
    histories[:, 1] = -5.0
    with pytest.raises(ValueError, match="leading"):
        model(base, base, instructions, histories)
    histories[:, :] = -5.0
    with pytest.raises(ValueError, match="at least one"):
        model(base, base, instructions, histories)


def test_base_only_uses_fresh_two_input_fusion():
    model = _model(use_wrist=False)
    base, histories, instructions = _batch()
    output = model(base, None, instructions, histories)
    assert model.semantic_fusion.in_features == 32
    assert output["semantic_embedding"].shape == (3, 16)


def test_local_bidirectional_loss_matches_hand_computed_identity_case():
    model = _model()
    logits = torch.tensor([[2.0, 0.0], [0.0, 2.0]])
    loss, metrics = model.contrastive_loss({"semantic_to_action_logits": logits, "action_to_semantic_logits": logits})
    expected = torch.nn.functional.cross_entropy(logits, torch.tensor([0, 1]))
    assert torch.allclose(loss, expected)
    assert metrics["retrieval_top1"] == 1.0
