from __future__ import annotations

import pytest
import torch

from lap.verifiers.cover.model import TinyFrozenBackbone
from lap.verifiers.cover.model import VerifierConfig
from lap.verifiers.cover.model import VerifierModel


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
