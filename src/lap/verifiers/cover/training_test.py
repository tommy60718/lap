from __future__ import annotations

import torch

from lap.verifiers.cover.model import TinyFrozenBackbone
from lap.verifiers.cover.model import VerifierConfig
from lap.verifiers.cover.model import VerifierModel
from lap.verifiers.cover.protocol import RunProtocol
from lap.verifiers.cover.training import create_optimizer
from lap.verifiers.cover.training import make_base_only_config
from lap.verifiers.cover.training import train_one_batch


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


def _batch(batch_size=2):
    histories = torch.rand(batch_size, 10, 7)
    histories[:, :6] = -5.0
    images = torch.rand(batch_size, 3, 384, 384)
    return {
        "base_rgb": images,
        "wrist_rgb": images.clone(),
        "instructions": [f"peg {i}" for i in range(batch_size)],
        "action_histories": histories,
    }


def test_one_batch_updates_trainable_state_only():
    model = _model()
    before = {name: value.detach().clone() for name, value in model.named_parameters()}
    optimizer, _ = create_optimizer(model, RunProtocol(epochs=2, warmup_epochs=1))
    metrics = train_one_batch(model, _batch(), optimizer)
    assert metrics["loss"] > 0
    assert any(not torch.equal(before[name], value) for name, value in model.named_parameters() if value.requires_grad)
    assert all(torch.equal(before[name], value) for name, value in model.named_parameters() if not value.requires_grad)


def test_optimizer_contains_only_trainable_verifier_parameters():
    model = _model()
    optimizer, _ = create_optimizer(model, RunProtocol())

    optimized = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}
    trainable = {id(parameter) for parameter in model.parameters() if parameter.requires_grad}
    frozen = {id(parameter) for parameter in model.backbone.parameters()}

    assert optimized == trainable
    assert optimized.isdisjoint(frozen)


def test_base_only_config_changes_only_wrist_fusion_contract():
    two_view = _model().config
    base = make_base_only_config(two_view)
    assert two_view.use_wrist is True
    assert base.use_wrist is False
    assert base.fusion_input_width == 32
    assert base.action_width == two_view.action_width
