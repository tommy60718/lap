from __future__ import annotations

import numpy as np
import pytest
import torch

from lap.verifiers.cover.checkpoint import load_deployment_bundle
from lap.verifiers.cover.checkpoint import load_training_checkpoint
from lap.verifiers.cover.checkpoint import publish_deployment_bundle
from lap.verifiers.cover.checkpoint import save_training_checkpoint
from lap.verifiers.cover.data import preprocess_rgb
from lap.verifiers.cover.model import TinyFrozenBackbone
from lap.verifiers.cover.model import VerifierConfig
from lap.verifiers.cover.model import VerifierModel


def _model():
    return VerifierModel(
        VerifierConfig(
            backbone_width=32,
            embedding_width=16,
            visual_tokens=8,
            num_heads=4,
            pooling_layers=1,
            trajectory_layers=1,
            feed_forward_width=32,
        ),
        TinyFrozenBackbone(width=32, tokens=8),
    )


def test_training_checkpoint_roundtrip_restores_outputs_and_rejects_contract_drift(tmp_path):
    model = _model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
    contract = {"model": model.config.to_dict(), "run": {"seed": 4203}}
    path = tmp_path / "latest.pt"
    save_training_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        progress={"epoch": 1},
        contract=contract,
        sampler_state={"epoch": 1},
    )
    loaded = load_training_checkpoint(
        path, model=model, optimizer=optimizer, scheduler=scheduler, expected_contract=contract
    )
    assert loaded["progress"] == {"epoch": 1}
    with pytest.raises(ValueError, match="contract mismatch"):
        load_training_checkpoint(
            path, model=model, optimizer=optimizer, scheduler=scheduler, expected_contract={"changed": True}
        )


def test_deployment_bundle_loads_strictly_and_scores_candidates_in_order(tmp_path):
    model = _model()
    root = tmp_path / "deployment"
    normalization_hash = "a" * 64
    metadata = {
        "deployable": True,
        "model_config": model.config.to_dict(),
        "scorer_compatibility": {
            "model_schema_version": "osx_cover_verifier_checkpoint_v1",
            "views": ["base_rgb", "wrist_rgb"],
            "preprocessing_contract": "openclip_siglip2_384_center_crop_v1",
            "action_dimension": 7,
            "action_order": ["dx", "dy", "dz", "rotation_x", "rotation_y", "rotation_z", "gripper"],
            "history_length": 10,
            "representation_id": "ur5e_cover_relative_eef_v1",
            "normalization_artifact_hash": normalization_hash,
            "input_dtype": "float32",
            "output_shape_rank": 1,
        },
    }
    publish_deployment_bundle(root, model=model, metadata=metadata)
    loaded = load_deployment_bundle(
        root, model_factory=_model, expected_normalization_hash=normalization_hash, preprocessing=preprocess_rgb
    )
    scores = loaded.score(
        base_rgb=np.zeros((384, 384, 3), dtype=np.uint8),
        wrist_rgb=np.zeros((384, 384, 3), dtype=np.uint8),
        instruction="insert peg",
        action_histories=np.zeros((3, 10, 7), dtype=np.float32),
    )
    assert scores.shape == (3,)
    assert np.isfinite(scores).all()


def test_nondeployable_bundle_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="nondeployable"):
        publish_deployment_bundle(tmp_path / "deployment", model=_model(), metadata={"deployable": False})
