from __future__ import annotations

import copy
import json
from pathlib import Path
import random

import numpy as np
import pytest
import torch

from lap.verifiers.cover.checkpoint import ACCEPTED_DEPLOYMENT_MARKER
from lap.verifiers.cover.checkpoint import FIXTURE_DEPLOYMENT_MARKER
from lap.verifiers.cover.checkpoint import RESUME_KIND
from lap.verifiers.cover.checkpoint import build_checkpoint_contract
from lap.verifiers.cover.checkpoint import build_four_state_inventory
from lap.verifiers.cover.checkpoint import build_progress
from lap.verifiers.cover.checkpoint import load_deployment_bundle
from lap.verifiers.cover.checkpoint import load_training_checkpoint
from lap.verifiers.cover.checkpoint import publish_deployment_bundle
from lap.verifiers.cover.checkpoint import save_training_checkpoint
from lap.verifiers.cover.checkpoint import select_best_checkpoint
from lap.verifiers.cover.data import preprocess_rgb
from lap.verifiers.cover.model import TinyFrozenBackbone
from lap.verifiers.cover.model import VerifierConfig
from lap.verifiers.cover.model import VerifierModel
from lap.verifiers.cover.w3_contracts import canonical_bytes
from lap.verifiers.cover.w3_contracts import content_hash
from lap.verifiers.cover.w3_contracts import sha256_file


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


def _contract(**overrides):
    inventory = build_four_state_inventory()
    contract = build_checkpoint_contract(
        audit_manifest_sha256="a" * 64,
        bridge_artifact_sha256="b" * 64,
        target_fingerprint="c" * 64,
        protocol_content_hash="d" * 64,
        protocol_version="w3-g02-accepted-run-v1",
        model_config=_model().config.to_dict(),
        four_state_inventory=inventory,
        w2_identities={
            "train_manifest_hash": "e" * 64,
            "phrase_manifest_hash": "f" * 64,
            "normalization_artifact_hash": "1" * 64,
        },
        environment={
            "torch": torch.__version__,
            "cuda_available": False,
            "cuda_rng_device_count": 0,
        },
    )
    contract.update(overrides)
    return contract


def _cpu_rng_payload() -> dict:
    numpy_state = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": {
            "algorithm": numpy_state[0],
            "keys": numpy_state[1].tolist(),
            "position": int(numpy_state[2]),
            "has_gauss": int(numpy_state[3]),
            "cached_gaussian": float(numpy_state[4]),
        },
        "torch": torch.get_rng_state(),
    }


def _fake_cuda_entries(count: int) -> list[torch.Tensor]:
    return [torch.zeros(8, dtype=torch.uint8) for _ in range(count)]


def _progress(**overrides):
    progress = build_progress(epoch=3, global_step=120, best_metric=0.42, world_size=1)
    progress.update(overrides)
    return progress


def _forward(model):
    was_training = model.training
    model.eval()
    images = torch.zeros(2, 3, 384, 384)
    histories = torch.zeros(2, 10, 7)
    with torch.no_grad():
        output = model(images, images, ["insert peg", "insert peg"], histories)
    model.train(was_training)
    return {
        "semantic": output["semantic_embedding"].detach().cpu().clone(),
        "action": output["action_embedding"].detach().cpu().clone(),
        "logits": output["semantic_to_action_logits"].detach().cpu().clone(),
        "loss": float(model.contrastive_loss(output)[0].detach().cpu()),
    }


def _scorer_metadata(*, normalization_hash: str = "1" * 64, deployable: bool = True):
    return {
        "deployable": deployable,
        "model_config": _model().config.to_dict(),
        "w3_01_initialization": {
            "audit_manifest_sha256": "a" * 64,
            "bridge_artifact_sha256": "b" * 64,
            "target_fingerprint": "c" * 64,
        },
        "w3_03_protocol": {
            "protocol_content_hash": "d" * 64,
            "protocol_version": "w3-g02-accepted-run-v1",
        },
        "four_state_inventory": build_four_state_inventory(),
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
        "package_scope": "test_scoped_w3_06_not_canonical_w3_09",
    }


def test_save_resume_roundtrip_preserves_outputs_and_rejects_before_mutation(tmp_path):
    model = _model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
    contract = _contract()
    progress = _progress()
    path = tmp_path / "latest.pt"

    before = _forward(model)
    saved_state = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
    torch.manual_seed(7)
    np.random.seed(7)
    save_training_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        progress=progress,
        contract=contract,
        sampler_state={"epoch": 2, "rank": 0, "world_size": 1},
        rank=0,
    )

    # Mutate live state after save; exact resume must restore the saved point.
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(1.0)
    optimizer.param_groups[0]["lr"] = 9.9
    scheduler.step()
    torch.manual_seed(99)
    np.random.seed(99)

    loaded = load_training_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        expected_contract=contract,
        rank=0,
    )
    assert loaded["progress"] == progress
    assert loaded["sampler_state"] == {"epoch": 2, "rank": 0, "world_size": 1}
    assert loaded["contract"]["resume_kind"] == RESUME_KIND
    assert loaded["contract"]["resume_kind"] != "bridge_warm_start"
    for name, tensor in model.state_dict().items():
        torch.testing.assert_close(tensor.cpu(), saved_state[name], atol=0.0, rtol=0.0)
    after = _forward(model)
    for key in ("semantic", "action", "logits"):
        torch.testing.assert_close(after[key], before[key], atol=0.0, rtol=0.0)
    assert after["loss"] == before["loss"]

    pristine = {name: tensor.clone() for name, tensor in model.state_dict().items()}
    with pytest.raises(ValueError, match="contract mismatch"):
        load_training_checkpoint(
            path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            expected_contract={**contract, "protocol_version": "drifted"},
            rank=0,
        )
    for name, tensor in model.state_dict().items():
        torch.testing.assert_close(tensor, pristine[name], atol=0.0, rtol=0.0)


def test_training_checkpoint_rejects_partial_and_unexpected_keys_before_mutation(tmp_path):
    model = _model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
    contract = _contract()
    path = tmp_path / "latest.pt"
    save_training_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        progress=_progress(),
        contract=contract,
        sampler_state={"epoch": 1, "rank": 0, "world_size": 1},
        rank=0,
    )
    payload = torch.load(path, map_location="cpu", weights_only=False)
    first_key = next(iter(payload["model_state"]))
    del payload["model_state"][first_key]
    torch.save(payload, path)
    pristine = {name: tensor.clone() for name, tensor in model.state_dict().items()}
    with pytest.raises(ValueError, match="model state keys mismatch"):
        load_training_checkpoint(
            path, model=model, optimizer=optimizer, scheduler=scheduler, expected_contract=contract, rank=0
        )
    for name, tensor in model.state_dict().items():
        torch.testing.assert_close(tensor, pristine[name], atol=0.0, rtol=0.0)

    weights_only = tmp_path / "weights_only.pt"
    torch.save({"schema": "osx_cover_verifier_checkpoint_v1", "model_state": model.state_dict()}, weights_only)
    with pytest.raises(ValueError, match=r"incomplete|weights-only|missing"):
        load_training_checkpoint(
            weights_only, model=model, optimizer=optimizer, scheduler=scheduler, expected_contract=contract, rank=0
        )


def test_two_rank_rng_states_resume_exactly(tmp_path):
    model = _model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
    contract = _contract()
    path = tmp_path / "rank_bundle.pt"

    torch.manual_seed(11)
    np.random.seed(11)
    save_training_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        progress=_progress(world_size=2),
        contract=contract,
        sampler_state={"epoch": 0, "rank": 0, "world_size": 2},
        rank=0,
        rng_states={
            0: {
                "python": __import__("random").getstate(),
                "numpy": {
                    "algorithm": np.random.get_state()[0],
                    "keys": np.random.get_state()[1].tolist(),
                    "position": int(np.random.get_state()[2]),
                    "has_gauss": int(np.random.get_state()[3]),
                    "cached_gaussian": float(np.random.get_state()[4]),
                },
                "torch": torch.get_rng_state(),
            },
            1: {
                "python": __import__("random").getstate(),
                "numpy": {
                    "algorithm": np.random.get_state()[0],
                    "keys": (np.random.get_state()[1] + 1).tolist(),
                    "position": int(np.random.get_state()[2]),
                    "has_gauss": int(np.random.get_state()[3]),
                    "cached_gaussian": float(np.random.get_state()[4]),
                },
                "torch": torch.get_rng_state(),
            },
        },
    )
    payload = torch.load(path, map_location="cpu", weights_only=False)
    assert set(payload["rng_states"]) == {0, 1}
    assert payload["progress"]["world_size"] == 2
    load_training_checkpoint(
        path, model=model, optimizer=optimizer, scheduler=scheduler, expected_contract=contract, rank=1
    )
    restored = np.random.get_state()[1]
    assert restored.tolist() == payload["rng_states"][1]["numpy"]["keys"]


def test_latest_remains_resumable_and_best_uses_earliest_epoch_tie_break(tmp_path):
    records = [
        {"path": tmp_path / "epoch0.pt", "epoch": 0, "validation_loss": 1.5},
        {"path": tmp_path / "epoch1.pt", "epoch": 1, "validation_loss": 0.9},
        {"path": tmp_path / "epoch2.pt", "epoch": 2, "validation_loss": 0.9},
        {"path": tmp_path / "latest.pt", "epoch": 3, "validation_loss": 1.1},
    ]
    for record in records:
        record["path"].write_text("checkpoint", encoding="utf-8")
    selected = select_best_checkpoint(records)
    assert Path(selected) == tmp_path / "epoch1.pt"
    latest = next(record for record in records if record["path"].name == "latest.pt")
    assert latest["path"].is_file()


def test_load_deployment_bundle_score_requires_accepted_marker_and_validates_before_mutation(tmp_path):
    model = _model()
    root = tmp_path / "deployment"
    normalization_hash = "1" * 64
    publish_deployment_bundle(root, model=model, metadata=_scorer_metadata(), accepted_marker=True)
    assert (root / ACCEPTED_DEPLOYMENT_MARKER).is_file()
    assert not (root / FIXTURE_DEPLOYMENT_MARKER).exists()
    model_payload = torch.load(root / "model.pt", map_location="cpu", weights_only=True)
    assert "optimizer_state" not in model_payload
    assert set(model_payload) <= {"schema", "model_state"}

    metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["w3_01_initialization"]["audit_manifest_sha256"] == "a" * 64
    assert metadata["w3_03_protocol"]["protocol_content_hash"] == "d" * 64
    assert set(metadata["four_state_inventory"]) == {"frozen", "warm_started", "fresh", "forbidden"}
    assert metadata["package_scope"] == "test_scoped_w3_06_not_canonical_w3_09"

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

    fixture_root = tmp_path / "fixture"
    publish_deployment_bundle(fixture_root, model=_model(), metadata=_scorer_metadata(), accepted_marker=False)
    assert (fixture_root / FIXTURE_DEPLOYMENT_MARKER).is_file()
    with pytest.raises(ValueError, match=r"ACCEPTED_W3_DEPLOYMENT|fixture"):
        load_deployment_bundle(
            fixture_root,
            model_factory=_model,
            expected_normalization_hash=normalization_hash,
            preprocessing=preprocess_rgb,
        )

    # Corrupt a hashed file after publication; rejection must happen before model mutation.
    probe = _model()
    before = copy.deepcopy(probe.state_dict())
    (root / "metadata.json").write_text('{"schema":"tampered"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match=r"deployable W3 schema|content hash mismatch"):
        load_deployment_bundle(
            root,
            model_factory=lambda: probe,
            expected_normalization_hash=normalization_hash,
            preprocessing=preprocess_rgb,
        )
    for name, tensor in probe.state_dict().items():
        torch.testing.assert_close(tensor, before[name], atol=0.0, rtol=0.0)


def test_nondeployable_bundle_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="nondeployable"):
        publish_deployment_bundle(tmp_path / "deployment", model=_model(), metadata={"deployable": False})


def test_invalid_optimizer_state_rejects_without_mutating_caller_owned_state(tmp_path):
    """W3-06-R1: incompatible optimizer/scheduler/RNG must fail closed before mutation."""

    source = _model()
    optimizer = torch.optim.AdamW(source.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
    loss = sum(parameter.sum() for parameter in source.parameters())
    loss.backward()
    optimizer.step()
    contract = _contract()
    path = tmp_path / "latest.pt"
    save_training_checkpoint(
        path,
        model=source,
        optimizer=optimizer,
        scheduler=scheduler,
        progress=_progress(),
        contract=contract,
        sampler_state={"epoch": 1, "rank": 0, "world_size": 1},
        rank=0,
    )
    payload = torch.load(path, map_location="cpu", weights_only=False)
    payload["optimizer_state"] = {"invalid": True}
    torch.save(payload, path)

    target = _model()
    target_optimizer = torch.optim.AdamW(target.parameters(), lr=1e-3)
    target_scheduler = torch.optim.lr_scheduler.StepLR(target_optimizer, step_size=1)
    pristine_model = {name: tensor.detach().cpu().clone() for name, tensor in target.state_dict().items()}
    pristine_optimizer = copy.deepcopy(target_optimizer.state_dict())
    pristine_scheduler = copy.deepcopy(target_scheduler.state_dict())
    torch.manual_seed(123)
    np.random.seed(123)
    pristine_torch_rng = torch.get_rng_state().clone()
    pristine_numpy_rng = np.random.get_state()

    with pytest.raises(ValueError, match=r"optimizer|incompatible"):
        load_training_checkpoint(
            path,
            model=target,
            optimizer=target_optimizer,
            scheduler=target_scheduler,
            expected_contract=contract,
            rank=0,
        )

    for name, tensor in target.state_dict().items():
        torch.testing.assert_close(tensor.cpu(), pristine_model[name], atol=0.0, rtol=0.0)
    assert target_optimizer.state_dict() == pristine_optimizer
    assert target_scheduler.state_dict() == pristine_scheduler
    assert torch.equal(torch.get_rng_state(), pristine_torch_rng)
    assert np.random.get_state()[1].tolist() == list(pristine_numpy_rng[1])


def test_coherently_rehashed_incomplete_or_extra_content_index_is_rejected(tmp_path):
    """W3-06-R2: complete content-index membership is required before model construction."""

    normalization_hash = "1" * 64
    constructed: list[str] = []

    def tracking_factory():
        constructed.append("built")
        return _model()

    root = tmp_path / "deployment"
    publish_deployment_bundle(root, model=_model(), metadata=_scorer_metadata(), accepted_marker=True)

    omitted = tmp_path / "omitted"
    omitted.mkdir()
    for path in root.iterdir():
        (omitted / path.name).write_bytes(path.read_bytes())
    index = json.loads((omitted / "content_index.json").read_text(encoding="utf-8"))
    del index["files"]["model.pt"]
    index["content_hash"] = content_hash(index)
    (omitted / "content_index.json").write_bytes(canonical_bytes(index) + b"\n")
    with pytest.raises(ValueError, match=r"content index|required|incomplete|model\.pt"):
        load_deployment_bundle(
            omitted,
            model_factory=tracking_factory,
            expected_normalization_hash=normalization_hash,
            preprocessing=preprocess_rgb,
        )
    assert constructed == []

    extra = tmp_path / "extra"
    extra.mkdir()
    for path in root.iterdir():
        (extra / path.name).write_bytes(path.read_bytes())
    (extra / "unexpected.bin").write_bytes(b"extra")
    index = json.loads((extra / "content_index.json").read_text(encoding="utf-8"))
    index["files"]["unexpected.bin"] = sha256_file(extra / "unexpected.bin")
    index["content_hash"] = content_hash(index)
    (extra / "content_index.json").write_bytes(canonical_bytes(index) + b"\n")
    # Extra indexed file beyond the required sealed set must also fail closed.
    with pytest.raises(ValueError, match=r"content index|unexpected|extra|required"):
        load_deployment_bundle(
            extra,
            model_factory=tracking_factory,
            expected_normalization_hash=normalization_hash,
            preprocessing=preprocess_rgb,
        )
    assert constructed == []


def _snapshot_caller_owned_state(model, optimizer, scheduler):
    return {
        "model": {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()},
        "optimizer": copy.deepcopy(optimizer.state_dict()),
        "scheduler": copy.deepcopy(scheduler.state_dict()),
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state().clone(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _assert_caller_owned_state_unchanged(model, optimizer, scheduler, pristine):
    for name, tensor in model.state_dict().items():
        torch.testing.assert_close(tensor.cpu(), pristine["model"][name], atol=0.0, rtol=0.0)
    assert optimizer.state_dict() == pristine["optimizer"]
    assert scheduler.state_dict() == pristine["scheduler"]
    assert random.getstate() == pristine["python"]
    assert np.random.get_state()[1].tolist() == list(pristine["numpy"][1])
    assert torch.equal(torch.get_rng_state(), pristine["torch"])
    if pristine["cuda"] is not None:
        current = torch.cuda.get_rng_state_all()
        assert len(current) == len(pristine["cuda"])
        for left, right in zip(current, pristine["cuda"], strict=True):
            assert torch.equal(left, right)


def test_malformed_rng_state_rejects_without_mutating_caller_owned_state(tmp_path):
    """W3-06-R3: incompatible Torch RNG must fail closed before any live restore."""

    source = _model()
    optimizer = torch.optim.AdamW(source.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
    contract = _contract()
    path = tmp_path / "latest.pt"
    save_training_checkpoint(
        path,
        model=source,
        optimizer=optimizer,
        scheduler=scheduler,
        progress=_progress(),
        contract=contract,
        sampler_state={"epoch": 1, "rank": 0, "world_size": 1},
        rank=0,
    )
    payload = torch.load(path, map_location="cpu", weights_only=False)
    payload["rng_states"][0]["torch"] = torch.tensor([1, 2, 3], dtype=torch.int64)
    torch.save(payload, path)

    target = _model()
    target_optimizer = torch.optim.AdamW(target.parameters(), lr=1e-3)
    target_scheduler = torch.optim.lr_scheduler.StepLR(target_optimizer, step_size=1)
    torch.manual_seed(321)
    np.random.seed(321)
    pristine = _snapshot_caller_owned_state(target, target_optimizer, target_scheduler)

    with pytest.raises(ValueError, match=r"rng|RNG|ByteTensor|incompatible"):
        load_training_checkpoint(
            path,
            model=target,
            optimizer=target_optimizer,
            scheduler=target_scheduler,
            expected_contract=contract,
            rank=0,
        )
    _assert_caller_owned_state_unchanged(target, target_optimizer, target_scheduler, pristine)


def test_incomplete_or_incompatible_sampler_state_rejects_without_mutation(tmp_path):
    """W3-06-R3: sampler schema/compatibility is validated before mutation."""

    source = _model()
    optimizer = torch.optim.AdamW(source.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
    contract = _contract()
    path = tmp_path / "latest.pt"
    save_training_checkpoint(
        path,
        model=source,
        optimizer=optimizer,
        scheduler=scheduler,
        progress=_progress(),
        contract=contract,
        sampler_state={"epoch": 1, "rank": 0, "world_size": 1},
        rank=0,
    )

    target = _model()
    target_optimizer = torch.optim.AdamW(target.parameters(), lr=1e-3)
    target_scheduler = torch.optim.lr_scheduler.StepLR(target_optimizer, step_size=1)

    for broken in (
        {},
        {"epoch": 1},
        {"epoch": 1, "rank": 0, "world_size": 2},
        {"epoch": 1, "rank": 0, "world_size": 1, "extra": True},
    ):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        payload["sampler_state"] = broken
        torch.save(payload, path)
        torch.manual_seed(11)
        np.random.seed(11)
        pristine = _snapshot_caller_owned_state(target, target_optimizer, target_scheduler)
        with pytest.raises(ValueError, match=r"sampler"):
            load_training_checkpoint(
                path,
                model=target,
                optimizer=target_optimizer,
                scheduler=target_scheduler,
                expected_contract=contract,
                rank=0,
            )
        _assert_caller_owned_state_unchanged(target, target_optimizer, target_scheduler, pristine)


def test_valid_complete_checkpoint_still_resumes_exactly_after_r3_guards(tmp_path):
    """Successful exact resume remains available after R3 validate-before-mutate guards."""

    model = _model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
    contract = _contract()
    progress = _progress()
    path = tmp_path / "latest.pt"
    before = _forward(model)
    saved_state = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
    save_training_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        progress=progress,
        contract=contract,
        sampler_state={"epoch": 4, "rank": 0, "world_size": 1},
        rank=0,
    )
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(0.5)
    loaded = load_training_checkpoint(
        path, model=model, optimizer=optimizer, scheduler=scheduler, expected_contract=contract, rank=0
    )
    assert loaded["progress"] == progress
    assert loaded["sampler_state"] == {"epoch": 4, "rank": 0, "world_size": 1}
    for name, tensor in model.state_dict().items():
        torch.testing.assert_close(tensor.cpu(), saved_state[name], atol=0.0, rtol=0.0)
    after = _forward(model)
    for key in ("semantic", "action", "logits"):
        torch.testing.assert_close(after[key], before[key], atol=0.0, rtol=0.0)
    assert after["loss"] == before["loss"]


def test_type_invalid_sampler_fields_reject_before_mutation(tmp_path):
    """W3-06-R4: sampler fields must be actual integers; no int(...) coercion."""

    source = _model()
    optimizer = torch.optim.AdamW(source.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
    contract = _contract()
    path = tmp_path / "latest.pt"
    save_training_checkpoint(
        path,
        model=source,
        optimizer=optimizer,
        scheduler=scheduler,
        progress=_progress(),
        contract=contract,
        sampler_state={"epoch": 1, "rank": 0, "world_size": 1},
        rank=0,
    )

    target = _model()
    target_optimizer = torch.optim.AdamW(target.parameters(), lr=1e-3)
    target_scheduler = torch.optim.lr_scheduler.StepLR(target_optimizer, step_size=1)

    invalid_cases = (
        {"epoch": "1", "rank": 0, "world_size": 1},
        {"epoch": 1.0, "rank": 0, "world_size": 1},
        {"epoch": True, "rank": 0, "world_size": 1},
        {"epoch": 1, "rank": "0", "world_size": 1},
        {"epoch": 1, "rank": 0.0, "world_size": 1},
        {"epoch": 1, "rank": False, "world_size": 1},
        {"epoch": 1, "rank": 0, "world_size": "1"},
        {"epoch": 1, "rank": 0, "world_size": 1.0},
        {"epoch": 1, "rank": 0, "world_size": True},
    )
    for broken in invalid_cases:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        payload["sampler_state"] = broken
        torch.save(payload, path)
        torch.manual_seed(17)
        np.random.seed(17)
        pristine = _snapshot_caller_owned_state(target, target_optimizer, target_scheduler)
        with pytest.raises(ValueError, match=r"sampler|integer"):
            load_training_checkpoint(
                path,
                model=target,
                optimizer=target_optimizer,
                scheduler=target_scheduler,
                expected_contract=contract,
                rank=0,
            )
        _assert_caller_owned_state_unchanged(target, target_optimizer, target_scheduler, pristine)


def test_cpu_contract_rejects_cuda_rng_payload_before_mutation(tmp_path):
    """W3-06-R4: cuda_rng_device_count=0 forbids any cuda RNG payload."""

    source = _model()
    optimizer = torch.optim.AdamW(source.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
    contract = _contract()
    path = tmp_path / "latest.pt"
    save_training_checkpoint(
        path,
        model=source,
        optimizer=optimizer,
        scheduler=scheduler,
        progress=_progress(),
        contract=contract,
        sampler_state={"epoch": 1, "rank": 0, "world_size": 1},
        rank=0,
    )
    payload = torch.load(path, map_location="cpu", weights_only=False)
    payload["rng_states"][0]["cuda"] = _fake_cuda_entries(1)
    torch.save(payload, path)

    target = _model()
    target_optimizer = torch.optim.AdamW(target.parameters(), lr=1e-3)
    target_scheduler = torch.optim.lr_scheduler.StepLR(target_optimizer, step_size=1)
    torch.manual_seed(19)
    np.random.seed(19)
    pristine = _snapshot_caller_owned_state(target, target_optimizer, target_scheduler)
    with pytest.raises(ValueError, match=r"cuda|CPU-only|cuda_rng_device_count"):
        load_training_checkpoint(
            path,
            model=target,
            optimizer=target_optimizer,
            scheduler=target_scheduler,
            expected_contract=contract,
            rank=0,
        )
    _assert_caller_owned_state_unchanged(target, target_optimizer, target_scheduler, pristine)


def test_cuda_contract_rejects_incomplete_or_malformed_cuda_rng_before_mutation(tmp_path):
    """W3-06-R4: positive cuda_rng_device_count requires exact per-rank CUDA cardinality."""

    source = _model()
    optimizer = torch.optim.AdamW(source.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
    contract = _contract()
    contract["environment"] = {
        **contract["environment"],
        "cuda_available": True,
        "cuda_rng_device_count": 2,
    }
    path = tmp_path / "latest.pt"
    # Build a valid CPU payload first, then rewrite contract + rng for CUDA cases.
    save_training_checkpoint(
        path,
        model=source,
        optimizer=optimizer,
        scheduler=scheduler,
        progress=_progress(),
        contract=_contract(),
        sampler_state={"epoch": 1, "rank": 0, "world_size": 1},
        rank=0,
    )
    base = torch.load(path, map_location="cpu", weights_only=False)

    target = _model()
    target_optimizer = torch.optim.AdamW(target.parameters(), lr=1e-3)
    target_scheduler = torch.optim.lr_scheduler.StepLR(target_optimizer, step_size=1)

    broken_rng_cases = (
        {0: _cpu_rng_payload()},  # missing cuda
        {0: {**_cpu_rng_payload(), "cuda": []}},  # empty
        {0: {**_cpu_rng_payload(), "cuda": _fake_cuda_entries(1)}},  # truncated
        {0: {**_cpu_rng_payload(), "cuda": _fake_cuda_entries(3)}},  # oversized
        {0: {**_cpu_rng_payload(), "cuda": [torch.zeros(8, dtype=torch.float32), torch.zeros(8, dtype=torch.uint8)]}},
    )
    for rng_states in broken_rng_cases:
        payload = copy.deepcopy(base)
        payload["contract"] = contract
        payload["rng_states"] = rng_states
        torch.save(payload, path)
        torch.manual_seed(23)
        np.random.seed(23)
        pristine = _snapshot_caller_owned_state(target, target_optimizer, target_scheduler)
        with pytest.raises(ValueError, match=r"cuda|RNG|ByteTensor|cardinality|device"):
            load_training_checkpoint(
                path,
                model=target,
                optimizer=target_optimizer,
                scheduler=target_scheduler,
                expected_contract=contract,
                rank=0,
            )
        _assert_caller_owned_state_unchanged(target, target_optimizer, target_scheduler, pristine)


def test_cuda_runtime_unavailable_or_cardinality_mismatch_rejects_before_mutation(tmp_path, monkeypatch):
    """W3-06-R4: load-time CUDA availability/count is a compatibility gate only."""

    source = _model()
    optimizer = torch.optim.AdamW(source.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
    path = tmp_path / "latest.pt"
    save_training_checkpoint(
        path,
        model=source,
        optimizer=optimizer,
        scheduler=scheduler,
        progress=_progress(),
        contract=_contract(),
        sampler_state={"epoch": 1, "rank": 0, "world_size": 1},
        rank=0,
    )
    contract = _contract()
    contract["environment"] = {
        **contract["environment"],
        "cuda_available": True,
        "cuda_rng_device_count": 2,
    }
    payload = torch.load(path, map_location="cpu", weights_only=False)
    payload["contract"] = contract
    payload["rng_states"] = {0: {**_cpu_rng_payload(), "cuda": _fake_cuda_entries(2)}}
    torch.save(payload, path)

    target = _model()
    target_optimizer = torch.optim.AdamW(target.parameters(), lr=1e-3)
    target_scheduler = torch.optim.lr_scheduler.StepLR(target_optimizer, step_size=1)

    torch.manual_seed(29)
    np.random.seed(29)
    pristine = _snapshot_caller_owned_state(target, target_optimizer, target_scheduler)

    # Force-unavailable must reject before mutation on any host, including 2-GPU.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(ValueError, match=r"CUDA|cuda|unavailable|device"):
        load_training_checkpoint(
            path,
            model=target,
            optimizer=target_optimizer,
            scheduler=target_scheduler,
            expected_contract=contract,
            rank=0,
        )
    _assert_caller_owned_state_unchanged(target, target_optimizer, target_scheduler, pristine)

    # Visible cardinality mismatch must also reject before mutation.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    with pytest.raises(ValueError, match=r"CUDA|cuda|cardinality|device"):
        load_training_checkpoint(
            path,
            model=target,
            optimizer=target_optimizer,
            scheduler=target_scheduler,
            expected_contract=contract,
            rank=0,
        )
    _assert_caller_owned_state_unchanged(target, target_optimizer, target_scheduler, pristine)


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 2,
    reason="W3-06-R4 real two-GPU resume requires two visible CUDA devices",
)
def test_real_two_gpu_checkpoint_resumes_exactly(tmp_path):
    """W3-06-R4: complete two-GPU CUDA checkpoint resumes exactly on supported host."""

    device_count = torch.cuda.device_count()
    model = _model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
    contract = _contract()
    contract["environment"] = {
        **contract["environment"],
        "cuda_available": True,
        "cuda_rng_device_count": device_count,
    }
    path = tmp_path / "latest.pt"
    before = _forward(model)
    saved_state = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
    torch.manual_seed(41)
    np.random.seed(41)
    torch.cuda.manual_seed_all(41)
    save_training_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        progress=_progress(),
        contract=contract,
        sampler_state={"epoch": 2, "rank": 0, "world_size": 1},
        rank=0,
    )
    payload = torch.load(path, map_location="cpu", weights_only=False)
    assert "cuda" in payload["rng_states"][0]
    assert len(payload["rng_states"][0]["cuda"]) == device_count

    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(1.0)
    torch.manual_seed(99)
    np.random.seed(99)
    torch.cuda.manual_seed_all(99)
    loaded = load_training_checkpoint(
        path, model=model, optimizer=optimizer, scheduler=scheduler, expected_contract=contract, rank=0
    )
    assert loaded["sampler_state"] == {"epoch": 2, "rank": 0, "world_size": 1}
    for name, tensor in model.state_dict().items():
        torch.testing.assert_close(tensor.cpu(), saved_state[name], atol=0.0, rtol=0.0)
    after = _forward(model)
    for key in ("semantic", "action", "logits"):
        torch.testing.assert_close(after[key], before[key], atol=0.0, rtol=0.0)
    assert after["loss"] == before["loss"]
