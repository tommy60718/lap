from __future__ import annotations

import json
from pathlib import Path
import subprocess

import numpy as np
import pytest
from torch.utils.data.distributed import DistributedSampler

from lap.verifiers.cover import protocol as protocol_module
from lap.verifiers.cover.bridge_audit import build_production_target_inventory
from lap.verifiers.cover.model import VerifierConfig
from lap.verifiers.cover.model import fingerprint_siglip2_assets
from lap.verifiers.cover.protocol import APPROVED_SHUFFLED_EXCLUSIONS
from lap.verifiers.cover.protocol import EVALUATION_SEED
from lap.verifiers.cover.protocol import PROBE_ORDER
from lap.verifiers.cover.protocol import TRAINING_SEED
from lap.verifiers.cover.protocol import WORLD_SIZE
from lap.verifiers.cover.protocol import _is_runtime_relevant_path
from lap.verifiers.cover.protocol import build_bootstrap_indices
from lap.verifiers.cover.protocol import build_phrase_manifest
from lap.verifiers.cover.protocol import build_shuffled_pairs
from lap.verifiers.cover.protocol import materialize_protocol
from lap.verifiers.cover.protocol import probe_batch_sizes
from lap.verifiers.cover.protocol import sampler_indices
from lap.verifiers.cover.protocol import select_training_phrase
from lap.verifiers.cover.protocol import validate_batch_probe_receipt
from lap.verifiers.cover.protocol import validate_protocol_directory
from lap.verifiers.cover.w3_contracts import BACKBONE_ID
from lap.verifiers.cover.w3_contracts import BACKBONE_REVISION
from lap.verifiers.cover.w3_contracts import content_hash
from lap.verifiers.cover.w3_contracts import sha256_file
from lap.verifiers.cover.w3_contracts import write_canonical_json

CANONICAL_EXPORT = Path(__file__).resolve().parents[6] / ".w2_canonical_export_v3"
CANONICAL_RECEIPT = Path(__file__).resolve().parents[4] / "artifacts/w3/protocol/batch_probe_receipt.json"


def _canonical_validation_rows():
    return json.loads((CANONICAL_EXPORT / "validation_samples.json").read_text(encoding="utf-8"))


def _canonical_model_identity(*, audit_manifest_sha256: str = "1" * 64, snapshot: Path | None = None):
    configuration = VerifierConfig().to_dict()
    identity = {
        "backbone": "hf-hub:timm/ViT-L-16-SigLIP2-384",
        "canonical_target": True,
        "configuration": configuration,
        "configuration_hash": content_hash(configuration),
        "audit_manifest_sha256": audit_manifest_sha256,
        "target_inventory_fingerprint": build_production_target_inventory().fingerprint,
        "preprocessing_fingerprint": "2" * 64,
        "tokenizer_fingerprint": "3" * 64,
    }
    if snapshot is not None:
        identity.update(fingerprint_siglip2_assets(snapshot))
        identity["siglip2_snapshot"] = {
            "backbone_id": BACKBONE_ID,
            "revision": BACKBONE_REVISION,
            "local_snapshot": str(snapshot),
        }
    return identity


def _asset_snapshot(tmp_path: Path):
    snapshot = tmp_path / BACKBONE_REVISION
    snapshot.mkdir()
    (snapshot / "open_clip_config.json").write_text(
        json.dumps({"preprocess_cfg": {"mean": [0.5]}, "model_cfg": {"text_cfg": {"context_length": 64}}}),
        encoding="utf-8",
    )
    for filename in ("special_tokens_map.json", "tokenizer.json", "tokenizer_config.json"):
        (snapshot / filename).write_text(json.dumps({"source": filename}), encoding="utf-8")
    return snapshot


def _canonical_gpu_snapshots():
    return [
        {
            "index": 0,
            "name": "NVIDIA RTX 6000 Ada Generation",
            "driver_version": "570.124.06",
            "uuid": "GPU-test-0",
            "physical_total_memory_mib": 49140,
            "allocatable_total_memory_mib": 48502,
            "free_memory_mib": 47000,
        },
        {
            "index": 1,
            "name": "NVIDIA RTX 6000 Ada Generation",
            "driver_version": "570.124.06",
            "uuid": "GPU-test-1",
            "physical_total_memory_mib": 49140,
            "allocatable_total_memory_mib": 48510,
            "free_memory_mib": 47100,
        },
    ]


def _rank_success(batch_size: int, rank: int, _gpu):
    return {
        "rank": rank,
        "status": "passed",
        "forward": "passed",
        "backward": "passed",
        "optimizer_step": "passed",
        "loss": 1.25,
        "gradient_norm": 0.5,
        "gradients_finite": True,
        "parameters_finite": True,
        "trainable_state_changed": True,
        "frozen_state_unchanged": True,
        "gradient_fingerprint": "a" * 64,
        "local_negative_pool_size": batch_size,
        "embedding_all_gather": False,
        "frozen_backbone_dtype": "bfloat16",
        "trainable_dtype": "float32",
        "logits_dtype": "float32",
    }


def test_revision_drift_classification_limits_capacity_defining_paths():
    assert not _is_runtime_relevant_path("AGENTS.md")
    assert not _is_runtime_relevant_path("artifacts/w3/protocol/run_protocol.json")
    assert not _is_runtime_relevant_path(".understand-anything/graph.json")
    assert not _is_runtime_relevant_path("src/lap/verifiers/cover/checkpoint.py")
    assert not _is_runtime_relevant_path("src/lap/verifiers/cover/checkpoint_test.py")
    assert not _is_runtime_relevant_path("src/lap/verifiers/cover/pipeline.py")
    assert not _is_runtime_relevant_path("src/lap/verifiers/cover/command.py")
    assert not _is_runtime_relevant_path("src/lap/verifiers/cover/evaluator.py")
    assert not _is_runtime_relevant_path("scripts/w3_cover.py")
    assert not _is_runtime_relevant_path("src/lap/verifiers/cover/protocol.py")
    assert not _is_runtime_relevant_path("src/lap/verifiers/cover/bridge_audit.py")
    assert _is_runtime_relevant_path("src/lap/verifiers/cover/model.py")
    assert _is_runtime_relevant_path("src/lap/verifiers/cover/training.py")
    assert _is_runtime_relevant_path("src/lap/verifiers/cover/batch_probe.py")
    assert _is_runtime_relevant_path("src/lap/verifiers/cover/data.py")
    assert _is_runtime_relevant_path("src/lap/verifiers/cover/w3_contracts.py")
    assert _is_runtime_relevant_path("scripts/w3_batch_probe.py")
    assert _is_runtime_relevant_path("pyproject.toml")
    assert _is_runtime_relevant_path("uv.lock")


def test_accepted_receipt_allows_w3_02_audit_loader_ownership_move():
    """Live public-seam proof: dirty W3-02 bridge_audit loader move must not invalidate."""
    audit = json.loads(Path("artifacts/w3/bridge_audit_manifest.json").read_text(encoding="utf-8"))
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain", "--", "src/lap/verifiers/cover/bridge_audit.py"], text=True
    )
    assert dirty.strip(), "expected preserved W3-02 bridge_audit ownership move in the worktree"
    bridge_diff = subprocess.check_output(
        ["git", "diff", "--", "src/lap/verifiers/cover/bridge_audit.py"],
        text=True,
    )
    assert "def load_and_validate_audit_manifest" in bridge_diff
    validated = validate_protocol_directory(
        Path("artifacts/w3/protocol"),
        require_complete=True,
        expected_audit_manifest_sha256=audit["manifest_sha256"],
        expected_target_inventory_fingerprint=audit["target"]["fingerprint"],
    )
    assert validated["batch_probe_receipt"]["content_hash"] == (
        "57878ecccba5e657eb8ccc3526693b4c2769e2e6c3da4bab412e148f7632d7a4"
    )


def test_canonical_receipt_rejects_bridge_audit_producer_symbol_drift(monkeypatch):
    receipt = json.loads(Path("artifacts/w3/protocol/batch_probe_receipt.json").read_text(encoding="utf-8"))
    committed = subprocess.check_output(
        ["git", "show", "HEAD:src/lap/verifiers/cover/bridge_audit.py"],
        text=True,
    )
    mutated = committed.replace(
        "def apply_audited_initialization(",
        "def apply_audited_initialization(  # capacity-defining drift\n",
        1,
    )
    assert mutated != committed
    monkeypatch.setattr(protocol_module, "_current_lap_revision", lambda: "b" * 40)

    def fake_run(args, **kwargs):
        if "merge-base" in args:
            return subprocess.CompletedProcess(args=args, returncode=0, stdout="", stderr="")
        raise AssertionError(args)

    def fake_check_output(args, **kwargs):
        if "diff" in args and "--name-only" in args:
            return ""
        if "status" in args:
            return " M src/lap/verifiers/cover/bridge_audit.py\n"
        raise AssertionError(args)

    monkeypatch.setattr(protocol_module.subprocess, "run", fake_run)
    monkeypatch.setattr(protocol_module.subprocess, "check_output", fake_check_output)
    monkeypatch.setattr(
        protocol_module,
        "_read_bridge_audit_source",
        lambda **kwargs: mutated if kwargs.get("worktree") else committed,
    )
    with pytest.raises(ValueError, match="uncommitted runtime-relevant changes"):
        validate_batch_probe_receipt(receipt, require_canonical=True)


def test_canonical_receipt_allows_unrelated_downstream_commits(monkeypatch):
    receipt = json.loads(Path("artifacts/w3/protocol/batch_probe_receipt.json").read_text(encoding="utf-8"))
    monkeypatch.setattr(protocol_module, "_current_lap_revision", lambda: "b" * 40)

    def fake_run(args, **kwargs):
        if "merge-base" in args:
            return subprocess.CompletedProcess(args=args, returncode=0, stdout="", stderr="")
        raise AssertionError(args)

    def fake_check_output(args, **kwargs):
        if "diff" in args and "--name-only" in args:
            return "\n".join(
                [
                    "artifacts/w3/protocol/batch_probe_receipt.json",
                    "src/lap/verifiers/cover/checkpoint.py",
                    "src/lap/verifiers/cover/checkpoint_test.py",
                    "src/lap/verifiers/cover/pipeline.py",
                ]
            )
        if "status" in args:
            return ""
        raise AssertionError(args)

    monkeypatch.setattr(protocol_module.subprocess, "run", fake_run)
    monkeypatch.setattr(protocol_module.subprocess, "check_output", fake_check_output)
    validate_batch_probe_receipt(receipt, require_canonical=True)


def test_canonical_receipt_rejects_capacity_defining_drift(monkeypatch):
    receipt = json.loads(Path("artifacts/w3/protocol/batch_probe_receipt.json").read_text(encoding="utf-8"))
    monkeypatch.setattr(protocol_module, "_current_lap_revision", lambda: "b" * 40)

    def fake_run(args, **kwargs):
        if "merge-base" in args:
            return subprocess.CompletedProcess(args=args, returncode=0, stdout="", stderr="")
        raise AssertionError(args)

    def fake_check_output(args, **kwargs):
        if "diff" in args and "--name-only" in args:
            return "src/lap/verifiers/cover/model.py\n"
        raise AssertionError(args)

    monkeypatch.setattr(protocol_module.subprocess, "run", fake_run)
    monkeypatch.setattr(protocol_module.subprocess, "check_output", fake_check_output)
    with pytest.raises(ValueError, match="relevant code or environment drift"):
        validate_batch_probe_receipt(receipt, require_canonical=True)


def test_phrase_manifest_has_exact_approved_bank():
    manifest = build_phrase_manifest()
    assert len(manifest["phrases"]) == 16
    assert manifest["phrases"][0] == "reach to the hole and insert the circular peg"
    assert manifest["phrases"][-1] == "place the square peg into the hole"
    assert len(set(manifest["phrases"])) == 16
    assert manifest["authorship"]["review_status"] == "approved"
    assert manifest["authorship"]["source"].endswith("#confirmed-offline-language-bank")
    assert manifest["content_hash"]


def test_language_selection_is_deterministic_and_shape_specific():
    first = select_training_phrase(seed=42, epoch=3, sample_id="sample", shape="circular")
    second = select_training_phrase(seed=42, epoch=3, sample_id="sample", shape="circular")
    assert first == second
    assert "circular" in first


def test_bootstrap_indices_use_fixed_shape_and_seed():
    indices = build_bootstrap_indices([f"episode_{i}" for i in range(8)], replicates=4)
    expected = np.array(
        [[7, 2, 0, 5, 5, 7, 7, 6], [2, 7, 3, 1, 5, 0, 5, 0], [7, 1, 1, 3, 6, 1, 1, 2], [5, 1, 4, 6, 0, 3, 5, 6]]
    )
    assert indices.shape == (4, 8)
    assert np.array_equal(indices, expected)


def test_sampler_indices_match_pytorch_distributed_sampler():
    sample_ids = [f"sample-{index}" for index in range(11)]
    actual = sampler_indices(sample_ids, epoch=3, rank=1)
    expected_sampler = DistributedSampler(
        list(range(len(sample_ids))), num_replicas=WORLD_SIZE, rank=1, shuffle=True, seed=TRAINING_SEED
    )
    expected_sampler.set_epoch(3)
    expected = [sample_ids[index] for index in list(expected_sampler)]
    assert actual["sample_ids"] == expected
    assert actual["seed"] == TRAINING_SEED
    assert actual["world_size"] == WORLD_SIZE
    assert actual["collision_diagnostics"]["adapts_batches"] is False


def test_shuffled_pairs_use_the_six_approved_exclusions():
    pairs, exclusions = build_shuffled_pairs(_canonical_validation_rows())
    assert len(pairs) == 1112
    assert exclusions == list(APPROVED_SHUFFLED_EXCLUSIONS)
    assert {pair["semantic_sample_id"] for pair in pairs} == {
        row["sample_id"] for row in _canonical_validation_rows()
    } - set(APPROVED_SHUFFLED_EXCLUSIONS)


def test_batch_probe_records_two_gpu_attempts_and_first_success():
    snapshots = [
        {
            "index": 0,
            "name": "NVIDIA RTX 6000 Ada Generation",
            "physical_total_memory_mib": 49140.0,
            "allocatable_total_memory_mib": 48502.69,
            "free_memory_mib": 47000.0,
        },
        {
            "index": 1,
            "name": "NVIDIA RTX 6000 Ada Generation",
            "physical_total_memory_mib": 49140.0,
            "allocatable_total_memory_mib": 48510.94,
            "free_memory_mib": 47100.0,
        },
    ]
    calls = []

    def step(batch_size, rank, _gpu):
        calls.append((batch_size, rank))
        if batch_size == 64:
            raise RuntimeError("CUDA out of memory")
        return _rank_success(batch_size, rank, _gpu)

    receipt = probe_batch_sizes(
        step,
        snapshot_fn=lambda: snapshots,
        model_identity={
            "backbone": "hf-hub:timm/ViT-L-16-SigLIP2-384",
            "canonical_target": True,
            "configuration_hash": "0" * 64,
            "audit_manifest_sha256": "1" * 64,
            "target_inventory_fingerprint": "2" * 64,
        },
    )
    assert receipt["attempted_batch_sizes"] == list(PROBE_ORDER[:2])
    assert receipt["selected_per_rank_batch_size"] == 32
    assert receipt["gpu_snapshot"] == snapshots
    assert receipt["successful_two_rank_optimizer_step"]["ranks"] == [0, 1]
    assert calls == [(64, 0), (32, 0), (32, 1)]


def test_tiny_probe_cannot_be_materialized_as_canonical(tmp_path):
    receipt = probe_batch_sizes(
        _rank_success,
        snapshot_fn=lambda: [
            {
                "index": 0,
                "name": "fixture",
                "driver_version": "test",
                "uuid": "GPU-test-0",
                "physical_total_memory_mib": 1,
                "allocatable_total_memory_mib": 1,
                "free_memory_mib": 1,
            },
            {
                "index": 1,
                "name": "fixture",
                "driver_version": "test",
                "uuid": "GPU-test-1",
                "physical_total_memory_mib": 1,
                "allocatable_total_memory_mib": 1,
                "free_memory_mib": 1,
            },
        ],
        model_identity={"backbone": "TinyFrozenBackbone", "canonical_target": False},
    )
    with pytest.raises(ValueError, match=r"canonical|Tiny"):
        materialize_protocol(
            tmp_path,
            train_manifest_hash="a" * 64,
            validation=_canonical_validation_rows(),
            batch_probe_receipt=receipt,
        )


def test_protocol_core_neither_requires_nor_invents_capacity_evidence(tmp_path):
    hashes = materialize_protocol(
        tmp_path,
        train_manifest_hash="a" * 64,
        validation=_canonical_validation_rows(),
    )

    assert "batch_probe_receipt" not in hashes
    assert not (tmp_path / "batch_probe_receipt.json").exists()
    validated = validate_protocol_directory(tmp_path, require_complete=False)
    assert "batch_probe_receipt" not in validated
    assert validated["protocol"]["batch"]["per_rank"] is None


def test_protocol_core_versions_the_complete_accepted_training_semantics(tmp_path):
    materialize_protocol(
        tmp_path,
        train_manifest_hash="a" * 64,
        validation=_canonical_validation_rows(),
    )

    protocol = validate_protocol_directory(tmp_path, require_complete=False)["protocol"]
    assert protocol["training"]["gradient_synchronization"] == "two_rank_ddp"
    assert protocol["training"]["embedding_all_gather"] is False
    assert protocol["training"]["gradient_accumulation_enlarges_negative_pool"] is False
    assert protocol["training"]["nonfinite_policy"] == "stop_run_and_prevent_accepted_publication"
    assert protocol["logit_scale"] == {
        "initial_logit_scale": 2.6592,
        "exponentiated_scale_bounds": [1.0, 100.0],
    }
    phrase_manifest = json.loads((tmp_path / "rephrase_manifest.json").read_text(encoding="utf-8"))
    assert phrase_manifest["selection"]["method"] == "sha256(seed:epoch:sample_id:shape) modulo variants_for_shape"


def test_protocol_core_rejects_rehashed_optimizer_semantic_drift(tmp_path):
    materialize_protocol(
        tmp_path,
        train_manifest_hash="a" * 64,
        validation=_canonical_validation_rows(),
    )
    protocol_path = tmp_path / "run_protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    protocol["optimization"]["learning_rate"] = 2e-6
    write_canonical_json(protocol_path, protocol)

    with pytest.raises(ValueError, match="optimization"):
        validate_protocol_directory(tmp_path, require_complete=False)


@pytest.mark.parametrize(
    ("field_path", "drifted_value"),
    [
        (("protocol_version",), "validation-selected-v2"),
        (("sampler", "shuffle"), False),
        (("training", "embedding_all_gather"), True),
        (("batch", "selection"), "validation_selected"),
        (("logit_scale", "initial_logit_scale"), 3.0),
        (("evaluation", "top_k"), [1]),
        (("status",), "complete"),
    ],
)
def test_protocol_core_rejects_rehashed_semantic_drift(tmp_path, field_path, drifted_value):
    materialize_protocol(
        tmp_path,
        train_manifest_hash="a" * 64,
        validation=_canonical_validation_rows(),
    )
    protocol_path = tmp_path / "run_protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    target = protocol
    for field in field_path[:-1]:
        target = target[field]
    target[field_path[-1]] = drifted_value
    write_canonical_json(protocol_path, protocol)

    with pytest.raises(ValueError, match=r"contract|status"):
        validate_protocol_directory(tmp_path, require_complete=False)


def test_protocol_core_rejects_rehashed_shuffled_pair_semantic_drift(tmp_path):
    validation = _canonical_validation_rows()
    materialize_protocol(
        tmp_path,
        train_manifest_hash="a" * 64,
        validation=validation,
    )
    shuffled_path = tmp_path / "shuffled_pairs.json"
    shuffled = json.loads(shuffled_path.read_text(encoding="utf-8"))
    shuffled["pairs"][0]["history_sample_id"] = shuffled["pairs"][0]["semantic_sample_id"]
    shuffled_hash = write_canonical_json(shuffled_path, shuffled)
    protocol_path = tmp_path / "run_protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    protocol["artifacts"]["shuffled_pairs"] = shuffled_hash
    write_canonical_json(protocol_path, protocol)

    with pytest.raises(ValueError, match="shuffled"):
        validate_protocol_directory(tmp_path, require_complete=False)


def test_protocol_core_rejects_rehashed_nearby_pair_semantic_drift(tmp_path):
    validation = _canonical_validation_rows()
    materialize_protocol(
        tmp_path,
        train_manifest_hash="a" * 64,
        validation=validation,
    )
    nearby_path = tmp_path / "nearby_pairs.json"
    nearby = json.loads(nearby_path.read_text(encoding="utf-8"))
    nearby["pairs"][0]["history_sample_id"] = nearby["pairs"][0]["semantic_sample_id"]
    nearby_hash = write_canonical_json(nearby_path, nearby)
    protocol_path = tmp_path / "run_protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    protocol["artifacts"]["nearby_pairs"] = nearby_hash
    write_canonical_json(protocol_path, protocol)

    with pytest.raises(ValueError, match="nearby"):
        validate_protocol_directory(tmp_path, require_complete=False, validation=validation)


def test_protocol_core_rejects_rehashed_bootstrap_semantic_drift(tmp_path):
    validation = _canonical_validation_rows()
    materialize_protocol(
        tmp_path,
        train_manifest_hash="a" * 64,
        validation=validation,
    )
    bootstrap_path = tmp_path / "bootstrap_indices.npy"
    bootstrap = np.load(bootstrap_path, allow_pickle=False)
    bootstrap[0, 0] = (bootstrap[0, 0] + 1) % 8
    np.save(bootstrap_path, bootstrap, allow_pickle=False)
    protocol_path = tmp_path / "run_protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    protocol["artifacts"]["bootstrap_indices"] = sha256_file(bootstrap_path)
    write_canonical_json(protocol_path, protocol)

    with pytest.raises(ValueError, match="bootstrap"):
        validate_protocol_directory(tmp_path, require_complete=False, validation=validation)


def test_protocol_core_binds_the_immutable_w2_validation_semantics(tmp_path):
    validation = _canonical_validation_rows()
    materialize_protocol(
        tmp_path,
        train_manifest_hash="a" * 64,
        validation=validation,
    )

    protocol = validate_protocol_directory(tmp_path, require_complete=False, validation=validation)["protocol"]
    assert (
        protocol["identities"]["validation_semantics_hash"]
        == "4d7d9ad47ab47f5c8ada380e8b9e7ceef6a22bc32d2c519d901b0e7b909f8e4f"
    )


def test_protocol_core_rejects_coherently_rehashed_noncanonical_w2_order(tmp_path):
    validation = _canonical_validation_rows()
    validation[0], validation[1] = validation[1], validation[0]
    materialize_protocol(
        tmp_path,
        train_manifest_hash="a" * 64,
        validation=validation,
    )

    with pytest.raises(ValueError, match="immutable W2"):
        validate_protocol_directory(tmp_path, require_complete=False)


def test_protocol_core_repeated_materialization_is_byte_identical(tmp_path):
    validation = _canonical_validation_rows()
    first = tmp_path / "first"
    second = tmp_path / "second"
    first_hashes = materialize_protocol(
        first,
        train_manifest_hash="a" * 64,
        validation=validation,
    )
    second_hashes = materialize_protocol(
        second,
        train_manifest_hash="a" * 64,
        validation=validation,
    )

    assert first_hashes == second_hashes
    assert {path.name: path.read_bytes() for path in first.iterdir()} == {
        path.name: path.read_bytes() for path in second.iterdir()
    }


def test_protocol_core_rejects_rehashed_phrase_contract_drift(tmp_path):
    materialize_protocol(
        tmp_path,
        train_manifest_hash="a" * 64,
        validation=_canonical_validation_rows(),
    )
    phrase_path = tmp_path / "rephrase_manifest.json"
    phrase_manifest = json.loads(phrase_path.read_text(encoding="utf-8"))
    phrase_manifest["validation_language"] = "training_rephrase"
    phrase_hash = write_canonical_json(phrase_path, phrase_manifest)
    protocol_path = tmp_path / "run_protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    protocol["artifacts"]["rephrase_manifest"] = phrase_hash
    protocol["identities"]["phrase_manifest_hash"] = phrase_hash
    write_canonical_json(protocol_path, protocol)

    with pytest.raises(ValueError, match="phrase"):
        validate_protocol_directory(tmp_path, require_complete=False)


def test_materialized_protocol_is_hashable_and_rejects_drift(tmp_path):
    snapshot = _asset_snapshot(tmp_path)
    receipt = probe_batch_sizes(
        _rank_success,
        snapshot_fn=_canonical_gpu_snapshots,
        model_identity=_canonical_model_identity(snapshot=snapshot),
    )
    materialize_protocol(
        tmp_path,
        train_manifest_hash="a" * 64,
        validation=_canonical_validation_rows(),
        batch_probe_receipt=receipt,
    )
    validated = validate_protocol_directory(tmp_path)
    assert validated["protocol"]["training"]["seed"] == TRAINING_SEED
    assert validated["protocol"]["evaluation"]["seed"] == EVALUATION_SEED
    assert validated["protocol"]["sampler"]["world_size"] == WORLD_SIZE

    run_protocol = tmp_path / "run_protocol.json"
    payload = json.loads(run_protocol.read_text(encoding="utf-8"))
    payload["evaluation"]["seed"] = 7
    run_protocol.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match=r"hash|protocol|seed"):
        validate_protocol_directory(tmp_path)


def test_canonical_batch_receipt_rejects_rehashed_identity_drift():
    receipt = probe_batch_sizes(
        _rank_success,
        snapshot_fn=_canonical_gpu_snapshots,
        model_identity=_canonical_model_identity(),
    )

    receipt["environment"]["uv_lock_sha256"] = "f" * 64
    unsigned = {key: value for key, value in receipt.items() if key != "content_hash"}
    receipt["content_hash"] = content_hash(unsigned)

    with pytest.raises(ValueError, match=r"uv.lock"):
        validate_batch_probe_receipt(receipt, require_canonical=True)


@pytest.mark.parametrize("field", ["tokenizer_fingerprint", "preprocessing_fingerprint"])
def test_canonical_batch_receipt_rejects_rehashed_runtime_asset_drift(field):
    receipt = json.loads(CANONICAL_RECEIPT.read_text(encoding="utf-8"))
    receipt["environment"]["lap_revision"] = subprocess.check_output(
        ["git", "-C", str(Path(__file__).resolve().parents[4]), "rev-parse", "HEAD"],
        text=True,
    ).strip()
    receipt["model"]["preprocessing_fingerprint"] = "fd7ffb201d3e3eec5770fa313d96af1c031403cab51ae43ee734ed1d431c3b14"
    receipt["model"]["tokenizer_fingerprint"] = "e97a61cde12740e6031fa02bcd4aa39d9b0e75f1b8fe6c1ef9dad92bc4b09575"
    receipt["model"][field] = "0" * 64
    unsigned = {key: value for key, value in receipt.items() if key != "content_hash"}
    receipt["content_hash"] = content_hash(unsigned)

    with pytest.raises(ValueError, match=field.replace("_", " ")):
        validate_batch_probe_receipt(receipt, require_canonical=True)


def test_canonical_batch_receipt_must_match_supplied_audit(tmp_path):
    snapshot = _asset_snapshot(tmp_path)
    receipt = probe_batch_sizes(
        _rank_success,
        snapshot_fn=_canonical_gpu_snapshots,
        model_identity=_canonical_model_identity(audit_manifest_sha256="a" * 64, snapshot=snapshot),
    )

    with pytest.raises(ValueError, match="audit manifest"):
        validate_batch_probe_receipt(
            receipt,
            require_canonical=True,
            expected_audit_manifest_sha256="b" * 64,
        )
