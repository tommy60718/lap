import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from lap.verifiers.cover.action_adapter import NormalizationArtifact
from lap.verifiers.cover.action_adapter import NormalizationProvenance
from lap.verifiers.cover.action_adapter import build_action_histories
from lap.verifiers.cover.action_adapter import convert_absolute_targets_to_relative_rows
from lap.verifiers.cover.action_adapter import derive_normalization_artifact

CONSUMER_FIXTURE = Path(__file__).parent / "testdata" / "ur5e_cover_action_adapter_v1.json"


def test_builds_one_candidate_history_with_empty_past() -> None:
    candidate_chunks = np.zeros((1, 5, 7), dtype=np.float64)
    candidate_chunks[0, :, 6] = [0.1, 0.2, 0.3, 0.4, 0.9]
    normalization = NormalizationArtifact(
        q01=np.zeros(6, dtype=np.float64),
        q99=np.ones(6, dtype=np.float64),
    )

    result = build_action_histories(
        reference_position=np.zeros(3, dtype=np.float64),
        reference_rotation_vector=np.zeros(3, dtype=np.float64),
        candidate_chunks=candidate_chunks,
        processed_past=np.empty((0, 7), dtype=np.float64),
        normalization=normalization,
    )

    expected = np.full((1, 10, 7), -5.0, dtype=np.float32)
    expected[0, 6:10, :6] = -1.0
    expected[0, 6:10, 6] = [0.1, 0.2, 0.3, 0.4]

    np.testing.assert_array_equal(result.histories, expected)
    assert result.histories.dtype == np.float32
    assert result.first_future_index == 6


def test_rejects_empty_candidate_batch() -> None:
    with pytest.raises(ValueError, match="at least one candidate"):
        build_action_histories(
            reference_position=np.zeros(3, dtype=np.float64),
            reference_rotation_vector=np.zeros(3, dtype=np.float64),
            candidate_chunks=np.empty((0, 4, 7), dtype=np.float64),
            processed_past=np.empty((0, 7), dtype=np.float64),
            normalization=NormalizationArtifact(
                q01=np.zeros(6, dtype=np.float64),
                q99=np.ones(6, dtype=np.float64),
            ),
        )


def test_rejects_candidate_with_fewer_than_four_future_actions() -> None:
    with pytest.raises(ValueError, match="at least four future actions"):
        build_action_histories(
            reference_position=np.zeros(3, dtype=np.float64),
            reference_rotation_vector=np.zeros(3, dtype=np.float64),
            candidate_chunks=np.zeros((1, 3, 7), dtype=np.float64),
            processed_past=np.empty((0, 7), dtype=np.float64),
            normalization=NormalizationArtifact(
                q01=np.zeros(6, dtype=np.float64),
                q99=np.ones(6, dtype=np.float64),
            ),
        )


def test_rejects_candidate_with_wrong_action_width() -> None:
    with pytest.raises(ValueError, match=r"shape \[M, H, 7\]"):
        build_action_histories(
            reference_position=np.zeros(3, dtype=np.float64),
            reference_rotation_vector=np.zeros(3, dtype=np.float64),
            candidate_chunks=np.zeros((1, 4, 6), dtype=np.float64),
            processed_past=np.empty((0, 7), dtype=np.float64),
            normalization=NormalizationArtifact(
                q01=np.zeros(6, dtype=np.float64),
                q99=np.ones(6, dtype=np.float64),
            ),
        )


def test_rejects_nonfinite_candidate_action() -> None:
    candidate_chunks = np.zeros((1, 4, 7), dtype=np.float64)
    candidate_chunks[0, 0, 0] = np.nan

    with pytest.raises(ValueError, match="candidate_chunks must be finite"):
        build_action_histories(
            reference_position=np.zeros(3, dtype=np.float64),
            reference_rotation_vector=np.zeros(3, dtype=np.float64),
            candidate_chunks=candidate_chunks,
            processed_past=np.empty((0, 7), dtype=np.float64),
            normalization=NormalizationArtifact(
                q01=np.zeros(6, dtype=np.float64),
                q99=np.ones(6, dtype=np.float64),
            ),
        )


def test_rejects_malformed_processed_past() -> None:
    with pytest.raises(ValueError, match=r"processed_past must have shape \[P, 7\]"):
        build_action_histories(
            reference_position=np.zeros(3, dtype=np.float64),
            reference_rotation_vector=np.zeros(3, dtype=np.float64),
            candidate_chunks=np.zeros((1, 4, 7), dtype=np.float64),
            processed_past=np.empty((0, 6), dtype=np.float64),
            normalization=NormalizationArtifact(
                q01=np.zeros(6, dtype=np.float64),
                q99=np.ones(6, dtype=np.float64),
            ),
        )


@pytest.mark.parametrize(
    ("reference_position", "reference_rotation_vector"),
    [
        (np.zeros(2, dtype=np.float64), np.zeros(3, dtype=np.float64)),
        (np.zeros(3, dtype=np.float64), np.zeros(2, dtype=np.float64)),
    ],
)
def test_rejects_malformed_reference_pose(
    reference_position: np.ndarray,
    reference_rotation_vector: np.ndarray,
) -> None:
    with pytest.raises(ValueError, match="reference pose must contain two finite 3-vectors"):
        build_action_histories(
            reference_position=reference_position,
            reference_rotation_vector=reference_rotation_vector,
            candidate_chunks=np.zeros((1, 4, 7), dtype=np.float64),
            processed_past=np.empty((0, 7), dtype=np.float64),
            normalization=NormalizationArtifact(
                q01=np.zeros(6, dtype=np.float64),
                q99=np.ones(6, dtype=np.float64),
            ),
        )


def test_rejects_nonfinite_reference_pose() -> None:
    reference_position = np.zeros(3, dtype=np.float64)
    reference_position[0] = np.nan

    with pytest.raises(ValueError, match="reference pose must contain two finite 3-vectors"):
        build_action_histories(
            reference_position=reference_position,
            reference_rotation_vector=np.zeros(3, dtype=np.float64),
            candidate_chunks=np.zeros((1, 4, 7), dtype=np.float64),
            processed_past=np.empty((0, 7), dtype=np.float64),
            normalization=NormalizationArtifact(
                q01=np.zeros(6, dtype=np.float64),
                q99=np.ones(6, dtype=np.float64),
            ),
        )


def test_rejects_more_than_six_processed_past_actions() -> None:
    with pytest.raises(ValueError, match="at most six actions"):
        build_action_histories(
            reference_position=np.zeros(3, dtype=np.float64),
            reference_rotation_vector=np.zeros(3, dtype=np.float64),
            candidate_chunks=np.zeros((1, 4, 7), dtype=np.float64),
            processed_past=np.zeros((7, 7), dtype=np.float64),
            normalization=NormalizationArtifact(
                q01=np.zeros(6, dtype=np.float64),
                q99=np.ones(6, dtype=np.float64),
            ),
        )


def test_rejects_nonfinite_processed_past_action() -> None:
    processed_past = np.zeros((1, 7), dtype=np.float64)
    processed_past[0, 0] = np.inf

    with pytest.raises(ValueError, match="processed_past must be finite"):
        build_action_histories(
            reference_position=np.zeros(3, dtype=np.float64),
            reference_rotation_vector=np.zeros(3, dtype=np.float64),
            candidate_chunks=np.zeros((1, 4, 7), dtype=np.float64),
            processed_past=processed_past,
            normalization=NormalizationArtifact(
                q01=np.zeros(6, dtype=np.float64),
                q99=np.ones(6, dtype=np.float64),
            ),
        )


def test_converts_worked_ur5e_geometry_and_preserves_gripper() -> None:
    reference_position = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    reference_rotation_vector = np.array([np.pi / 2.0, 0.0, 0.0], dtype=np.float64)

    # Analytic composition: R_target = Rz(90 deg) * Rx(90 deg), whose
    # rotation vector is 120 deg about [1, 1, 1].
    target_rotation_component = 2.0 * np.pi / (3.0 * np.sqrt(3.0))
    candidate_chunks = np.zeros((1, 4, 7), dtype=np.float64)
    candidate_chunks[0, :, :3] = [1.25, 1.5, 4.0]
    candidate_chunks[0, :, 3:6] = target_rotation_component
    candidate_chunks[0, :, 6] = [0.2, 0.4, 0.6, 0.8]

    normalization = NormalizationArtifact(
        q01=np.zeros(6, dtype=np.float64),
        q99=np.array(
            [
                2.0 - 1e-6,
                2.0 - 1e-6,
                2.0 - 1e-6,
                np.pi - 1e-6,
                np.pi - 1e-6,
                np.pi - 1e-6,
            ],
            dtype=np.float64,
        ),
    )

    result = build_action_histories(
        reference_position=reference_position,
        reference_rotation_vector=reference_rotation_vector,
        candidate_chunks=candidate_chunks,
        processed_past=np.empty((0, 7), dtype=np.float64),
        normalization=normalization,
    )

    expected_future = np.array(
        [-0.75, -1.5, 0.0, -1.0, -1.0, 0.0],
        dtype=np.float32,
    )
    expected_futures = np.repeat(expected_future[None, :], repeats=4, axis=0)
    np.testing.assert_allclose(result.histories[0, 6:10, :6], expected_futures, atol=1e-6)
    np.testing.assert_allclose(result.histories[0, 6:10, 6], [0.2, 0.4, 0.6, 0.8], atol=1e-6)


def test_ignores_later_futures_without_mutating_inputs() -> None:
    candidate_chunks = np.zeros((1, 5, 7), dtype=np.float64)
    candidate_chunks[0, 4, :6] = 9.0
    candidate_chunks[0, 4, 6] = 0.9
    processed_past = np.empty((0, 7), dtype=np.float64)
    original_candidates = candidate_chunks.copy()
    original_past = processed_past.copy()
    normalization = NormalizationArtifact(
        q01=np.zeros(6, dtype=np.float64),
        q99=np.ones(6, dtype=np.float64),
    )

    first = build_action_histories(
        reference_position=np.zeros(3, dtype=np.float64),
        reference_rotation_vector=np.zeros(3, dtype=np.float64),
        candidate_chunks=candidate_chunks,
        processed_past=processed_past,
        normalization=normalization,
    )
    later_future_changed = candidate_chunks.copy()
    later_future_changed[0, 4, :6] = -9.0
    second = build_action_histories(
        reference_position=np.zeros(3, dtype=np.float64),
        reference_rotation_vector=np.zeros(3, dtype=np.float64),
        candidate_chunks=later_future_changed,
        processed_past=processed_past,
        normalization=normalization,
    )

    np.testing.assert_array_equal(first.histories, second.histories)
    np.testing.assert_array_equal(processed_past, original_past)
    np.testing.assert_array_equal(candidate_chunks, original_candidates)
    assert (
        first.histories.tobytes()
        == build_action_histories(
            reference_position=np.zeros(3, dtype=np.float64),
            reference_rotation_vector=np.zeros(3, dtype=np.float64),
            candidate_chunks=candidate_chunks,
            processed_past=processed_past,
            normalization=normalization,
        ).histories.tobytes()
    )


def test_rejects_malformed_normalization_quantiles() -> None:
    with pytest.raises(ValueError, match="normalization quantiles must be finite 6-vectors"):
        build_action_histories(
            reference_position=np.zeros(3, dtype=np.float64),
            reference_rotation_vector=np.zeros(3, dtype=np.float64),
            candidate_chunks=np.zeros((1, 4, 7), dtype=np.float64),
            processed_past=np.empty((0, 7), dtype=np.float64),
            normalization=NormalizationArtifact(
                q01=np.zeros(5, dtype=np.float64),
                q99=np.ones(6, dtype=np.float64),
            ),
        )


def test_rejects_nonfinite_normalization_quantiles() -> None:
    q99 = np.ones(6, dtype=np.float64)
    q99[0] = np.nan

    with pytest.raises(ValueError, match="normalization quantiles must be finite 6-vectors"):
        build_action_histories(
            reference_position=np.zeros(3, dtype=np.float64),
            reference_rotation_vector=np.zeros(3, dtype=np.float64),
            candidate_chunks=np.zeros((1, 4, 7), dtype=np.float64),
            processed_past=np.empty((0, 7), dtype=np.float64),
            normalization=NormalizationArtifact(
                q01=np.zeros(6, dtype=np.float64),
                q99=q99,
            ),
        )


def test_derives_linear_quantiles_from_first_six_training_dimensions() -> None:
    values = np.arange(100, dtype=np.float64)
    training_rows = np.column_stack([values + offset for offset in range(6)] + [values * 1000.0])
    provenance = NormalizationProvenance(
        dataset_id="reaching_peg_in_hole_64",
        dataset_schema_version="osx_dual_verifier_raw_v0",
        split_manifest_hash="a" * 64,
        shared_15hz_index_hash="b" * 64,
    )

    artifact = derive_normalization_artifact(
        training_relative_rows=training_rows,
        provenance=provenance,
    )

    np.testing.assert_allclose(artifact.q01, np.arange(6) + 0.99)
    np.testing.assert_allclose(artifact.q99, np.arange(6) + 98.01)
    assert artifact.fit_row_count == 100
    assert artifact.quantile_method == "numpy.quantile(method='linear')"
    assert artifact.quantile_library_version == f"numpy=={np.__version__}"


@pytest.mark.parametrize(
    "training_rows",
    [
        np.empty((0, 7), dtype=np.float64),
        np.empty((4, 6), dtype=np.float64),
    ],
)
def test_rejects_empty_or_malformed_training_rows(training_rows: np.ndarray) -> None:
    with pytest.raises(ValueError, match=r"training_relative_rows must have shape \[N, 7\] with N >= 1"):
        derive_normalization_artifact(
            training_relative_rows=training_rows,
            provenance=NormalizationProvenance(
                dataset_id="reaching_peg_in_hole_64",
                dataset_schema_version="osx_dual_verifier_raw_v0",
                split_manifest_hash="a" * 64,
                shared_15hz_index_hash="b" * 64,
            ),
        )


def test_rejects_nonfinite_training_rows() -> None:
    training_rows = np.zeros((10, 7), dtype=np.float64)
    training_rows[4, 2] = np.nan

    with pytest.raises(ValueError, match="training_relative_rows must be finite"):
        derive_normalization_artifact(
            training_relative_rows=training_rows,
            provenance=NormalizationProvenance(
                dataset_id="reaching_peg_in_hole_64",
                dataset_schema_version="osx_dual_verifier_raw_v0",
                split_manifest_hash="a" * 64,
                shared_15hz_index_hash="b" * 64,
            ),
        )


def test_rejects_padding_rows_in_fit_population() -> None:
    values = np.arange(100, dtype=np.float64)
    training_rows = np.column_stack([values + offset for offset in range(6)] + [values])
    training_rows[50] = -5.0

    with pytest.raises(ValueError, match="padding rows must be excluded"):
        derive_normalization_artifact(
            training_relative_rows=training_rows,
            provenance=NormalizationProvenance(
                dataset_id="reaching_peg_in_hole_64",
                dataset_schema_version="osx_dual_verifier_raw_v0",
                split_manifest_hash="a" * 64,
                shared_15hz_index_hash="b" * 64,
            ),
        )


def test_rejects_degenerate_quantile_dimension() -> None:
    values = np.arange(100, dtype=np.float64)
    training_rows = np.column_stack([values + offset for offset in range(6)] + [values])
    training_rows[:, 2] = 0.25

    with pytest.raises(ValueError, match=r"q99 must be greater than q01.*dimension 2"):
        derive_normalization_artifact(
            training_relative_rows=training_rows,
            provenance=NormalizationProvenance(
                dataset_id="reaching_peg_in_hole_64",
                dataset_schema_version="osx_dual_verifier_raw_v0",
                split_manifest_hash="a" * 64,
                shared_15hz_index_hash="b" * 64,
            ),
        )


@pytest.mark.parametrize(
    ("dataset_id", "dataset_schema_version", "split_hash", "index_hash"),
    [
        ("", "osx_dual_verifier_raw_v0", "a" * 64, "b" * 64),
        ("reaching_peg_in_hole_64", "", "a" * 64, "b" * 64),
        ("reaching_peg_in_hole_64", "osx_dual_verifier_raw_v0", "short", "b" * 64),
        ("reaching_peg_in_hole_64", "osx_dual_verifier_raw_v0", "a" * 64, "G" * 64),
    ],
)
def test_rejects_missing_or_invalid_provenance(
    dataset_id: str,
    dataset_schema_version: str,
    split_hash: str,
    index_hash: str,
) -> None:
    with pytest.raises(ValueError, match="normalization provenance"):
        NormalizationProvenance(
            dataset_id=dataset_id,
            dataset_schema_version=dataset_schema_version,
            split_manifest_hash=split_hash,
            shared_15hz_index_hash=index_hash,
        )


def test_serializes_complete_deterministic_artifact_and_round_trips() -> None:
    values = np.arange(100, dtype=np.float64)
    training_rows = np.column_stack([values + offset for offset in range(6)] + [values])
    artifact = derive_normalization_artifact(
        training_relative_rows=training_rows,
        provenance=NormalizationProvenance(
            dataset_id="reaching_peg_in_hole_64",
            dataset_schema_version="osx_dual_verifier_raw_v0",
            split_manifest_hash="a" * 64,
            shared_15hz_index_hash="b" * 64,
        ),
    )

    serialized = artifact.to_json()
    payload = json.loads(serialized)

    assert serialized == artifact.to_json()
    assert payload["schema"] == "osx_cover_normalization_v0"
    assert payload["representation"]["id"] == "ur5e_cover_relative_eef_v1"
    assert payload["representation"]["action_order"] == [
        "dx",
        "dy",
        "dz",
        "rotation_x",
        "rotation_y",
        "rotation_z",
        "gripper",
    ]
    assert payload["normalization"]["epsilon"] == 1e-6
    assert payload["normalization"]["clipping"] is False
    assert payload["normalization"]["gripper"] == "unchanged"
    assert payload["fit"] == {
        "padding_excluded": True,
        "row_count": 100,
        "split": "train",
    }
    assert payload["provenance"]["split_manifest_hash"] == "a" * 64
    assert payload["provenance"]["shared_15hz_index_hash"] == "b" * 64
    assert len(payload["adapter_configuration_hash"]) == 64
    assert len(payload["content_hash"]) == 64

    loaded = NormalizationArtifact.from_json(serialized)
    np.testing.assert_array_equal(loaded.q01, artifact.q01)
    np.testing.assert_array_equal(loaded.q99, artifact.q99)
    assert loaded.to_json() == serialized
    repeated = derive_normalization_artifact(
        training_relative_rows=training_rows,
        provenance=artifact.provenance,
    )
    assert repeated.to_json() == serialized


def test_rejects_tampered_artifact_content_hash() -> None:
    values = np.arange(100, dtype=np.float64)
    artifact = derive_normalization_artifact(
        training_relative_rows=np.column_stack([values + offset for offset in range(6)] + [values]),
        provenance=NormalizationProvenance(
            dataset_id="reaching_peg_in_hole_64",
            dataset_schema_version="osx_dual_verifier_raw_v0",
            split_manifest_hash="a" * 64,
            shared_15hz_index_hash="b" * 64,
        ),
    )
    payload = json.loads(artifact.to_json())
    payload["normalization"]["q01"][0] += 1.0

    with pytest.raises(ValueError, match="content hash mismatch"):
        NormalizationArtifact.from_json(json.dumps(payload))


def test_rejects_incompatible_adapter_configuration() -> None:
    values = np.arange(100, dtype=np.float64)
    artifact = derive_normalization_artifact(
        training_relative_rows=np.column_stack([values + offset for offset in range(6)] + [values]),
        provenance=NormalizationProvenance(
            dataset_id="reaching_peg_in_hole_64",
            dataset_schema_version="osx_dual_verifier_raw_v0",
            split_manifest_hash="a" * 64,
            shared_15hz_index_hash="b" * 64,
        ),
    )
    payload = json.loads(artifact.to_json())
    payload["adapter_configuration_hash"] = "c" * 64
    payload_without_hash = {key: value for key, value in payload.items() if key != "content_hash"}
    canonical = json.dumps(payload_without_hash, sort_keys=True, separators=(",", ":"))
    payload["content_hash"] = hashlib.sha256(canonical.encode()).hexdigest()

    with pytest.raises(ValueError, match="adapter configuration mismatch"):
        NormalizationArtifact.from_json(json.dumps(payload))


def test_rejects_artifact_with_missing_metadata() -> None:
    values = np.arange(100, dtype=np.float64)
    artifact = derive_normalization_artifact(
        training_relative_rows=np.column_stack([values + offset for offset in range(6)] + [values]),
        provenance=NormalizationProvenance(
            dataset_id="reaching_peg_in_hole_64",
            dataset_schema_version="osx_dual_verifier_raw_v0",
            split_manifest_hash="a" * 64,
            shared_15hz_index_hash="b" * 64,
        ),
    )
    payload = json.loads(artifact.to_json())
    del payload["provenance"]["dataset_id"]
    payload_without_hash = {key: value for key, value in payload.items() if key != "content_hash"}
    canonical = json.dumps(payload_without_hash, sort_keys=True, separators=(",", ":"))
    payload["content_hash"] = hashlib.sha256(canonical.encode()).hexdigest()

    with pytest.raises(ValueError, match="missing or invalid metadata"):
        NormalizationArtifact.from_json(json.dumps(payload))


def test_rejects_inconsistent_source_artifact_hashes() -> None:
    values = np.arange(100, dtype=np.float64)
    artifact = derive_normalization_artifact(
        training_relative_rows=np.column_stack([values + offset for offset in range(6)] + [values]),
        provenance=NormalizationProvenance(
            dataset_id="reaching_peg_in_hole_64",
            dataset_schema_version="osx_dual_verifier_raw_v0",
            split_manifest_hash="a" * 64,
            shared_15hz_index_hash="b" * 64,
        ),
    )
    payload = json.loads(artifact.to_json())
    payload["provenance"]["source_artifact_hashes"]["split_manifest"] = "c" * 64
    payload_without_hash = {key: value for key, value in payload.items() if key != "content_hash"}
    canonical = json.dumps(payload_without_hash, sort_keys=True, separators=(",", ":"))
    payload["content_hash"] = hashlib.sha256(canonical.encode()).hexdigest()

    with pytest.raises(ValueError, match="source artifact hash mismatch"):
        NormalizationArtifact.from_json(json.dumps(payload))


def test_rejects_loaded_artifact_with_degenerate_quantiles() -> None:
    values = np.arange(100, dtype=np.float64)
    artifact = derive_normalization_artifact(
        training_relative_rows=np.column_stack([values + offset for offset in range(6)] + [values]),
        provenance=NormalizationProvenance(
            dataset_id="reaching_peg_in_hole_64",
            dataset_schema_version="osx_dual_verifier_raw_v0",
            split_manifest_hash="a" * 64,
            shared_15hz_index_hash="b" * 64,
        ),
    )
    payload = json.loads(artifact.to_json())
    payload["normalization"]["q99"][3] = payload["normalization"]["q01"][3]
    payload_without_hash = {key: value for key, value in payload.items() if key != "content_hash"}
    canonical = json.dumps(payload_without_hash, sort_keys=True, separators=(",", ":"))
    payload["content_hash"] = hashlib.sha256(canonical.encode()).hexdigest()

    with pytest.raises(ValueError, match="q99 must be greater than q01"):
        NormalizationArtifact.from_json(json.dumps(payload))


def test_rejects_incompatible_embedded_representation() -> None:
    values = np.arange(100, dtype=np.float64)
    artifact = derive_normalization_artifact(
        training_relative_rows=np.column_stack([values + offset for offset in range(6)] + [values]),
        provenance=NormalizationProvenance(
            dataset_id="reaching_peg_in_hole_64",
            dataset_schema_version="osx_dual_verifier_raw_v0",
            split_manifest_hash="a" * 64,
            shared_15hz_index_hash="b" * 64,
        ),
    )
    payload = json.loads(artifact.to_json())
    payload["representation"]["id"] = "different_representation"
    payload_without_hash = {key: value for key, value in payload.items() if key != "content_hash"}
    canonical = json.dumps(payload_without_hash, sort_keys=True, separators=(",", ":"))
    payload["content_hash"] = hashlib.sha256(canonical.encode()).hexdigest()

    with pytest.raises(ValueError, match="embedded representation mismatch"):
        NormalizationArtifact.from_json(json.dumps(payload))


def test_places_one_committed_past_action_before_every_candidate_future() -> None:
    candidate_chunks = np.zeros((2, 4, 7), dtype=np.float64)
    processed_past = np.array([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]], dtype=np.float64)

    result = build_action_histories(
        reference_position=np.zeros(3, dtype=np.float64),
        reference_rotation_vector=np.zeros(3, dtype=np.float64),
        candidate_chunks=candidate_chunks,
        processed_past=processed_past,
        normalization=NormalizationArtifact(
            q01=np.zeros(6, dtype=np.float64),
            q99=np.ones(6, dtype=np.float64),
        ),
    )

    np.testing.assert_array_equal(result.histories[:, :5], -5.0)
    expected_past = np.repeat(processed_past, repeats=2, axis=0)
    np.testing.assert_allclose(result.histories[:, 5], expected_past)
    np.testing.assert_array_equal(result.histories[0, :6], result.histories[1, :6])
    assert result.first_future_index == 6


def test_rejects_degenerate_runtime_normalization_artifact() -> None:
    with pytest.raises(ValueError, match="q99 must be greater than q01"):
        build_action_histories(
            reference_position=np.zeros(3, dtype=np.float64),
            reference_rotation_vector=np.zeros(3, dtype=np.float64),
            candidate_chunks=np.zeros((1, 4, 7), dtype=np.float64),
            processed_past=np.empty((0, 7), dtype=np.float64),
            normalization=NormalizationArtifact(
                q01=np.zeros(6, dtype=np.float64),
                q99=np.zeros(6, dtype=np.float64),
            ),
        )


def test_keeps_six_committed_past_actions_in_chronological_order() -> None:
    processed_past = np.arange(42, dtype=np.float64).reshape(6, 7)
    processed_past[:, 6] = np.linspace(0.0, 1.0, 6)
    result = build_action_histories(
        reference_position=np.zeros(3, dtype=np.float64),
        reference_rotation_vector=np.zeros(3, dtype=np.float64),
        candidate_chunks=np.zeros((3, 4, 7), dtype=np.float64),
        processed_past=processed_past,
        normalization=NormalizationArtifact(
            q01=np.zeros(6, dtype=np.float64),
            q99=np.ones(6, dtype=np.float64),
        ),
    )

    expected_past = np.repeat(processed_past[None, :, :], repeats=3, axis=0)
    np.testing.assert_allclose(result.histories[:, :6], expected_past, atol=1e-6)
    assert result.first_future_index == 6


def test_matches_frozen_exporter_and_runtime_consumer_fixture() -> None:
    fixture = json.loads(CONSUMER_FIXTURE.read_text())
    candidate_chunks = np.asarray(fixture["input"]["candidate_chunks"], dtype=np.float64)
    processed_past = np.asarray(fixture["input"]["processed_past"], dtype=np.float64)
    original_candidates = candidate_chunks.copy()
    original_past = processed_past.copy()

    result = build_action_histories(
        reference_position=np.asarray(fixture["input"]["reference_position"], dtype=np.float64),
        reference_rotation_vector=np.asarray(
            fixture["input"]["reference_rotation_vector"],
            dtype=np.float64,
        ),
        candidate_chunks=candidate_chunks,
        processed_past=processed_past,
        normalization=NormalizationArtifact(
            q01=np.asarray(fixture["input"]["q01"], dtype=np.float64),
            q99=np.asarray(fixture["input"]["q99"], dtype=np.float64),
        ),
    )

    expected = np.asarray(fixture["expected"]["histories"], dtype=np.float32)
    np.testing.assert_allclose(result.histories, expected, atol=1e-6)
    np.testing.assert_array_equal(candidate_chunks, original_candidates)
    np.testing.assert_array_equal(processed_past, original_past)
    assert result.histories.shape == (2, 10, 7)
    assert result.histories.dtype == np.float32
    assert result.first_future_index == fixture["expected"]["first_future_index"]
    selected_index = fixture["expected"]["selected_candidate_index"]
    np.testing.assert_allclose(
        result.histories[selected_index, result.first_future_index],
        fixture["expected"]["selected_pending_action"],
        atol=1e-6,
    )
    np.testing.assert_array_equal(
        result.histories[:, :, 0] == -5.0,
        fixture["expected"]["padding_mask"],
    )


def test_rejects_candidate_gripper_outside_absolute_command_range() -> None:
    candidate_chunks = np.zeros((1, 4, 7), dtype=np.float64)
    candidate_chunks[0, 0, 6] = 1.1

    with pytest.raises(ValueError, match=r"candidate gripper must be within \[0, 1\]"):
        build_action_histories(
            reference_position=np.zeros(3, dtype=np.float64),
            reference_rotation_vector=np.zeros(3, dtype=np.float64),
            candidate_chunks=candidate_chunks,
            processed_past=np.empty((0, 7), dtype=np.float64),
            normalization=NormalizationArtifact(
                q01=np.zeros(6, dtype=np.float64),
                q99=np.ones(6, dtype=np.float64),
            ),
        )


def test_rejects_committed_past_gripper_outside_absolute_command_range() -> None:
    processed_past = np.zeros((1, 7), dtype=np.float64)
    processed_past[0, 6] = -0.1

    with pytest.raises(ValueError, match=r"processed past gripper must be within \[0, 1\]"):
        build_action_histories(
            reference_position=np.zeros(3, dtype=np.float64),
            reference_rotation_vector=np.zeros(3, dtype=np.float64),
            candidate_chunks=np.zeros((1, 4, 7), dtype=np.float64),
            processed_past=processed_past,
            normalization=NormalizationArtifact(
                q01=np.zeros(6, dtype=np.float64),
                q99=np.ones(6, dtype=np.float64),
            ),
        )


def test_relative_rows_subtract_translation_and_preserve_absolute_gripper() -> None:
    absolute_targets = np.array(
        [
            [1.25, 1.5, 4.0, 0.0, 0.0, 0.0, 0.2],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.9],
        ],
        dtype=np.float64,
    )

    relative = convert_absolute_targets_to_relative_rows(
        reference_positions=np.array([1.0, 2.0, 3.0], dtype=np.float64),
        reference_rotation_vectors=np.zeros(3, dtype=np.float64),
        absolute_targets=absolute_targets,
    )

    expected = np.array(
        [
            [0.25, -0.5, 1.0, 0.0, 0.0, 0.0, 0.2],
            [-1.0, -2.0, -3.0, 0.0, 0.0, 0.0, 0.9],
        ],
        dtype=np.float64,
    )
    np.testing.assert_allclose(relative, expected)
    assert relative.dtype == np.float64
    np.testing.assert_array_equal(absolute_targets[:, 6], [0.2, 0.9])


def test_relative_rows_use_independent_so3_composition() -> None:
    from scipy.spatial.transform import Rotation

    reference_rotation_vector = np.array([np.pi / 2.0, 0.0, 0.0], dtype=np.float64)
    # Analytic composition: R_target = Rz(90 deg) * Rx(90 deg).
    target_rotation_component = 2.0 * np.pi / (3.0 * np.sqrt(3.0))
    absolute_targets = np.array(
        [[0.0, 0.0, 0.0, target_rotation_component, target_rotation_component, target_rotation_component, 0.5]],
        dtype=np.float64,
    )

    relative = convert_absolute_targets_to_relative_rows(
        reference_positions=np.zeros(3, dtype=np.float64),
        reference_rotation_vectors=reference_rotation_vector,
        absolute_targets=absolute_targets,
    )

    independent = (
        Rotation.from_rotvec(absolute_targets[0, 3:6])
        * Rotation.from_rotvec(reference_rotation_vector).inv()
    ).as_rotvec()
    np.testing.assert_allclose(relative[0, 3:6], independent, atol=1e-12)
    assert relative[0, 6] == 0.5


def test_relative_rows_support_one_to_one_historical_references() -> None:
    reference_positions = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    absolute_targets = np.array(
        [
            [0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1],
            [1.25, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2],
        ],
        dtype=np.float64,
    )

    relative = convert_absolute_targets_to_relative_rows(
        reference_positions=reference_positions,
        reference_rotation_vectors=np.zeros((2, 3), dtype=np.float64),
        absolute_targets=absolute_targets,
    )

    expected = np.array(
        [
            [0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1],
            [0.25, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2],
        ],
        dtype=np.float64,
    )
    np.testing.assert_allclose(relative, expected)


def test_relative_rows_match_builder_futures_after_same_normalization() -> None:
    reference_position = np.array([0.1, -0.2, 0.3], dtype=np.float64)
    reference_rotation_vector = np.array([0.0, 0.0, np.pi / 4.0], dtype=np.float64)
    candidate_chunks = np.zeros((1, 4, 7), dtype=np.float64)
    candidate_chunks[0, :, :3] = [[0.2, -0.1, 0.4], [0.25, -0.05, 0.45], [0.3, 0.0, 0.5], [0.35, 0.05, 0.55]]
    candidate_chunks[0, :, 3:6] = [0.0, 0.0, np.pi / 3.0]
    candidate_chunks[0, :, 6] = [0.1, 0.2, 0.3, 0.4]
    normalization = NormalizationArtifact(
        q01=np.full(6, -1.0, dtype=np.float64),
        q99=np.full(6, 1.0, dtype=np.float64),
    )

    built = build_action_histories(
        reference_position=reference_position,
        reference_rotation_vector=reference_rotation_vector,
        candidate_chunks=candidate_chunks,
        processed_past=np.empty((0, 7), dtype=np.float64),
        normalization=normalization,
    )
    unnormalized = convert_absolute_targets_to_relative_rows(
        reference_positions=reference_position,
        reference_rotation_vectors=reference_rotation_vector,
        absolute_targets=candidate_chunks[0, :4],
    )
    normalized = unnormalized.copy()
    normalized[:, :6] = (normalized[:, :6] - normalization.q01) / (
        normalization.q99 - normalization.q01 + 1e-6
    ) * 2.0 - 1.0

    np.testing.assert_allclose(built.histories[0, 6:10], normalized.astype(np.float32), atol=1e-6)


def test_relative_rows_reject_invalid_shapes_nonfinite_and_gripper() -> None:
    valid_reference = np.zeros(3, dtype=np.float64)
    valid_targets = np.zeros((2, 7), dtype=np.float64)
    valid_targets[:, 6] = 0.5

    with pytest.raises(ValueError, match=r"absolute_targets must have shape \[N, 7\]"):
        convert_absolute_targets_to_relative_rows(
            reference_positions=valid_reference,
            reference_rotation_vectors=valid_reference,
            absolute_targets=np.zeros((2, 6), dtype=np.float64),
        )
    with pytest.raises(ValueError, match="at least one row"):
        convert_absolute_targets_to_relative_rows(
            reference_positions=valid_reference,
            reference_rotation_vectors=valid_reference,
            absolute_targets=np.empty((0, 7), dtype=np.float64),
        )
    bad_targets = valid_targets.copy()
    bad_targets[0, 0] = np.nan
    with pytest.raises(ValueError, match="absolute_targets must be finite"):
        convert_absolute_targets_to_relative_rows(
            reference_positions=valid_reference,
            reference_rotation_vectors=valid_reference,
            absolute_targets=bad_targets,
        )
    bad_gripper = valid_targets.copy()
    bad_gripper[1, 6] = 1.5
    with pytest.raises(ValueError, match=r"absolute target gripper must be within \[0, 1\]"):
        convert_absolute_targets_to_relative_rows(
            reference_positions=valid_reference,
            reference_rotation_vectors=valid_reference,
            absolute_targets=bad_gripper,
        )
    with pytest.raises(ValueError, match="reference poses must be shape"):
        convert_absolute_targets_to_relative_rows(
            reference_positions=np.zeros((3, 3), dtype=np.float64),
            reference_rotation_vectors=valid_reference,
            absolute_targets=valid_targets,
        )
    nonfinite_reference = np.array([0.0, np.inf, 0.0], dtype=np.float64)
    with pytest.raises(ValueError, match="reference poses must be finite"):
        convert_absolute_targets_to_relative_rows(
            reference_positions=nonfinite_reference,
            reference_rotation_vectors=valid_reference,
            absolute_targets=valid_targets,
        )


def test_relative_rows_are_deterministic_and_non_mutating() -> None:
    reference_positions = np.array([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]], dtype=np.float64)
    reference_rotation_vectors = np.zeros((2, 3), dtype=np.float64)
    absolute_targets = np.array(
        [
            [0.5, 1.5, 2.5, 0.0, 0.0, 0.0, 0.3],
            [3.5, 4.5, 5.5, 0.0, 0.0, 0.0, 0.7],
        ],
        dtype=np.float64,
    )
    original_positions = reference_positions.copy()
    original_targets = absolute_targets.copy()

    first = convert_absolute_targets_to_relative_rows(
        reference_positions=reference_positions,
        reference_rotation_vectors=reference_rotation_vectors,
        absolute_targets=absolute_targets,
    )
    second = convert_absolute_targets_to_relative_rows(
        reference_positions=reference_positions,
        reference_rotation_vectors=reference_rotation_vectors,
        absolute_targets=absolute_targets,
    )

    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(reference_positions, original_positions)
    np.testing.assert_array_equal(absolute_targets, original_targets)
