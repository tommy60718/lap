from __future__ import annotations

import dataclasses
import hashlib
import json

import numpy as np
from scipy.spatial.transform import Rotation

ARTIFACT_SCHEMA = "osx_cover_normalization_v0"
REPRESENTATION_ID = "ur5e_cover_relative_eef_v1"
ACTION_ORDER = ("dx", "dy", "dz", "rotation_x", "rotation_y", "rotation_z", "gripper")
NORMALIZATION_EPSILON = 1e-6


def _canonical_json(value: dict) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _sha256_json(value: dict) -> str:
    return hashlib.sha256(_canonical_json(value).encode()).hexdigest()


def _adapter_configuration() -> dict:
    return {
        "representation": {
            "id": REPRESENTATION_ID,
            "action_order": list(ACTION_ORDER),
            "translation": {
                "units": "meters",
                "frame": "UR5e_base",
                "formula": "target_position_minus_reference_position",
            },
            "rotation": {
                "units": "radians",
                "representation": "rotation_vector",
                "formula": "rotvec(R_target_times_inverse_R_reference)",
            },
            "gripper": {
                "representation": "absolute_command",
                "expected_range": [0.0, 1.0],
                "transform": "unchanged",
            },
        },
        "normalization": {
            "dimensions": list(range(6)),
            "formula": "(x-q01)/(q99-q01+1e-6)*2-1",
            "epsilon": NORMALIZATION_EPSILON,
            "clipping": False,
            "gripper": "unchanged",
        },
    }


@dataclasses.dataclass(frozen=True)
class NormalizationProvenance:
    dataset_id: str
    dataset_schema_version: str
    split_manifest_hash: str
    shared_15hz_index_hash: str

    def __post_init__(self) -> None:
        hashes = (self.split_manifest_hash, self.shared_15hz_index_hash)
        valid_hashes = all(
            len(value) == 64 and all(character in "0123456789abcdef" for character in value) for value in hashes
        )
        if not self.dataset_id or not self.dataset_schema_version or not valid_hashes:
            raise ValueError("normalization provenance requires dataset identifiers and lowercase SHA-256 hashes")


@dataclasses.dataclass(frozen=True)
class NormalizationArtifact:
    q01: np.ndarray
    q99: np.ndarray
    fit_row_count: int | None = None
    quantile_method: str | None = None
    quantile_library_version: str | None = None
    provenance: NormalizationProvenance | None = None

    def to_json(self) -> str:
        if self.fit_row_count is None or self.quantile_method is None or self.quantile_library_version is None:
            raise ValueError("normalization artifact is missing fit metadata")
        if self.provenance is None:
            raise ValueError("normalization artifact is missing provenance")

        configuration = _adapter_configuration()
        payload = {
            "schema": ARTIFACT_SCHEMA,
            "representation": configuration["representation"],
            "normalization": {
                **configuration["normalization"],
                "q01": np.asarray(self.q01, dtype=np.float64).tolist(),
                "q99": np.asarray(self.q99, dtype=np.float64).tolist(),
                "quantile_method": self.quantile_method,
                "quantile_library_version": self.quantile_library_version,
            },
            "fit": {
                "row_count": self.fit_row_count,
                "split": "train",
                "padding_excluded": True,
            },
            "provenance": {
                "dataset_id": self.provenance.dataset_id,
                "dataset_schema_version": self.provenance.dataset_schema_version,
                "split_manifest_hash": self.provenance.split_manifest_hash,
                "shared_15hz_index_hash": self.provenance.shared_15hz_index_hash,
                "source_artifact_hashes": {
                    "split_manifest": self.provenance.split_manifest_hash,
                    "shared_15hz_index": self.provenance.shared_15hz_index_hash,
                },
            },
            "adapter_configuration_hash": _sha256_json(configuration),
        }
        payload["content_hash"] = _sha256_json(payload)
        return _canonical_json(payload)

    @classmethod
    def from_json(cls, serialized: str) -> NormalizationArtifact:
        payload = json.loads(serialized)
        content_hash = payload.pop("content_hash", None)
        if content_hash != _sha256_json(payload):
            raise ValueError("normalization artifact content hash mismatch")
        configuration = _adapter_configuration()
        if payload.get("adapter_configuration_hash") != _sha256_json(configuration):
            raise ValueError("normalization artifact adapter configuration mismatch")
        if payload.get("schema") != ARTIFACT_SCHEMA or payload.get("representation") != configuration["representation"]:
            raise ValueError("normalization artifact embedded representation mismatch")
        embedded_normalization = payload.get("normalization", {})
        if any(embedded_normalization.get(key) != value for key, value in configuration["normalization"].items()):
            raise ValueError("normalization artifact embedded normalization mismatch")
        try:
            normalization = payload["normalization"]
            provenance = payload["provenance"]
            source_hashes = provenance["source_artifact_hashes"]
            if (
                source_hashes["split_manifest"] != provenance["split_manifest_hash"]
                or source_hashes["shared_15hz_index"] != provenance["shared_15hz_index_hash"]
            ):
                raise ValueError("normalization artifact source artifact hash mismatch")
            q01 = np.asarray(normalization["q01"], dtype=np.float64)
            q99 = np.asarray(normalization["q99"], dtype=np.float64)
            if q01.shape != (6,) or q99.shape != (6,) or not np.isfinite(q01).all() or not np.isfinite(q99).all():
                raise ValueError("normalization artifact quantiles must be finite 6-vectors")
            if np.any(q99 <= q01):
                raise ValueError("normalization artifact q99 must be greater than q01")
            return cls(
                q01=q01,
                q99=q99,
                fit_row_count=int(payload["fit"]["row_count"]),
                quantile_method=str(normalization["quantile_method"]),
                quantile_library_version=str(normalization["quantile_library_version"]),
                provenance=NormalizationProvenance(
                    dataset_id=str(provenance["dataset_id"]),
                    dataset_schema_version=str(provenance["dataset_schema_version"]),
                    split_manifest_hash=str(provenance["split_manifest_hash"]),
                    shared_15hz_index_hash=str(provenance["shared_15hz_index_hash"]),
                ),
            )
        except ValueError:
            raise
        except (KeyError, TypeError) as error:
            raise ValueError("normalization artifact has missing or invalid metadata") from error


@dataclasses.dataclass(frozen=True)
class ActionHistoryBatch:
    histories: np.ndarray
    first_future_index: int = 6


def convert_absolute_targets_to_relative_rows(
    *,
    reference_positions: np.ndarray,
    reference_rotation_vectors: np.ndarray,
    absolute_targets: np.ndarray,
) -> np.ndarray:
    """Convert absolute UR5e 7-D targets into unnormalized relative rows.

    Supports one-to-one historical references (`[N, 3]` poses) and one current
    reference broadcast across `N` targets (`[3]` poses).
    """
    targets = np.asarray(absolute_targets, dtype=np.float64)
    if targets.ndim != 2 or targets.shape[1] != 7:
        raise ValueError("absolute_targets must have shape [N, 7]")
    if targets.shape[0] == 0:
        raise ValueError("absolute_targets must contain at least one row")
    if not np.isfinite(targets).all():
        raise ValueError("absolute_targets must be finite")
    if np.any((targets[:, 6] < 0.0) | (targets[:, 6] > 1.0)):
        raise ValueError("absolute target gripper must be within [0, 1]")

    positions = np.asarray(reference_positions, dtype=np.float64)
    rotation_vectors = np.asarray(reference_rotation_vectors, dtype=np.float64)
    row_count = targets.shape[0]
    if positions.shape == (3,) and rotation_vectors.shape == (3,):
        positions = np.broadcast_to(positions, (row_count, 3)).copy()
        rotation_vectors = np.broadcast_to(rotation_vectors, (row_count, 3)).copy()
    elif positions.shape != (row_count, 3) or rotation_vectors.shape != (row_count, 3):
        raise ValueError(
            "reference poses must be shape [3] for broadcast or [N, 3] matching absolute_targets"
        )
    else:
        positions = np.asarray(positions, dtype=np.float64, order="C")
        rotation_vectors = np.asarray(rotation_vectors, dtype=np.float64, order="C")
    if not np.isfinite(positions).all() or not np.isfinite(rotation_vectors).all():
        raise ValueError("reference poses must be finite")

    relative = np.empty((row_count, 7), dtype=np.float64)
    relative[:, :3] = targets[:, :3] - positions
    reference_rotations = Rotation.from_rotvec(np.ascontiguousarray(rotation_vectors))
    target_rotations = Rotation.from_rotvec(np.ascontiguousarray(targets[:, 3:6]))
    relative[:, 3:6] = (target_rotations * reference_rotations.inv()).as_rotvec()
    relative[:, 6] = targets[:, 6]
    return relative


def derive_normalization_artifact(
    *,
    training_relative_rows: np.ndarray,
    provenance: NormalizationProvenance,
) -> NormalizationArtifact:
    """Derive CoVer quantiles from unpadded training-relative rows."""
    rows = np.asarray(training_relative_rows, dtype=np.float64)
    if rows.ndim != 2 or rows.shape[0] == 0 or rows.shape[1] != 7:
        raise ValueError("training_relative_rows must have shape [N, 7] with N >= 1")
    if not np.isfinite(rows).all():
        raise ValueError("training_relative_rows must be finite")
    if np.any(np.all(rows == -5.0, axis=1)):
        raise ValueError("padding rows must be excluded from training_relative_rows")
    q01, q99 = np.quantile(rows[:, :6], [0.01, 0.99], axis=0, method="linear")
    degenerate_dimensions = np.flatnonzero(q99 <= q01)
    if degenerate_dimensions.size:
        dimensions = ", ".join(str(int(index)) for index in degenerate_dimensions)
        raise ValueError(f"q99 must be greater than q01 for every dimension; invalid dimension {dimensions}")
    return NormalizationArtifact(
        q01=q01,
        q99=q99,
        fit_row_count=rows.shape[0],
        quantile_method="numpy.quantile(method='linear')",
        quantile_library_version=f"numpy=={np.__version__}",
        provenance=provenance,
    )


def build_action_histories(
    *,
    reference_position: np.ndarray,
    reference_rotation_vector: np.ndarray,
    candidate_chunks: np.ndarray,
    processed_past: np.ndarray,
    normalization: NormalizationArtifact,
) -> ActionHistoryBatch:
    """Convert absolute UR5e candidates into CoVer action histories."""
    candidates = np.asarray(candidate_chunks, dtype=np.float64)
    if candidates.ndim != 3 or candidates.shape[2] != 7:
        raise ValueError("candidate_chunks must have shape [M, H, 7]")
    if not np.isfinite(candidates).all():
        raise ValueError("candidate_chunks must be finite")
    if np.any((candidates[..., 6] < 0.0) | (candidates[..., 6] > 1.0)):
        raise ValueError("candidate gripper must be within [0, 1]")
    if candidates.shape[0] == 0:
        raise ValueError("candidate_chunks must contain at least one candidate")
    if candidates.shape[1] < 4:
        raise ValueError("candidate_chunks must contain at least four future actions")

    past = np.asarray(processed_past, dtype=np.float64)
    if past.ndim != 2 or past.shape[1] != 7:
        raise ValueError("processed_past must have shape [P, 7]")
    if not np.isfinite(past).all():
        raise ValueError("processed_past must be finite")
    if np.any((past[:, 6] < 0.0) | (past[:, 6] > 1.0)):
        raise ValueError("processed past gripper must be within [0, 1]")
    if past.shape[0] > 6:
        raise ValueError("processed_past must contain at most six actions")

    position = np.asarray(reference_position, dtype=np.float64)
    rotation_vector = np.asarray(reference_rotation_vector, dtype=np.float64)
    if (
        position.shape != (3,)
        or rotation_vector.shape != (3,)
        or not np.isfinite(position).all()
        or not np.isfinite(rotation_vector).all()
    ):
        raise ValueError("reference pose must contain two finite 3-vectors")

    q01 = np.asarray(normalization.q01, dtype=np.float64)
    q99 = np.asarray(normalization.q99, dtype=np.float64)
    if q01.shape != (6,) or q99.shape != (6,) or not np.isfinite(q01).all() or not np.isfinite(q99).all():
        raise ValueError("normalization quantiles must be finite 6-vectors")
    if np.any(q99 <= q01):
        raise ValueError("normalization artifact q99 must be greater than q01")

    futures = candidates[:, :4]
    relative = convert_absolute_targets_to_relative_rows(
        reference_positions=position,
        reference_rotation_vectors=rotation_vector,
        absolute_targets=futures.reshape(-1, 7),
    ).reshape(futures.shape[0], 4, 7)

    relative[..., :6] = (relative[..., :6] - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0

    histories = np.full((futures.shape[0], 10, 7), -5.0, dtype=np.float32)
    if past.shape[0]:
        histories[:, 6 - past.shape[0] : 6] = past.astype(np.float32)
    histories[:, 6:10] = relative.astype(np.float32)
    return ActionHistoryBatch(histories=histories)


def assemble_training_action_history(
    *,
    relative_rows: np.ndarray,
    normalization: NormalizationArtifact,
) -> np.ndarray:
    """Normalize unpadded relative rows and left-pad to float32[10, 7]."""
    rows = np.asarray(relative_rows, dtype=np.float64)
    if rows.ndim != 2 or rows.shape[1] != 7 or rows.shape[0] == 0:
        raise ValueError("relative_rows must have shape [K, 7] with K >= 1")
    if rows.shape[0] > 10:
        raise ValueError("relative_rows exceed history length 10")
    if not np.isfinite(rows).all():
        raise ValueError("relative_rows must be finite")
    if np.any((rows[:, 6] < 0.0) | (rows[:, 6] > 1.0)):
        raise ValueError("relative row gripper must be within [0, 1]")
    if rows.shape[0] < 4:
        raise ValueError("relative_rows must contain four future actions")
    past_count = rows.shape[0] - 4
    if past_count > 6:
        raise ValueError("relative_rows may contain at most six past actions")

    q01 = np.asarray(normalization.q01, dtype=np.float64)
    q99 = np.asarray(normalization.q99, dtype=np.float64)
    if q01.shape != (6,) or q99.shape != (6,) or not np.isfinite(q01).all() or not np.isfinite(q99).all():
        raise ValueError("normalization quantiles must be finite 6-vectors")
    if np.any(q99 <= q01):
        raise ValueError("normalization artifact q99 must be greater than q01")

    normalized = rows.copy()
    normalized[:, :6] = (normalized[:, :6] - q01) / (q99 - q01 + NORMALIZATION_EPSILON) * 2.0 - 1.0
    history = np.full((10, 7), -5.0, dtype=np.float32)
    history[6 - past_count : 10] = normalized.astype(np.float32)
    return history
