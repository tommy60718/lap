"""W2 v3 consumer gateway and deterministic two-view sampling."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import importlib.util
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torch.utils.data.distributed import DistributedSampler

from lap.verifiers.cover.protocol import select_training_phrase
from lap.verifiers.cover.w3_contracts import HISTORY_SHAPE
from lap.verifiers.cover.w3_contracts import W2_TRAIN_COUNT
from lap.verifiers.cover.w3_contracts import W2_VALIDATION_COUNT
from lap.verifiers.cover.w3_contracts import require_history


@dataclass(frozen=True)
class W3Sample:
    sample_id: str
    episode_id: str
    instruction: str
    base_image: Path
    wrist_image: Path
    action_history: np.ndarray
    condition: dict[str, Any]
    split: str


def _load_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _load_outer_validator(path: Path):
    spec = importlib.util.spec_from_file_location("osx_vla_w2_validator", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load W2 validator from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_w2_export(
    export_root: Path, *, validator_path: Path | None = None, fixture: bool = False
) -> dict[str, Any]:
    export_root = Path(export_root)
    if validator_path is None:
        validator_path = Path(__file__).resolve().parents[6] / "scripts" / "export_cover_training.py"
    if fixture:
        return _load_json(export_root / "export_manifest.json")
    return _load_outer_validator(validator_path).validate_exported_package(export_root)


def _sample_from_record(export_root: Path, record: dict[str, Any], split: str) -> W3Sample:
    if record.get("split") != split:
        raise ValueError(f"{record.get('sample_id')}: split mismatch")
    history = require_history(record.get("action_history"), name=f"{record.get('sample_id')}.action_history")
    trace = record.get("traceability", {})
    condition = {"peg_shape": trace.get("peg_shape"), "approach_direction": trace.get("approach_direction")}
    if not condition["peg_shape"] or not condition["approach_direction"]:
        raise ValueError(f"{record.get('sample_id')}: missing condition metadata")
    return W3Sample(
        sample_id=str(record["sample_id"]),
        episode_id=str(record["episode_id"]),
        instruction=str(record["instruction"]),
        base_image=export_root / str(record["base_image"]),
        wrist_image=export_root / str(record["wrist_image"]),
        action_history=history,
        condition=condition,
        split=split,
    )


class W2DatasetGateway:
    """Loads the already-partitioned W2 manifests without resplitting."""

    def __init__(self, export_root: Path, *, validator_path: Path | None = None, fixture: bool = False) -> None:
        self.export_root = Path(export_root)
        self.validation_receipt = validate_w2_export(self.export_root, validator_path=validator_path, fixture=fixture)
        train_records = _load_json(self.export_root / "train_samples.json")
        validation_records = _load_json(self.export_root / "validation_samples.json")
        expected_train = W2_TRAIN_COUNT if not fixture else len(train_records)
        expected_validation = W2_VALIDATION_COUNT if not fixture else len(validation_records)
        if len(train_records) != expected_train or len(validation_records) != expected_validation:
            raise ValueError(f"W2 counts mismatch: train={len(train_records)}, validation={len(validation_records)}")
        self.train = [_sample_from_record(self.export_root, row, "train") for row in train_records]
        self.validation = [_sample_from_record(self.export_root, row, "validation") for row in validation_records]

    @property
    def train_manifest_hash(self) -> str:
        return str(self.validation_receipt.get("artifact_content_hashes", {}).get("train_samples.json", ""))


def decode_rgb(path: Path) -> Image.Image:
    if not Path(path).is_file():
        raise FileNotFoundError(path)
    try:
        return Image.open(path).convert("RGB")
    except Exception as error:
        raise ValueError(f"cannot decode RGB image: {path}") from error


def preprocess_rgb(image: Image.Image | np.ndarray, *, size: int = 384) -> torch.Tensor:
    if isinstance(image, np.ndarray):
        image = Image.fromarray(np.asarray(image, dtype=np.uint8)).convert("RGB")
    image = image.convert("RGB")
    image = image.resize((size, size), Image.Resampling.BICUBIC)
    array = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous()
    return (tensor - 0.5) / 0.5


class TwoViewDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        samples: list[W3Sample],
        *,
        seed: int = 42,
        epoch: int = 0,
        training: bool = True,
        preprocess: Callable[[Image.Image], torch.Tensor] = preprocess_rgb,
    ) -> None:
        self.samples = samples
        self.seed = seed
        self.epoch = epoch
        self.training = training
        self.preprocess = preprocess

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        shape = sample.condition["peg_shape"]
        instruction = (
            sample.instruction
            if not self.training
            else select_training_phrase(seed=self.seed, epoch=self.epoch, sample_id=sample.sample_id, shape=shape)
        )
        return {
            "sample_id": sample.sample_id,
            "episode_id": sample.episode_id,
            "base_rgb": self.preprocess(decode_rgb(sample.base_image)),
            "wrist_rgb": self.preprocess(decode_rgb(sample.wrist_image)),
            "instruction": instruction,
            "canonical_instruction": sample.instruction,
            "action_history": torch.from_numpy(np.asarray(sample.action_history, dtype=np.float32).copy()),
            "condition": sample.condition,
        }


def collate_two_view_batch(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Collate canonical W2 rows without changing either view or provenance."""

    return {
        "sample_ids": [row["sample_id"] for row in rows],
        "episode_ids": [row["episode_id"] for row in rows],
        "base_rgb": torch.stack([row["base_rgb"] for row in rows]),
        "wrist_rgb": torch.stack([row["wrist_rgb"] for row in rows]),
        "instructions": [row["instruction"] for row in rows],
        "action_histories": torch.stack([row["action_history"] for row in rows]),
        "conditions": [row["condition"] for row in rows],
    }


_DISTANCE_BIN_EDGES = np.asarray(
    [0.0, 1e-6, 1e-4, 1e-3, 1e-2, 1e-1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, np.inf],
    dtype=np.float64,
)


def _pair_metric(count: int, denominator: int) -> dict[str, int | float]:
    return {
        "count": count,
        "denominator": denominator,
        "rate": float(count / denominator) if denominator else 0.0,
    }


def _matching_pair_count(values: Sequence[Any]) -> int:
    return sum(count * (count - 1) // 2 for count in Counter(values).values())


def _distance_distribution(counts: np.ndarray, denominator: int) -> dict[str, Any]:
    return {
        "denominator": denominator,
        "histogram_bin_edges": [*map(float, _DISTANCE_BIN_EDGES[:-1]), None],
        "histogram_counts": [int(count) for count in counts],
    }


def build_epoch_collision_report(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Describe collisions over every unordered off-diagonal epoch-row pair."""

    sample_ids = [str(row["sample_id"]) for row in rows]
    episode_ids = [str(row["episode_id"]) for row in rows]
    instructions = [str(row["instruction"]) for row in rows]
    validated_histories = [
        require_history(row["history"], name=f"epoch_rows[{index}].history") for index, row in enumerate(rows)
    ]
    histories = (
        np.stack(validated_histories) if validated_histories else np.empty((0, *HISTORY_SHAPE), dtype=np.float32)
    )
    row_count = len(rows)
    pair_denominator = row_count * (row_count - 1) // 2
    history_keys = [history.tobytes() for history in histories]
    repeated_instruction_count = _matching_pair_count(instructions)
    exact_history_count = _matching_pair_count(history_keys)
    repeated_language_history_count = _matching_pair_count(list(zip(instructions, history_keys, strict=True)))
    same_episode_count = _matching_pair_count(episode_ids)

    all_histogram = np.zeros(len(_DISTANCE_BIN_EDGES) - 1, dtype=np.int64)
    same_episode_histogram = np.zeros_like(all_histogram)
    flattened = histories.astype(np.float64, copy=False).reshape(row_count, HISTORY_SHAPE[0] * HISTORY_SHAPE[1])
    squared_norms = np.einsum("ij,ij->i", flattened, flattened)
    block_size = 256
    for start in range(0, row_count, block_size):
        stop = min(start + block_size, row_count)
        squared_distances = (
            squared_norms[start:stop, None] + squared_norms[None, :] - 2.0 * flattened[start:stop] @ flattened.T
        )
        np.maximum(squared_distances, 0.0, out=squared_distances)
        normalized_distances = np.sqrt(squared_distances / flattened.shape[1])
        left_indices, right_indices = np.nonzero(np.arange(row_count)[None, :] > np.arange(start, stop)[:, None])
        off_diagonal = normalized_distances[left_indices, right_indices]
        all_histogram += np.histogram(off_diagonal, bins=_DISTANCE_BIN_EDGES)[0]
        same_episode_mask = (
            np.asarray(episode_ids, dtype=object)[right_indices]
            == np.asarray(episode_ids[start:stop], dtype=object)[left_indices]
        )
        same_episode_histogram += np.histogram(off_diagonal[same_episode_mask], bins=_DISTANCE_BIN_EDGES)[0]

    return {
        "sampler_rows": row_count,
        "unique_sampler_rows": len(set(sample_ids)),
        "sampler_added_duplicate_rows": row_count - len(set(sample_ids)),
        "off_diagonal_population": {
            "pair_definition": "unordered_distinct_row_positions_i_lt_j",
            "denominator": pair_denominator,
        },
        "repeated_instruction_pairs": _pair_metric(repeated_instruction_count, pair_denominator),
        "exact_duplicate_history_pairs": _pair_metric(exact_history_count, pair_denominator),
        "repeated_language_history_pairs": _pair_metric(repeated_language_history_count, pair_denominator),
        "same_episode_pairs": _pair_metric(same_episode_count, pair_denominator),
        "repeated_instruction_pair_rate": _pair_metric(repeated_instruction_count, pair_denominator)["rate"],
        "exact_duplicate_history_pair_rate": _pair_metric(exact_history_count, pair_denominator)["rate"],
        "same_episode_pair_rate": _pair_metric(same_episode_count, pair_denominator)["rate"],
        "normalized_history_distances": {
            "normalization": "root_mean_square_over_fixed_10x7_history",
            "all_pairs": _distance_distribution(all_histogram, pair_denominator),
            "same_episode_pairs": _distance_distribution(same_episode_histogram, same_episode_count),
        },
        "adapts_batches": False,
    }


def make_sampler(
    dataset: Dataset[Any], *, seed: int = 42, world_size: int = 1, rank: int = 0
) -> DistributedSampler[Any]:
    return DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=seed, drop_last=False)
