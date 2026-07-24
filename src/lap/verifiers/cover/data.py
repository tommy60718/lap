"""W2 v3 consumer gateway and deterministic two-view sampling."""

from __future__ import annotations

from collections.abc import Callable
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


def make_sampler(
    dataset: Dataset[Any], *, seed: int = 42, world_size: int = 1, rank: int = 0
) -> DistributedSampler[Any]:
    return DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=seed, drop_last=False)
