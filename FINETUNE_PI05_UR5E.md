# Finetuning Pi0.5 on UR5e — Complete Plan

This document describes how to finetune Pi0.5 (from the OpenPI codebase) on a UR5e teleoperation dataset and deploy the finetuned model back on the real robot.

## Overview

```
Dataset (LeRobot)                     OpenPI Training                     Deployment
┌──────────────────┐    ┌──────────────────────────────┐    ┌──────────────────────┐
│ 100 episodes     │    │ 1. ur5e_policy.py            │    │ serve_policy.py      │
│ 2 cameras        │───>│ 2. LeRobotUR5eDataConfig     │───>│ (checkpoint)         │
│ 7D actions       │    │ 3. TrainConfig (pi05_ur5e)   │    │         │            │
│ joints + EEF     │    │ 4. compute_norm_stats        │    │         v            │
│ prompt: "pick.." │    │ 5. train.py                  │    │ deploy_lap_ur5e.py   │
└──────────────────┘    └──────────────────────────────┘    │ (--model pi0)        │
                                                            └──────────────────────┘
```

## Prerequisites

- OpenPI codebase (cloned at `lap/third_party/openpi/`)
- Compiled LeRobot dataset: `jsiburian/ur5e_pick_carrot` on HuggingFace (or local copy)
- GPU: RTX 4090 (24GB) for LoRA, or 4x H200 for full finetuning
- `uv` package manager (used by OpenPI for dependency management)

---

## Dataset

**HuggingFace:** `jsiburian/ur5e_pick_carrot`

| Key | Shape | Description |
|-----|-------|-------------|
| `observation.image.extra_camera` | VideoFrame (480x640) | External camera RGB |
| `observation.image.wrist_camera` | VideoFrame (480x640) | Wrist-mounted camera RGB |
| `observation.joints` | (6,) float32 | Joint positions (radians) |
| `observation.gripper` | (1,) float32 | Gripper state (raw robotiq value) |
| `observation.eef_pos` | (3,) float32 | EEF position in robot base frame (meters) |
| `observation.eef_rot_axis_angle` | (3,) float32 | EEF rotation in robot base frame (axis-angle, radians) |
| `actions` | (7,) float32 | **Absolute** EEF target: [pos(3), rot_aa(3), gripper(1)] |

- 100 episodes, ~150 frames each, 15000 total frames
- Recorded at ~16.7 Hz (50 Hz control loop, recording every 3rd step)
- Single task prompt: `"pick up the carrot"`

---

## Critical Design Decisions

### 1. State vector must match action space (IMPORTANT)

`DeltaActions` computes: `actions[:6] -= state[:6]`

This means the first 6 dimensions of `state` and `actions` must be in the **same space**.

- **Actions** = `[eef_pos(3), eef_rot_axis_angle(3), gripper(1)]` (Cartesian)
- **State** must therefore be `[eef_pos(3), eef_rot_axis_angle(3), gripper(1)]` (Cartesian)

Do NOT use `[joints(6), gripper(1)]` as state — joints are in radians/joint-space, actions are in meters/Cartesian-space. Subtracting joints from EEF positions is meaningless.

### 2. Pretrained weights: Pi0.5 Base (not DROID)

Use `gs://openpi-assets/checkpoints/pi05_base/params`. The UR5e has different kinematics, camera layout, and action space from DROID (Franka). Base weights are embodiment-agnostic.

### 3. FPS metadata

The dataset `info.json` currently says `fps: 50` but the actual recording rate is ~16.7 Hz. The timestamps in the Arrow data are correct (from the collection script). Verify the video decoder handles this correctly. If not, recompile with `--fps 16`.

### 4. Prompt handling

The dataset does NOT have a `task_index` column or `tasks.json` metadata file. Therefore `prompt_from_task=True` will NOT work out of the box. Two options:

- **Option A (recommended):** Set a `default_prompt` in the data config and skip `prompt_from_task`
- **Option B:** Add `task_index` and `tasks.json` to the compilation script and recompile

### 5. No DeltaActions alternative

If delta actions cause issues (e.g., the axis-angle delta subtraction is not meaningful because axis-angle is not a linear space), you can skip the `DeltaActions` transform entirely and train on absolute targets. Modify `LeRobotUR5eDataConfig` to remove the `.push(DeltaActions(...))` block. At inference, the model will directly output absolute EEF targets.

---

## Implementation: 3 files to create/modify

### File 1: `src/openpi/policies/ur5e_policy.py` (NEW)

```python
import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def make_ur5e_example() -> dict:
    return {
        "base_rgb": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "wrist_rgb": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "eef_pos": np.random.rand(3).astype(np.float32),
        "eef_rot": np.random.rand(3).astype(np.float32),
        "gripper": np.random.rand(1).astype(np.float32),
        "prompt": "pick up the object",
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class UR5eInputs(transforms.DataTransformFn):
    """Maps UR5e observation dict to Pi0.5 model input format.

    State vector: [eef_pos(3), eef_rot_axis_angle(3), gripper(1)] = 7D
    This MUST match the action space so DeltaActions can compute meaningful deltas.
    """

    model_type: _model.ModelType = _model.ModelType.PI05

    def __call__(self, data: dict) -> dict:
        eef_pos = np.asarray(data["eef_pos"], dtype=np.float32)
        eef_rot = np.asarray(data["eef_rot"], dtype=np.float32)
        gripper = np.asarray(data["gripper"], dtype=np.float32)
        if gripper.ndim == 0:
            gripper = gripper[np.newaxis]

        # 7D state: [eef_pos(3), eef_rot_aa(3), gripper(1)]
        # Matches action format so DeltaActions computes: delta = action - state
        state = np.concatenate([eef_pos, eef_rot, gripper])

        base_image = _parse_image(data["base_rgb"])
        wrist_image = _parse_image(data["wrist_rgb"])

        match self.model_type:
            case _model.ModelType.PI0 | _model.ModelType.PI05:
                names = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
                images = (base_image, wrist_image, np.zeros_like(base_image))
                image_masks = (np.True_, np.True_, np.False_)
            case _model.ModelType.PI0_FAST:
                names = ("base_0_rgb", "base_1_rgb", "wrist_0_rgb")
                images = (base_image, np.zeros_like(base_image), wrist_image)
                image_masks = (np.True_, np.True_, np.True_)
            case _:
                raise ValueError(f"Unsupported model type: {self.model_type}")

        inputs = {
            "state": state,
            "image": dict(zip(names, images, strict=True)),
            "image_mask": dict(zip(names, image_masks, strict=True)),
        }

        if "actions" in data:
            inputs["actions"] = np.asarray(data["actions"])

        if "prompt" in data:
            if isinstance(data["prompt"], bytes):
                data["prompt"] = data["prompt"].decode("utf-8")
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class UR5eOutputs(transforms.DataTransformFn):
    """Extracts UR5e actions from model output.

    Pi0.5 outputs 32D padded actions; we return only the first 7:
    [eef_pos(3), eef_rot_axis_angle(3), gripper(1)]
    """

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, :7])}
```

**Key difference from the old README:** State is now `[eef_pos(3), eef_rot_aa(3), gripper(1)]` instead of `[joints(6), gripper(1)]`, so it matches the action space for `DeltaActions`.

### File 2: `LeRobotUR5eDataConfig` — add to `src/openpi/training/config.py`

Add the import at the top:

```python
from openpi.policies import ur5e_policy
```

Add the class after `LeRobotDROIDDataConfig` (around line 454):

```python
@dataclasses.dataclass(frozen=True)
class LeRobotUR5eDataConfig(DataConfigFactory):
    """Data config for UR5e dataset in LeRobot format.

    Actions are absolute EEF targets: [pos(3), rot_aa(3), gripper(1)].
    State uses the same EEF representation so DeltaActions computes meaningful deltas.
    """

    # Set a default prompt for the task (used if dataset lacks task_index).
    default_prompt: str = "pick up the carrot"

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "base_rgb": "observation.image.extra_camera",
                        "wrist_rgb": "observation.image.wrist_camera",
                        "eef_pos": "observation.eef_pos",
                        "eef_rot": "observation.eef_rot_axis_angle",
                        "gripper": "observation.gripper",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                )
            ]
        )

        data_transforms = _transforms.Group(
            inputs=[ur5e_policy.UR5eInputs(model_type=model_config.model_type)],
            outputs=[ur5e_policy.UR5eOutputs()],
        )

        # DeltaActions: actions[:6] -= state[:6]
        # state = [eef_pos(3), eef_rot_aa(3), gripper(1)]
        # action = [eef_pos(3), eef_rot_aa(3), gripper(1)]
        # Result: [delta_pos(3), delta_rot_aa(3), abs_gripper(1)]
        # mask: first 6 = delta (True), last 1 = absolute (False)
        delta_action_mask = _transforms.make_bool_mask(6, -1)
        data_transforms = data_transforms.push(
            inputs=[_transforms.DeltaActions(delta_action_mask)],
            outputs=[_transforms.AbsoluteActions(delta_action_mask)],
        )

        model_transforms = ModelTransformFactory(
            default_prompt=self.default_prompt,
        )(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )
```

**Key differences from old README:**
- State uses `eef_pos` + `eef_rot_axis_angle` instead of `joints`
- Added `default_prompt` field (handles missing `task_index` in dataset)
- Passes `default_prompt` to `ModelTransformFactory` so `InjectDefaultPrompt` fills it when missing

### File 3: `TrainConfig` entries — add to `_CONFIGS` list in `config.py`

```python
    #
    # UR5e configs
    #
    TrainConfig(
        name="pi05_ur5e",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=16,
        ),
        data=LeRobotUR5eDataConfig(
            repo_id="jsiburian/ur5e_pick_carrot",
            default_prompt="pick up the carrot",
            assets=AssetsConfig(asset_id="ur5e"),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi05_base/params"
        ),
        num_train_steps=20_000,
        batch_size=32,
    ),
    TrainConfig(
        name="pi05_ur5e_lora",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=16,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ),
        data=LeRobotUR5eDataConfig(
            repo_id="jsiburian/ur5e_pick_carrot",
            default_prompt="pick up the carrot",
            assets=AssetsConfig(asset_id="ur5e"),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi05_base/params"
        ),
        freeze_filter=pi0_config.Pi0Config(
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        ema_decay=None,
        num_train_steps=20_000,
        batch_size=8,
    ),
```

**Notes:**
- `repo_id="jsiburian/ur5e_pick_carrot"` — HuggingFace dataset; replace with local path if downloaded
- `default_prompt` — used as fallback when dataset samples lack a `prompt` field
- `asset_id="ur5e"` — norm stats will be saved under `assets/pi05_ur5e/ur5e/`
- `action_horizon=16` — model predicts 16 future action steps (~1 second at 16.7 Hz)
- `action_dim=32` — Pi0.5 internally pads to 32D; `UR5eOutputs` slices back to 7

---

## Training Procedure

### Step 1: Download dataset (if not using HuggingFace directly)

```bash
huggingface-cli download jsiburian/ur5e_pick_carrot \
    --repo-type dataset \
    --local-dir /path/to/datasets/ur5e_pick_carrot
```

Then set `repo_id` to the local path in both `TrainConfig` entries.

### Step 2: Apply code changes

1. Create `src/openpi/policies/ur5e_policy.py` (File 1 above)
2. Add `from openpi.policies import ur5e_policy` to imports in `config.py`
3. Add `LeRobotUR5eDataConfig` class to `config.py` (File 2 above)
4. Add both `TrainConfig` entries to `_CONFIGS` list (File 3 above)

### Step 3: Compute normalization statistics

```bash
cd /path/to/openpi

# For full finetuning config:
uv run scripts/compute_norm_stats.py --config-name pi05_ur5e

# For LoRA config (same dataset, but saves under its own config dir):
uv run scripts/compute_norm_stats.py --config-name pi05_ur5e_lora
```

This computes per-key q01/q99 percentiles (quantile normalization) and saves to `assets/<config_name>/ur5e/norm_stats.json`.

**Verify:** The norm stats should show reasonable ranges for EEF position (order of ~0.1-0.5 meters) and axis-angle rotation (order of ~0-3 radians).

### Step 4: Launch training

**Option A: Full finetuning** (requires >= 40GB VRAM, e.g. 4x H200)

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_ur5e \
    --exp-name=pick_carrot_v1 \
    --overwrite
```

With multi-GPU FSDP:
```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_ur5e \
    --exp-name=pick_carrot_v1 \
    --overwrite \
    --fsdp-devices=4
```

**Option B: LoRA finetuning** (fits on RTX 4090 24GB)

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_ur5e_lora \
    --exp-name=pick_carrot_lora_v1 \
    --overwrite
```

### Step 5: Monitor training

- Loss logged every 100 steps to console
- Checkpoints saved every 1000 steps to `checkpoints/<config_name>/<exp_name>/`
- WandB logging on by default (disable: `--wandb-enabled=false`)
- Expect loss to decrease steadily; if it plateaus immediately, check norm stats

### Expected training time

| Setup | Config | Batch | Trainable | Time (20k steps) |
|-------|--------|-------|-----------|-------------------|
| 1x RTX 4090 (24GB) | `pi05_ur5e_lora` | 8 | ~50-100M (3%) | 6-12 hours |
| 4x H200 (141GB each) | `pi05_ur5e` | 32-128 | ~3B (100%) | 30-60 minutes |

---

## Deployment

### Step 1: Serve the finetuned checkpoint

```bash
cd /path/to/openpi

# Full finetuning checkpoint:
uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config=pi05_ur5e \
    --policy.dir=checkpoints/pi05_ur5e/pick_carrot_v1/20000

# LoRA checkpoint:
uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config=pi05_ur5e_lora \
    --policy.dir=checkpoints/pi05_ur5e_lora/pick_carrot_lora_v1/20000
```

This starts a WebSocket policy server on port 8000.

### Step 2: Run on the UR5e

The existing `deploy_lap_ur5e.py` with `--model pi0` can be used. However, the current Pi0 mode sends DROID-style observation keys and interprets actions as scaled deltas. For the finetuned model, the deployment must:

1. **Send observations** matching the training format:
   - `observation/exterior_image_1_left` -> external camera (224x224)
   - `observation/wrist_image_left` -> wrist camera (224x224)
   - `observation/joint_position` -> NOT USED for state (but may be required by the server)
   - `observation/gripper_position` -> gripper value
   - `prompt` -> `"pick up the carrot"`

2. **Interpret returned actions** as absolute EEF targets (not scaled deltas):
   - `actions[:3]` = target position (meters, robot base frame)
   - `actions[3:6]` = target rotation (axis-angle, robot base frame)
   - `actions[6]` = gripper command

The current `_pi0_actions_to_env` treats actions as normalized deltas with empirical scaling. For the finetuned model, **you need to modify this function** (or add a new `--model pi0_ur5e` mode) to treat actions as absolute EEF targets. The `AbsoluteActions` output transform in the server already converts deltas back to absolute targets, so the server returns absolute EEF poses.

**Deployment command:**
```bash
python deploy_lap_ur5e.py \
    --model pi0 \
    --prompt "pick up the carrot" \
    --remote_host <server-ip> \
    --remote_port 8000
```

### Step 3: Safety

- Keep `SafetyMonitor` active (max_delta_pos, max_force, max_torque)
- Start with `--open_loop_horizon 1` (re-query every step, most reactive)
- Test multiple checkpoints (5k, 10k, 15k, 20k) — earlier checkpoints may generalize better
- Have e-stop ready; finetuned models can produce unpredictable actions

---

## Important Considerations

### Camera consistency
The external camera MUST be at the same position, angle, and height as during data collection. Wrist camera is fixed to the robot. Significant lighting changes may also degrade performance.

### Prompt consistency
Use the exact training prompt (`"pick up the carrot"`) at inference. On a 100-episode single-task dataset, even minor wording changes ("grab the carrot") can hurt performance.

### Axis-angle delta subtraction caveat
`DeltaActions` computes `delta_rot = target_rot_aa - current_rot_aa`. Axis-angle subtraction is NOT the true geodesic rotation difference — it's an approximation that works well for small rotations but may introduce errors for large rotational changes. If the model struggles with orientation, consider:
- Training without DeltaActions (absolute actions only)
- Switching to quaternion or rotation matrix representation

### Norm stats must match data
If you add more episodes and retrain, you MUST recompute norm stats. Stale statistics will cause the model to see out-of-distribution inputs.

### Adding more data later
To collect more episodes, run:
```bash
python collect_vla_data.py finetuning_ur5e --num_episodes 200
```
Then recompile: `python compile_vla_to_lerobot.py --dataset-dir <path>` and re-upload to HuggingFace.
