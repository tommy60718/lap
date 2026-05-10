# Finetuning LAP-3B on UR5e — Complete Plan

This document describes how to finetune the LAP-3B model on the UR5e teleoperation dataset and deploy on the real robot.

## LAP vs Pi0.5: Key Differences

| Aspect | Pi0.5 | LAP-3B |
|--------|-------|--------|
| Architecture | PaliGemma (SigLIP + Gemma) | Same base, extends Pi0 |
| Training losses | Action flow matching only | Action flow + language CoT + prediction |
| Inference modes | Flow only | Flow (action expert) + AR (language reasoning) |
| Image slots | 3 (base, left_wrist, right_wrist) | 2 (base, left_wrist) by default |
| State input | Discretized into prompt (Pi0.5) | Same (discrete_state_input=True) |
| Training script | `third_party/openpi/scripts/train.py` | `scripts/train.py` (LAP's own) |
| Config location | `third_party/openpi/src/openpi/training/config.py` | `src/lap/training/config.py` |
| Data format | LeRobot (torch) | RLDS (primary) or LeRobot (fallback) |

LAP's `TrainConfig` extends OpenPI's `TrainConfig` and includes all upstream configs. LAP's data loader falls back to OpenPI's LeRobot/torch loader when `rlds_data_dir` is not set.

---

## Dataset

**HuggingFace:** `jsiburian/ur5e_pick_carrot`

Same dataset as the Pi0.5 plan:
- 100 episodes, ~150 frames each, 15000 total frames
- 2 cameras (extra_camera + wrist_camera)
- 7D absolute EEF actions: [pos(3), rot_axis_angle(3), gripper(1)]
- Single task prompt: `"pick up the carrot"`

---

## Finetuning Strategy

LAP has multiple training losses that can be enabled independently:

| Loss | Flag | Description | Use for UR5e? |
|------|------|-------------|---------------|
| Action flow | `enable_action_training=True` | Flow matching on action chunks | **Yes** — primary objective |
| Language CoT | `enable_langact_training=True` | Cross-entropy on language reasoning tokens | **No** — dataset lacks CoT annotations |
| Prediction | `enable_prediction_training=True` | Predict movement between frames | Optional |
| VQA | `enable_vqa_training=True` | Visual question answering | No |

**Recommended approach:** Enable action training only. This finetunes LAP's action expert (flow matching head) on the UR5e data while preserving the pretrained VLM and language reasoning capabilities. This is essentially the same as Pi0.5 finetuning but starting from LAP weights.

**Freeze options:**
- `get_freeze_filter()` — LoRA-style: freeze all non-LoRA params
- `get_vlm_freeze_filter()` — Freeze VLM + image encoder, train action expert only (efficient, preserves language reasoning)

For 100 episodes of a single task, **freezing the VLM and training only the action expert** is the most practical approach. This is the cheapest option and avoids catastrophic forgetting of the VLM's language capabilities.

---

## Implementation: 2 files to create/modify

### File 1: `src/openpi/policies/ur5e_policy.py` (NEW — same as Pi0.5 plan)

Create this file in the **OpenPI** directory. It's shared between LAP and Pi0.5 since both use the same model input format.

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
    """Maps UR5e observation dict to LAP/Pi0.5 model input format.

    State: [eef_pos(3), eef_rot_axis_angle(3), gripper(1)] = 7D
    Matches action space so DeltaActions computes meaningful deltas.
    """

    # Use _model.ModelType for standard Pi0/Pi0.5, or any LAP extended type
    model_type: _model.ModelType = _model.ModelType.PI05

    def __call__(self, data: dict) -> dict:
        eef_pos = np.asarray(data["eef_pos"], dtype=np.float32)
        eef_rot = np.asarray(data["eef_rot"], dtype=np.float32)
        gripper = np.asarray(data["gripper"], dtype=np.float32)
        if gripper.ndim == 0:
            gripper = gripper[np.newaxis]

        state = np.concatenate([eef_pos, eef_rot, gripper])

        base_image = _parse_image(data["base_rgb"])
        wrist_image = _parse_image(data["wrist_rgb"])

        # LAP uses 2 image slots (no right_wrist by default)
        # Pi0/Pi0.5 uses 3 image slots
        # We provide all 3 and mask the unused one
        names = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
        images = (base_image, wrist_image, np.zeros_like(base_image))
        image_masks = (np.True_, np.True_, np.False_)

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
    """Extracts UR5e actions from model output (first 7 of 32D padded)."""

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, :7])}
```

### File 2: Add configs to `src/lap/training/config.py`

Add the import at the top of LAP's config file:

```python
from openpi.policies import ur5e_policy
```

Add a `LeRobotUR5eDataConfig` class (can reuse the same class from the Pi0.5 plan, or define it here). Then add these `TrainConfig` entries to the `_CONFIGS` list:

```python
    #
    # UR5e LAP configs
    #
    TrainConfig(
        name="lap_ur5e",
        model=lap_config.LAPConfig(
            action_dim=7,
            action_horizon=10,
            max_token_len=180,
            enable_action_training=True,
            enable_langact_training=False,    # No CoT annotations in dataset
            enable_prediction_training=False,
            stop_action_to_vlm_grad=False,
            language_loss_weight=0.0,
            enable_image_augmentation=False,
            use_bimanual=False,               # Single-arm UR5e
        ),
        data=LeRobotUR5eDataConfig(
            repo_id="jsiburian/ur5e_pick_carrot",
            default_prompt="pick up the carrot",
            assets=AssetsConfig(asset_id="ur5e"),
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1000,
            peak_lr=5e-5,
            decay_steps=20_000,
            decay_lr=5e-5,
        ),
        weight_loader=weight_loaders.WeightLoaderChoice(
            kind="checkpoint",
            params_path="checkpoints/lap/params",
        ),
        num_train_steps=20_000,
        batch_size=32,
    ),
    TrainConfig(
        name="lap_ur5e_action_expert_only",
        model=lap_config.LAPConfig(
            action_dim=7,
            action_horizon=10,
            max_token_len=180,
            enable_action_training=True,
            enable_langact_training=False,
            enable_prediction_training=False,
            stop_action_to_vlm_grad=False,
            language_loss_weight=0.0,
            enable_image_augmentation=False,
            use_bimanual=False,
        ),
        data=LeRobotUR5eDataConfig(
            repo_id="jsiburian/ur5e_pick_carrot",
            default_prompt="pick up the carrot",
            assets=AssetsConfig(asset_id="ur5e"),
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=500,
            peak_lr=5e-5,
            decay_steps=20_000,
            decay_lr=5e-5,
        ),
        weight_loader=weight_loaders.WeightLoaderChoice(
            kind="checkpoint",
            params_path="checkpoints/lap/params",
        ),
        # Freeze VLM + image encoder, only train action expert
        freeze_filter=lap_config.LAPConfig().get_vlm_freeze_filter(),
        ema_decay=None,
        num_train_steps=20_000,
        batch_size=8,
    ),
```

**Note on `LeRobotUR5eDataConfig`:** This is the same class from the Pi0.5 plan. If you already added it to `third_party/openpi/src/openpi/training/config.py`, import it in LAP's config:

```python
from openpi.training.config import LeRobotUR5eDataConfig
```

If not, define it in LAP's `config.py` using the same code from the Pi0.5 README.

---

## Training Procedure

### Step 0: Obtain LAP base weights

Download the LAP-3B checkpoint from HuggingFace (linked in LAP README) and place at:
```
lap/checkpoints/lap/params/
```

### Step 1: Apply code changes

1. Create `third_party/openpi/src/openpi/policies/ur5e_policy.py`
2. Add `LeRobotUR5eDataConfig` (if not already done for Pi0.5)
3. Add `lap_ur5e` and `lap_ur5e_action_expert_only` configs to `src/lap/training/config.py`

### Step 2: Compute normalization statistics

```bash
cd /path/to/lap

# Uses LAP's config system (which includes OpenPI configs)
uv run scripts/compute_norm_stats.py --config-name lap_ur5e
```

**Important:** LAP's `compute_norm_stats.py` may not exist as a standalone script. If not, use OpenPI's:

```bash
# Make sure LAP's configs are importable
uv run third_party/openpi/scripts/compute_norm_stats.py --config-name lap_ur5e
```

### Step 3: Launch training

**Option A: Full finetuning** (all parameters)

```bash
cd /path/to/lap

JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
    uv run scripts/train.py lap_ur5e \
    --exp-name=pick_carrot_v1 \
    --overwrite
```

**Option B: Action expert only** (freeze VLM, much cheaper)

```bash
JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
    uv run scripts/train.py lap_ur5e_action_expert_only \
    --exp-name=pick_carrot_ae_v1 \
    --overwrite
```

This freezes the VLM (Gemma 2B) and image encoder (SigLIP), training only the action expert (~300M params). Should fit on an RTX 4090.

### Step 4: Monitor

- Loss logged every 50 steps (LAP default)
- Checkpoints every 2000 steps
- WandB on by default

---

## Deployment

### Serve the finetuned model

Add a new `EnvMode` in `scripts/serve_policy.py` or use the checkpoint directly:

```bash
cd /path/to/lap

JAX_PLATFORMS=cuda uv run scripts/serve_policy.py \
    --env LAP \
    --checkpoint.config lap_ur5e \
    --checkpoint.dir checkpoints/lap_ur5e/pick_carrot_v1/20000 \
    --checkpoint.type flow
```

Or for the action-expert-only checkpoint:
```bash
JAX_PLATFORMS=cuda uv run scripts/serve_policy.py \
    --env LAP \
    --checkpoint.config lap_ur5e_action_expert_only \
    --checkpoint.dir checkpoints/lap_ur5e_action_expert_only/pick_carrot_ae_v1/20000 \
    --checkpoint.type flow
```

### Run on the UR5e

```bash
python deploy_lap_ur5e.py \
    --model lap \
    --prompt "pick up the carrot" \
    --remote_host <server-ip> \
    --remote_port 8000
```

The existing LAP flow mode in `deploy_lap_ur5e.py` should work since the finetuned model uses the same flow-matching inference path. The output is a (horizon, 7) action chunk which the deployment script already handles.

**However:** The current deployment script expects LAP's default state encoding (EEF position + rot6d). The finetuned model expects [eef_pos(3) + rot_axis_angle(3) + gripper(1)]. You will need to update `_make_request` in `deploy_lap_ur5e.py` to send the state in the format matching the training data (axis-angle instead of rot6d, and 7D instead of 10D).

---

## LAP vs Pi0.5: Which to finetune?

| Consideration | LAP | Pi0.5 |
|---------------|-----|-------|
| Base model quality | Strong (newer, multi-task pretrained) | Strong (general purpose) |
| Language reasoning preserved | Yes (if VLM frozen) | N/A (no language reasoning) |
| Action expert only option | Yes (`get_vlm_freeze_filter`) | Yes (LoRA) |
| Future AR inference | Possible if language reasoning preserved | Not applicable |
| Data format flexibility | RLDS or LeRobot | LeRobot only |
| Training script | `lap/scripts/train.py` | `openpi/scripts/train.py` |

**Recommendation:** If you want to eventually use LAP's language reasoning (AR mode) for interpretability or debugging, finetune LAP with the VLM frozen. If you only care about action quality, either model works — try both and compare on the real robot.

---

## Estimated Training Time

| Setup | Config | Trainable params | Time (20k steps) |
|-------|--------|-----------------|-------------------|
| 1x RTX 4090 | `lap_ur5e_action_expert_only` | ~300M (action expert) | 4-8 hours |
| 4x H200 | `lap_ur5e` | ~3B (all) | 30-60 minutes |
