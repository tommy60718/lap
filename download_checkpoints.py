#!/usr/bin/env python3
"""Download all checkpoints from HuggingFace."""

import os
os.environ.setdefault("HF_TOKEN", os.environ.get("HF_TOKEN", ""))

from huggingface_hub import snapshot_download

path = snapshot_download(
    "jsiburian/pi05-ur5e-pick-carrot-lora",
    local_dir="checkpoints/pi05_ur5e_lora",
)
print(f"Done: {path}")
