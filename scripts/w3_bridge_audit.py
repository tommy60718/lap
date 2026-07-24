#!/usr/bin/env python3
"""Emit the canonical W3-01 production Bridge audit manifest."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from lap.verifiers.cover.bridge_audit import audit_bridge_checkpoint
from lap.verifiers.cover.bridge_audit import build_production_target_inventory
from lap.verifiers.cover.bridge_audit import canonical_json_bytes


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path, help="pre-acquired cover_verifier_bridge.pt")
    parser.add_argument("--output", type=Path, help="optional path for the canonical JSON manifest")
    args = parser.parse_args(argv)

    manifest = audit_bridge_checkpoint(
        args.artifact,
        target_inventory=build_production_target_inventory(),
    )
    payload = canonical_json_bytes(manifest) + b"\n"
    if args.output is not None:
        args.output.write_bytes(payload)
    sys.stdout.buffer.write(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
