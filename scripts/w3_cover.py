#!/usr/bin/env python3
"""CLI for the ROS-independent W3 CoVer training boundary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lap.verifiers.cover.command import preflight_w3
from lap.verifiers.cover.command import run_protocol_mode
from lap.verifiers.cover.pipeline import run_canonical_acceptance
from lap.verifiers.cover.pipeline import run_fixture_end_to_end


def parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("preflight", "protocol", "train", "evaluate", "package", "accept", "fixture-acceptance"),
        required=True,
    )
    parser.add_argument("--w2-root", type=Path, required=True)
    parser.add_argument("--bridge-artifact", type=Path, required=True)
    parser.add_argument("--audit-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--protocol-dir", type=Path, required=True)
    parser.add_argument("--validator", type=Path, default=None)
    parser.add_argument("--fixture", action="store_true")
    parser.add_argument("--per-rank-batch-size", type=int, default=16)
    parser.add_argument("--batch-probe-receipt", type=Path, default=None)
    return parser


def main() -> None:
    args = parser().parse_args()
    if args.mode == "fixture-acceptance" or (args.mode in {"train", "evaluate", "package", "accept"} and args.fixture):
        result = run_fixture_end_to_end(output_root=args.output_root)
    elif args.mode == "preflight":
        result = preflight_w3(
            w2_root=args.w2_root,
            bridge_artifact=args.bridge_artifact,
            audit_manifest=args.audit_manifest,
            output_root=args.output_root,
            validator_path=args.validator,
            fixture=args.fixture,
        )
    elif args.mode == "protocol":
        result = run_protocol_mode(
            w2_root=args.w2_root,
            protocol_dir=args.protocol_dir,
            validator_path=args.validator,
            fixture=args.fixture,
            per_rank_batch_size=args.per_rank_batch_size,
            batch_probe_hash=(
                None
                if args.batch_probe_receipt is None
                else __import__("hashlib").sha256(args.batch_probe_receipt.read_bytes()).hexdigest()
            ),
        )
    else:
        result = run_canonical_acceptance(
            w2_root=args.w2_root,
            bridge_artifact=args.bridge_artifact,
            audit_manifest=args.audit_manifest,
            protocol_dir=args.protocol_dir,
            output_root=args.output_root,
            validator_path=args.validator,
        )
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
