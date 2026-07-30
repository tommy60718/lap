#!/usr/bin/env python3
"""CLI for the ROS-independent W3 CoVer training boundary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lap.verifiers.cover.command import preflight_w3
from lap.verifiers.cover.command import run_protocol_mode
from lap.verifiers.cover.pipeline import run_canonical_acceptance
from lap.verifiers.cover.pipeline import run_evaluate_mode
from lap.verifiers.cover.pipeline import run_fixture_end_to_end
from lap.verifiers.cover.pipeline import run_package_mode
from lap.verifiers.cover.pipeline import run_train_mode


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
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--evidence-root", type=Path, default=None)
    return parser


def dispatch(args: argparse.Namespace) -> dict:
    if args.mode == "fixture-acceptance":
        return run_fixture_end_to_end(output_root=args.output_root)
    if args.fixture and args.mode in {"train", "evaluate", "package", "accept"}:
        if args.mode == "accept":
            raise ValueError("accept mode does not alias fixture execution; use --mode fixture-acceptance")
        raise ValueError("fixture train/evaluate/package modes require their explicit stage inputs")
    if args.mode == "preflight":
        return preflight_w3(
            w2_root=args.w2_root,
            bridge_artifact=args.bridge_artifact,
            audit_manifest=args.audit_manifest,
            output_root=args.output_root,
            validator_path=args.validator,
            fixture=args.fixture,
            protocol_dir=args.protocol_dir,
        )
    if args.mode == "protocol":
        return run_protocol_mode(
            w2_root=args.w2_root,
            protocol_dir=args.protocol_dir,
            validator_path=args.validator,
            fixture=args.fixture,
            per_rank_batch_size=args.per_rank_batch_size,
            batch_probe_receipt=args.batch_probe_receipt,
        )
    if args.mode == "train":
        return run_train_mode(
            w2_root=args.w2_root,
            bridge_artifact=args.bridge_artifact,
            audit_manifest=args.audit_manifest,
            protocol_dir=args.protocol_dir,
            output_root=args.output_root,
            validator_path=args.validator,
        )
    if args.mode == "evaluate":
        if args.checkpoint is None:
            raise ValueError("evaluate mode requires --checkpoint")
        return run_evaluate_mode(
            checkpoint=args.checkpoint,
            output_root=args.output_root,
            w2_root=args.w2_root,
            protocol_dir=args.protocol_dir,
            validator_path=args.validator,
        )
    if args.mode == "package":
        if args.evidence_root is None:
            raise ValueError("package mode requires --evidence-root")
        return run_package_mode(evidence_root=args.evidence_root, output_root=args.output_root)
    if args.mode == "accept":
        return run_canonical_acceptance(
            w2_root=args.w2_root,
            bridge_artifact=args.bridge_artifact,
            audit_manifest=args.audit_manifest,
            protocol_dir=args.protocol_dir,
            output_root=args.output_root,
            validator_path=args.validator,
        )

    raise ValueError(f"unsupported W3 mode: {args.mode}")


def main() -> None:
    result = dispatch(parser().parse_args())
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
