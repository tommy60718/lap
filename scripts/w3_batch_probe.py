#!/usr/bin/env python3
"""Run the real pinned two-rank W3-03 SigLIP2 batch-size preflight."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lap.verifiers.cover.batch_probe import run_probe_worker
from lap.verifiers.cover.batch_probe import run_real_batch_probe


def parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--w2-root", type=Path, required=True)
    parser.add_argument("--bridge-artifact", type=Path, required=True)
    parser.add_argument("--audit-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--worker-mode", choices=("independent", "ddp"))
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--result-dir", type=Path)
    return parser


def main() -> None:
    args = parser().parse_args()
    if args.worker:
        if args.worker_mode is None or args.batch_size is None or args.model_dir is None or args.result_dir is None:
            raise ValueError("worker mode requires --worker-mode, --batch-size, --model-dir, and --result-dir")
        run_probe_worker(
            mode=args.worker_mode,
            batch_size=args.batch_size,
            w2_root=args.w2_root,
            bridge_artifact=args.bridge_artifact,
            audit_manifest=args.audit_manifest,
            model_dir=args.model_dir,
            result_dir=args.result_dir,
        )
        return
    if args.output is None:
        raise ValueError("orchestrator mode requires --output")
    receipt = run_real_batch_probe(
        worker_script=Path(__file__).resolve(),
        w2_root=args.w2_root,
        bridge_artifact=args.bridge_artifact,
        audit_manifest=args.audit_manifest,
        output_path=args.output,
        timeout_seconds=args.timeout_seconds,
    )
    print(json.dumps(receipt, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
