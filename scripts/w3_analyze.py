#!/usr/bin/env python3
"""Deterministic CLI: read an accepted W3 package → write a separate analysis directory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lap.verifiers.cover.analysis import analyze_accepted_w3_package


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--package-root", type=Path, required=True, help="Accepted W3 package root")
    p.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="Analysis output directory (must be outside the package)",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    receipt = analyze_accepted_w3_package(
        package_root=args.package_root,
        output_root=args.output_root,
    )
    print(json.dumps(receipt, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
