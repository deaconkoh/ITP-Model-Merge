"""Validate exact and structural independence of FJSP dataset splits."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate FJSP train/validation/test separation")
    parser.add_argument("--train", required=True)
    parser.add_argument("--validation", required=True)
    parser.add_argument("--test", required=True)
    parser.add_argument("--out", help="Optional JSON fingerprint report")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "daniel"))
    from data_manifest import assert_disjoint_splits

    fingerprints = assert_disjoint_splits(
        train=args.train,
        validation=args.validation,
        test=args.test,
    )
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(fingerprints, indent=2), encoding="utf-8")
    print("PASS: train, validation, and test splits are exact- and structural-hash disjoint")


if __name__ == "__main__":
    main()
