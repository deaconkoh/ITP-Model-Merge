"""Create deterministic grouped train/validation/final-test dataset splits.

The split assignment uses the structural FJSP hash, so differently annotated
carbon/priority variants of the same base instance receive the same split even
when this tool is invoked on separate dataset directories.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build leakage-resistant FJSP splits")
    parser.add_argument("--sources", nargs="+", required=True, help="Dataset directories to split")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--data-source", default="SD2")
    parser.add_argument("--seed", type=int, default=20260831)
    parser.add_argument("--train-fraction", type=float, default=0.7)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    return parser.parse_args()


def assignment(structural_hash: str, seed: int, train_fraction: float, validation_fraction: float) -> str:
    token = hashlib.sha256(f"{seed}:{structural_hash}".encode("utf-8")).digest()
    value = int.from_bytes(token[:8], "big") / 2**64
    if value < train_fraction:
        return "train"
    if value < train_fraction + validation_fraction:
        return "validation"
    return "final_test"


def main() -> None:
    args = parse_args()
    if not 0 < args.train_fraction < 1:
        raise ValueError("--train-fraction must be between zero and one")
    if not 0 < args.validation_fraction < 1 - args.train_fraction:
        raise ValueError("--validation-fraction must leave a non-empty final-test fraction")

    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "daniel"))
    from data_manifest import dataset_fingerprint, structural_fjsp_sha256

    output_root = Path(args.output_root).resolve()
    split_dirs = {
        "train": "data_train",
        "validation": "data_validation",
        "final_test": "data_final_test",
    }
    manifest = {"seed": args.seed, "fractions": {
        "train": args.train_fraction,
        "validation": args.validation_fraction,
        "final_test": 1 - args.train_fraction - args.validation_fraction,
    }, "datasets": {}}

    for source_value in args.sources:
        source = Path(source_value).resolve()
        if not source.is_dir():
            raise FileNotFoundError(source)
        dataset_name = source.name
        rows = []
        for path in sorted(source.glob("*.fjs")):
            structural_hash = structural_fjsp_sha256(path)
            split = assignment(structural_hash, args.seed, args.train_fraction, args.validation_fraction)
            destination_dir = output_root / split_dirs[split] / args.data_source / dataset_name
            destination_dir.mkdir(parents=True, exist_ok=True)
            destination = destination_dir / path.name
            if destination.exists():
                raise FileExistsError(f"Refusing to overwrite existing split file: {destination}")
            shutil.copy2(path, destination)
            rows.append({"source": str(path), "split": split, "structural_sha256": structural_hash})
        manifest["datasets"][dataset_name] = rows

    for split, directory in split_dirs.items():
        split_root = output_root / directory / args.data_source
        if split_root.exists():
            manifest[f"{split}_fingerprints"] = {
                child.name: dataset_fingerprint(child) for child in sorted(split_root.iterdir()) if child.is_dir()
            }
    manifest_path = output_root / "manifests" / f"split_seed_{args.seed}.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote split manifest: {manifest_path}")


if __name__ == "__main__":
    main()
