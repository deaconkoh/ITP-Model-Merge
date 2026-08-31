"""Generate carbon+priority versions of DANIEL SD2 .fjs datasets.

The original SD2 format for each operation is:

    machine_count machine_id processing_time machine_id processing_time ...

The generated extended format is:

    machine_count priority machine_id processing_time carbon machine_id processing_time carbon ...

Priority and carbon values are synthetic integers from 1 to 100. A fixed
seed is used so the generated files are reproducible.
"""

from __future__ import annotations

import argparse
import hashlib
import random
from pathlib import Path


DEFAULT_SIZES = ["20x5", "15x10", "20x10", "30x10", "40x10"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create SD2 carbon+priority datasets")
    parser.add_argument(
        "--data-root",
        default="daniel/data",
        help="Root containing canonical split folders",
    )
    parser.add_argument(
        "--sizes",
        nargs="+",
        default=DEFAULT_SIZES,
        help="Dataset sizes to convert, without suffix. Example: 20x5 15x10",
    )
    parser.add_argument("--source-suffix", default="mix", help="Input dataset suffix")
    parser.add_argument("--target-suffix", default="carbon+priority", help="Output dataset suffix")
    parser.add_argument("--seed", type=int, default=20260629, help="Base seed for deterministic generation")
    parser.add_argument("--priority-min", type=int, default=1)
    parser.add_argument("--priority-max", type=int, default=100)
    parser.add_argument("--carbon-min", type=int, default=1)
    parser.add_argument("--carbon-max", type=int, default=100)
    parser.add_argument("--splits", nargs="+",
                        default=["data_train", "data_validation", "data_final_test"],
                        choices=["data_train", "data_validation", "data_final_test"],
                        help="Canonical split folders to annotate independently")
    parser.add_argument("--include-legacy", action="store_true",
                        help="Also convert the inherited SD2 folder for legacy comparison")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output files")
    return parser.parse_args()


def stable_seed(base_seed: int, label: str) -> int:
    digest = hashlib.sha256(label.encode("utf-8")).hexdigest()
    return base_seed + int(digest[:12], 16)


def convert_line(line: str, rng: random.Random, args: argparse.Namespace) -> str:
    parts = line.strip().split()
    if not parts:
        return ""

    job_op_count = parts[0]
    output = [job_op_count]
    idx = 1
    while idx < len(parts):
        machine_count = int(parts[idx])
        expected_end = idx + 1 + machine_count * 2
        if expected_end > len(parts):
            raise ValueError(f"Malformed operation near token {idx}: {line!r}")

        priority = rng.randint(args.priority_min, args.priority_max)
        output.extend([str(machine_count), str(priority)])
        idx += 1

        for _ in range(machine_count):
            machine_id = parts[idx]
            processing_time = parts[idx + 1]
            carbon = rng.randint(args.carbon_min, args.carbon_max)
            output.extend([machine_id, processing_time, str(carbon)])
            idx += 2

    return " ".join(output)


def convert_file(src: Path, dst: Path, root: Path, args: argparse.Namespace) -> None:
    if dst.exists() and not args.overwrite:
        return

    label = str(src.relative_to(root)).replace("\\", "/")
    rng = random.Random(stable_seed(args.seed, label))
    lines = src.read_text(encoding="utf-8").splitlines()
    if not lines:
        raise ValueError(f"Empty input file: {src}")

    converted = [lines[0]]
    converted.extend(convert_line(line, rng, args) for line in lines[1:])
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text("\n".join(converted) + "\n", encoding="utf-8")


def convert_dataset(root: Path, source_dir: Path, target_dir: Path, args: argparse.Namespace) -> int:
    if not source_dir.exists():
        print(f"skip missing source: {source_dir}")
        return 0

    count = 0
    for src in sorted(source_dir.glob("*.fjs")):
        dst = target_dir / src.name
        existed = dst.exists()
        convert_file(src, dst, root, args)
        if args.overwrite or not existed:
            count += 1
    print(f"{source_dir} -> {target_dir}: wrote {count} files")
    return count


def main() -> None:
    args = parse_args()
    root = Path(args.data_root).resolve()

    for size in args.sizes:
        source_name = f"{size}+{args.source_suffix}"
        target_name = f"{size}+{args.target_suffix}"

        for split in args.splits:
            convert_dataset(
                root,
                root / split / "SD2" / source_name,
                root / split / "SD2" / target_name,
                args,
            )

        if args.include_legacy:
            convert_dataset(root, root / "SD2" / source_name, root / "SD2" / target_name, args)


if __name__ == "__main__":
    main()
