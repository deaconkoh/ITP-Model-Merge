"""Validate and summarize complete canonical multi-seed evaluation tables."""

from __future__ import annotations

import argparse
import csv
import re
import statistics
from collections import defaultdict
from pathlib import Path


SEED_RE = re.compile(r"^(?P<family>.+)_s(?P<seed>\d+)$")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize a complete canonical seed set")
    parser.add_argument("--input", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--expected-seeds", nargs="+", type=int, default=[300, 301, 302, 303, 304])
    args = parser.parse_args()

    with Path(args.input).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or "model" not in rows[0] or "test_data" not in rows[0]:
        raise ValueError("Canonical evaluation CSV must contain model and test_data columns")
    groups = defaultdict(list)
    bad = []
    for row in rows:
        match = SEED_RE.match(row["model"])
        if match is None:
            bad.append(row["model"])
            continue
        row["model_family"] = match.group("family")
        row["seed"] = int(match.group("seed"))
        groups[(row["model_family"], row["test_data"])].append(row)
    if bad:
        raise ValueError(f"Models do not end in a recorded _s<seed> suffix: {bad[:10]}")
    expected = set(args.expected_seeds)
    errors = []
    for keys, group in groups.items():
        seeds = [row["seed"] for row in group]
        actual = set(seeds)
        if actual != expected or len(seeds) != len(actual):
            errors.append(f"{keys}: expected {sorted(expected)}, found {sorted(actual)}")
    if errors:
        raise ValueError("Incomplete or duplicated canonical seed sets:\n" + "\n".join(errors))

    metric_columns = []
    for column in rows[0]:
        if not column.startswith("mean_"):
            continue
        try:
            [float(row[column]) for row in rows]
        except (KeyError, TypeError, ValueError):
            continue
        metric_columns.append(column)
    if not metric_columns:
        raise ValueError("No numeric mean_* metric columns found")
    summary = []
    for (family, test_data), group in sorted(groups.items()):
        result = {"model_family": family, "test_data": test_data}
        for metric in metric_columns:
            values = [float(row[metric]) for row in group]
            result[f"{metric}_across_seeds"] = statistics.mean(values)
            result[f"{metric}_std_across_seeds"] = statistics.stdev(values)
        summary.append(result)
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    print(f"Wrote {len(summary)} complete multi-seed summaries to {output}")


if __name__ == "__main__":
    main()
