"""Create a single weight-merged DANIEL checkpoint.

This performs a simple "model soup" merge:

    merged = w1 * checkpoint_1 + w2 * checkpoint_2 + ...

The output is one normal `.pth` checkpoint that can be evaluated with the
existing single-model evaluation path.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge DANIEL checkpoints by weighted averaging")
    parser.add_argument("--data-source", default="SD1", help="Checkpoint folder under daniel/trained_network")
    parser.add_argument("--models", nargs="+", required=True, help="Checkpoint names without .pth")
    parser.add_argument(
        "--weights",
        nargs="+",
        type=float,
        help="Optional merge weights. Defaults to uniform weights.",
    )
    parser.add_argument("--out-name", required=True, help="Output checkpoint name without .pth")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    checkpoint_dir = repo_root / "daniel" / "trained_network" / args.data_source
    out_path = checkpoint_dir / f"{args.out_name}.pth"

    if args.weights is None:
        weights = [1.0 / len(args.models)] * len(args.models)
    else:
        if len(args.weights) != len(args.models):
            raise ValueError("--weights must have the same count as --models")
        weight_sum = sum(args.weights)
        if weight_sum == 0:
            raise ValueError("--weights must not sum to zero")
        weights = [weight / weight_sum for weight in args.weights]

    import torch

    checkpoints = []
    for model_name in args.models:
        model_path = checkpoint_dir / f"{model_name}.pth"
        if not model_path.exists():
            raise FileNotFoundError(model_path)
        checkpoints.append(torch.load(model_path, map_location="cpu"))

    base_keys = list(checkpoints[0].keys())
    for model_name, checkpoint in zip(args.models[1:], checkpoints[1:]):
        if list(checkpoint.keys()) != base_keys:
            raise ValueError(f"{model_name} has different checkpoint keys; cannot average directly")

    merged = {}
    for key in base_keys:
        values = [checkpoint[key] for checkpoint in checkpoints]
        first = values[0]
        if not all(value.shape == first.shape for value in values):
            raise ValueError(f"Tensor shape mismatch for key {key}")
        if first.is_floating_point() or first.is_complex():
            merged[key] = sum(weight * value for weight, value in zip(weights, values))
        else:
            if not all(torch.equal(first, value) for value in values[1:]):
                raise ValueError(f"Non-floating tensor differs for key {key}; cannot safely merge")
            merged[key] = first.clone()

    torch.save(merged, out_path)
    model_weights = ", ".join(f"{name}:{weight:.4f}" for name, weight in zip(args.models, weights))
    print(f"Wrote {out_path}")
    print(f"Merged {model_weights}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
