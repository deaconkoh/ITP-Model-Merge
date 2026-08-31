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
    parser.add_argument(
        "--feature-schema",
        choices=["canonical_f11_p9_v2", "legacy_f11_p9_v1", "legacy_f11_p8_v1", "legacy_f10_p8_v1"],
        help="Required when source checkpoints have no embedded schema",
    )
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
    checkpoint_metadata = []
    parent_records = []
    sys.path.insert(0, str(repo_root / "daniel"))
    from checkpointing import CHECKPOINT_SCHEMA_VERSION, file_sha256, unwrap_checkpoint
    for model_name in args.models:
        model_path = checkpoint_dir / f"{model_name}.pth"
        if not model_path.exists():
            raise FileNotFoundError(model_path)
        state_dict, metadata = unwrap_checkpoint(torch.load(model_path, map_location="cpu"))
        checkpoints.append(state_dict)
        checkpoint_metadata.append(metadata)
        parent_records.append({"path": str(model_path.resolve()), "sha256": file_sha256(model_path)})
    for parent, weight in zip(parent_records, weights):
        parent["weight"] = weight

    base_keys = list(checkpoints[0].keys())
    for model_name, checkpoint in zip(args.models[1:], checkpoints[1:]):
        if list(checkpoint.keys()) != base_keys:
            raise ValueError(f"{model_name} has different checkpoint keys; cannot average directly")

    declared_schemas = {
        metadata.get("config", {}).get("feature_schema")
        for metadata in checkpoint_metadata
        if metadata.get("config", {}).get("feature_schema")
    }
    if len(declared_schemas) > 1:
        raise ValueError(f"Checkpoint feature schemas differ: {declared_schemas}")
    declared_schema = next(iter(declared_schemas), None)
    if args.feature_schema and declared_schema and args.feature_schema != declared_schema:
        raise ValueError(f"Requested schema {args.feature_schema} conflicts with {declared_schema}")
    feature_schema = declared_schema or args.feature_schema
    if feature_schema is None:
        raise ValueError("--feature-schema is required for legacy state-dict-only sources")

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

    merged_config = dict(checkpoint_metadata[0].get("config", {}))
    merged_config["feature_schema"] = feature_schema
    torch.save({
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "artifact_type": "merged_checkpoint",
        "state_dict": merged,
        "config": merged_config,
        "parents": parent_records,
        "merge": {"method": "soup", "weights": weights},
    }, out_path)
    model_weights = ", ".join(f"{name}:{weight:.4f}" for name, weight in zip(args.models, weights))
    print(f"Wrote {out_path}")
    print(f"Merged {model_weights}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
