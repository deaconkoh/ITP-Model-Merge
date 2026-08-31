"""Merge DANIEL checkpoints into one unified checkpoint.

This script is intentionally reusable for the final project stage:

1. Plain/model-soup merge:
       merged = w1 * model_1 + w2 * model_2 + ...

2. Task-arithmetic merge:
       merged = base + w1 * (model_1 - base) + w2 * (model_2 - base) + ...

All input checkpoints must have the same architecture. For the current
job-priority models this normally means all checkpoints should use the
11-feature priority-aware DANIEL model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge DANIEL checkpoints into one .pth file")
    parser.add_argument(
        "--method",
        choices=["soup", "task_arithmetic"],
        default="soup",
        help="Merge method. 'soup' is weighted averaging; 'task_arithmetic' adds specialist changes to a base.",
    )
    parser.add_argument("--data-source", default="SD2", help="Checkpoint folder under daniel/trained_network")
    parser.add_argument(
        "--checkpoints",
        nargs="+",
        required=True,
        help="Checkpoint names under daniel/trained_network/<data-source>, or direct .pth paths",
    )
    parser.add_argument(
        "--weights",
        nargs="+",
        type=float,
        help="Weights for checkpoints. Soup weights are normalized. Task-arithmetic weights are used as given.",
    )
    parser.add_argument(
        "--base",
        help="Base checkpoint for task_arithmetic. Name or direct .pth path.",
    )
    parser.add_argument("--out-name", help="Output checkpoint name without .pth, saved under daniel/trained_network/<data-source>")
    parser.add_argument("--out", help="Direct output .pth path. Overrides --out-name.")
    parser.add_argument(
        "--feature-schema",
        choices=["canonical_f11_p9_v2", "legacy_f11_p9_v1", "legacy_f11_p8_v1", "legacy_f10_p8_v1"],
        help="Required when merging state-dict-only legacy checkpoints",
    )
    parser.add_argument(
        "--metadata-out",
        help="Optional metadata JSON path. Defaults to the output checkpoint path with .json suffix.",
    )
    parser.add_argument(
        "--allow-legacy-task-arithmetic",
        action="store_true",
        help="Allow Task Arithmetic without verifiable shared-base provenance (legacy reproduction only)",
    )
    return parser.parse_args()


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def checkpoint_dir(data_source: str) -> Path:
    return repo_root() / "daniel" / "trained_network" / data_source


def resolve_checkpoint(value: str, data_source: str) -> Path:
    path = Path(value)
    if path.exists():
        return path.resolve()
    if path.suffix == ".pth" or any(part in value for part in ("/", "\\")):
        return path.resolve()
    return (checkpoint_dir(data_source) / f"{value}.pth").resolve()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_weights(raw_weights: list[float], count: int, method: str) -> list[float]:
    if raw_weights is None:
        if method == "soup":
            return [1.0 / count] * count
        return [1.0] * count
    if len(raw_weights) != count:
        raise ValueError("--weights must have the same count as --checkpoints")
    if method == "soup":
        weight_sum = sum(raw_weights)
        if weight_sum == 0:
            raise ValueError("Soup weights must not sum to zero")
        return [weight / weight_sum for weight in raw_weights]
    return raw_weights


def validate_compatible(reference: dict, candidate: dict, candidate_name: str) -> None:
    if list(candidate.keys()) != list(reference.keys()):
        raise ValueError(f"{candidate_name} has different checkpoint keys")
    for key, ref_value in reference.items():
        value = candidate[key]
        if not hasattr(ref_value, "shape"):
            if value != ref_value:
                raise ValueError(f"{candidate_name}:{key} is non-tensor and differs from the reference")
            continue
        if value.shape != ref_value.shape:
            raise ValueError(
                f"{candidate_name}:{key} shape mismatch: {tuple(value.shape)} != {tuple(ref_value.shape)}"
            )


def merge_soup(checkpoints: list[dict], weights: list[float]) -> dict:
    import torch
    sys.path.insert(0, str(repo_root() / "daniel"))
    from checkpointing import CHECKPOINT_SCHEMA_VERSION, unwrap_checkpoint

    merged = {}
    for key in checkpoints[0].keys():
        values = [checkpoint[key] for checkpoint in checkpoints]
        first = values[0]
        if torch.is_tensor(first) and (first.is_floating_point() or first.is_complex()):
            merged[key] = sum(weight * value for weight, value in zip(weights, values))
        elif torch.is_tensor(first):
            if not all(torch.equal(first, value) for value in values[1:]):
                raise ValueError(f"Non-floating tensor differs for key {key}; cannot safely merge")
            merged[key] = first.clone()
        else:
            merged[key] = first
    return merged


def merge_task_arithmetic(base: dict, checkpoints: list[dict], weights: list[float]) -> dict:
    import torch

    merged = {}
    for key in base.keys():
        base_value = base[key]
        values = [checkpoint[key] for checkpoint in checkpoints]
        if torch.is_tensor(base_value) and (base_value.is_floating_point() or base_value.is_complex()):
            delta = sum(weight * (value - base_value) for weight, value in zip(weights, values))
            merged[key] = base_value + delta
        elif torch.is_tensor(base_value):
            if not all(torch.equal(base_value, value) for value in values):
                raise ValueError(f"Non-floating tensor differs for key {key}; cannot safely merge")
            merged[key] = base_value.clone()
        else:
            merged[key] = base_value
    return merged


def main() -> None:
    args = parse_args()
    if args.method == "task_arithmetic" and not args.base:
        raise ValueError("--base is required for task_arithmetic")
    if args.out:
        out_path = Path(args.out).resolve()
    elif args.out_name:
        out_path = (checkpoint_dir(args.data_source) / f"{args.out_name}.pth").resolve()
    else:
        raise ValueError("Provide either --out-name or --out")

    import torch

    paths = [resolve_checkpoint(value, args.data_source) for value in args.checkpoints]
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(path)
    weights = normalize_weights(args.weights, len(paths), args.method)

    loaded = [unwrap_checkpoint(torch.load(path, map_location="cpu")) for path in paths]
    checkpoints = [item[0] for item in loaded]
    source_metadata = [item[1] for item in loaded]
    base_path = resolve_checkpoint(args.base, args.data_source) if args.base else None
    if base_path:
        base_checkpoint, base_metadata = unwrap_checkpoint(torch.load(base_path, map_location="cpu"))
    else:
        base_checkpoint, base_metadata = None, None

    reference = base_checkpoint if base_checkpoint is not None else checkpoints[0]
    for path, checkpoint in zip(paths, checkpoints):
        validate_compatible(reference, checkpoint, str(path))

    if args.method == "soup":
        merged = merge_soup(checkpoints, weights)
    else:
        validate_compatible(reference, base_checkpoint, str(base_path))
        base_hash = sha256(base_path)
        shared_base_verified = all(
            any(parent.get("sha256") == base_hash for parent in metadata.get("parents", []))
            for metadata in source_metadata
        )
        if not shared_base_verified and not args.allow_legacy_task_arithmetic:
            raise ValueError(
                "Task Arithmetic requires every specialist checkpoint to record the selected base "
                "as a parent. Use --allow-legacy-task-arithmetic only to reproduce inherited artifacts."
            )
        merged = merge_task_arithmetic(base_checkpoint, checkpoints, weights)

    metadata = {
        "method": args.method,
        "output": str(out_path),
        "data_source": args.data_source,
        "checkpoints": [
            {"path": str(path), "weight": weight, "sha256": sha256(path)}
            for path, weight in zip(paths, weights)
        ],
    }
    if base_path is not None:
        metadata["base"] = {"path": str(base_path), "sha256": sha256(base_path)}
        metadata["shared_base_verified"] = shared_base_verified

    source_schemas = {
        item.get("config", {}).get("feature_schema")
        for item in source_metadata
        if item.get("config", {}).get("feature_schema")
    }
    if len(source_schemas) > 1:
        raise ValueError(f"Source checkpoints declare different feature schemas: {source_schemas}")
    declared_schema = next(iter(source_schemas), None)
    if args.feature_schema and declared_schema and args.feature_schema != declared_schema:
        raise ValueError(
            f"--feature-schema {args.feature_schema} conflicts with checkpoint metadata {declared_schema}"
        )
    feature_schema = declared_schema or args.feature_schema
    if feature_schema is None:
        raise ValueError("--feature-schema is required for legacy state-dict-only merge sources")
    metadata["feature_schema"] = feature_schema

    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged_config = dict(source_metadata[0].get("config", {}))
    merged_config["feature_schema"] = feature_schema
    bundle = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "artifact_type": "merged_checkpoint",
        "state_dict": merged,
        "config": merged_config,
        "parents": metadata["checkpoints"],
        "merge": metadata,
    }
    torch.save(bundle, out_path)

    metadata_path = Path(args.metadata_out).resolve() if args.metadata_out else out_path.with_suffix(".json")
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Wrote merged checkpoint: {out_path}")
    print(f"Wrote merge metadata: {metadata_path}")
    for item in metadata["checkpoints"]:
        print(f"  {item['weight']:.4f}  {item['path']}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
