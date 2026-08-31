"""Versioned checkpoint bundles with backward-compatible legacy loading."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from importlib import metadata as importlib_metadata
from datetime import datetime, timezone
from pathlib import Path


CHECKPOINT_SCHEMA_VERSION = 2


def runtime_environment() -> dict[str, object]:
    packages = {}
    for package in ("numpy", "torch", "pandas", "scipy", "ortools"):
        try:
            packages[package] = importlib_metadata.version(package)
        except importlib_metadata.PackageNotFoundError:
            packages[package] = "NOT_INSTALLED"
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": packages,
    }


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_state(repo_root: str | Path = ".") -> dict[str, object]:
    root = Path(repo_root)
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True, text=True
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"], cwd=root, check=True, capture_output=True, text=True
            ).stdout.strip()
        )
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = "UNKNOWN", True
    return {"commit": commit, "dirty": dirty}


def make_checkpoint_bundle(
    state_dict,
    *,
    config: dict,
    data_fingerprints: dict,
    update: int,
    validation_metrics: dict[str, float],
    selection_metric: float,
    parents: list[dict] | None = None,
    repo_root: str | Path = ".",
) -> dict:
    return {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "state_dict": state_dict,
        "config": config,
        "data_fingerprints": data_fingerprints,
        "update": int(update),
        "validation_metrics": {key: float(value) for key, value in validation_metrics.items()},
        "selection_metric": float(selection_metric),
        "parents": parents or [],
        "git": git_state(repo_root),
        "runtime": runtime_environment(),
    }


def unwrap_checkpoint(checkpoint) -> tuple[object, dict]:
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint and "schema_version" in checkpoint:
        return checkpoint["state_dict"], checkpoint
    return checkpoint, {"schema_version": 1, "legacy": True, "provenance": "UNKNOWN"}


def infer_state_dict_family(state_dict) -> tuple[int, int, int] | None:
    """Infer DANIEL operation/machine/pair widths from stable tensor shapes."""
    op_key = "feature_exact.op_attention_blocks.0.attention_0.W"
    machine_key = "feature_exact.mch_attention_blocks.0.attention_0.W"
    actor_key = "actor.linears.0.weight"
    if not all(key in state_dict for key in (op_key, machine_key, actor_key)):
        return None
    operation_dim = int(state_dict[op_key].shape[0])
    machine_dim = int(state_dict[machine_key].shape[0])
    actor_input_dim = int(state_dict[actor_key].shape[1])
    pair_dim = actor_input_dim - 32  # four pooled vectors, final DAN embedding width 8
    return operation_dim, machine_dim, pair_dim


def validate_checkpoint_schema(metadata: dict, expected_schema: str, state_dict=None) -> None:
    from feature_schemas import get_feature_schema

    declared = metadata.get("config", {}).get("feature_schema")
    if metadata.get("legacy"):
        if not expected_schema.startswith("legacy_"):
            raise ValueError(
                "State-dict-only checkpoint has unknown feature semantics. Select an explicit "
                "legacy feature schema before loading it."
            )
        if state_dict is not None:
            inferred = infer_state_dict_family(state_dict)
            schema = get_feature_schema(expected_schema)
            expected = (schema.operation_dim, schema.machine_dim, schema.pair_dim)
            if inferred is not None and inferred != expected:
                raise ValueError(
                    f"Legacy checkpoint tensor family {inferred} does not match schema "
                    f"{expected_schema} {expected}"
                )
        return
    if declared is None:
        raise ValueError("Versioned checkpoint is missing config.feature_schema")
    if declared != expected_schema:
        raise ValueError(f"Checkpoint schema {declared!r} does not match requested {expected_schema!r}")


def json_safe_config(config) -> dict:
    safe = {}
    for key, value in vars(config).items():
        try:
            json.dumps(value)
            safe[key] = value
        except TypeError:
            safe[key] = str(value)
    return safe
