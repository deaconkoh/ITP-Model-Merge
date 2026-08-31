"""Freeze model and split choices before accessing final-test performance."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Create an immutable final-evaluation manifest")
    parser.add_argument("--config", required=True)
    parser.add_argument("--train", required=True)
    parser.add_argument("--validation", required=True)
    parser.add_argument("--test", required=True)
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "daniel"))
    from checkpointing import file_sha256, git_state
    from data_manifest import assert_disjoint_splits

    output = Path(args.out).resolve()
    if output.exists():
        raise FileExistsError(f"Evaluation manifests are immutable; refusing to overwrite {output}")
    config_path = Path(args.config).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    splits = assert_disjoint_splits(train=args.train, validation=args.validation, test=args.test)
    checkpoints = []
    for value in args.checkpoints:
        path = Path(value).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        checkpoints.append({"path": str(path), "sha256": file_sha256(path)})

    manifest = {
        "schema_version": 1,
        "status": "FROZEN_BEFORE_FINAL_TEST",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": config,
        "config_path": str(config_path),
        "config_sha256": file_sha256(config_path),
        "splits": splits,
        "checkpoints": checkpoints,
        "git": git_state(root),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote frozen evaluation manifest: {output}")


if __name__ == "__main__":
    main()
