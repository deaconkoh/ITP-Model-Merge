"""Create a checksummed catalog without modifying inherited artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Catalog inherited checkpoints and result artifacts")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "daniel"))
    from checkpointing import file_sha256, infer_state_dict_family, unwrap_checkpoint

    entries = []
    checkpoint_root = root / "daniel" / "trained_network"
    try:
        import torch
    except ImportError:
        torch = None

    for path in sorted(checkpoint_root.rglob("*.pth")):
        entry = {
            "path": str(path.relative_to(root)),
            "sha256": file_sha256(path),
            "kind": "checkpoint",
            "scientific_use": "LEGACY_REFERENCE_OR_EXPLORATION_ONLY",
            "objective": "UNKNOWN_UNLESS_CORROBORATED_EXTERNALLY",
            "seed": "UNKNOWN",
        }
        if torch is not None:
            state_dict, metadata = unwrap_checkpoint(torch.load(path, map_location="cpu"))
            entry["architecture_family"] = infer_state_dict_family(state_dict) or "UNKNOWN"
            entry["checkpoint_schema_version"] = metadata.get("schema_version", 1)
            entry["provenance"] = "RECORDED" if not metadata.get("legacy") else "INCOMPLETE"
        else:
            entry["architecture_family"] = "REQUIRES_PYTORCH_INSPECTION"
            entry["provenance"] = "INCOMPLETE"
        entries.append(entry)

    result_roots = [root / "daniel" / "test_results", root / "experiments", root / "test_results"]
    for result_root in result_roots:
        if not result_root.exists():
            continue
        for path in sorted(item for item in result_root.rglob("*") if item.is_file()):
            entries.append({
                "path": str(path.relative_to(root)),
                "sha256": file_sha256(path),
                "kind": "historical_result",
                "scientific_use": "EXPLORATORY_OR_PROVENANCE_REVIEW_ONLY",
            })

    output = Path(args.out).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing catalog: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"schema_version": 1, "artifacts": entries}, indent=2), encoding="utf-8")
    print(f"Cataloged {len(entries)} artifacts in {output}")


if __name__ == "__main__":
    main()
