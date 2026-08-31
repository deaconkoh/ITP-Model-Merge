"""Verification gate for frozen final-test evaluation manifests."""

from __future__ import annotations

import json
from pathlib import Path

from checkpointing import file_sha256
from data_manifest import dataset_fingerprint


def verify_frozen_evaluation(manifest_path, checkpoint_path, test_directory) -> None:
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    if manifest.get("status") != "FROZEN_BEFORE_FINAL_TEST":
        raise ValueError("Evaluation manifest is not frozen for final-test use")

    checkpoint_hash = file_sha256(checkpoint_path)
    allowed_checkpoints = {item["sha256"] for item in manifest.get("checkpoints", [])}
    if checkpoint_hash not in allowed_checkpoints:
        raise ValueError(f"Checkpoint {checkpoint_path} is not frozen in the evaluation manifest")

    actual_test = dataset_fingerprint(test_directory)
    expected_test = manifest.get("splits", {}).get("test")
    if expected_test is None:
        raise ValueError("Evaluation manifest has no test split fingerprint")
    if actual_test["sha256"] != expected_test["sha256"]:
        raise ValueError(
            f"Test dataset fingerprint changed: {actual_test['sha256']} != {expected_test['sha256']}"
        )
