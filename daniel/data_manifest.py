"""Dataset identity and split-leakage safeguards."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dataset_files(directory: str | Path) -> list[Path]:
    root = Path(directory).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {root}")
    return sorted(path for path in root.rglob("*") if path.is_file() and not path.name.startswith("."))


def dataset_fingerprint(directory: str | Path) -> dict:
    root = Path(directory).resolve()
    files = dataset_files(root)
    entries = []
    for path in files:
        entry = {"path": str(path.relative_to(root)), "sha256": sha256_file(path)}
        if path.suffix.lower() == ".fjs":
            entry["structural_sha256"] = structural_fjsp_sha256(path)
        entries.append(entry)
    digest = hashlib.sha256(
        json.dumps(entries, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {"root": str(root), "sha256": digest, "files": entries}


def assert_disjoint_splits(**splits: str | Path) -> dict[str, dict]:
    """Reject path aliases and exact file-content overlap among dataset splits."""
    fingerprints = {name: dataset_fingerprint(path) for name, path in splits.items()}
    roots = [entry["root"] for entry in fingerprints.values()]
    if len(roots) != len(set(roots)):
        raise ValueError("Training, validation, and test dataset directories must be distinct")

    owners: dict[str, tuple[str, str]] = {}
    structural_owners: dict[str, tuple[str, str]] = {}
    overlaps: list[str] = []
    for split, fingerprint in fingerprints.items():
        for entry in fingerprint["files"]:
            previous = owners.get(entry["sha256"])
            if previous is not None:
                overlaps.append(
                    f"{previous[0]}:{previous[1]} == {split}:{entry['path']} ({entry['sha256']})"
                )
            else:
                owners[entry["sha256"]] = (split, entry["path"])
            structural_hash = entry.get("structural_sha256")
            if structural_hash is not None:
                previous_structural = structural_owners.get(structural_hash)
                if previous_structural is not None:
                    overlaps.append(
                        f"structural instance {previous_structural[0]}:{previous_structural[1]} == "
                        f"{split}:{entry['path']} ({structural_hash})"
                    )
                else:
                    structural_owners[structural_hash] = (split, entry["path"])
    if overlaps:
        raise ValueError("Exact dataset leakage detected:\n" + "\n".join(overlaps[:20]))
    return fingerprints


def _parse_job_line(tokens: list[int], extended: bool) -> list[tuple[int, list[tuple[int, int]]]]:
    operation_count = tokens[0]
    idx = 1
    operations = []
    for _ in range(operation_count):
        machine_count = tokens[idx]
        idx += 1
        if extended:
            idx += 1  # operation priority
        pairs = []
        for _ in range(machine_count):
            machine = tokens[idx]
            processing_time = tokens[idx + 1]
            pairs.append((machine, processing_time))
            idx += 3 if extended else 2  # extended rows also contain carbon
        operations.append((machine_count, pairs))
    if idx != len(tokens):
        raise ValueError("Unexpected tokens after final operation")
    return operations


def structural_fjsp_sha256(path: str | Path) -> str:
    """Hash jobs, compatibility, and processing times while ignoring annotations."""
    lines = [line.strip() for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(lines) < 2:
        raise ValueError(f"Malformed FJSP instance: {path}")
    parsed_jobs = []
    for line in lines[1:]:
        tokens = [int(float(token)) for token in line.split()]
        parsed = None
        for extended in (False, True):
            try:
                parsed = _parse_job_line(tokens, extended)
                break
            except (IndexError, ValueError):
                continue
        if parsed is None:
            raise ValueError(f"Cannot parse standard or extended FJSP line in {path}: {line}")
        parsed_jobs.append(parsed)
    canonical = json.dumps(parsed_jobs, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
