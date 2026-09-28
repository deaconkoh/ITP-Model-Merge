"""Frozen reference values for the v2 reward (results/v2_fixes.md, A3).

The v2 reward normalises each objective by the Step 4 score's references, so that an episode's return is minus its
Step 4 score:
  MKref_i, CBref_i  the seed's Step 4 centroid composition (simplex_m333c333t334_<dd tag>_s<seed>) evaluated greedily
                    on instance i (canonical per-instance evaluation);
  Tbar_s            that composition's mean total tardiness over the 100 train_vali instances (Step 4's normaliser).
They are computed ONCE by tools/build_reward_refs.py and frozen here with a SHA-256 of every instance file; training
READS them and fails loudly on a missing instance, seed or hash mismatch (same protocol as due_dates.py).
"""
from __future__ import annotations
import json
from pathlib import Path

from data_utils import sorted_instance_files
from due_dates import DATA_ROOT, file_sha256, instance_key

MANIFEST = DATA_ROOT / "reward_refs" / "reward_ref_manifest.json"


def load_manifest(path=None) -> dict:
    path = Path(path or MANIFEST)
    if not path.exists():
        raise FileNotFoundError(f"reward-reference manifest not found: {path}")
    return json.loads(path.read_text())


def refs_for_directory(directory, manifest: dict, seed: int):
    """(list of (MKref, CBref) in loader order, Tbar) for every instance in `directory`, hash-checked."""
    s = str(int(seed))
    if s not in manifest["seeds"]:
        raise KeyError(f"reward references have no seed {s} (have {sorted(manifest['seeds'])})")
    out = []
    for f in sorted_instance_files(str(directory)):
        key = instance_key(f)
        rec = manifest["instances"].get(key)
        if rec is None:
            raise KeyError(f"reward references missing instance {key}")
        if rec["sha256"] != file_sha256(f):
            raise ValueError(f"instance file changed since the reward references were frozen: {key}")
        out.append((float(rec["mk"][s]), float(rec["cb"][s])))
    return out, float(manifest["seeds"][s]["tbar"])
