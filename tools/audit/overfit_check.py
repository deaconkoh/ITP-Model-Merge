"""Audit diagnostic: does training on a fixed pool of 63 instances overfit? (results/code_audit.md). EVALUATION ONLY.

Every 10x5 arm since Step 4 trains on the 63-file train split (train_from_files=True, sampled with replacement), where
the original DANIEL trains on freshly generated SD2 instances. For each saved snapshot (seed 111: the v2 re-run arms
and the Step 4 makespan and scratch arms) this scores greedy schedules with the Step 4 score on
  TRAIN       the 63 training instances (references: the frozen reward-reference manifest, seed 111 centroid),
  VALIDATION  the 18 validation instances (same references),
  HELD-OUT    the 49 HELD-OUT instances (Step 4 references).
A widening TRAIN-vs-HELD-OUT gap with more updates is overfitting. No repo code is changed.

    python tools/audit/overfit_check.py
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tests/audit"))
import audit_common as A                                # noqa: E402
sys.path.insert(0, str(REPO / "daniel"))
import reward_refs as RR                                # noqa: E402

SEED = 111
ARMS = {"V2-M": ("v2", [0, 50, 100, 250, 500, 1000, 1500, 2000, 2500, 3000]),
        "V2-S": ("v2", [0, 50, 100, 250, 500, 1000, 1500, 2000, 2500, 3000]),
        "V2-T": ("v2", [0, 50, 100, 250, 500, 1000, 1500, 2000, 2500, 3000]),
        "V1-CONT": ("v1", [1000, 1500, 2000, 2500, 3000]),
        "Step4 spec_m": ("v1", [0, 50, 100, 250, 500, 1000]),
        "Step4 scratch": ("v1", [0, 50, 100, 250, 500, 1000])}


def name(arm, u):
    if arm.startswith("Step4"):
        return f"10x5+carbon+priority+s4t_{arm.split()[1]}_k125s060_s{SEED}@u{u}"
    return f"v2rr/v2rr_{arm}_s{SEED}@u{u}"


def main():
    man = RR.load_manifest()
    pools = {}
    for split in ("data_train", "data_validation"):
        data, due = A.pool(split)
        pairs, tbar = RR.refs_for_directory(A.DAN / f"data/{split}/SD2/10x5+carbon+priority", man, SEED)
        pools[split] = (data, due, np.array([p[0] for p in pairs]), np.array([p[1] for p in pairs]), tbar)
    hdata, hdue = A.pool()
    out = {}
    for arm, (obs, ups) in ARMS.items():
        for u in ups:
            pol = A.V.load_policy(name(arm, u), obs)
            row = {}
            for split, (data, due, mkr, cbr, tbar) in pools.items():
                o = A.V.run(pol, data, due, np.arange(len(data[0])), obs=obs)
                row[split] = float(np.mean((o["mk"] / mkr + o["cb"] / cbr + o["tard"] / tbar) / 3))
            o = A.V.run(pol, hdata, hdue, A.HELD, obs=obs)
            row["held_out"] = float(np.mean([A.SC.score(SEED, i, o["mk"][k], o["cb"][k], o["tard"][k]) for k, i in enumerate(A.HELD)]))
            out.setdefault(arm, {})[u] = row
            print(f"{arm:<14} u{u:<5} train {row['data_train']:.4f}  validation {row['data_validation']:.4f}  "
                  f"held-out {row['held_out']:.4f}  gap(held-train) {row['held_out'] - row['data_train']:+.4f}", flush=True)
    A.write_result("overfit_check", out)


if __name__ == "__main__":
    main()
