"""Build the frozen v2 reward-reference manifest (results/v2_fixes.md, A3). EVALUATION ONLY.

For each seed, the Step 4 centroid composition simplex_m333c333t334_k125s060_s<seed> (v1 checkpoint, v1 env) is
evaluated greedily on every instance of the 10x5 train, validation and train_vali pools (canonical: batches only of
instances sharing the maximum processing time). train_vali must reproduce the Step 4 result files exactly (else
abort). Writes daniel/data/reward_refs/reward_ref_manifest.json with a SHA-256 per instance file.
Never touches the final-test pools.

    python tools/build_reward_refs.py
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))
import ceiling_sampling as CS                          # noqa: E402  (canonical grouped greedy rollouts)
import complementarity as P                            # noqa: E402
from data_utils import load_priority_carbon_data_from_files, sorted_instance_files   # noqa: E402
from due_dates import due_dates_for_directory, load_manifest, instance_key, file_sha256, manifest_tag  # noqa: E402

DAN = REPO / "daniel"
OUT = DAN / "data/reward_refs/reward_ref_manifest.json"
TAG = "k125s060"
SEEDS = [111, 222, 333, 444]
POOLS = {"data_train": DAN / "data/data_train/SD2/10x5+carbon+priority",
         "data_validation": DAN / "data/data_validation/SD2/10x5+carbon+priority",
         "data_train_vali": DAN / "data/data_train_vali/SD2/10x5+carbon+priority"}


def main():
    if OUT.exists():
        raise SystemExit(f"{OUT} exists; references are frozen (delete deliberately to rebuild)")
    dd = load_manifest()
    inst, seeds = {}, {}
    for s in SEEDS:
        name = f"10x5+carbon+priority+simplex_m333c333t334_{TAG}_s{s}"
        pol = P.load_policy(name)
        for pool, path in POOLS.items():
            assert manifest_tag(path, dd) == TAG
            data = load_priority_carbon_data_from_files(str(path))
            due = due_dates_for_directory(path, dd)
            o = CS.run(pol, data, due, np.arange(len(data[0])))
            files = sorted_instance_files(str(path))
            if pool == "data_train_vali":
                ref = np.load(REPO / f"daniel/test_results/SD2/trainvali-{TAG}_10x5+carbon+priority/Result_DANIELG+"
                              f"{name}_trainvali-{TAG}_10x5+carbon+priority.npy")[:, [0, 1, 3]]
                mine = np.stack([o["mk"], o["cb"], o["tard"]], 1)
                if not np.array_equal(mine, ref):
                    raise SystemExit(f"seed {s}: centroid rollout does not reproduce Step 4 on train_vali")
                seeds[str(s)] = {"tbar": float(np.mean(o["tard"])), "checkpoint": name + ".pth",
                                 "checkpoint_sha256": file_sha256(P.CK / f"{name}.pth")}
            for k, f in enumerate(files):
                rec = inst.setdefault(instance_key(f), {"sha256": file_sha256(f), "pool": pool, "mk": {}, "cb": {}})
                rec["mk"][str(s)], rec["cb"][str(s)] = float(o["mk"][k]), float(o["cb"][k])
            print(f"seed {s} {pool}: {len(files)} instances, mean mk {o['mk'].mean():.2f} cb {o['cb'].mean():.2f}", flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "purpose": "v2 reward normalisers (results/v2_fixes.md, A3): per-instance MKref/CBref and per-seed Tbar",
        "composition": f"simplex_m333c333t334_{TAG}_s<seed> (Step 4 centroid), greedy, canonical per-instance",
        "tbar_definition": "mean total tardiness of the composition over the 100 train_vali instances (Step 4)",
        "dd_tag": TAG, "due_date_manifest_sha256": file_sha256(DAN / "data/due_dates/due_date_manifest.json"),
        "train_vali_reproduces_step4": True, "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "seeds": seeds, "instances": inst}, indent=1))
    print(f"wrote {OUT} ({len(inst)} instances); Tbar: " + ", ".join(f"{s}: {v['tbar']:.4f}" for s, v in seeds.items()))


if __name__ == "__main__":
    main()
