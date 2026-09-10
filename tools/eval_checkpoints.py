"""Evaluate arbitrary checkpoints on a named pool, 1 greedy repeat, saving per-instance .npy.

Greedy decoding is deterministic, so 1 repeat is sufficient (verified: the 5 repeats in
test_trained_model.py are bit-identical on all metric columns).

Usage (from repo root):
    python tools/eval_checkpoints.py --pool-root ./data/data_train_vali --pool 10x5+carbon+priority \
        --out-tag trainvali --models A B C
"""
import sys, os, time, argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "daniel"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool-root", default="./data/data_train_vali")
    ap.add_argument("--pool", required=True, help="dataset dir name under <pool-root>/SD2")
    ap.add_argument("--out-tag", required=True, help="prefix for the output results dir")
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--seed-test", type=int, default=50)
    args = ap.parse_args()

    sys.argv = ["eval_checkpoints", "--device", "cuda"]
    import numpy as np
    from data_utils import load_priority_carbon_data_from_files
    import test_trained_model as T

    pool = f"{args.pool_root}/SD2/{args.pool}"
    data_set = load_priority_carbon_data_from_files(pool)
    n = len(data_set[0])
    out_dir = f"./test_results/SD2/{args.out_tag}_{args.pool}"
    os.makedirs(out_dir, exist_ok=True)
    print(f"pool={pool} n={n} -> {out_dir}", flush=True)

    for i, name in enumerate(args.models, 1):
        out = f"{out_dir}/Result_DANIELG+{name}_{args.out_tag}_{args.pool}.npy"
        if os.path.exists(out):
            print(f"  [{i}/{len(args.models)}] skip (exists) {name}", flush=True)
            continue
        ck = f"./trained_network/SD2/{name}.pth"
        if not os.path.exists(ck):
            print(f"  [{i}/{len(args.models)}] MISSING {ck}", flush=True)
            continue
        t0 = time.time()
        res = T.test_greedy_strategy(data_set, ck, args.seed_test)
        np.save(out, res)
        print(f"  [{i}/{len(args.models)}] {name}: {time.time()-t0:.1f}s "
              f"makespan={res[:,0].mean():.1f} carbon={res[:,1].mean():.1f}", flush=True)
    print("EVAL_CHECKPOINTS_DONE", flush=True)


if __name__ == "__main__":
    main()
