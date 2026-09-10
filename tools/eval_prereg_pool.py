"""Pre-registered Track A evaluation: 20 checkpoints x 100 instances, 1 greedy repeat.

Evaluates on the data_train_vali pools (hash-verified disjoint from train/validation/
development and from BOTH reserved final-test sets). Writes per-instance .npy in the
same [makespan, carbon, priority_weighted_completion, seconds] format used elsewhere.

Uses 1 greedy repeat rather than test_trained_model.py's 5: the repeats were verified
bit-identical on all metric columns (max abs diff 0.0), so they only average the timing.

Run from the daniel/ directory.
"""
import sys, os, time
from pathlib import Path

# running as tools/eval_prereg_pool.py puts tools/ on sys.path[0], not daniel/
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "daniel"))
sys.argv = ["eval_prereg_pool", "--device", "cuda"]
import numpy as np
from data_utils import load_priority_carbon_data_from_files
import test_trained_model as T

SEEDS = [111, 222, 333, 444]
POOL_ROOT = "./data/data_train_vali/SD2"
OUT_ROOT = "./test_results/SD2"
SEED_TEST = 50  # configs.seed_test default, matches all prior evaluations


def models_for(size):
    names = []
    for s in SEEDS:
        names.append(f"{size}+carbon+priority+m_s{s}")   # lambda = 0.0
        for lam in (30, 50, 70):
            names.append(f"{size}+carbon+priority+soup_c{lam}_s{s}")
        names.append(f"{size}+carbon+priority+c_s{s}")   # lambda = 1.0
    return names


def main():
    for size in ["10x5", "20x10"]:
        pool = f"{POOL_ROOT}/{size}+carbon+priority"
        data_set = load_priority_carbon_data_from_files(pool)
        n = len(data_set[0])
        out_dir = f"{OUT_ROOT}/trainvali_{size}+carbon+priority"
        os.makedirs(out_dir, exist_ok=True)
        print(f"### {size}: {n} instances from {pool}", flush=True)

        for i, name in enumerate(models_for(size), 1):
            out = f"{out_dir}/Result_DANIELG+{name}_trainvali_{size}+carbon+priority.npy"
            if os.path.exists(out):
                print(f"  [{i:2d}/20] skip (exists) {name}", flush=True)
                continue
            t0 = time.time()
            res = T.test_greedy_strategy(data_set, f"./trained_network/SD2/{name}.pth", SEED_TEST)
            np.save(out, res)
            print(f"  [{i:2d}/20] {name}: {time.time()-t0:.1f}s  "
                  f"mean_makespan={res[:,0].mean():.1f} mean_carbon={res[:,1].mean():.1f}", flush=True)
    print("PREREG_EVAL_DONE", flush=True)


if __name__ == "__main__":
    main()
