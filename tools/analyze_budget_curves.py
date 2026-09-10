"""Track C3: RQ1 quality-vs-budget curves.

Arms:
  merge      - zero-budget baseline, the lambda=0.3 soup (best static lambda under the
               target preference, unanimous across seeds)
  ftmerge    - fine-tune from that merged checkpoint
  scratch    - train from scratch
Both trained arms use goal 'mc' with the calibrated carbon_reward_weight = 0.002766, so
they optimise the same preference the merge targets (see preregistration/calibration).

Quality metric: the same scalarized preference used throughout,
    obj = 0.5 * carbon/carbon_ref + 0.5 * makespan/makespan_ref
with the per-instance reference being that seed's lambda=0.5 merge. Lower is better.

Run from the repository root.
"""
import sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[1]
POOL = "trainvali_10x5+carbon+priority"
SEEDS = [111, 222, 333]
BUDGETS = [0, 50, 100, 250, 500, 1000]
ARMS = ["ftmerge", "scratch"]
W = 0.5

D = REPO / "daniel/test_results/SD2" / POOL


def load(name):
    f = D / f"Result_DANIELG+{name}_{POOL}.npy"
    return np.load(f) if f.exists() else None


def objective(a, mref, cref):
    return W * (a[:, 1] / cref) + (1 - W) * (a[:, 0] / mref)


def main():
    # per-seed per-instance reference = that seed's lambda=0.5 merge
    ref = {}
    for s in SEEDS:
        r = load(f"10x5+carbon+priority+soup_c50_s{s}")
        if r is None:
            print(f"missing reference for seed {s}"); return
        ref[s] = (r[:, 0], r[:, 1])

    # zero-budget merging baseline (lambda = 0.3)
    merge = {}
    for s in SEEDS:
        a = load(f"10x5+carbon+priority+soup_c30_s{s}")
        merge[s] = objective(a, *ref[s]).mean()
    merge_vals = np.array([merge[s] for s in SEEDS])
    print("=" * 78)
    print("RQ1 BUDGET CURVES — 10x5, 3 seeds, preference w_carbon=0.5 (lower is better)")
    print("=" * 78)
    print(f"\nMERGING BASELINE (zero budget, lambda=0.3): "
          f"{merge_vals.mean():.4f} +/- {merge_vals.std(ddof=1):.4f}   per-seed {np.round(merge_vals,4)}")

    curves = {}
    print(f"\n{'budget':>8} | {'ftmerge mean+/-sd':>22} | {'scratch mean+/-sd':>22}")
    print("-" * 78)
    for b in BUDGETS:
        row = {}
        for arm in ARMS:
            vals = []
            for s in SEEDS:
                a = load(f"10x5+carbon+priority+c3_{arm}_s{s}@u{b}")
                vals.append(objective(a, *ref[s]).mean() if a is not None else np.nan)
            row[arm] = np.array(vals)
        curves[b] = row
        f_ = row["ftmerge"]; sc = row["scratch"]
        print(f"{b:>8} | {np.nanmean(f_):>10.4f} +/- {np.nanstd(f_, ddof=1):<8.4f} | "
              f"{np.nanmean(sc):>10.4f} +/- {np.nanstd(sc, ddof=1):<8.4f}")

    print("\n" + "-" * 78)
    print("CROSSOVER vs the merging baseline (first budget whose mean beats merging)")
    for arm in ARMS:
        cross = None
        for b in BUDGETS:
            if np.nanmean(curves[b][arm]) < merge_vals.mean():
                cross = b; break
        if cross is None:
            print(f"  {arm:8s}: NEVER crosses within {max(BUDGETS)} updates")
        else:
            print(f"  {arm:8s}: crosses at {cross} updates "
                  f"({np.nanmean(curves[cross][arm]):.4f} vs merge {merge_vals.mean():.4f})")
        # per-seed crossover, to show it is not a single-seed artifact
        per = []
        for i, s in enumerate(SEEDS):
            c = next((b for b in BUDGETS if curves[b][arm][i] < merge_vals[i]), None)
            per.append(f"s{s}:{c if c is not None else '>1000'}")
        print(f"            per-seed crossover: {', '.join(per)}")

    print("\n" + "-" * 78)
    print("FINE-TUNE-FROM-MERGE vs FROM-SCRATCH at matched budget")
    for b in BUDGETS:
        f_ = np.nanmean(curves[b]["ftmerge"]); sc = np.nanmean(curves[b]["scratch"])
        better = "ftmerge" if f_ < sc else "scratch"
        print(f"  budget {b:>4}: ftmerge={f_:.4f}  scratch={sc:.4f}  -> {better} better "
              f"(gap {abs(f_-sc):.4f})")

    print("\n" + "-" * 78)
    print("PRACTICAL READ: improvement over merging, in % of the merge baseline")
    for b in BUDGETS:
        for arm in ARMS:
            v = np.nanmean(curves[b][arm])
            print(f"  budget {b:>4} {arm:8s}: {100*(merge_vals.mean()-v)/merge_vals.mean():+6.2f}%")


if __name__ == "__main__":
    main()
