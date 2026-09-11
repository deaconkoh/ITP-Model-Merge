"""Step 7 analysis: budget curves, crossovers (pooled AND per-seed), and Pareto fronts.

Budget curves     quality vs PPO updates for every arm, with per-seed spread
Crossovers        reported pooled AND per-seed; a crossover carried by 2 of 3 seeds is NOT
                  a crossover, so both are printed side by side
Pareto fronts     in raw objective space (makespan, carbon, priority), because the reframe
                  is geometric and the evidence should be too
Trajectories      where each initialisation starts (budget 0) and ends (budget 1000), so
                  "cheap start, short path" is visible rather than asserted

Reads existing .npy evaluations; runs nothing itself.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from preference3 import Preference, scalarise  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
BUDGETS = [0, 50, 100, 250, 500, 1000]
ARMS = ["merge", "scratch", "spec_m", "spec_c", "spec_p"]


def load(size, pool_tag, tag):
    f = (REPO / "daniel/test_results/SD2" / f"{pool_tag}_{size}+carbon+priority" /
         f"Result_DANIELG+{size}+carbon+priority+{tag}_{pool_tag}_{size}+carbon+priority.npy")
    return np.load(f)[:, :3] if f.exists() else None


def pareto_front(pts):
    """Indices of non-dominated rows (all objectives minimised)."""
    keep = []
    for i, p in enumerate(pts):
        if not np.any(np.all(pts <= p, axis=1) & np.any(pts < p, axis=1)):
            keep.append(i)
    return keep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", default="10x5")
    ap.add_argument("--pool-tag", default="trainvali")
    ap.add_argument("--prefix", default="s4")
    ap.add_argument("--arms", nargs="+", default=ARMS)
    ap.add_argument("--seeds", nargs="+", type=int, default=[111, 222, 333, 444])
    ap.add_argument("--ref", required=True,
                    help="checkpoint tag of the reference composition (per-instance normaliser)")
    ap.add_argument("--pref", nargs=3, type=float, default=[1 / 3, 1 / 3, 1 / 3])
    args = ap.parse_args()
    pref = Preference(*args.pref)

    refs = {s: load(args.size, args.pool_tag, args.ref.format(seed=s)) for s in args.seeds}
    if any(v is None for v in refs.values()):
        print("reference composition evaluations missing; run Step 3 evaluation first"); return 1

    # ---------------- budget curves ----------------
    print("=" * 88)
    print(f"BUDGET CURVES  {args.size}  preference {pref.tag}  (lower is better)")
    print("=" * 88)
    curve = {}
    hdr = f"{'budget':>8}" + "".join(f"{a:>18}" for a in args.arms)
    print(hdr); print("-" * len(hdr))
    for b in BUDGETS:
        row = {}
        for a in args.arms:
            vals = []
            for s in args.seeds:
                m = load(args.size, args.pool_tag, f"{args.prefix}_{a}_s{s}@u{b}")
                vals.append(scalarise(m, refs[s], pref).mean() if m is not None else np.nan)
            row[a] = np.array(vals)
        curve[b] = row
        print(f"{b:>8}" + "".join(f"{np.nanmean(row[a]):11.4f}+-{np.nanstd(row[a], ddof=1):<5.4f}"
                                  for a in args.arms))

    # ---------------- crossovers ----------------
    print("\n" + "=" * 88)
    print("CROSSOVERS vs the merge arm at budget 0 (the zero-cost composition)")
    print("=" * 88)
    base = curve[0]["merge"]
    for a in args.arms:
        if a == "merge":
            continue
        pooled = next((b for b in BUDGETS if np.nanmean(curve[b][a]) < np.nanmean(base)), None)
        per = []
        for i, s in enumerate(args.seeds):
            c = next((b for b in BUDGETS if curve[b][a][i] < base[i]), None)
            per.append(f"s{s}:{c if c is not None else '>max'}")
        unanimous = all("＞" not in p and ">max" not in p for p in per)
        print(f"  {a:<8} pooled: {str(pooled) if pooled is not None else 'never':>6}"
              f"   per-seed: {', '.join(per)}"
              f"   {'(unanimous)' if unanimous else '(NOT unanimous -- do not call this a crossover)'}")

    # ---------------- sanity checks ----------------
    print("\n" + "=" * 88)
    print("SANITY CHECKS")
    print("=" * 88)
    mref = [load(args.size, args.pool_tag, args.ref.format(seed=s)) for s in args.seeds]
    m0 = [load(args.size, args.pool_tag, f"{args.prefix}_merge_s{s}@u0") for s in args.seeds]
    if all(x is not None for x in m0) and args.ref.find("{seed}") >= 0:
        d = max(float(np.abs(a - b).max()) for a, b in zip(mref, m0) if a is not None and b is not None)
        print(f"  merge@u0 reproduces the composition exactly : "
              f"{'PASS' if d < 1e-9 else f'FAIL (max diff {d:.3e})'}")
    sc0 = np.nanmean(curve[0]["scratch"]) if "scratch" in curve[0] else np.nan
    print(f"  scratch@u0 sits at random-init quality       : obj={sc0:.4f} "
          f"({'PASS -- far worse than merge' if sc0 > 1.15 else 'CHECK -- suspiciously good'})")

    # ---------------- Pareto fronts in objective space ----------------
    print("\n" + "=" * 88)
    print("PARETO FRONT in raw objective space (pool means per arm/budget)")
    print("=" * 88)
    pts, lbl = [], []
    for a in args.arms:
        for b in BUDGETS:
            v = [load(args.size, args.pool_tag, f"{args.prefix}_{a}_s{s}@u{b}") for s in args.seeds]
            v = [x for x in v if x is not None]
            if v:
                pts.append(np.stack([x.mean(0) for x in v]).mean(0)); lbl.append(f"{a}@{b}")
    if pts:
        P = np.stack(pts); front = set(pareto_front(P))
        print(f"  {'point':<18}{'makespan':>11}{'carbon':>11}{'priority':>11}   on front")
        for i, (l, p) in enumerate(zip(lbl, P)):
            print(f"  {l:<18}{p[0]:11.1f}{p[1]:11.1f}{p[2]:11.1f}   {'YES' if i in front else ''}")
        print(f"\n  {len(front)} of {len(P)} (arm, budget) points are Pareto-non-dominated")

    # ---------------- start -> end trajectories ----------------
    print("\n" + "=" * 88)
    print("TRAJECTORY: where each initialisation starts and ends")
    print("=" * 88)
    print(f"  {'arm':<10}{'obj@0':>10}{'obj@1000':>11}{'improvement':>13}{'budget to reach merge@0':>26}")
    for a in args.arms:
        o0, o1 = np.nanmean(curve[0][a]), np.nanmean(curve[1000][a])
        reach = next((b for b in BUDGETS if np.nanmean(curve[b][a]) <= np.nanmean(base)), None)
        print(f"  {a:<10}{o0:10.4f}{o1:11.4f}{100*(o0-o1)/o0:12.2f}%"
              f"{(str(reach) if reach is not None else 'never'):>26}")


if __name__ == "__main__":
    raise SystemExit(main() or 0)
