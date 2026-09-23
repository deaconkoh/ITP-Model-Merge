"""Gate 2 supplement: is the interior optimum decisive, or a coin-flip over the edge?

analyze_simplex.py reports WHERE the optimum sits. This asks HOW MUCH it is worth:
  * the winning interior composition vs the winning edge composition, paired across the
    100 instances (both are maxima, so this is an upper bound on the true advantage --
    reported as such, not as an unbiased effect)
  * how flat the neighbourhood is: how many compositions sit within 0.5% of the best
  * what the optimum buys over the centroid, which is the composition Step 4 would
    otherwise start from
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from preference3 import Preference                      # noqa: E402
from analyze_simplex import load_all, REPO              # noqa: E402
from preference3 import scalarise                       # noqa: E402

CENTROID = (333, 333, 334)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", default="10x5")
    ap.add_argument("--pool-tag", default="trainvali")
    ap.add_argument("--prefs", nargs="+", default=["0.3333,0.3333,0.3334", "0.25,0.5,0.25"])
    ap.add_argument("--edge-tol", type=int, default=50)
    ap.add_argument("--third", choices=["p", "t"], default="p")
    ap.add_argument("--tag")
    args = ap.parse_args()

    data = load_all(args.size, args.pool_tag, args.third, args.tag)
    pooled = args.third == "t"
    pts = sorted(data)
    seeds = sorted(data[CENTROID])

    for tag in args.prefs:
        pref = Preference(*[float(x) for x in tag.split(",")], third=args.third)
        print("\n" + "=" * 78)
        print(f"preference {pref.tag}")
        print("=" * 78)
        for s in seeds:
            ref = data[CENTROID][s]
            per_inst = {p: scalarise(data[p][s], ref, pref, pooled) for p in pts}   # (n_inst,) each
            means = {p: float(v.mean()) for p, v in per_inst.items()}
            interior = [p for p in pts if min(p) >= args.edge_tol]
            edge = [p for p in pts if min(p) < args.edge_tol]
            bi = min(interior, key=lambda p: means[p])
            be = min(edge, key=lambda p: means[p])
            d = per_inst[be] - per_inst[bi]                   # >0 means interior better
            t, pv = stats.ttest_rel(per_inst[be], per_inst[bi])
            wins = float((d > 0).mean())
            flat = sum(1 for p in pts if means[p] <= means[bi] * 1.005)
            cen = means[CENTROID]
            print(f"  seed {s}:")
            print(f"    best interior m{bi[0]:03d}c{bi[1]:03d}p{bi[2]:03d} = {means[bi]:.4f}   "
                  f"best edge m{be[0]:03d}c{be[1]:03d}p{be[2]:03d} = {means[be]:.4f}")
            print(f"    paired over {len(d)} instances: interior wins {100*wins:.0f}%   "
                  f"t={t:.2f}  p={pv:.4f}   (upper bound: both are selected maxima)")
            print(f"    compositions within 0.5% of the best: {flat} of {len(pts)}")
            print(f"    optimum vs centroid: {100*(cen-means[bi])/cen:+.2f}%  "
                  f"(centroid {cen:.4f} -> {means[bi]:.4f})")


if __name__ == "__main__":
    raise SystemExit(main())
