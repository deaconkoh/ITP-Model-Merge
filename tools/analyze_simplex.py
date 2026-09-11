"""GATE 2: does the third specialist earn its place, or does the optimum sit on an edge?

If the pooled optimum has ~zero weight on one specialist, the composition collapses to the
2-specialist case already covered, and the three-objective reframe loses its central claim.
That is a go/no-go before committing the remaining budget to Steps 4-6.

Also reports where PER-INSTANCE optima sit, per seed (never seed-averaged -- seed-averaging
suppresses composition headroom by 58-65%).

Reads existing .npy evaluations of the simplex compositions; runs nothing itself.
"""
from __future__ import annotations
import argparse, re
from pathlib import Path
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from preference3 import Preference, scalarise  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
SEEDS = [111, 222, 333, 444]
PAT = re.compile(r"simplex_m(\d{3})c(\d{3})p(\d{3})_s(\d+)")


def load_all(size, pool_tag):
    d = REPO / "daniel/test_results/SD2" / f"{pool_tag}_{size}+carbon+priority"
    out = {}
    for f in sorted(d.glob("*simplex_*.npy")):
        m = PAT.search(f.name)
        if not m:
            continue
        pts = (int(m.group(1)), int(m.group(2)), int(m.group(3)))
        out.setdefault(pts, {})[int(m.group(4))] = np.load(f)[:, :3]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", default="10x5")
    ap.add_argument("--pool-tag", default="trainvali")
    ap.add_argument("--pref", nargs=3, type=float, default=[1 / 3, 1 / 3, 1 / 3])
    ap.add_argument("--edge-tol", type=int, default=50,
                    help="permille below which a specialist counts as contributing nothing")
    args = ap.parse_args()
    pref = Preference(*args.pref)

    data = load_all(args.size, args.pool_tag)
    if not data:
        print(f"no simplex evaluations found for {args.size}/{args.pool_tag} -- run Step 3 first")
        return 1
    pts = sorted(data)
    centroid = min(pts, key=lambda p: sum(abs(a - b) for a, b in zip(p, (333, 333, 334))))
    print(f"{args.size}: {len(pts)} compositions, preference {pref.tag}, "
          f"reference composition = m{centroid[0]}c{centroid[1]}p{centroid[2]}")

    seeds = sorted(set().union(*[set(v) for v in data.values()]))
    pooled_best, per_inst_rows = {}, {}
    for s in seeds:
        ref = data[centroid][s]                      # (n_inst, 3) per-instance reference
        M = np.stack([scalarise(data[p][s], ref, pref) for p in pts])   # (n_pts, n_inst)
        pooled_best[s] = pts[int(np.argmin(M.mean(1)))]
        per_inst_rows[s] = [pts[i] for i in M.argmin(0)]

    print(f"\n  pooled optimum per seed (never seed-averaged):")
    for s in seeds:
        b = pooled_best[s]
        edge = [n for n, w in zip("mcp", b) if w < args.edge_tol]
        print(f"    seed {s}: m={b[0]/10:5.1f}% c={b[1]/10:5.1f}% p={b[2]/10:5.1f}%"
              f"   {'EDGE (' + ','.join(edge) + ' ~ 0)' if edge else 'INTERIOR'}")

    # ---- the gate
    on_edge = [s for s in seeds if any(w < args.edge_tol for w in pooled_best[s])]
    p_weights = [pooled_best[s][2] / 10 for s in seeds]
    print(f"\n  priority-specialist weight at the pooled optimum: "
          f"{np.mean(p_weights):.1f}% (per seed {[round(w,1) for w in p_weights]})")

    print(f"\n  per-instance optima, mean composition per seed:")
    for s in seeds:
        arr = np.array(per_inst_rows[s]) / 10
        print(f"    seed {s}: m={arr[:,0].mean():5.1f}% c={arr[:,1].mean():5.1f}% "
              f"p={arr[:,2].mean():5.1f}%   distinct compositions used: {len(set(map(tuple, (arr*10).astype(int))))}")

    interior = len(on_edge) == 0
    print("\n" + "=" * 78)
    if interior:
        print("GATE 2: PASS -- pooled optimum is INTERIOR at every seed.")
        print("  All three specialists contribute; the three-objective composition is justified.")
    else:
        print(f"GATE 2: FAIL -- optimum sits on an EDGE for seed(s) {on_edge}.")
        print("  At least one specialist contributes ~nothing for this preference, so the")
        print("  composition reduces to the 2-specialist case already covered by Track C.")
        print("  STOP and reconsider before committing Steps 4-6.")
    return 0 if interior else 1


if __name__ == "__main__":
    raise SystemExit(main())
