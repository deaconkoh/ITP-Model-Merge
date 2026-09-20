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
PAT = re.compile(r"simplex_m(\d+)c(\d+)p(\d+)_s(\d+)")  # \d+ not \d{3}: the pure vertices are m1000c000p000 etc.


def load_all(size, pool_tag):
    d = REPO / "daniel/test_results/SD2" / f"{pool_tag}_{size}+carbon+priority"
    out = {}
    for f in sorted(d.glob("*simplex_*.npy")):
        m = PAT.search(f.name)
        if not m:
            continue
        pts = (int(m.group(1)), int(m.group(2)), int(m.group(3)))
        assert sum(pts) == 1000, f"parsed weights do not sum to 1000: {f.name} -> {pts}"
        out.setdefault(pts, {})[int(m.group(4))] = np.load(f)[:, :3]
    # A composition present for only some seeds would be scored on an uneven footing, and a
    # silently dropped one (as the pure vertices once were) understates the edge case.
    seeds = sorted(set().union(*[set(v) for v in out.values()]))
    ragged = {p: sorted(v) for p, v in out.items() if sorted(v) != seeds}
    if ragged:
        print(f"WARNING: {len(ragged)} composition(s) are missing some seeds; expected {seeds}")
        for p, s in list(ragged.items())[:5]:
            print(f"    m{p[0]:03d}c{p[1]:03d}p{p[2]:03d}: has {s}")
    print(f"loaded {len(out)} compositions x {len(seeds)} seeds "
          f"({sum(len(v) for v in out.values())} result files)")
    return out


def analyse(data, pts, centroid, size, pref, args):
    print("\n" + "#" * 78)
    print(f"{size}: {len(pts)} compositions, preference {pref.tag}, "
          f"reference composition = m{centroid[0]}c{centroid[1]}p{centroid[2]}")
    print("#" * 78)

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

    # ---- MARGIN: how far from the edge, and is the priority weight meaningfully non-zero?
    print(f"\n  MARGIN (how decisive is the verdict, not just pass/fail):")
    for s in seeds:
        b = pooled_best[s]
        dist_edge = min(b)                       # permille distance to the nearest face
        print(f"    seed {s}: distance to nearest edge = {dist_edge/10:5.1f} percentage points"
              f"   (a point ON an edge has 0)")
    margins = [min(pooled_best[s]) for s in seeds]
    print(f"    across seeds: min={min(margins)/10:.1f}pp  mean={np.mean(margins)/10:.1f}pp  "
          f"max={max(margins)/10:.1f}pp")

    # how much worse is the best EDGE composition than the best interior one?
    print(f"\n  cost of being forced onto an edge (best interior vs best edge composition):")
    for s in seeds:
        ref = data[centroid][s]
        vals = {p: float(scalarise(data[p][s], ref, pref).mean()) for p in pts}
        inter = {p: v for p, v in vals.items() if min(p) >= args.edge_tol}
        edge = {p: v for p, v in vals.items() if min(p) < args.edge_tol}
        if inter and edge:
            bi, be = min(inter.values()), min(edge.values())
            print(f"    seed {s}: best interior {bi:.4f}   best edge {be:.4f}   "
                  f"interior advantage {100*(be-bi)/be:+.2f}%"
                  f"{'  (edge is as good or better)' if bi >= be else ''}")

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
    return interior


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", default="10x5")
    ap.add_argument("--pool-tag", default="trainvali")
    ap.add_argument("--pref", nargs=3, type=float, default=[1 / 3, 1 / 3, 1 / 3])
    ap.add_argument("--prefs", nargs="+",
                    help="extra preferences as m,c,p triples e.g. 0.25,0.5,0.25")
    ap.add_argument("--edge-tol", type=int, default=50,
                    help="permille below which a specialist counts as contributing nothing")
    ap.add_argument("--min-points", type=int, default=40,
                    help="refuse a verdict below this many evaluated compositions")
    args = ap.parse_args()

    data = load_all(args.size, args.pool_tag)
    if not data:
        print(f"no simplex evaluations found for {args.size}/{args.pool_tag} -- run Step 3 first")
        return 1
    pts = sorted(data)
    centroid = min(pts, key=lambda p: sum(abs(a - b) for a, b in zip(p, (333, 333, 334))))

    # Guard against a spurious verdict from a partially-complete sweep: with only a handful
    # of compositions the "optimum" is whatever happens to have been evaluated, and an
    # interior point can win by default rather than on merit.
    if len(pts) < args.min_points:
        print(f"REFUSING TO RENDER A VERDICT: only {len(pts)} compositions evaluated "
              f"(need >= {args.min_points}).")
        print("  A sparse sweep can return INTERIOR simply because no edge point was")
        print("  evaluated. Let Step 3 finish, then re-run.")
        return 2

    prefs = [Preference(*args.pref)]
    if args.prefs:
        prefs += [Preference(*[float(x) for x in t.split(",")]) for t in args.prefs]

    verdicts = {}
    for pref in prefs:
        verdicts[pref.tag] = analyse(data, pts, centroid, args.size, pref, args)

    if len(prefs) > 1:
        print("\n" + "=" * 78)
        print("SUMMARY ACROSS PREFERENCES  (scoring is free -- same evaluations, re-weighted)")
        print("=" * 78)
        for tag, v in verdicts.items():
            print(f"  {tag}: {'INTERIOR' if v else 'EDGE'}")
    # the gate is decided by the PRIMARY preference (the first one)
    return 0 if verdicts[prefs[0].tag] else 1


if __name__ == "__main__":
    raise SystemExit(main())
