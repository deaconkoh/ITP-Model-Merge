"""Full Gate 2 report for one composition triple, in one fixed format so triples can be compared.

Per preference and per seed (never seed-averaged): the pooled optimum and its weight on each
specialist, which face of the simplex it sits on, distance to the nearest edge, the interior-vs-edge
margin (paired across instances, then with seeds as the unit), how flat the optimum is, and the gain
over the centroid. Then:
  * stability across seeds -- how much worse each seed's optimum is when applied to the other seeds;
  * agreement between the two preferences, seed by seed;
  * a weight profile for one chosen specialist: the best composition at each fixed weight on it,
    relative to the best composition with that specialist at zero.

Reads existing evaluations only.

    python tools/gate2_full_report.py --third t --tag k125s060 --pool-tag trainvali-k125s060 --profile m
    python tools/gate2_full_report.py --third p --pool-tag trainvali --profile p
"""
from __future__ import annotations
import argparse, itertools, sys
from pathlib import Path
import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_simplex import load_all          # noqa: E402
from preference3 import Preference, scalarise  # noqa: E402

CENTROID = (333, 333, 334)
SEEDS = [111, 222, 333, 444]
PREFS = [(1 / 3, 1 / 3, 1 / 3), (0.25, 0.5, 0.25)]
NAMES = {"m": "makespan", "c": "carbon", "p": "priority", "t": "tardiness"}


def face(p, letters):
    zero = [letters[i] for i in range(3) if p[i] == 0]
    if not zero:
        return "interior"
    if len(zero) == 2:
        return f"vertex {next(l for l in letters if l not in zero)}"
    keep = [l for l in letters if l not in zero]
    return f"{keep[0]}-{keep[1]} edge"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--third", choices=["p", "t"], required=True)
    ap.add_argument("--tag")
    ap.add_argument("--pool-tag", required=True)
    ap.add_argument("--profile", default="m", help="specialist letter whose weight is profiled")
    args = ap.parse_args()
    L = "mc" + args.third
    data = load_all("10x5", args.pool_tag, args.third, args.tag)
    pts = sorted(data)
    grid = [p for p in pts if p != CENTROID]
    pooled = args.third == "t"
    opt = {}
    for w in PREFS:
        pref = Preference(*w, third=args.third)
        print("\n" + "=" * 96)
        print(f"PREFERENCE {pref.tag}   ({len(grid)} grid compositions + centroid, {len(SEEDS)} seeds)")
        print("=" * 96)
        per_inst, V = {}, {}
        for s in SEEDS:
            ref = data[CENTROID][s]
            per_inst[s] = {p: scalarise(data[p][s], ref, pref, pooled) for p in pts}
            V[s] = {p: float(v.mean()) for p, v in per_inst[s].items()}
        print(f"  {'seed':<6}{'optimum ' + '/'.join(L):<18}{'face':<16}{'edge dist':>10}"
              f"{'obj':>8}{'vs centroid':>13}{'within 0.5%':>13}")
        adv, pvals = [], []
        for s in SEEDS:
            b = min(grid, key=V[s].get)
            opt[(pref.tag, s)] = b
            print(f"  {s:<6}{'/'.join(f'{x // 10}' for x in b):<18}{face(b, L):<16}{min(b) / 10:>9.1f}pp"
                  f"{V[s][b]:>8.4f}{100 * (V[s][CENTROID] - V[s][b]) / V[s][CENTROID]:>+12.2f}%"
                  f"{sum(V[s][p] <= V[s][b] * 1.005 for p in grid):>10} of {len(grid)}")
        print(f"\n  weight on each specialist at the optimum (mean over seeds): "
              + "  ".join(f"{NAMES[l]} {np.mean([opt[(pref.tag, s)][i] for s in SEEDS]) / 10:.1f}%"
                          f" (sd {np.std([opt[(pref.tag, s)][i] for s in SEEDS], ddof=1) / 10:.1f})"
                          for i, l in enumerate(L)))
        print(f"\n  INTERIOR vs EDGE (best interior vs best edge composition; biased toward the interior,"
              f" both are selected maxima)")
        for s in SEEDS:
            inter = [p for p in grid if min(p) > 0]
            edge = [p for p in grid if min(p) == 0]
            bi, be = min(inter, key=V[s].get), min(edge, key=V[s].get)
            a = 100 * (V[s][be] - V[s][bi]) / V[s][be]
            t, pv = stats.ttest_rel(per_inst[s][be], per_inst[s][bi])
            adv.append(a); pvals.append(pv)
            print(f"    seed {s}: interior {'/'.join(str(x // 10) for x in bi)} {V[s][bi]:.4f}   "
                  f"edge {'/'.join(str(x // 10) for x in be)} ({face(be, L)}) {V[s][be]:.4f}   "
                  f"interior advantage {a:+.2f}%   paired over instances p={pv:.3g}")
        t, pv = stats.ttest_1samp(adv, 0)
        print(f"    seeds as the unit: mean {np.mean(adv):+.2f}%, {sum(x > 0 for x in adv)}/4 seeds favour "
              f"the interior, t={t:.2f}, p={pv:.2f}")

        # weight profile for the chosen specialist
        i = L.index(args.profile)
        print(f"\n  {NAMES[args.profile].upper()} WEIGHT PROFILE: best composition with {NAMES[args.profile]} "
              f"weight fixed, relative to the best with it at ZERO (positive = adding it costs)")
        levels = [0, 100, 200, 300, 500, 700]
        print(f"    {'seed':<6}" + "".join(f"{f'w={l // 10}%':>10}" for l in levels))
        rows = []
        for s in SEEDS:
            base = min(V[s][p] for p in grid if p[i] == 0)
            r = [100 * (min(V[s][p] for p in grid if p[i] == l) - base) / base for l in levels]
            rows.append(r)
            print(f"    {s:<6}" + "".join(f"{x:>+9.2f}%" for x in r))
        rows = np.array(rows)
        print(f"    {'mean':<6}" + "".join(f"{x:>+9.2f}%" for x in rows.mean(0)))
        for j, l in enumerate(levels[1:3], start=1):
            t, pv = stats.ttest_1samp(rows[:, j], 0)
            print(f"    seeds as the unit, w={l // 10}%: mean {rows[:, j].mean():+.2f}%, t={t:.2f}, p={pv:.2f}")

        # stability across seeds
        reg = np.array([[100 * (V[b][opt[(pref.tag, a)]] - V[b][opt[(pref.tag, b)]]) / V[b][opt[(pref.tag, b)]]
                         for b in SEEDS] for a in SEEDS])
        off = reg[~np.eye(4, dtype=bool)]
        print(f"\n  STABILITY: one seed's optimum applied to another seed, % worse than that seed's own optimum")
        print(f"    rows = optimum from seed, columns = evaluated on seed {SEEDS}")
        for a, row in zip(SEEDS, reg):
            print(f"    {a:<6}" + "".join(f"{x:>8.2f}%" for x in row))
        print(f"    off-diagonal mean {off.mean():.2f}%, max {off.max():.2f}%")

    print("\n" + "=" * 96)
    print("AGREEMENT BETWEEN THE TWO PREFERENCES, seed by seed")
    print("=" * 96)
    a, b = [Preference(*w, third=args.third).tag for w in PREFS]
    for s in SEEDS:
        pa, pb = opt[(a, s)], opt[(b, s)]
        d = sum(abs(x - y) for x, y in zip(pa, pb)) / 2 / 10
        print(f"  seed {s}: {a} {'/'.join(str(x // 10) for x in pa)} ({face(pa, L)})   "
              f"{b} {'/'.join(str(x // 10) for x in pb)} ({face(pb, L)})   moved {d:.0f}pp   "
              f"{'SAME face' if face(pa, L) == face(pb, L) else 'different face'}")


if __name__ == "__main__":
    main()
