"""GATE 1: do the newly trained priority specialists actually behave as specialists?

A priority specialist must achieve the BEST priority-weighted completion of the three, and
pay for it in makespan and/or carbon. If it does not, the merge has nothing meaningful to
compose and Steps 2-6 are pointless.

Reads existing per-instance .npy evaluations; runs nothing itself.
Columns of each .npy: [makespan, carbon, priority_weighted_completion, seconds].
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[1]
SEEDS = [111, 222, 333, 444]
OBJ = {"m": "makespan", "c": "carbon", "p": "priority"}
COL = {"makespan": 0, "carbon": 1, "priority": 2}


def load(size, tag, pool_tag):
    f = (REPO / "daniel/test_results/SD2" / f"{pool_tag}_{size}+carbon+priority" /
         f"Result_DANIELG+{size}+carbon+priority+{tag}_{pool_tag}_{size}+carbon+priority.npy")
    return np.load(f) if f.exists() else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", nargs="+", default=["10x5", "20x10"])
    ap.add_argument("--pool-tag", default="trainvali")
    args = ap.parse_args()

    overall_ok = True
    for size in args.sizes:
        print("=" * 78)
        print(f"{size}: specialist behaviour check (pool={args.pool_tag})")
        print("=" * 78)
        means = {}
        missing = []
        for o in ("m", "c", "p"):
            per_seed = []
            for s in SEEDS:
                a = load(size, f"{o}_s{s}", args.pool_tag)
                if a is None:
                    missing.append(f"{o}_s{s}")
                else:
                    per_seed.append(a[:, :3].mean(0))
            if per_seed:
                means[o] = np.stack(per_seed)
        if missing:
            print(f"  MISSING evaluations: {missing}\n  -> run the Step 1 evaluation first\n")
            overall_ok = False
            continue

        print(f"  {'specialist':<12}{'makespan':>12}{'carbon':>12}{'priority':>12}")
        for o in ("m", "c", "p"):
            mu = means[o].mean(0); sd = means[o].std(0, ddof=1)
            print(f"  {OBJ[o]+' spec':<12}" + "".join(f"{mu[i]:9.1f}+-{sd[i]:<4.1f}" for i in range(3)))

        # each specialist must win its own objective
        print("\n  specialist must be best on its own objective:")
        for o in ("m", "c", "p"):
            col = COL[OBJ[o]]
            vals = {k: means[k].mean(0)[col] for k in means}
            best = min(vals, key=vals.get)
            ok = best == o
            overall_ok &= ok
            print(f"    {OBJ[o]:<9}: best is '{OBJ[best]}' specialist  -> {'PASS' if ok else 'FAIL'}"
                  f"   ({', '.join(f'{k}={v:.1f}' for k, v in vals.items())})")

        # the priority specialist must pay for it somewhere
        pm = means["p"].mean(0)
        ok_cost = (pm[0] > means["m"].mean(0)[0]) or (pm[1] > means["c"].mean(0)[1])
        overall_ok &= ok_cost
        print(f"\n  priority specialist pays in makespan or carbon: "
              f"{'PASS' if ok_cost else 'FAIL (it dominates -- suspicious, investigate)'}")
        print()

    print("=" * 78)
    print(f"GATE 1: {'PASS -- proceed to Step 2' if overall_ok else 'FAIL -- STOP, do not proceed'}")
    return 0 if overall_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
