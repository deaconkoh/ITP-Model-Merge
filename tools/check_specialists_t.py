"""GATE 1 for the tardiness triple: does each specialist win its own objective, and how far
apart are the three specialists in objective space -- next to the retired priority triple?

Distances follow the Gate 1 method exactly: per objective, the seed-mean pool means of the three
specialists are normalised to [0, 1] across those three specialists, then pairwise Euclidean
distances are taken. The priority numbers are recomputed from the original evaluations so the
two triples are measured by one piece of code (it must reproduce 1.665 / 1.651 / 0.169 at 10x5).

Reads existing evaluations only; runs nothing.
"""
from __future__ import annotations
import argparse, itertools, json
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[1]
SEEDS = [111, 222, 333, 444]
RES = REPO / "daniel/test_results/SD2"


def load(size, pool_tag, spec, cols):
    f = (RES / f"{pool_tag}_{size}+carbon+priority" /
         f"Result_DANIELG+{size}+carbon+priority+{spec}_{pool_tag}_{size}+carbon+priority.npy")
    return np.load(f)[:, cols] if f.exists() else None


def triple(size, pool_tag, specs, cols):
    """specs: key -> name template with {seed}. Returns key -> (n_seeds, 3) pool means."""
    out, missing = {}, []
    for k, tmpl in specs.items():
        rows = []
        for s in SEEDS:
            a = load(size, pool_tag, tmpl.format(seed=s), cols)
            if a is None:
                missing.append(tmpl.format(seed=s))
            else:
                rows.append(a.mean(0))
        out[k] = np.array(rows) if rows else None
    return out, missing


def distances(means, keys):
    V = np.array([means[k].mean(0) for k in keys])        # rows specialists, cols objectives
    rng = V.max(0) - V.min(0)
    Vn = (V - V.min(0)) / np.where(rng > 0, rng, 1.0)
    return {f"{a}{b}": float(np.linalg.norm(Vn[i] - Vn[j]))
            for (i, a), (j, b) in itertools.combinations(enumerate(keys), 2)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", nargs="+", default=["10x5", "20x10"])
    args = ap.parse_args()
    manifest = json.loads((REPO / "daniel/data/due_dates/due_date_manifest.json").read_text())
    ok_all = True
    for size in args.sizes:
        tag = manifest["dd_tag_by_size"][size]
        pool_t = f"trainvali-{tag}"
        print("=" * 84)
        print(f"{size}: tardiness triple, due-date tag {tag}, pool {pool_t}")
        print("=" * 84)
        mt, miss = triple(size, pool_t, {"m": "m_s{seed}", "c": "c_s{seed}",
                                         "t": f"t_{tag}_s{{seed}}"}, [0, 1, 3])
        if miss:
            print(f"  MISSING evaluations: {miss}"); ok_all = False; continue
        names = {"m": "makespan", "c": "carbon", "t": "tardiness"}
        print(f"  {'specialist':<16}{'makespan':>16}{'carbon':>16}{'tardiness':>16}")
        for k in "mct":
            mu, sd = mt[k].mean(0), mt[k].std(0, ddof=1)
            print(f"  {names[k]+' spec':<16}" + "".join(f"{mu[i]:>10.1f} +-{sd[i]:<4.1f}" for i in range(3)))
        print("\n  each specialist must be best on its own objective:")
        for i, k in enumerate("mct"):
            vals = {j: mt[j].mean(0)[i] for j in "mct"}
            best = min(vals, key=vals.get)
            ok = best == k; ok_all &= ok
            print(f"    {names[k]:<10} best = {names[best]:<10} -> {'PASS' if ok else 'FAIL'}   "
                  + ", ".join(f"{names[j]}={v:.1f}" for j, v in vals.items()))
        # per-seed win check for the new specialist
        wins = sum(mt["t"][s][2] < min(mt["m"][s][2], mt["c"][s][2]) for s in range(len(SEEDS)))
        print(f"    tardiness specialist wins tardiness at {wins}/{len(SEEDS)} seeds (seed-matched)")
        # Per-SPECIALIST criterion: every trained tardiness specialist must beat its own seed's
        # makespan and carbon specialists. A seed-mean win can hide specialists that lose outright.
        if wins < len(SEEDS):
            ok_all = False
            print(f"    -> FAIL on the per-specialist criterion: {len(SEEDS) - wins} tardiness "
                  f"specialist(s) do not win their own objective")
        tm = mt["t"].mean(0)
        pays = tm[0] > mt["m"].mean(0)[0] or tm[1] > mt["c"].mean(0)[1]
        print(f"    tardiness specialist pays in makespan or carbon: {'yes' if pays else 'NO (dominates)'}")
        adv = 100 * (mt["m"].mean(0)[2] - tm[2]) / mt["m"].mean(0)[2]
        print(f"    advantage over the makespan specialist on tardiness: {adv:+.1f}%")

        mp, missp = triple(size, "trainvali", {"m": "m_s{seed}", "c": "c_s{seed}", "p": "p_s{seed}"},
                           [0, 1, 2])
        dt = distances(mt, "mct")
        print("\n  objective-space distances (Gate 1 method):")
        print(f"    {'triple':<34}{'m<->c':>8}{'c<->X':>8}{'m<->X':>8}{'m<->c / m<->X':>16}")
        if not missp:
            dp = distances(mp, "mcp")
            print(f"    {'makespan, carbon, priority (old)':<34}{dp['mc']:>8.3f}{dp['cp']:>8.3f}"
                  f"{dp['mp']:>8.3f}{dp['mc']/dp['mp']:>15.1f}x")
        print(f"    {'makespan, carbon, tardiness (new)':<34}{dt['mc']:>8.3f}{dt['ct']:>8.3f}"
              f"{dt['mt']:>8.3f}{dt['mc']/dt['mt']:>15.1f}x")
        print()
    print("=" * 84)
    print(f"GATE 1 (tardiness): {'PASS' if ok_all else 'FAIL -- STOP'}")
    return 0 if ok_all else 1


if __name__ == "__main__":
    raise SystemExit(main())
