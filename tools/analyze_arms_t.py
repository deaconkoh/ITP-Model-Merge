"""STEP 4 analysis (tardiness triple, 10x5): budget curves and comparisons on the n=100 evaluation pool.

Objective: (1/3,1/3,1/3) scalarisation, per seed, relative to that seed's centroid composition
(makespan and carbon per instance, tardiness by the centroid's pool mean). Lower is better.
Never seed-averaged before comparing: every comparison is also reported per seed.

    python tools/analyze_arms_t.py [--sanity-only] [--seeds 111 ...]
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from preference3 import Preference, scalarise  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
TAG = "k125s060"
RES = REPO / f"daniel/test_results/SD2/trainvali-{TAG}_10x5+carbon+priority"
BUDGETS = [0, 50, 100, 250, 500, 1000]
ARMS = ["merge", "scratch", "spec_m", "spec_c", "spec_t"]
LABEL = {"merge": "merge (a)", "scratch": "scratch (b)", "spec_m": "makespan spec (c)",
         "spec_c": "carbon spec (d)", "spec_t": "tardiness spec (e)"}
PREF = Preference(1 / 3, 1 / 3, 1 / 3, third="t")


def load(name, cols=(0, 1, 3)):
    f = RES / f"Result_DANIELG+10x5+carbon+priority+{name}_trainvali-{TAG}_10x5+carbon+priority.npy"
    return np.load(f)[:, list(cols)] if f.exists() else None


def arm_name(arm, seed, b):
    return f"s4t_{arm}_{TAG}_s{seed}@u{b}"


def fmt_c(p):
    return "/".join(str(x // 10) for x in p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=[111, 222, 333, 444])
    ap.add_argument("--sanity-only", action="store_true")
    args = ap.parse_args()
    seeds = args.seeds
    sel = json.loads((REPO / "results/step4/merge_init_selection.json").read_text())
    ref = {s: load(f"simplex_m333c333t334_{TAG}_s{s}") for s in seeds}

    # ------------------------------------------------------------- sanity checks
    print("=" * 100)
    print("SANITY CHECKS")
    print("=" * 100)
    ok = True
    for s in seeds:
        comp = sel[str(s)]["checkpoint"].split("+")[-1]
        a = load(comp, cols=(0, 1, 2, 3)); b = load(arm_name("merge", s, 0), cols=(0, 1, 2, 3))
        if a is None or b is None:
            print(f"  seed {s}: merge@u0 or its composition not evaluated yet"); continue
        d = float(np.abs(a - b).max())
        ok &= d == 0.0
        print(f"  seed {s}: merge@u0 vs composition {fmt_c(sel[str(s)]['composition_permille'])}: "
              f"max |difference| over all metrics = {d:.3g} -> {'PASS (exact)' if d == 0 else 'FAIL'}")
    for s in seeds:
        sc = load(arm_name("scratch", s, 0))
        if sc is None:
            continue
        o = float(scalarise(sc, ref[s], PREF, True).mean())
        warm = [float(scalarise(load(arm_name(a, s, 0)), ref[s], PREF, True).mean())
                for a in ARMS if a != "scratch" and load(arm_name(a, s, 0)) is not None]
        # Performance screen only: scratch@u0 must be clearly worse than the centroid and the merge.
        # NOT "worse than every warm start": the carbon specialist starts even worse on this objective.
        # The decisive test is tools/verify_scratch_init.py (weights identical to a fresh seeded init).
        m0 = float(scalarise(load(arm_name("merge", s, 0)), ref[s], PREF, True).mean())
        good = o > 1.15 and o > m0
        ok &= good
        print(f"  seed {s}: scratch@u0 objective {o:.3f} (centroid = 1.000; warm starts at u0: "
              f"{', '.join(f'{w:.3f}' for w in warm)}) -> {'PASS (random-init level)' if good else 'FAIL'}"
              f"   makespan {sc[:, 0].mean():.0f}, carbon {sc[:, 1].mean():.0f}, tardiness {sc[:, 2].mean():.0f}")
    print(f"\n  SANITY: {'PASS' if ok else 'FAIL'}")
    if args.sanity_only:
        return 0 if ok else 1

    # ------------------------------------------------------------- objective table
    O, I = {}, {}                       # O[arm][b] -> array over seeds; I[arm][b][s] -> per-instance
    for a in ARMS:
        O[a], I[a] = {}, {}
        for b in BUDGETS:
            vals, inst = [], {}
            for s in seeds:
                m = load(arm_name(a, s, b))
                if m is None:
                    vals.append(np.nan); continue
                v = scalarise(m, ref[s], PREF, True); inst[s] = v; vals.append(float(v.mean()))
            O[a][b], I[a][b] = np.array(vals), inst

    print("\n" + "=" * 100)
    print("BUDGET CURVES (objective, lower is better; mean +- sd across seeds)")
    print("=" * 100)
    print(f"  {'updates':>8}" + "".join(f"{LABEL[a]:>20}" for a in ARMS))
    for b in BUDGETS:
        print(f"  {b:>8}" + "".join(f"{np.nanmean(O[a][b]):>12.4f} +-{np.nanstd(O[a][b], ddof=1):<6.4f}" for a in ARMS))
    print("\n  per seed:")
    for s_i, s in enumerate(seeds):
        print(f"   seed {s}")
        for b in BUDGETS:
            print(f"    {b:>8}" + "".join(f"{O[a][b][s_i]:>20.4f}" for a in ARMS))

    # ------------------------------------------------------------- crossovers
    print("\n" + "=" * 100)
    print("CROSSOVERS: first budget at which an arm is at least as good as the MERGE arm at the SAME budget")
    print("=" * 100)
    for a in ARMS[1:]:
        pooled = next((b for b in BUDGETS if np.nanmean(O[a][b]) <= np.nanmean(O["merge"][b])), None)
        per = [next((b for b in BUDGETS if O[a][b][i] <= O["merge"][b][i]), None) for i in range(len(seeds))]
        n_cross = sum(p is not None for p in per)
        print(f"  {LABEL[a]:<20} pooled: {str(pooled) if pooled is not None else 'never':>6}   per seed: "
              + ", ".join(f"s{s}:{p if p is not None else 'never'}" for s, p in zip(seeds, per))
              + f"   ({n_cross}/{len(seeds)} seeds cross"
              + ("" if n_cross in (0, len(seeds)) else " -- NOT a crossover") + ")")

    # ------------------------------------------------------------- merge worth vs scratch
    print("\n" + "=" * 100)
    print("MERGE vs SCRATCH: scratch updates needed to reach the merge arm's quality at each budget")
    print("=" * 100)
    xs = np.log1p(BUDGETS)

    def needed(curve, target):
        """Interpolated scratch budget (in log(1+u)) at which `curve` first reaches `target`."""
        for j in range(1, len(BUDGETS)):
            if curve[j] <= target:
                y0, y1 = curve[j - 1], curve[j]
                f = 0.0 if y0 == y1 else (y0 - target) / (y0 - y1)
                return float(np.expm1(xs[j - 1] + f * (xs[j] - xs[j - 1])))
        return None
    for b in BUDGETS:
        rows = []
        for i, s in enumerate(seeds):
            n = needed([O["scratch"][x][i] for x in BUDGETS], O["merge"][b][i])
            rows.append(n)
        pn = needed([np.nanmean(O["scratch"][x]) for x in BUDGETS], np.nanmean(O["merge"][b]))
        diff = O["scratch"][b] - O["merge"][b]
        t, p = stats.ttest_1samp(diff, 0)
        print(f"  merge@{b:<5} scratch needs: pooled {('>1000' if pn is None else f'{pn:.0f}'):>6}   per seed "
              + ", ".join('>1000' if r is None else f"{r:.0f}" for r in rows)
              + f"   | matched budget: scratch worse by {100*np.mean(diff/O['merge'][b]):+.1f}% "
              f"({sum(diff > 0)}/{len(seeds)} seeds, seeds-as-unit p={p:.3g})")

    # ------------------------------------------------------------- merge vs single specialists
    print("\n" + "=" * 100)
    print("MERGE vs EACH SINGLE SPECIALIST at matched budget (positive = merge better)")
    print("   seeds as the unit (n=4) and paired over all instance x seed pairs (n=400, not independent)")
    print("=" * 100)
    for a in ["spec_m", "spec_c", "spec_t"]:
        print(f"  {LABEL[a]}")
        for b in BUDGETS:
            diff = O[a][b] - O["merge"][b]
            t, p = stats.ttest_1samp(diff, 0)
            inst = np.concatenate([I[a][b][s] - I["merge"][b][s] for s in seeds if s in I[a][b]])
            pi = stats.ttest_1samp(inst, 0).pvalue
            print(f"    @{b:<5} merge advantage {100*np.mean(diff/O[a][b]):+6.2f}%   seeds favouring merge "
                  f"{sum(diff > 0)}/{len(seeds)}  p(seeds)={p:.3g}   p(instances)={pi:.3g}   "
                  f"per seed: " + ", ".join(f"{100*d/o:+.1f}%" for d, o in zip(diff, O[a][b])))

    # ------------------------------------------------------------- operating points
    print("\n" + "=" * 100)
    print("OPERATING POINTS (raw pool means, mean +- sd across seeds): where each arm starts and ends")
    print("=" * 100)
    print(f"  {'arm':<20}{'budget':>7}{'makespan':>18}{'carbon':>20}{'tardiness':>18}{'objective':>12}")
    for a in ARMS:
        for b in (0, 1000):
            M = np.stack([load(arm_name(a, s, b)).mean(0) for s in seeds])
            mu, sd = M.mean(0), M.std(0, ddof=1)
            print(f"  {LABEL[a]:<20}{b:>7}" + "".join(f"{mu[i]:>11.1f} +-{sd[i]:<5.1f}" for i in range(3))
                  + f"{np.nanmean(O[a][b]):>12.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
