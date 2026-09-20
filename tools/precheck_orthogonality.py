"""STEP 2 -- HARD GATE: is the replacement third objective actually separable from makespan?

This is the check operation-priority failed, applied BEFORE any training this time. CPU only,
no checkpoints touched.

Method: one greedy dispatcher and a shared FAMILY of scoring rules. Each objective's specialist
is whichever family member scores best on that objective, so no candidate axis is handicapped by
a weak hand-picked heuristic. Every schedule is scored on EVERY objective, so objectives are
compared on identical schedules. Two readouts:

  1. Gate-1-analogous distance: normalise each objective to [0,1] across the three specialists of
     a candidate triple, then take pairwise Euclidean distances. Gate 1 measured, from TRAINED
     specialists: m<->c = 1.665 (orthogonal reference), m<->p = 0.169 (degenerate) at 10x5.
  2. Correlation between two objectives across schedules from sweeping a blend of their two
     rules, per-instance normalised so instance scale cannot manufacture correlation.

The RETIRED priority axis is measured in the same frame as a positive control: a screen that
cannot reproduce priority's known degeneracy is not worth trusting.

Load balance (spread of machine utilisation) is the zero-assumption control: no data
augmentation, no due-date rule, nothing to defend if it separates.
"""
from __future__ import annotations
import argparse, sys, itertools
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dispatcher import (load_pool, job_work_content, dispatch, priority_weighted_completion,
                        ect_rule, carbon_rule, edd_rule, slack_rule, wspt_rule, balance_rule,
                        atc_rule, watc_rule, blend, due_dates)

REPO = Path(__file__).resolve().parents[1]
FAMILY = {"ECT": ect_rule, "carbon-greedy": carbon_rule, "EDD": edd_rule, "MinSlack": slack_rule,
          "ATC(k=20)": atc_rule(20.0), "ATC(k=50)": atc_rule(50.0), "ATC(k=100)": atc_rule(100.0),
          "WATC": watc_rule(2.0), "wECT": wspt_rule, "least-loaded": balance_rule}
# Rules whose SCHEDULE depends on the due dates; the rest are scheduled once and their
# tardiness recomputed per (k, spread) from stored job completion times.
DUE_DEPENDENT = {"EDD", "MinSlack", "ATC(k=20)", "ATC(k=50)", "ATC(k=100)"}
NAME = {"m": "makespan", "c": "carbon", "t": "tardiness", "p": "priority(retired)",
        "b": "load-balance"}
PAIRS = [("m", "c"), ("m", "t"), ("c", "t"), ("m", "p"), ("m", "b"), ("c", "b")]
LAMS = [0.25, 0.5, 0.75]


def score_all(res, prios, due):
    return {"m": res["makespan"], "c": res["carbon"],
            "t": float(np.maximum(0.0, res["job_completion"] - due).sum()),
            "p": priority_weighted_completion(res["op_ct"], prios),
            "b": res["load_imbalance"]}


def run_family(inst, k_ref, spread):
    """Every family rule on every instance, once. Returns rule -> objective -> per-instance array.
    Due dates use k_ref for the rules that consult them; tardiness VALUES are recomputed per k
    from stored job completion times (EDD/ATC orderings shift little, checked via --ks)."""
    out = {r: {o: [] for o in NAME} for r in FAMILY}
    raw = {r: [] for r in FAMILY}
    for i, (_, _, n_mch, jobs, prios) in enumerate(inst):
        P = job_work_content(jobs)
        d = due_dates(P, k_ref, spread, seed=i)
        for rname, rule in FAMILY.items():
            res = dispatch(jobs, n_mch, rule, due=d, job_prios=prios)
            raw[rname].append((res, prios, d))
            for o, v in score_all(res, prios, d).items():
                out[rname][o].append(v)
    return {r: {o: np.array(v) for o, v in d.items()} for r, d in out.items()}, raw


def tardiness_at(raw_rows, k=None):
    """Due dates are stored per instance as produced, so tardiness is read straight off them."""
    return np.array([float(np.maximum(0.0, res["job_completion"] - d).sum())
                     for res, _, d in raw_rows])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", nargs="+", default=["10x5", "20x10"])
    ap.add_argument("--ks", nargs="+", type=float, default=[0.9, 1.0, 1.1, 1.2])
    ap.add_argument("--k-ref", type=float, default=1.0)
    ap.add_argument("--spread", type=float, default=0.8,
                    help="per-job due-date tightness dispersion; 0 = uniform TWK")
    ap.add_argument("--n-inst", type=int, default=0, help="limit instances (0 = all)")
    args = ap.parse_args()

    trend = {}
    for size in args.sizes:
        inst = load_pool(REPO, size)
        if args.n_inst:
            inst = inst[:args.n_inst]
        fam, raw = run_family(inst, args.k_ref, args.spread)
        print("=" * 104)
        print(f"{size}: {len(inst)} instances, {len(FAMILY)} rules in the shared family "
              f"(due dates k={args.k_ref}, dispersion spread={args.spread})")
        print("=" * 104)

        print(f"\n  POOL MEANS per rule (specialist for each objective = the column winner)")
        print(f"    {'rule':<15}" + "".join(f"{NAME[o]:>20}" for o in NAME))
        winner = {o: min(FAMILY, key=lambda r: fam[r][o].mean()) for o in NAME}
        for r in FAMILY:
            marks = "".join(f"{fam[r][o].mean():>19.2f}" + ("*" if winner[o] == r else " ")
                            for o in NAME)
            print(f"    {r:<15}{marks}")
        print("    * = best on that objective; that rule is used as its specialist")
        print("    specialists: " + ", ".join(f"{NAME[o]}={winner[o]}" for o in NAME))

        print(f"\n  SPECIALIST ADVANTAGE -- how much its own specialist beats the MAKESPAN")
        print(f"  specialist on each objective. This is the comparable orthogonality measure:")
        print(f"  a genuinely separate axis has something to gain that makespan does not deliver.")
        for o in NAME:
            if o == "m":
                continue
            base = fam[winner["m"]][o].mean()
            best = fam[winner[o]][o].mean()
            gain = 100 * (base - best) / base if base > 0 else 0.0
            print(f"    {NAME[o]:<18} makespan-specialist {base:>10.2f}   own specialist "
                  f"({winner[o]}) {best:>10.2f}   advantage {gain:+6.1f}%")

        for third in ("t", "p", "b"):
            ks = args.ks if third == "t" else [args.k_ref]
            for k in ks:
                triple = ("m", "c", third)
                V = []
                for s in triple:
                    row = []
                    for o in triple:
                        if o == "t":
                            row.append(tardiness_at(raw[winner[s]]).mean())
                        else:
                            row.append(fam[winner[s]][o].mean())
                    V.append(row)
                V = np.array(V)
                rng = V.max(0) - V.min(0)
                Vn = (V - V.min(0)) / np.where(rng > 0, rng, 1.0)
                d = {f"{a}{b}": float(np.linalg.norm(Vn[i] - Vn[j]))
                     for (i, a), (j, b) in itertools.combinations(enumerate(triple), 2)}
                same = winner["m"] == winner[third]
                ratio = d["mc"] / d[f"m{third}"] if d[f"m{third}"] > 1e-12 else float("inf")
                print(f"\n  TRIPLE (makespan, carbon, {NAME[third]})"
                      f"{f' k={k}' if third == 't' else ''}")
                note = ("   [SAME RULE wins makespan and this objective: no trade-off exists "
                        "within the family]" if same else "")
                print(f"    m<->c {d['mc']:.3f}   c<->{third} {d[f'c{third}']:.3f}   "
                      f"m<->{third} {d[f'm{third}']:.3f}   -> m<->c is {ratio:.1f}x m<->{third}{note}")
                trend.setdefault((third, k), {})[size] = (d[f"m{third}"], ratio)

        print(f"\n  CORRELATION across blended schedules (per-instance normalised; +1 = same axis)")
        for a, b in PAIRS:
            if winner[a] == winner[b]:
                print(f"    {NAME[a]:>16} vs {NAME[b]:<18} -- same specialist rule, "
                      f"no blend to sweep")
                continue
            ra, rb = FAMILY[winner[a]], FAMILY[winner[b]]
            A, B = [], []
            for lam in LAMS:
                va, vb = [], []
                for i, (_, _, n_mch, jobs, prios) in enumerate(inst):
                    P = job_work_content(jobs)
                    dd = due_dates(P, args.k_ref, args.spread, seed=i)
                    res = dispatch(jobs, n_mch, blend(ra, rb, lam), due=dd, job_prios=prios)
                    sc = score_all(res, prios, dd)
                    va.append(sc[a]); vb.append(sc[b])
                base_a = fam[winner["m"]][a]; base_b = fam[winner["m"]][b]
                A.append(np.array(va) / np.where(base_a > 0, base_a, 1))
                B.append(np.array(vb) / np.where(base_b > 0, base_b, 1))
            r = float(np.corrcoef(np.concatenate(A), np.concatenate(B))[0, 1])
            print(f"    {NAME[a]:>16} vs {NAME[b]:<18} r = {r:+.3f}")

    print("\n" + "=" * 104)
    print("SIZE TREND (Gate 1: m<->p separation got WORSE with size, 9.8x -> 13.3x)")
    print("=" * 104)
    for (third, k), row in sorted(trend.items()):
        if len(row) < 2:
            continue
        (d10, r10), (d20, r20) = row["10x5"], row["20x10"]
        print(f"  third={NAME[third]:<18} k={k:<5} m<->X {d10:.3f} -> {d20:.3f}   "
              f"ratio {r10:.1f}x -> {r20:.1f}x   "
              f"{'improves' if r20 < r10 else 'WORSENS'} with size")


if __name__ == "__main__":
    raise SystemExit(main())
