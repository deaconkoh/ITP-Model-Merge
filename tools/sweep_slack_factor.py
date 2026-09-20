"""STEP 1: choose the due-date slack factor k by a rule fixed in advance, not by hand.

Due dates follow the TWK (total-work-content) rule, the standard workload-dependent assignment:

    d_j = r_j + k * P_j ,   r_j = 0 ,   P_j = sum over j's operations of the MEAN
                                              processing time across eligible machines

r_j = 0 because SD2 instances are static: every job is available at time zero. Published TWK
factors sit around 3-5, but those come from DYNAMIC shops where jobs queue behind each other;
these instances have no queueing inflation, so k is swept rather than borrowed.

Objective: unweighted total tardiness T = sum_j max(0, C_j - d_j). NOT the weighted form -- the
only weights available are the retired operation-priority values, and folding them in would
smuggle the retired axis back into its replacement.

DECISION RULE (fixed before looking at the numbers):
  choose the SMALLEST k whose mean tardy fraction lies in [0.2, 0.6] at BOTH sizes AND which
  leaves at most 10% of instances with exactly zero tardiness. If no k qualifies at both sizes,
  report that and stop rather than tuning k per size.

Tardy fraction is a property of the SCHEDULE, not of the instance, so it is reported under four
reference dispatchers spanning weak-to-strong and due-date-blind-to-aware.
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dispatcher import (load_pool, job_work_content, dispatch, time_rule, ect_rule,  # noqa: E402
                        edd_rule, slack_rule)

REPO = Path(__file__).resolve().parents[1]
KS = [1.2, 1.5, 2.0, 2.5, 3.0]
RULES = {"SPT (weak, blind)": time_rule, "ECT (strong, blind)": ect_rule,
         "EDD (due-date aware)": edd_rule, "MinSlack (aware)": slack_rule}
TARDY_LO, TARDY_HI, ZERO_MAX = 0.2, 0.6, 0.10


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", nargs="+", default=["10x5", "20x10"])
    ap.add_argument("--ks", nargs="+", type=float, default=KS)
    args = ap.parse_args()

    qualifies = {}
    stats_for_scaling = {}
    for size in args.sizes:
        inst = load_pool(REPO, size)
        works = [job_work_content(jobs) for _, _, _, jobs in inst]
        print("=" * 100)
        print(f"{size}: {len(inst)} instances (train_vali pool), TWK due dates d_j = k * P_j")
        print("=" * 100)
        print(f"  {'k':>4}  {'rule':<22}{'tardy frac':>11}{'zero-T inst':>13}"
              f"{'median T':>10}{'IQR T':>16}{'T / makespan':>14}")
        for k in args.ks:
            for label, rule in RULES.items():
                T, TF, REL = [], [], []
                for (_, _, n_mch, jobs), P in zip(inst, works):
                    r = dispatch(jobs, n_mch, rule, due=k * P)
                    T.append(r["tardiness"]); TF.append(r["tardy_fraction"])
                    REL.append(r["tardiness"] / r["makespan"])
                T, TF = np.array(T), np.array(TF)
                zero = float((T <= 1e-9).mean())
                q1, q3 = np.percentile(T, [25, 75])
                print(f"  {k:>4.1f}  {label:<22}{TF.mean():>11.3f}{zero:>13.2f}"
                      f"{np.median(T):>10.1f}{f'[{q1:.0f}, {q3:.0f}]':>16}{np.mean(REL):>14.2f}")
                if label.startswith("ECT"):
                    qualifies.setdefault(k, {})[size] = (TF.mean(), zero)
                    stats_for_scaling.setdefault(k, {})[size] = (TF, T)
            print()

    # ---- the pre-stated decision rule -------------------------------------------------------
    print("=" * 100)
    print("DECISION RULE applied (reference = ECT, the strongest due-date-blind dispatcher,")
    print(f"  since a weak reference flatters tight due dates): mean tardy fraction in "
          f"[{TARDY_LO}, {TARDY_HI}] at BOTH sizes, zero-tardiness instances <= {ZERO_MAX:.0%}")
    print("=" * 100)
    chosen = None
    for k in sorted(qualifies):
        rows = qualifies[k]
        ok = all(TARDY_LO <= tf <= TARDY_HI and z <= ZERO_MAX for tf, z in rows.values())
        detail = "  ".join(f"{s}: tardy={tf:.3f} zeroT={z:.2f}" for s, (tf, z) in rows.items())
        print(f"  k={k:<4} {'QUALIFIES' if ok else 'no       '}  {detail}")
        if ok and chosen is None:
            chosen = k
    print()
    if chosen is None:
        print("NO k QUALIFIES AT BOTH SIZES -- stopping rather than tuning k per size.")
    else:
        print(f"CHOSEN k = {chosen} (smallest qualifying value)")

    # ---- how does the per-instance spread scale with size? ----------------------------------
    print("\n" + "=" * 100)
    print("SCALING of per-instance spread, 10x5 -> 20x10 (driver is n_jobs 10->20, not n_ops)")
    print("=" * 100)
    print("  If the instance-level statistic were a mean of n_jobs i.i.d. terms, its spread would")
    print("  fall by sqrt(2) = 1.41x. The retired per-operation priority averaged 50 -> 200 ops,")
    print("  i.e. sqrt(4) = 2.0x -- the faster concentration that made instances look alike.")
    for k in sorted(stats_for_scaling):
        row = stats_for_scaling[k]
        if len(row) < 2:
            continue
        tf10, t10 = row["10x5"]; tf20, t20 = row["20x10"]
        cv = lambda a: float(np.std(a) / np.mean(a)) if np.mean(a) > 0 else float("nan")
        print(f"  k={k:<4} tardy-fraction sd: {np.std(tf10):.3f} -> {np.std(tf20):.3f} "
              f"(ratio {np.std(tf10)/np.std(tf20):.2f}x)   "
              f"total-T CV: {cv(t10):.3f} -> {cv(t20):.3f} (ratio {cv(t10)/cv(t20):.2f}x)")


if __name__ == "__main__":
    raise SystemExit(main())
