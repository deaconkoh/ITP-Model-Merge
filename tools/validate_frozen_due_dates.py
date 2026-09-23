"""STEP 4: validation screen on the FROZEN due dates. Confirms the generation; does not choose
the objective.

Same training-free dispatcher and shared rule family as tools/precheck_orthogonality.py, but the
due dates are READ from the frozen manifest (hash-checked), never derived. Reports, per size:
  * each objective's specialist advantage over the makespan specialist on that objective --
    tardiness next to the carbon reference and the retired priority case;
  * the makespan<->tardiness correlation across blended schedules (per-instance normalised).
If tardiness is near-degenerate here (advantage close to 0%), the generation is wrong: STOP.

Optional before/after for the augmentation finding: priority with PER-JOB weights (each job's
weight is the priority value already attached to its first operation, so no new data is drawn)
instead of i.i.d. per-operation weights. Recorded only.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools")); sys.path.insert(0, str(REPO / "daniel"))
from dispatcher import (load_pool, dispatch, priority_weighted_completion, ect_rule, carbon_rule,  # noqa
                        edd_rule, slack_rule, wspt_rule, balance_rule, atc_rule, watc_rule, blend)
from due_dates import load_manifest, due_dates_for_directory, instance_key, manifest_tag  # noqa
from data_utils import sorted_instance_files  # noqa

FAMILY = {"ECT": ect_rule, "carbon-greedy": carbon_rule, "EDD": edd_rule, "MinSlack": slack_rule,
          "ATC(k=20)": atc_rule(20.0), "ATC(k=50)": atc_rule(50.0), "ATC(k=100)": atc_rule(100.0),
          "WATC": watc_rule(2.0), "wECT": wspt_rule, "least-loaded": balance_rule}


def job_weighted_rule(j, m, pt, cb, start, ctx):
    return (start + pt) / (ctx["prio"][j][0] + 1e-9)      # job weight = first-op priority


def job_weighted_completion(res, prios):
    w = np.array([p[0] for p in prios]); C = res["job_completion"]
    return float((w * C).sum() / w.sum())


FAMILY["job-wECT"] = job_weighted_rule
NAME = {"m": "makespan", "c": "carbon", "t": "tardiness", "p": "priority (per-op, retired)",
        "q": "priority (per-JOB)"}


def scores(res, prios, due):
    return {"m": res["makespan"], "c": res["carbon"],
            "t": float(np.maximum(0.0, res["job_completion"] - due).sum()),
            "p": priority_weighted_completion(res["op_ct"], prios),
            "q": job_weighted_completion(res, prios)}


def main():
    manifest = load_manifest()
    for size in ("10x5", "20x10"):
        pool_dir = REPO / f"daniel/data/data_train_vali/SD2/{size}+carbon+priority"
        frozen = dict(zip([instance_key(f) for f in sorted_instance_files(str(pool_dir))],
                          due_dates_for_directory(pool_dir, manifest)))       # hash-checked
        inst = load_pool(REPO, size)
        dues = [frozen[instance_key(f)] for f, *_ in inst]
        fam = {r: {o: [] for o in NAME} for r in FAMILY}
        for (f, _, n_mch, jobs, prios), due in zip(inst, dues):
            for rn, rule in FAMILY.items():
                sc = scores(dispatch(jobs, n_mch, rule, due=due, job_prios=prios), prios, due)
                for o, v in sc.items():
                    fam[rn][o].append(v)
        fam = {r: {o: np.array(v) for o, v in d.items()} for r, d in fam.items()}
        win = {o: min(FAMILY, key=lambda r: fam[r][o].mean()) for o in NAME}
        print("=" * 92)
        print(f"{size}: FROZEN due dates, tag {manifest_tag(pool_dir, manifest)}, {len(inst)} instances")
        print("=" * 92)
        print("  specialist advantage over the MAKESPAN specialist on each objective:")
        for o in ("c", "t", "p", "q"):
            base, best = fam[win["m"]][o].mean(), fam[win[o]][o].mean()
            print(f"    {NAME[o]:<28} makespan-spec {base:>10.2f}   own spec ({win[o]:<12}) "
                  f"{best:>10.2f}   advantage {100*(base-best)/base:+6.1f}%")
        # tardy fraction under the dispatcher's best time rule, for context
        A, B = [], []
        ra, rb = FAMILY[win["m"]], FAMILY[win["t"]]
        if win["m"] != win["t"]:
            for lam in (0.25, 0.5, 0.75):
                va, vb = [], []
                for (f, _, n_mch, jobs, prios), due in zip(inst, dues):
                    sc = scores(dispatch(jobs, n_mch, blend(ra, rb, lam), due=due, job_prios=prios),
                                prios, due)
                    va.append(sc["m"]); vb.append(sc["t"])
                A.append(np.array(va) / fam[win["m"]]["m"])
                B.append(np.array(vb) / np.maximum(fam[win["m"]]["t"], 1e-9))
            r = float(np.corrcoef(np.concatenate(A), np.concatenate(B))[0, 1])
            print(f"  makespan<->tardiness correlation across blended schedules: r = {r:+.3f}")
        else:
            print("  makespan<->tardiness: SAME rule wins both -- degenerate")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
