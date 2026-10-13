"""Part 2: build and validate the schedule set of every DEV instance and store its objective vectors.

Per instance (labels in the saved arrays):
  rule  <JOB>/<MACHINE> for JOB in {SPT, LPT, MOR, MWKR, FIFO} x MACHINE in {ECT, SPT, MINE, OLDC}   (20)
  edd   EDD/<MACHINE>@s for MACHINE in {ECT, SPT, MINE} and s in the s-grid (the EDD order depends on s only:
        d_j = k (1 + s (v_j - 0.5)) W_j, so k does not change the ranking)                            (12)
  rand  50 random dispatching schedules (default_rng(sha256(key|random_schedules)))                       (50)
Stored per schedule: makespan, workload (sum of chosen p), new energy-based carbon (kg CO2e), old i.i.d. carbon,
job completion times C_j. Every schedule passes the feasibility validator (else the run aborts).
Output: results/objective_validation/sets_<split>_<size>.npz

    python tools/objective_validation/build_sets.py
"""
from __future__ import annotations
import sys, time
from multiprocessing import Pool
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C                                       # noqa: E402
import schedules as S                                    # noqa: E402

N_RANDOM = 50


def one(args):
    split, size, i = args
    insts = C.instances(split, size); inst = insts[i]
    man = C.load_manifest(); rec = man["instances"][inst.key]
    assert rec["sha256"] == C.file_sha256(C.DAN / "data" / inst.key), "instance file changed"
    p = C.params_for(inst, man)
    cnew = C.energy_carbon(inst, p["power"]); cold = p["oldcarbon"]
    labels, rows = [], []

    def add(label, m, s):
        labels.append(label); rows.append(S.metrics(inst, m, s, cnew, cold))
    for jr in ("SPT", "LPT", "MOR", "MWKR", "FIFO"):
        for mr in ("ECT", "SPT", "MINE", "OLDC"):
            add(f"rule:{jr}/{mr}", *S.build(inst, jr, mr, energy=cnew, oldc=cold))
    for s in C.S_GRID:
        due = C.due_dates(inst, p["v"], 1.0, s)                 # ranking is k-invariant
        for mr in ("ECT", "SPT", "MINE"):
            add(f"edd:EDD/{mr}@s={s}", *S.build(inst, "EDD", mr, energy=cnew, oldc=cold, due=due))
    rng = C.keyed_rng(inst.key, "random_schedules")
    for q in range(N_RANDOM):
        add(f"rand:{q}", *S.build(inst, "RANDOM", None, rng=rng))
    return dict(labels=labels, mk=[r["mk"] for r in rows], workload=[r["workload"] for r in rows],
                c_new=[r["c_new"] for r in rows], c_old=[r["c_old"] for r in rows], Cj=[r["Cj"] for r in rows],
                W=inst.work_content(), v=p["v"], key=inst.key,
                cmin_new=float(np.where(inst.pt > 0, cnew, np.inf).min(1).sum()),
                cmin_old=float(np.where(inst.pt > 0, cold, np.inf).min(1).sum()))


def main():
    C.OUT.mkdir(parents=True, exist_ok=True)
    for split in ("cal", "check"):
        for size in ("10x5", "20x10"):
            f = C.OUT / f"sets_{split}_{size}.npz"
            if f.exists():
                print(f"skip {f.name}"); continue
            t0 = time.time()
            with Pool(10) as pool:
                res = pool.map(one, [(split, size, i) for i in range(C.COUNTS[split])])
            labels = res[0]["labels"]
            assert all(r["labels"] == labels for r in res)
            np.savez_compressed(f, labels=np.array(labels), keys=np.array([r["key"] for r in res]),
                                mk=np.array([r["mk"] for r in res]), workload=np.array([r["workload"] for r in res]),
                                c_new=np.array([r["c_new"] for r in res]), c_old=np.array([r["c_old"] for r in res]),
                                Cj=np.array([r["Cj"] for r in res]), W=np.array([r["W"] for r in res]),
                                v=np.array([r["v"] for r in res]), cmin_new=np.array([r["cmin_new"] for r in res]),
                                cmin_old=np.array([r["cmin_old"] for r in res]))
            print(f"{split} {size}: {len(res)} instances x {len(labels)} schedules, all feasible "
                  f"({time.time() - t0:.0f} s)", flush=True)


if __name__ == "__main__":
    main()
