"""Part 2 CP-SAT subset (results/objective_validation.md): the first 10 DEV-CAL instances per size, 60 s, 8 workers,
fixed seed, three problems:
  mk      minimum makespan;
  tard    minimum total tardiness at the chosen due-date configuration (integer model: TQ_j >= Q*C_j - ceil(Q*d_j),
          Q = 1000; the returned schedule is re-scored exactly);
  mineE   minimum makespan with every operation fixed to its minimum-energy machine (the analytic minimum carbon).
Every returned schedule passes the independent validator. Output results/objective_validation/cpsat_subset.json
(resume: existing entries are skipped).

    python tools/objective_validation/cpsat_subset.py
"""
from __future__ import annotations
import json, math, sys, time
from pathlib import Path
import numpy as np
from ortools.sat.python import cp_model

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C                                       # noqa: E402
import schedules as S                                    # noqa: E402

Q = 1000
N_INST, LIMIT, WORKERS, SEED = 10, 60, 8, 20261013


def model(inst, fixed=None, due=None):
    pt = inst.pt; N, M = pt.shape
    H = int(sum(pt[o][pt[o] > 0].max() for o in range(N)))
    md = cp_model.CpModel()
    st = [md.NewIntVar(0, H, f"s{o}") for o in range(N)]; en = [md.NewIntVar(0, H, f"e{o}") for o in range(N)]
    x, per_m = {}, {m: [] for m in range(M)}
    for o in range(N):
        ms = [fixed[o]] if fixed is not None else list(np.flatnonzero(pt[o] > 0))
        lits = []
        for m in ms:
            b = md.NewBoolVar(f"x{o}_{m}"); x[o, m] = b; lits.append(b)
            per_m[m].append(md.NewOptionalIntervalVar(st[o], int(pt[o, m]), en[o], b, f"i{o}_{m}"))
        md.AddExactlyOne(lits)
    for j in range(inst.n_j):
        for o in range(inst.first[j], inst.first[j] + inst.job_length[j] - 1):
            md.Add(st[o + 1] >= en[o])
    for m in range(M):
        md.AddNoOverlap(per_m[m])
    last = inst.first + inst.job_length - 1
    if due is None:
        mk = md.NewIntVar(0, H, "mk")
        for o in last:
            md.Add(mk >= en[o])
        md.Minimize(mk)
    else:
        tq = []
        for j, o in enumerate(last):
            t = md.NewIntVar(0, Q * H, f"t{j}"); md.Add(t >= Q * en[o] - math.ceil(Q * float(due[j]))); tq.append(t)
        md.Minimize(sum(tq))
    return md, st, x


def solve(inst, fixed=None, due=None):
    md, st, x = model(inst, fixed, due)
    sv = cp_model.CpSolver(); sv.parameters.max_time_in_seconds = LIMIT
    sv.parameters.num_workers = WORKERS; sv.parameters.random_seed = SEED
    t0 = time.time(); status = sv.Solve(md)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return {"status": sv.StatusName(status)}
    mach = np.array([next(m for (oo, m), b in x.items() if oo == o and sv.Value(b)) for o in range(inst.n_ops)])
    start = np.array([sv.Value(s) for s in st], dtype=float)
    return {"status": sv.StatusName(status), "wall_s": time.time() - t0, "machine": mach.tolist(), "start": start.tolist(),
            "bound": sv.BestObjectiveBound()}


def main():
    diag = json.loads((C.OUT / "diagnostics.json").read_text())
    man = C.load_manifest()
    f = C.OUT / "cpsat_subset.json"
    res = json.loads(f.read_text()) if f.exists() else {}
    for size in ("10x5", "20x10"):
        ch = diag[size]["chosen"]
        for inst in C.instances("cal", size)[:N_INST]:
            p = C.params_for(inst, man); cnew = C.energy_carbon(inst, p["power"])
            mine = np.array([np.flatnonzero(inst.pt[o] > 0)[np.argmin(np.where(inst.pt[o] > 0, cnew[o], np.inf)[inst.pt[o] > 0])]
                             for o in range(inst.n_ops)])
            due = C.due_dates(inst, p["v"], ch["k"], ch["s"])
            for prob in ("mk", "tard", "mineE"):
                key = f"{inst.key}|{prob}"
                if key in res:
                    continue
                r = solve(inst, fixed=mine if prob == "mineE" else None, due=due if prob == "tard" else None)
                if "machine" in r:
                    met = S.metrics(inst, np.array(r["machine"]), np.array(r["start"]), cnew, p["oldcarbon"])
                    r.update(mk=met["mk"], c_new=met["c_new"], workload=met["workload"],
                             tard=float(np.maximum(0, met["Cj"] - due).sum()), tardy_share=float((met["Cj"] > due).mean()))
                    del r["machine"], r["start"]
                res[key] = dict(r, size=size, problem=prob, k=ch["k"], s=ch["s"])
                f.write_text(json.dumps(res, indent=1))
                print(f"{size} {Path(inst.key).name} {prob:<6} {r['status']:<9} mk {r.get('mk', float('nan')):.0f} "
                      f"carbon {r.get('c_new', float('nan')):.3f} T {r.get('tard', float('nan')):.1f}", flush=True)
    print("CPSAT_SUBSET_DONE")


if __name__ == "__main__":
    main()
