"""Ceiling check, Part A: exact FJSP solver on the Step 4 score (results/ceiling_check.md). CPU only.

Model (OR-Tools CP-SAT), per (seed, HELD-OUT instance) -- the score's normalisers are seed-specific, so each seed
is its own objective:
  * every operation o: start/end integer vars; one optional interval per eligible machine m (length pt[o, m]),
    exactly one presence literal true; start/end tied to the chosen interval;
  * job precedence end(o) <= start(next op of the job); NoOverlap over the optional intervals of each machine;
  * makespan MK >= end(last op of j) for all j;
  * scaled tardiness TQ_j >= Q * end(last op of j) - ceil(Q * d_j), TQ_j >= 0   (Q = 1000);
  * objective = cm * MK + sum_{o,m} cc * carbon[o, m] * x[o, m] + ct * sum_j TQ_j, with
        cm = floor(S * w_m / mk_ref[i]), cc = floor(S * w_c / cb_ref[i]), ct = floor(S * w_t / (T_ref * Q)),
    S = 1e12. Every coefficient is rounded DOWN and the due dates UP, so for every schedule the integer objective
    is <= S * (exact Step 4 score): CP-SAT's proven bound / S is a valid lower bound on the exact score.
  * warm start: the seed's C1@1000 greedy schedule (Step 0) as a complete hint (machine literals, starts, ends,
    MK, TQ).
Every returned schedule is re-scored with the independent Step 0 scorer; that score is what is reported.

    python tools/ceiling_cpsat.py --limit 60  [--seeds ...] [--insts pool_idx ...]
    python tools/ceiling_cpsat.py --limit 600 --insts <the 10 pre-registered instances>
Outputs results/ceiling_check/raw/cpsat_t<limit>/s<seed>_i<idx>.json (resume skips existing units).
"""
from __future__ import annotations
import argparse, json, math, sys, time
from pathlib import Path
import numpy as np
from ortools.sat.python import cp_model

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))
import ceiling_scorer as SC                            # noqa: E402

S_SCALE = 10 ** 12
Q = 1000
WORKERS = 8
SOLVER_SEED = 20260928


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def coefficients(seed, i, ref):
    mk_ref, cb_ref, t_ref = ref
    return (math.floor(S_SCALE * SC.W[0] / mk_ref[i]), math.floor(S_SCALE * SC.W[1] / cb_ref[i]),
            math.floor(S_SCALE * SC.W[2] / (t_ref * Q)))


def solve(inst: SC.Instance, due, coef, limit, hint=None, workers=WORKERS):
    cm, cc, ct = coef
    horizon = sum(max(p.values()) for p in inst.pt)
    md = cp_model.CpModel()
    st = [md.NewIntVar(0, horizon, f"s{o}") for o in range(inst.n_ops)]
    en = [md.NewIntVar(0, horizon, f"e{o}") for o in range(inst.n_ops)]
    x, per_m = {}, {m: [] for m in range(inst.n_m)}
    for o in range(inst.n_ops):
        lits = []
        for m, p in inst.pt[o].items():
            b = md.NewBoolVar(f"x{o}_{m}"); x[o, m] = b; lits.append(b)
            per_m[m].append(md.NewOptionalIntervalVar(st[o], p, en[o], b, f"iv{o}_{m}"))
        md.AddExactlyOne(lits)
    for ops in inst.job_ops:
        for a, b in zip(ops, ops[1:]):
            md.Add(st[b] >= en[a])
    for m in range(inst.n_m):
        md.AddNoOverlap(per_m[m])
    MK = md.NewIntVar(0, horizon, "MK")
    TQ = []
    for j, ops in enumerate(inst.job_ops):
        md.Add(MK >= en[ops[-1]])
        t = md.NewIntVar(0, Q * horizon, f"T{j}")
        md.Add(t >= Q * en[ops[-1]] - math.ceil(Q * float(due[j])))
        TQ.append(t)
    md.Minimize(cm * MK + sum(cc * inst.cb[o][m] * b for (o, m), b in x.items()) + ct * sum(TQ))
    hint_obj = None
    if hint is not None:
        mach, start = [int(v) for v in hint[0]], [int(round(v)) for v in hint[1]]
        end = [start[o] + inst.pt[o][mach[o]] for o in range(inst.n_ops)]
        for (o, m), b in x.items():
            md.AddHint(b, int(mach[o] == m))
        for o in range(inst.n_ops):
            md.AddHint(st[o], start[o]); md.AddHint(en[o], end[o])
        md.AddHint(MK, max(end))
        tq = [max(0, Q * end[ops[-1]] - math.ceil(Q * float(due[j]))) for j, ops in enumerate(inst.job_ops)]
        for t, v in zip(TQ, tq):
            md.AddHint(t, v)
        hint_obj = cm * max(end) + sum(cc * inst.cb[o][mach[o]] for o in range(inst.n_ops)) + ct * sum(tq)
    sv = cp_model.CpSolver()
    sv.parameters.max_time_in_seconds = float(limit)
    sv.parameters.num_workers = workers
    sv.parameters.random_seed = SOLVER_SEED
    t0 = time.time()
    status = sv.Solve(md)
    wall = time.time() - t0
    out = {"status": sv.StatusName(status), "wall_s": wall, "objective_int": None, "bound_int": int(sv.BestObjectiveBound()),
           "hint_objective_int": hint_obj, "horizon": horizon}
    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        out["objective_int"] = int(sv.ObjectiveValue())
        out["machine"] = [next(m for m in inst.pt[o] if sv.Value(x[o, m])) for o in range(inst.n_ops)]
        out["start"] = [int(sv.Value(st[o])) for o in range(inst.n_ops)]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=60)
    ap.add_argument("--seeds", nargs="+", type=int, default=SC.SEEDS)
    ap.add_argument("--insts", nargs="+", type=int, default=None, help="pool indices (default: all HELD-OUT)")
    a = ap.parse_args()
    sys.argv = sys.argv[:1]                      # daniel's params module parses sys.argv on import (via SC loaders)
    insts_idx = a.insts if a.insts is not None else [int(i) for i in SC.HELD]
    assert set(insts_idx) <= set(int(i) for i in SC.HELD), "HELD-OUT instances only"
    out_dir = SC.OUT / "raw" / f"cpsat_t{a.limit}"; out_dir.mkdir(parents=True, exist_ok=True)
    insts, dues = SC.instances(), SC.due_dates()
    for s in a.seeds:
        ref = SC.reference(s)
        g = np.load(SC.OUT / "raw" / f"c1_greedy_s{s}.npz")
        held = [int(i) for i in g["held"]]
        for i in insts_idx:
            f = out_dir / f"s{s}_i{i}.json"
            if f.exists():
                continue
            k = held.index(i)
            hint = (g["machine"][k], g["start"][k])
            coef = coefficients(s, i, ref)
            r = solve(insts[i], dues[i], coef, a.limit, hint=hint)
            r.update(seed=s, inst=i, name=insts[i].name, limit_s=a.limit, workers=WORKERS, S=S_SCALE, Q=Q,
                     coef=list(coef), bound_score=r["bound_int"] / S_SCALE, hint_score=float(g["score"][k]))
            if r["objective_int"] is not None:
                mk, cb, td = SC.raw_objectives(insts[i], dues[i], r["machine"], r["start"])
                r.update(mk=mk, cb=cb, tard=td, score=SC.score(s, i, mk, cb, td, ref=ref))
            f.write_text(json.dumps(r))
            log(f"t{a.limit} seed {s} inst {i:>3}: {r['status']:<8} {r['wall_s']:6.1f}s  score {r.get('score', float('nan')):.4f}"
                f"  bound {r['bound_score']:.4f}  hint {r['hint_score']:.4f}")
    log(f"CPSAT_T{a.limit}_DONE")


if __name__ == "__main__":
    main()
