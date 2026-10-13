"""CP-SAT reference bounds for benchmark v3 (docs/evaluation_protocol_v3.md, Part B). CPU only; no training data.

B1  every instance of the split (default: evaluation):
      MK  minimise makespan (--limit, default 60 s); lexicographic stages (--lex_limit, default 20 s each, hinted):
          minimise carbon s.t. makespan <= MK*, then tardiness s.t. makespan <= MK*, carbon <= C*
      T   minimise total tardiness (--limit); then carbon s.t. T <= T*, then makespan s.t. T <= T*, carbon <= C*
      C   minimum carbon, analytic: every operation on its lowest-carbon eligible machine (ties: lowest machine index;
          OPTIMAL by construction); then makespan with that assignment fixed, then tardiness s.t. makespan <= MK*
B2  epsilon-constraint front on the first --first instances: minimise makespan s.t. carbon <= eps_c, tardiness <= eps_t
    on a --grid x --grid grid of budgets q in {0, 1/(g-1), ..., 1} between each instance's ideal and nadir from B1
    (ideal = best value of each objective over the three B1 points, nadir = worst); --limit s per solve. The front is the
    non-dominated set of the feasible B2 points and the three B1 points (exact objectives).

Integer scaling for CP-SAT: carbon in units of 1e-4 kg CO2e (c_int = round(1e4 * c_om)); tardiness T_j >= 1000 * C_j -
ceil(1000 * d_j) in units of 1e-3 time units. Budgets and lexicographic bounds are expressed in these integer units of
the incumbent schedule (so the incumbent stays feasible); every returned schedule is re-scored EXACTLY by the
independent scorer benchmark_v3.schedule_objectives (feasibility included). Proven optimality refers to the scaled model.

Per-instance JSON under results/baselines_v3/cpsat/<b1|b2>/<split>_<size>/<file stem>.json; existing files are skipped
(resume). The reserved final-test sets are not part of benchmark v3 and cannot be selected.

    python tools/cpsat_baselines_v3.py b1 --size 10x5 [--start 0 --count 10]
    python tools/cpsat_baselines_v3.py b2 --size 20x10 --first 10 --limit 60
"""
from __future__ import annotations
import argparse, json, math, sys, time
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.argv, _ARGV = sys.argv[:1], sys.argv                    # daniel/params.py parses argv on import
sys.path.insert(0, str(REPO / "daniel"))
import benchmark_v3 as B                                    # noqa: E402
sys.argv = _ARGV
sys.path.insert(0, str(REPO / "tools" / "objective_validation"))
import common as OV                                         # noqa: E402
import schedules as S                                       # noqa: E402
from ortools.sat.python import cp_model                     # noqa: E402

SC, Q = 10_000, 1_000                                       # carbon and tardiness integer scales
SEED = 20261013


class Inst:
    def __init__(self, data, i):
        self.key = data.keys[i]; self.name = Path(self.key).stem
        self.jl = np.asarray(data.job_length[i]); self.pt = np.asarray(data.op_pt[i]).astype(int)
        self.carbon = np.asarray(data.carbon[i]); self.due = np.asarray(data.due[i])
        self.n, self.m = self.pt.shape
        self.first = np.concatenate([[0], np.cumsum(self.jl)[:-1]]).astype(int)
        self.last = self.first + self.jl - 1
        self.c_int = np.rint(self.carbon * SC).astype(int)
        self.dq = np.array([math.ceil(Q * float(d)) for d in self.due])

    def ints(self, machine, start):
        """Integer (makespan, carbon_int, tardiness_int) of a schedule in the scaled model's units."""
        comp = start + self.pt[np.arange(self.n), machine]
        return (int(comp.max()), int(self.c_int[np.arange(self.n), machine].sum()),
                int(np.maximum(0, Q * comp[self.last] - self.dq).sum()))

    def exact(self, machine, start):
        r = B.schedule_objectives(self.jl, self.pt, self.carbon, self.due, machine, start)
        comp = r["completion"]
        return {"makespan": r["makespan"], "carbon": r["carbon"], "tardiness": r["tardiness"],
                "tardy_share": float((comp[self.last] > self.due).mean())}


def solve(inst: Inst, objective: str, limit: float, workers: int, ub=None, fixed=None, hint=None):
    """objective in {'mk', 'carbon', 'tard'}; ub: dict of integer upper bounds; fixed: machine per op; hint: (m, s)."""
    ub = ub or {}
    md = cp_model.CpModel()
    H = int(sum(inst.pt[o][inst.pt[o] > 0].max() for o in range(inst.n)))
    st = [md.NewIntVar(0, H, f"s{o}") for o in range(inst.n)]
    en = [md.NewIntVar(0, H, f"e{o}") for o in range(inst.n)]
    x, per_m = {}, {m: [] for m in range(inst.m)}
    for o in range(inst.n):
        lits = []
        for m in ([int(fixed[o])] if fixed is not None else np.flatnonzero(inst.pt[o] > 0)):
            b = md.NewBoolVar(f"x{o}_{m}"); x[o, int(m)] = b; lits.append(b)
            per_m[int(m)].append(md.NewOptionalIntervalVar(st[o], int(inst.pt[o, m]), en[o], b, f"i{o}_{m}"))
        md.AddExactlyOne(lits)
    for f, l in zip(inst.first, inst.last):
        for o in range(f, l):
            md.Add(st[o + 1] >= en[o])
    for m in range(inst.m):
        md.AddNoOverlap(per_m[m])
    mk = md.NewIntVar(0, H, "mk")
    md.AddMaxEquality(mk, [en[o] for o in inst.last])
    carbon = sum(int(inst.c_int[o, m]) * b for (o, m), b in x.items())
    tq = []
    for j, o in enumerate(inst.last):
        t = md.NewIntVar(0, Q * H, f"t{j}"); md.Add(t >= Q * en[o] - int(inst.dq[j])); tq.append(t)
    tard = sum(tq)
    expr = {"mk": mk, "carbon": carbon, "tard": tard}
    for k, v in ub.items():
        md.Add(expr[k] <= int(v))
    md.Minimize(expr[objective])
    if hint is not None:
        hm, hs = hint
        for o in range(inst.n):
            md.AddHint(st[o], int(hs[o]))
            for (oo, m), b in x.items():
                if oo == o:
                    md.AddHint(b, int(m == hm[o]))
    sv = cp_model.CpSolver()
    sv.parameters.max_time_in_seconds = float(limit)
    sv.parameters.num_workers = int(workers)
    sv.parameters.random_seed = SEED
    t0 = time.time(); status = sv.Solve(md); wall = time.time() - t0
    out = {"objective": objective, "status": sv.StatusName(status), "wall_s": wall, "limit_s": limit,
           "ub_int": {k: int(v) for k, v in ub.items()}}
    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        machine = np.array([next(m for (oo, m), b in x.items() if oo == o and sv.Value(b)) for o in range(inst.n)])
        start = np.array([sv.Value(s) for s in st])
        out.update(machine=machine.tolist(), start=start.tolist(), value_int=int(sv.ObjectiveValue()),
                   bound_int=float(sv.BestObjectiveBound()), exact=inst.exact(machine, start))
    return out


def lexicographic(inst, first, order, lex_limit, workers, fixed=None):
    """Stages after `first`: minimise order[0] with the earlier objectives bounded, then order[1]."""
    stages, cur = [], first
    bounded = {first["objective"]: inst.ints(np.array(first["machine"]), np.array(first["start"]))[
        ("mk", "carbon", "tard").index(first["objective"])]}
    for obj in order:
        r = solve(inst, obj, lex_limit, workers, ub=dict(bounded), fixed=fixed,
                  hint=(np.array(cur["machine"]), np.array(cur["start"])))
        stages.append(r)
        if "machine" in r:
            cur = r
        ints = inst.ints(np.array(cur["machine"]), np.array(cur["start"]))
        bounded[obj] = ints[("mk", "carbon", "tard").index(obj)]
    return stages, cur


def rule_hint(inst, objective):
    """Warm start: the better of the MWKR/ECT (reference) and EDD/ECT dispatching schedules for `objective`."""
    ov = OV.Instance(inst.key, inst.jl, inst.pt, np.repeat(np.arange(len(inst.jl)), inst.jl), inst.first)
    cands = [S.build(ov, "MWKR", "ECT"), S.build(ov, "EDD", "ECT", due=inst.due)]
    k = ("mk", "carbon", "tard").index(objective)
    return min(cands, key=lambda ms: inst.ints(np.asarray(ms[0]), np.asarray(ms[1]))[k])


def b1_instance(inst, limit, lex_limit, workers):
    rec = {"key": inst.key, "scales": {"carbon_per_kg": SC, "tardiness_per_time_unit": Q},
           "warm_start": "better of MWKR/ECT and EDD/ECT dispatching schedules", "points": {}}
    for obj, order in (("mk", ("carbon", "tard")), ("tard", ("carbon", "mk"))):
        first = solve(inst, obj, limit, workers, hint=rule_hint(inst, obj))
        if "machine" not in first:
            rec["points"][obj] = {"primary": first, "final": None}
            continue
        stages, final = lexicographic(inst, first, order, lex_limit, workers)
        rec["points"][obj] = {"primary": first, "lex_stages": stages,
                              "final": {"machine": final["machine"], "start": final["start"],
                                        "exact": inst.exact(np.array(final["machine"]), np.array(final["start"]))}}
    elig = inst.pt > 0
    fixed = np.array([np.flatnonzero(elig[o])[np.argmin(inst.carbon[o][elig[o]])] for o in range(inst.n)])
    cmin = float(inst.carbon[np.arange(inst.n), fixed].sum())
    first = solve(inst, "mk", lex_limit, workers, fixed=fixed)          # a schedule for the analytic assignment
    if "machine" not in first:
        rec["points"]["carbon"] = {"primary": {"objective": "carbon", "status": "OPTIMAL", "method": "analytic",
                                               "value": cmin, "wall_s": 0.0}, "lex_stages": [first], "final": None}
        return rec
    stages, final = lexicographic(inst, first, ("tard",), lex_limit, workers, fixed=fixed)
    rec["points"]["carbon"] = {"primary": {"objective": "carbon", "status": "OPTIMAL", "method": "analytic",
                                           "value": cmin, "wall_s": 0.0},
                               "lex_stages": [first] + stages,
                               "final": {"machine": final["machine"], "start": final["start"],
                                         "exact": inst.exact(np.array(final["machine"]), np.array(final["start"]))}}
    return rec


def nondominated(points):
    keep = []
    for i, p in enumerate(points):
        a = np.array([p["makespan"], p["carbon"], p["tardiness"]])
        dom = False
        for j, q in enumerate(points):
            b = np.array([q["makespan"], q["carbon"], q["tardiness"]])
            if j != i and np.all(b <= a + 1e-9) and np.any(b < a - 1e-9):
                dom = True; break
            if j < i and np.all(np.abs(b - a) <= 1e-9):
                dom = True; break                                       # exact duplicate: keep the first
        if not dom:
            keep.append(p)
    return keep


def b2_instance(inst, b1, grid, limit, workers):
    pts = {k: v["final"] for k, v in b1["points"].items() if v.get("final")}
    if len(pts) < 3:
        return {"key": inst.key, "error": "B1 incomplete for this instance"}
    ints = {k: inst.ints(np.array(v["machine"]), np.array(v["start"])) for k, v in pts.items()}
    c_lo, c_hi = min(v[1] for v in ints.values()), max(v[1] for v in ints.values())
    t_lo, t_hi = min(v[2] for v in ints.values()), max(v[2] for v in ints.values())
    qs = [i / (grid - 1) for i in range(grid)] if grid > 1 else [1.0]
    rec = {"key": inst.key, "grid": grid, "q": qs, "ideal_int": {"carbon": c_lo, "tard": t_lo},
           "nadir_int": {"carbon": c_hi, "tard": t_hi}, "points": []}
    for qc in qs:
        for qt in qs:
            ub = {"carbon": int(math.floor(c_lo + qc * (c_hi - c_lo))), "tard": int(math.floor(t_lo + qt * (t_hi - t_lo)))}
            hint = None
            ok = [k for k, v in ints.items() if v[1] <= ub["carbon"] and v[2] <= ub["tard"]]
            if ok:
                k = min(ok, key=lambda k: ints[k][0]); hint = (np.array(pts[k]["machine"]), np.array(pts[k]["start"]))
            r = solve(inst, "mk", limit, workers, ub=ub, hint=hint)
            r.update(q_carbon=qc, q_tard=qt, eps_carbon=ub["carbon"] / SC, eps_tard=ub["tard"] / Q)
            rec["points"].append(r)
    cand = [dict(r["exact"], source=f"b2 q=({r['q_carbon']:.2f},{r['q_tard']:.2f})", status=r["status"])
            for r in rec["points"] if "exact" in r]
    cand += [dict(v["exact"], source=f"b1 {k}", status="b1") for k, v in pts.items()]
    rec["front"] = nondominated(cand)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("part", choices=["b1", "b2"])
    ap.add_argument("--size", required=True, choices=list(B.SIZES))
    ap.add_argument("--split", default="evaluation", choices=list(B.SPLITS))
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--count", type=int, default=10 ** 9)
    ap.add_argument("--first", type=int, default=None, help="B2: the first N instances of the split")
    ap.add_argument("--limit", type=float, default=None, help="primary / epsilon solve limit in seconds")
    ap.add_argument("--lex_limit", type=float, default=20.0)
    ap.add_argument("--grid", type=int, default=5)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default=str(REPO / "results" / "baselines_v3" / "cpsat"))
    a = ap.parse_args()
    data = B.load_split(a.split, a.size)
    out = Path(a.out) / a.part / f"{a.split}_{a.size}"
    out.mkdir(parents=True, exist_ok=True)
    if a.part == "b1":
        limit = 60.0 if a.limit is None else a.limit
        idx = range(a.start, min(len(data), a.start + a.count))
    else:
        limit = a.limit if a.limit is not None else (30.0 if a.size == "10x5" else 60.0)
        n = a.first if a.first is not None else (20 if a.size == "10x5" else 10)
        idx = range(a.start, min(n, a.start + a.count))
    for i in idx:
        inst = Inst(data, i)
        f = out / f"{inst.name}.json"
        if f.exists():
            continue
        t0 = time.time()
        if a.part == "b1":
            rec = b1_instance(inst, limit, a.lex_limit, a.workers)
        else:
            b1f = Path(a.out) / "b1" / f"{a.split}_{a.size}" / f"{inst.name}.json"
            if not b1f.exists():
                print(f"skip {inst.name}: B1 result missing", flush=True); continue
            rec = b2_instance(inst, json.loads(b1f.read_text()), a.grid, limit, a.workers)
        rec.update(index=i, split=a.split, size=a.size, wall_total_s=time.time() - t0, workers=a.workers,
                   limits={"primary": limit, "lex": a.lex_limit}, manifest_sha256=B.file_sha256(B.MANIFEST))
        tmp = f.with_suffix(".tmp"); tmp.write_text(json.dumps(rec)); tmp.replace(f)
        summary = ({k: (v["primary"]["status"], v["final"]["exact"] if v.get("final") else None)
                    for k, v in rec["points"].items()} if a.part == "b1" else
                   {"front": len(rec.get("front", [])), "statuses": [p["status"] for p in rec.get("points", [])]})
        print(f"{a.part} {a.size} {inst.name} {time.time() - t0:.0f}s {json.dumps(summary)}", flush=True)
    print(f"CPSAT_{a.part.upper()}_BATCH_DONE")


if __name__ == "__main__":
    main()
