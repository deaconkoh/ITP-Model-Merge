"""The non-delay ceiling (docs/decoding_ceiling_protocol_v3.md, Part B): CP-SAT restricted to DANIEL's scheduling rule.

DANIEL's environment only allows an operation-machine pair whose start max(job ready, machine free) equals the
earliest such start over all eligible pairs, so every schedule it produces is non-delay and every non-delay schedule
can be produced (proof in the report). The model below is the CP-SAT model of tools/cpsat_baselines_v3.py plus the
non-delay condition: for every machine m, every operation o eligible on m and every idle gap of m (before its first
operation, between consecutive operations, after its last), the gap does not intersect [r_o, s_o), where r_o is the
completion of o's job predecessor (0 for a first operation). Consecutive operations come from a circuit per machine.

    python tools/nondelay_cpsat_v3.py check                      # equivalence checks (fixed schedules), B1 mask check
    python tools/nondelay_cpsat_v3.py pilot                      # time-limit pilot on 2 validation instances
    python tools/nondelay_cpsat_v3.py solve --limit 20 --workers 4 --start 0 --count 10
"""
from __future__ import annotations
import argparse, json, math, sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "daniel")); sys.path.insert(0, str(REPO / "tools"))
_argv = sys.argv
sys.argv = [sys.argv[0], "--device", "cpu", "--obs_version", "v3", "--reward_version", "v3", "--goal", "mct"]
import numpy as np                                                  # noqa: E402
from params import configs                                          # noqa: E402,F401
import benchmark_v3 as B                                            # noqa: E402
import mo_metrics as MM                                             # noqa: E402
sys.argv = _argv
import tardiness_diagnosis_v3 as TD                                 # noqa: E402
from ortools.sat.python import cp_model                             # noqa: E402

SC, Q = 10_000, 1_000                                               # carbon and tardiness integer scales (as B1/B2)
K = 10 ** 11                                                        # objective scale; coefficients rounded down
SEED = 20261113
OUT = REPO / "results" / "decoding_ceiling_v3" / "nondelay_cpsat"
PREFS = MM.simplex_grid(0.1)


class Inst:
    def __init__(self, data, i):
        self.i = i; self.key = data.keys[i]
        self.jl = np.asarray(data.job_length[i]); self.pt = np.asarray(data.op_pt[i]).astype(int)
        self.carbon = np.asarray(data.carbon[i]); self.due = np.asarray(data.due[i], dtype=float)
        self.n, self.m = self.pt.shape
        self.first = np.concatenate([[0], np.cumsum(self.jl)[:-1]]).astype(int)
        self.last = self.first + self.jl - 1
        self.pred = np.arange(self.n) - 1; self.pred[self.first] = -1
        self.c_int = np.rint(self.carbon * SC).astype(int)
        self.dq = np.array([math.ceil(Q * float(d)) for d in self.due])
        self.mk_ref, self.cb_ref, self.t_ref = data.mk_ref[i], data.cb_ref[i], data.t_ref

    def score(self, obj, w):
        return w[0] * obj[0] / self.mk_ref + w[1] * obj[1] / self.cb_ref + w[2] * obj[2] / self.t_ref

    def exact(self, machine, start):
        r = B.schedule_objectives(self.jl, self.pt, self.carbon, self.due, machine, start)
        return np.array([r["makespan"], r["carbon"], r["tardiness"]])

    def coef(self, w):
        """Integer objective coefficients (rounded down) for makespan, carbon (1e-4 kg) and tardiness (1e-3)."""
        return (int(math.floor(K * w[0] / self.mk_ref)), int(math.floor(K * w[1] / self.cb_ref / SC)),
                int(math.floor(K * w[2] / self.t_ref / Q)))

    def lower_bound(self, w, bound_int):
        """A proven lower bound on the true score from a proven bound on the integer objective: c_k <= K a_k / scale_k,
        carbon rounding adds at most n * 0.5e-4 kg, and the rounded-up due dates never overstate tardiness."""
        return bound_int / K - (w[1] / self.cb_ref) * self.n * 0.5 / SC


def build(inst: Inst, w=None, fix=None, hint=None, nondelay=True):
    """The model; fix = (machine, start) pins every variable (feasibility check); hint = (machine, start)."""
    md = cp_model.CpModel()
    H = int(sum(inst.pt[o][inst.pt[o] > 0].max() for o in range(inst.n)))
    st = [md.NewIntVar(0, H, f"s{o}") for o in range(inst.n)]
    en = [md.NewIntVar(0, H, f"e{o}") for o in range(inst.n)]
    x, per_m, elig = {}, {m: [] for m in range(inst.m)}, {m: [] for m in range(inst.m)}
    for o in range(inst.n):
        lits = []
        for m in np.flatnonzero(inst.pt[o] > 0):
            m = int(m)
            b = md.NewBoolVar(f"x{o}_{m}"); x[o, m] = b; lits.append(b); elig[m].append(o)
            per_m[m].append(md.NewOptionalIntervalVar(st[o], int(inst.pt[o, m]), en[o], b, f"i{o}_{m}"))
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
    arcs = {}
    if nondelay:
        cache = {}

        def le(a, b):
            """literal => st[a] <= en[b] (b = -1: <= 0)"""
            k = (a, b)
            if k not in cache:
                v = md.NewBoolVar(f"le{a}_{b}")
                md.Add(st[a] <= (en[b] if b >= 0 else 0)).OnlyEnforceIf(v)
                cache[k] = v
            return cache[k]
        for m in range(inst.m):
            ops = elig[m]
            node = {o: k + 1 for k, o in enumerate(ops)}
            arc_list = []
            empty = md.NewBoolVar(f"empty{m}")
            arc_list.append((0, 0, empty))
            for o in ops:
                arc_list.append((node[o], node[o], x[o, m].Not()))
                a0 = md.NewBoolVar(f"a0_{o}_{m}"); arc_list.append((0, node[o], a0)); arcs[m, -1, o] = a0
                b0 = md.NewBoolVar(f"b0_{o}_{m}"); arc_list.append((node[o], 0, b0)); arcs[m, o, -1] = b0
            for a in ops:
                for b in ops:
                    if a != b:
                        l = md.NewBoolVar(f"y{a}_{b}_{m}"); arc_list.append((node[a], node[b], l)); arcs[m, a, b] = l
                        md.Add(st[b] >= en[a]).OnlyEnforceIf(l)
            md.AddCircuit(arc_list)
            for o in ops:
                po = int(inst.pred[o])
                rd = le(o, po)                                                  # s_o <= r_o (no wait at all)
                md.AddBoolOr([empty.Not(), rd])
                for b in ops:                                                   # gap [0, s_b) before the first
                    md.AddBoolOr([arcs[m, -1, b].Not(), le(b, -1), le(b, po), le(o, -1), rd])
                for a in ops:                                                   # gap [e_a, inf) after the last
                    md.AddBoolOr([arcs[m, a, -1].Not(), le(o, a), rd])
                for a in ops:                                                   # gap [e_a, s_b) between a and b
                    if a == o:
                        continue
                    for b in ops:
                        if b == a:
                            continue
                        md.AddBoolOr([arcs[m, a, b].Not(), le(b, a), le(b, po), le(o, a), rd])
    if w is not None:
        c = inst.coef(w)
        md.Minimize(c[0] * mk + c[1] * carbon + c[2] * tard)
    if fix is not None or hint is not None:
        fm, fs = (fix if fix is not None else hint)
        fm = np.asarray(fm, int); fs = np.rint(np.asarray(fs, float)).astype(int)
        put = (lambda v, val: md.Add(v == int(val))) if fix is not None else (lambda v, val: md.AddHint(v, int(val)))
        for o in range(inst.n):
            put(st[o], fs[o])
            for m in np.flatnonzero(inst.pt[o] > 0):
                put(x[o, int(m)], int(fm[o] == m))
        if hint is not None and nondelay:
            for m in range(inst.m):
                seq = [o for o in np.argsort(fs, kind="stable") if fm[o] == m]
                on = set(seq)
                succ = {a: b for a, b in zip(seq, seq[1:])}
                for (mm, a, b), l in arcs.items():
                    if mm != m:
                        continue
                    if a == -1:
                        val = bool(seq) and seq[0] == b
                    elif b == -1:
                        val = bool(seq) and seq[-1] == a
                    else:
                        val = succ.get(a) == b
                    md.AddHint(l, int(val))
    return md, {"st": st, "x": x, "mk": mk, "carbon": carbon, "tard": tard}


def solve(inst, w, limit, workers, hint=None):
    md, v = build(inst, w, hint=hint)
    sv = cp_model.CpSolver()
    sv.parameters.max_time_in_seconds = float(limit)
    sv.parameters.num_workers = int(workers)
    sv.parameters.random_seed = SEED
    t0 = time.time(); status = sv.Solve(md); wall = time.time() - t0
    out = {"status": sv.StatusName(status), "wall_s": wall, "bound_int": float(sv.BestObjectiveBound()),
           "lower_bound": inst.lower_bound(w, float(sv.BestObjectiveBound())) if status in (cp_model.OPTIMAL, cp_model.FEASIBLE) else None}
    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        machine = np.array([next(m for (oo, m), b in v["x"].items() if oo == o and sv.Value(b)) for o in range(inst.n)])
        start = np.array([sv.Value(s) for s in v["st"]], dtype=float)
        obj = inst.exact(machine, start)
        out.update(machine=machine.tolist(), start=start.tolist(), obj=obj.tolist(), score=float(inst.score(obj, w)),
                   value_int=float(sv.ObjectiveValue()))
    return out


def feasible_fixed(inst, machine, start):
    md, _ = build(inst, fix=(machine, start))
    sv = cp_model.CpSolver(); sv.parameters.max_time_in_seconds = 30.0; sv.parameters.num_workers = 1
    st = sv.Solve(md)
    return sv.StatusName(st)


# ------------------------------------------------------------------------------------------------ checks

def mask_check(data, sub, keys, label):
    """DANIEL's mask (TD.NonDelay: the environment's scaled float arithmetic, bit-identical to it) against the same rule
    in exact integer arithmetic, at every step of the trajectories decoded from `keys`."""
    d = TD.NonDelay(data, sub)
    d.reset()
    mism, steps, extra, missing = 0, 0, 0, 0
    pol = TD.key_policy(keys)
    for _ in range(d.N):
        ok, nst = d.allowed()
        el = d.elig[d.ei[:, None], d.cand] & ~d.done[:, :, None]
        pf = np.maximum(d.rr[:, :, None], d.Fr[:, None, :])                     # raw (integer) times
        nst_r = np.min(np.where(el, pf, np.inf).reshape(d.E, -1), 1)
        ok_r = el & (pf == nst_r[:, None, None])
        diff = ok != ok_r
        mism += int(diff.any((1, 2)).sum()); steps += d.E
        extra += int((ok & ~ok_r).sum()); missing += int((~ok & ok_r).sum())
        d.step(pol(d, ok, nst))
    return {"label": label, "states": steps, "states_with_difference": mism, "pairs_allowed_only_by_float": extra,
            "pairs_allowed_only_by_exact": missing}


def check(n_inst=20):
    import gzip
    data = B.load_split("evaluation", "10x5")
    OUT.mkdir(parents=True, exist_ok=True)
    rep = {}
    # B1: the mask in float arithmetic against exact integers, on pool schedules and on random decodings
    pool = np.load(REPO / "results" / "cpsat_gap_v3" / "pool" / "fixed.npz")
    sub = np.arange(n_inst)
    rng = np.random.default_rng(7)
    res = []
    for r in range(25):                                                        # 25 x 20 random-key decodings
        res.append(mask_check(data, sub, rng.random((n_inst, 50, 5)), "random"))
    names = [k[: -len("__machine")] for k in pool.files if k.endswith("__machine")]
    for nm in names:
        m, s = pool[nm + "__machine"][:n_inst], pool[nm + "__start"][:n_inst]
        res.append(mask_check(data, sub, TD.keys_from_schedule(m, s, 5), nm))
    rep["mask_check"] = {k: int(sum(x[k] for x in res)) for k in ("states", "states_with_difference",
                                                                 "pairs_allowed_only_by_float", "pairs_allowed_only_by_exact")}
    print("mask check:", rep["mask_check"], flush=True)
    # B2 equivalence: DANIEL schedules feasible when fixed, CP-SAT left-shifted (not non-delay) infeasible
    with gzip.open(REPO / "results" / "cpsat_gap_v3" / "analysis.json.gz", "rt") as fh:
        AN = json.loads(fh.read())
    feas, infeas = [], []
    for i in range(n_inst):
        inst = Inst(data, i)
        for nm in names[:6]:
            st = feasible_fixed(inst, pool[nm + "__machine"][i], pool[nm + "__start"][i])
            feas.append(st)
    import cpsat_gap_v3_analysis as GA                                          # noqa: F401 (definitions only)
    from cpsat_gap_v3_analysis import Inst as GI, left_shift, is_nondelay      # noqa: E402
    for i in range(n_inst):
        inst = Inst(data, i); gi = GI(data, i)
        b2 = json.loads((REPO / "results/baselines_v3/cpsat/b2/evaluation_10x5" / f"{Path(inst.key).stem}.json").read_text())
        for p in [p for p in b2["points"] if "machine" in p][:6]:
            m = np.asarray(p["machine"]); s = left_shift(gi, m, np.asarray(p["start"], float))
            nd = is_nondelay(gi, m, s)
            infeas.append((feasible_fixed(inst, m, s), nd))
    rep["fixed_daniel_schedules"] = {s: feas.count(s) for s in set(feas)}
    rep["fixed_left_shifted_cpsat"] = {f"{s} (non-delay by the checker: {nd})": infeas.count((s, nd)) for s, nd in set(infeas)}
    print("fixed DANIEL schedules:", rep["fixed_daniel_schedules"], flush=True)
    print("fixed left-shifted CP-SAT schedules:", rep["fixed_left_shifted_cpsat"], flush=True)
    (OUT / "check.json").write_text(json.dumps(rep, indent=1) + "\n")


def best_known(i, w, split="evaluation"):
    """Best known DANIEL-reachable schedule for (instance, preference): CP-SAT audit pool plus Part A decoders."""
    cands = []
    pool = np.load(REPO / "results" / "cpsat_gap_v3" / "pool" / "fixed.npz")
    for k in pool.files:
        if k.endswith("__machine"):
            cands.append((pool[k][i], pool[k[: -len("__machine")] + "__start"][i]))
    for f in sorted((REPO / "results" / "cpsat_gap_v3" / "pool").glob("mo__*.npz")):
        z = np.load(f)
        for mode in ("greedy", "s16"):
            if f"{mode}_machine" in z.files:
                cands += list(zip(z[f"{mode}_machine"][i], z[f"{mode}_start"][i]))
    for f in sorted((REPO / "results" / "decoding_ceiling_v3" / split).glob("*.npz")):
        z = np.load(f)
        cands += list(zip(z["machine"][i], z["start"][i]))
    return cands


def solve_range(start, count, limit, workers, split="evaluation"):
    data = B.load_split(split, "10x5")
    out = OUT / split
    out.mkdir(parents=True, exist_ok=True)
    for i in range(start, min(start + count, 20 if split == "evaluation" else 18)):
        inst = Inst(data, i)
        f = out / f"{Path(inst.key).stem}.json"
        rec = json.loads(f.read_text()) if f.exists() else {"key": inst.key, "limit_s": limit, "workers": workers,
                                                            "solves": {}}
        cands = best_known(i, None, split)
        objs = np.stack([inst.exact(m, s) for m, s in cands])
        for g, w in enumerate(PREFS):
            if str(g) in rec["solves"]:
                continue
            sc = np.array([inst.score(o, w) for o in objs])
            h = int(np.argmin(sc))
            r = solve(inst, w, limit, workers, hint=cands[h])
            r["hint_score"] = float(sc[h])
            rec["solves"][str(g)] = r
            tmp = f.with_suffix(".tmp"); tmp.write_text(json.dumps(rec)); tmp.replace(f)
            print(f"[{time.strftime('%H:%M:%S')}] {split} {i} w{g} {r['status']} score {r.get('score', float('nan')):.4f} "
                  f"hint {r['hint_score']:.4f} bound {r['lower_bound'] if r['lower_bound'] is not None else float('nan'):.4f} "
                  f"({r['wall_s']:.1f} s)", flush=True)


# ------------------------------------------------------------------------------------------------ amendment: subset bounds

SUBSET_W = [(0, 0, 1), (0, 1, 0), (1, 0, 0), (.5, .5, 0), (.5, 0, .5), (0, .5, .5), (.6, .2, .2), (.2, .6, .2),
            (.2, .2, .6), (.4, .3, .3)]
SUBSET = [int(np.argmin(np.abs(PREFS - np.asarray(w)).sum(1))) for w in SUBSET_W]


def candidates(data, i, split="evaluation"):
    """Distinct known DANIEL schedules of instance i (machine [S, N], start [S, N], objectives [S, 3])."""
    cands = best_known(i, None, split)
    srch = REPO / "results" / "decoding_ceiling_v3" / "search" / f"{split}.npz"
    if srch.exists():
        z = np.load(srch)
        cands += list(zip(z["machine"][i], z["start"][i]))
    m = np.stack([c[0] for c in cands]).astype(int); s = np.stack([c[1] for c in cands]).astype(float)
    _, u = np.unique(np.concatenate([m, np.rint(s).astype(int)], 1), axis=0, return_index=True)
    m, s = m[u], s[u]
    inst = Inst(data, i)
    return m, s, np.stack([inst.exact(a, b) for a, b in zip(m, s)])


def bounds(which, limit, workers, start=0, count=20):
    data = B.load_split("evaluation", "10x5")
    out = OUT / "subset"
    out.mkdir(parents=True, exist_ok=True)
    for i in range(start, min(20, start + count)):
        inst = Inst(data, i)
        f = out / f"{which}_{Path(inst.key).stem}.json"
        rec = json.loads(f.read_text()) if f.exists() else {"key": inst.key, "limit_s": limit, "workers": workers, "solves": {}}
        if which == "unrestricted":
            b2 = json.loads((REPO / "results/baselines_v3/cpsat/b2/evaluation_10x5" / f"{Path(inst.key).stem}.json").read_text())
            pts = [(np.asarray(p["machine"]), np.asarray(p["start"], float)) for p in b2["points"] if "machine" in p]
            b1 = json.loads((REPO / "results/baselines_v3/cpsat/b1/evaluation_10x5" / f"{Path(inst.key).stem}.json").read_text())
            pts += [(np.asarray(v["final"]["machine"]), np.asarray(v["final"]["start"], float)) for v in b1["points"].values() if v.get("final")]
            objs = np.stack([inst.exact(m, s) for m, s in pts])
        else:
            m_, s_, objs = candidates(data, i)
            pts = list(zip(m_, s_))
        for g in SUBSET:
            if str(g) in rec["solves"]:
                continue
            w = PREFS[g]
            sc = np.array([inst.score(o, w) for o in objs]); h = int(np.argmin(sc))
            md, v = build(inst, w, hint=pts[h], nondelay=(which == "restricted"))
            sv = cp_model.CpSolver(); sv.parameters.max_time_in_seconds = float(limit)
            sv.parameters.num_workers = int(workers); sv.parameters.random_seed = SEED
            t0 = time.time(); stt = sv.Solve(md); wall = time.time() - t0
            r = {"status": sv.StatusName(stt), "wall_s": wall, "hint_score": float(sc[h])}
            if stt in (cp_model.OPTIMAL, cp_model.FEASIBLE):
                mach = np.array([next(m for (oo, m), b in v["x"].items() if oo == o and sv.Value(b)) for o in range(inst.n)])
                stv = np.array([sv.Value(x) for x in v["st"]], dtype=float)
                obj = inst.exact(mach, stv)
                r.update(machine=mach.tolist(), start=stv.tolist(), obj=obj.tolist(), score=float(inst.score(obj, w)),
                         bound_int=float(sv.BestObjectiveBound()),
                         lower_bound=inst.lower_bound(w, float(sv.BestObjectiveBound())))
            rec["solves"][str(g)] = r
            tmp = f.with_suffix(".tmp"); tmp.write_text(json.dumps(rec)); tmp.replace(f)
            print(f"[{time.strftime('%H:%M:%S')}] {which} {i} w{g} {r['status']} score {r.get('score', float('nan')):.4f} "
                  f"hint {r['hint_score']:.4f} bound {r.get('lower_bound', float('nan')):.4f} ({wall:.1f} s)", flush=True)


# ------------------------------------------------------------------------------------------------ amendment: search

def search(split="evaluation", pop=100, elite=10, gens=300, n_seed=20, quick=False):
    if quick:
        gens = 5
    """Random-key genetic search over DANIEL-decoded schedules, one preference at a time (all instances batched),
    fitness = the selection score; seeded with the best known schedules for that preference."""
    data = B.load_split(split, "10x5")
    I = 2 if quick else (20 if split == "evaluation" else 18)
    sub = np.arange(I)
    insts = [Inst(data, i) for i in sub]
    cand = [candidates(data, i, split) for i in sub]
    carbon = np.stack([np.asarray(data.carbon[i]) for i in sub])
    N, M = insts[0].n, insts[0].m
    out = {"machine": np.zeros((I, 66, N), int), "start": np.zeros((I, 66, N)), "obj": np.zeros((I, 66, 3)),
           "seed_score": np.zeros((I, 66)), "score": np.zeros((I, 66)), "generation": np.zeros((I, 66), int)}
    rng = np.random.default_rng(2026)
    gl = range(2) if quick else range(66)
    t_all = time.time()
    for g in gl:
        w = PREFS[g]
        seeds = np.zeros((I, n_seed, N, M))
        for i in range(I):
            sc = np.array([insts[i].score(o, w) for o in cand[i][2]])
            top = np.argsort(sc, kind="stable")[:n_seed]
            top = np.resize(top, n_seed)
            seeds[i] = TD.keys_from_schedule(cand[i][0][top], cand[i][1][top], M)
            out["seed_score"][i, g] = sc[top[0]]
        dec = TD.NonDelay(data, np.repeat(sub, pop))
        P = pop
        popu = rng.random((I, P, N, M)); popu[:, :n_seed] = seeds
        best = np.full(I, np.inf)
        mk_ref = np.array([x.mk_ref for x in insts]); cb_ref = np.array([x.cb_ref for x in insts])
        for gen in range(gens):
            res = dec.run(TD.key_policy(popu.reshape(I * P, N, M)))
            cb = carbon[np.repeat(np.arange(I), P)[:, None], np.arange(N)[None, :], res["machine"]].sum(1)
            fit = (w[0] * res["makespan"] / np.repeat(mk_ref, P) + w[1] * cb / np.repeat(cb_ref, P)
                   + w[2] * res["tardiness"] / data.t_ref).reshape(I, P)
            order = np.argsort(fit, 1, kind="stable")
            for i in range(I):
                v = fit[i, order[i, 0]]
                if v < best[i] - 1e-12:
                    best[i] = v; r = i * P + order[i, 0]
                    out["machine"][i, g] = res["machine"][r]; out["start"][i, g] = res["start"][r]
                    out["generation"][i, g] = gen
            elite_k = np.take_along_axis(popu, order[:, :elite, None, None], 1)
            nc = P - elite
            pa = elite_k[np.arange(I)[:, None], rng.integers(0, elite, (I, nc))]
            pb = elite_k[np.arange(I)[:, None], rng.integers(0, elite, (I, nc))]
            child = np.where(rng.random((I, nc, N, 1)) < 0.5, pa, pb)
            mut = rng.random(child.shape) < TD.SEARCH["mutation_rate"]
            child = np.where(mut, child + rng.normal(0, TD.SEARCH["mutation_sigma"], child.shape), child)
            imm = rng.random((I, nc)) < TD.SEARCH["immigrants"]
            child[imm] = rng.random((int(imm.sum()), N, M))
            popu = np.concatenate([elite_k, child], 1)
        for i in range(I):
            out["obj"][i, g] = insts[i].exact(out["machine"][i, g], out["start"][i, g])
            out["score"][i, g] = insts[i].score(out["obj"][i, g], w)
            assert out["score"][i, g] <= out["seed_score"][i, g] + 1e-9
        print(f"[{time.strftime('%H:%M:%S')}] search w{g}: mean score {out['score'][:, g].mean():.4f} "
              f"(seed {out['seed_score'][:, g].mean():.4f}; improved {(out['score'][:, g] < out['seed_score'][:, g] - 1e-9).sum()}/{I}) "
              f"{time.time() - t_all:.0f} s", flush=True)
    # every result replayed through the real environment and re-scored
    flat_m = out["machine"][:, list(gl)].reshape(-1, N); flat_s = out["start"][:, list(gl)].reshape(-1, N)
    idx = np.repeat(sub, len(list(gl)))
    d = TD.NonDelay(data, idx)
    res = d.run(TD.key_policy(TD.keys_from_schedule(flat_m, flat_s, M)))
    TD.check(data, idx, res, "search")
    assert np.array_equal(res["machine"], flat_m) and np.allclose(res["start"], flat_s), "search: not reproduced"
    dst = REPO / "results" / "decoding_ceiling_v3" / "search" / (f"{split}_quick.npz" if quick else f"{split}.npz")
    dst.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dst, **out, settings=json.dumps({"population": pop, "elite": elite, "generations": gens,
                                                         "seeds": n_seed, **TD.SEARCH}))
    print(f"search written: {dst}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("action", choices=["check", "pilot", "solve", "bounds", "search"])
    ap.add_argument("--which", default="unrestricted", choices=["unrestricted", "restricted"])
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--limit", type=float, default=20.0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--count", type=int, default=20)
    ap.add_argument("--split", default="evaluation", choices=["evaluation", "validation"])
    a = ap.parse_args()
    if a.action == "check":
        check()
    elif a.action == "solve":
        solve_range(a.start, a.count, a.limit, a.workers, a.split)
    elif a.action == "bounds":
        bounds(a.which, a.limit, a.workers, a.start, a.count)
    elif a.action == "search":
        search(a.split, quick=a.quick)
    else:
        data = B.load_split("validation", "10x5")
        for i in (0, 1):
            inst = Inst(data, i)
            for g in (0, 10, 30, 45, 65):
                for lim in (10, 30, 60):
                    r = solve(inst, PREFS[g], lim, a.workers)
                    print(f"pilot validation {i} w{g} limit {lim}: {r['status']} score {r.get('score', float('nan')):.4f} "
                          f"bound {r['lower_bound'] if r['lower_bound'] is not None else float('nan'):.4f} ({r['wall_s']:.1f} s)", flush=True)
