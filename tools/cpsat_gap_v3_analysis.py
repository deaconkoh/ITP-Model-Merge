"""Analysis stage of tools/cpsat_gap_v3.py (Parts 1-4). Pure numpy on committed CP-SAT results and the schedule pool.

Definitions (fixed before the run)
  ready time r_o      completion of the job predecessor of o in the schedule itself (0 for a job's first operation)
  idle period         on machine m: before its first operation, between consecutive operations, and from its last
                      operation to the makespan (a machine with no operation is idle on [0, makespan))
  deliberate waiting  the part of an idle period on m during which at least one operation o that is eligible on m is
                      ready (r_o <= t) and not yet started (t < s_o); whether o is later processed on m ("same
                      machine") or on another machine is recorded. The rest of the idle period is unavoidable waiting.
                      A schedule without deliberate waiting is exactly a schedule DANIEL's mask can produce
                      (non-delay); this is also checked by replaying it through DANIEL's own decision rule.
  left shift          same machine per operation and same order on every machine, every start as early as possible
                      (semi-active schedule)
  non-delay, fixed assignment
                      same machine per operation; whenever the operations that can start earliest on their assigned
                      machines tie, the one CP-SAT started first goes first; never waits for an operation assigned to
                      that machine (it may still wait while an operation assigned elsewhere is ready)
  through DANIEL's mask
                      CP-SAT's schedule decoded by DANIEL's own decision rule with keys from its machines and start
                      order (tools/tardiness_diagnosis_v3.keys_from_schedule): the nearest DANIEL-reachable schedule in
                      that sense, not the best one
  score               preference-weighted normalised benchmark score w_m MK/MKref + w_c C/Cref + w_t T/Tref (lower is
                      better), the selection score of best-of-16
  CP-SAT's schedule for preference w
                      the front schedule (B2 front, 20 instances) with the lowest score at w
  best known non-delay schedule for w
                      the pool schedule (tools/cpsat_gap_v3.py pool) with the lowest score at w
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np

import benchmark_v3 as B
import mo_metrics as MM
import tardiness_diagnosis_v3 as TD
C = sys.modules["__main__"]        # tools/cpsat_gap_v3.py (its constants and paths)

EPS = 1e-6


class Inst:
    def __init__(self, data, i):
        self.i = i; self.key = data.keys[i]
        self.jl = np.asarray(data.job_length[i]); self.pt = np.asarray(data.op_pt[i], dtype=float)
        self.carbon = np.asarray(data.carbon[i]); self.due = np.asarray(data.due[i], dtype=float)
        self.n, self.m = self.pt.shape
        self.first = np.concatenate([[0], np.cumsum(self.jl)[:-1]]).astype(int)
        self.last = self.first + self.jl - 1
        self.job = np.repeat(np.arange(len(self.jl)), self.jl)
        self.pred = np.arange(self.n) - 1
        self.pred[self.first] = -1
        self.elig = self.pt > 0
        self.mk_ref, self.cb_ref, self.t_ref = data.mk_ref[i], data.cb_ref[i], data.t_ref

    def objectives(self, machine, start):
        r = B.schedule_objectives(self.jl, self.pt, self.carbon, self.due, machine, start)
        return np.array([r["makespan"], r["carbon"], r["tardiness"]])

    def score(self, obj, w):
        obj = np.asarray(obj); w = np.asarray(w)
        return w[..., 0] * obj[..., 0] / self.mk_ref + w[..., 1] * obj[..., 1] / self.cb_ref + w[..., 2] * obj[..., 2] / self.t_ref


def union_len(iv):
    if not iv:
        return 0.0
    iv = sorted(iv)
    tot, (a, b) = 0.0, iv[0]
    for x, y in iv[1:]:
        if x > b:
            tot += b - a; a, b = x, y
        else:
            b = max(b, y)
    return tot + b - a


def idle_analysis(inst, machine, start, detail=False):
    """Idle periods of every machine split into deliberate and unavoidable waiting (definitions above)."""
    machine = np.asarray(machine, int); start = np.asarray(start, float)
    p = inst.pt[np.arange(inst.n), machine]
    comp = start + p
    mk = comp.max()
    r = np.where(inst.pred >= 0, comp[np.maximum(inst.pred, 0)], 0.0)
    out = {"idle": 0.0, "deliberate": 0.0, "deliberate_same": 0.0, "deliberate_other_only": 0.0, "gaps": 0,
           "internal_idle": 0.0, "internal_deliberate": 0.0, "internal_gaps": 0, "internal_deliberate_gaps": 0,
           "deliberate_gaps": 0, "fit_gaps": 0, "waits_for_next": 0, "durations": [], "periods": []}
    for m in range(inst.m):
        ops = np.flatnonzero(machine == m)
        ops = ops[np.argsort(start[ops], kind="stable")]
        gaps, prev = [], 0.0
        for o in ops:
            if start[o] > prev + EPS:
                gaps.append((prev, start[o], int(o)))
            prev = comp[o]
        if mk > prev + EPS:
            gaps.append((prev, mk, None))
        for a, b, nxt in gaps:
            out["idle"] += b - a; out["gaps"] += 1
            same, allv, fit, who = [], [], False, []
            for o in np.flatnonzero(inst.elig[:, m]):
                lo, hi = max(r[o], a), min(start[o], b)
                if hi > lo + EPS:
                    allv.append((lo, hi)); who.append(int(o))
                    if machine[o] == m:
                        same.append((lo, hi))
                    if lo + inst.pt[o, m] <= b + EPS:
                        fit = True
            d, ds = union_len(allv), union_len(same)
            if nxt is not None:                                  # before the machine's last operation
                out["internal_idle"] += b - a; out["internal_deliberate"] += d; out["internal_gaps"] += 1
                out["internal_deliberate_gaps"] += int(d > EPS)
            out["deliberate"] += d; out["deliberate_same"] += ds; out["deliberate_other_only"] += d - ds
            if d > EPS:
                out["deliberate_gaps"] += 1; out["durations"].append(d)
                out["fit_gaps"] += int(fit)
                if nxt is not None and abs(r[nxt] - b) <= EPS:
                    out["waits_for_next"] += 1           # the machine's next operation starts the moment it is ready
            if detail:
                out["periods"].append({"machine": m, "from": a, "to": b, "deliberate": d, "next_op": nxt,
                                       "ready_ops": who})
    return out


def is_nondelay(inst, machine, start):
    return idle_analysis(inst, machine, start)["deliberate"] <= EPS


def left_shift(inst, machine, start):
    machine = np.asarray(machine, int)
    order = np.lexsort((np.arange(inst.n), np.asarray(start, float)))
    s2, c2, free = np.zeros(inst.n), np.zeros(inst.n), np.zeros(inst.m)
    for o in order:
        jr = c2[inst.pred[o]] if inst.pred[o] >= 0 else 0.0
        s2[o] = max(jr, free[machine[o]]); c2[o] = s2[o] + inst.pt[o, machine[o]]; free[machine[o]] = c2[o]
    return s2


def nondelay_fixed(inst, machine, prio):
    """Non-delay dispatching with every operation's machine fixed; ties at the earliest start by prio."""
    machine = np.asarray(machine, int); prio = np.asarray(prio, float)
    nxt = inst.first.copy(); jr = np.zeros(len(inst.jl)); free = np.zeros(inst.m); s = np.zeros(inst.n)
    for _ in range(inst.n):
        js = np.flatnonzero(nxt <= inst.last)
        ops = nxt[js]
        es = np.maximum(jr[js], free[machine[ops]])
        tie = es <= es.min() + EPS
        k = np.flatnonzero(tie)[np.lexsort((ops[tie], prio[ops[tie]]))[0]]
        o, j = ops[k], js[k]
        s[o] = es[k]; c = s[o] + inst.pt[o, machine[o]]; jr[j] = c; free[machine[o]] = c; nxt[j] += 1
    return s


def through_mask(data, idx, machines, starts):
    """CP-SAT schedules decoded by DANIEL's decision rule (keys from machine and start order); replayed through the
    real environment and re-scored (TD.check)."""
    d = TD.NonDelay(data, idx)
    res = d.run(TD.key_policy(TD.keys_from_schedule(np.asarray(machines), np.asarray(starts, float), d.M)))
    TD.check(data, idx, res, "through the mask")
    return res


def env_reproduces(data, idx, machines, starts):
    """True where DANIEL's decision rule, led by the schedule's own keys, reproduces it exactly."""
    if len(idx) == 0:
        return np.zeros(0, bool)
    res = through_mask(data, idx, machines, starts)
    return (np.all(res["machine"] == np.asarray(machines), 1)
            & np.all(np.abs(res["start"] - np.asarray(starts, float)) < EPS, 1))


# ------------------------------------------------------------------------------------------------ inputs

def cpsat_front(inst):
    """Schedules of the B2 front points (machine, start, source, status) with re-scored objectives."""
    stem = Path(inst.key).stem
    b2 = json.loads((C.B2 / f"{stem}.json").read_text()); b1 = json.loads((C.B1 / f"{stem}.json").read_text())
    assert b2["key"] == inst.key == b1["key"]
    pts = {f"b2 q=({p['q_carbon']:.2f},{p['q_tard']:.2f})": p for p in b2["points"] if "machine" in p}
    out = []
    for f in b2["front"]:
        src = f["source"]
        if src.startswith("b1 "):
            k = src[3:]; rec = b1["points"][k]["final"]; status = b1["points"][k]["primary"]["status"]
            kind = {"mk": "B1 makespan", "tard": "B1 tardiness", "carbon": "B1 carbon"}[k]
        else:
            rec = pts[src]; status = rec["status"]; kind = "B2 epsilon-constraint"
        m, s = np.asarray(rec["machine"], int), np.asarray(rec["start"], float)
        obj = inst.objectives(m, s)
        assert np.allclose(obj, [f["makespan"], f["carbon"], f["tardiness"]], atol=1e-9), (inst.key, src)
        out.append({"source": src, "kind": kind, "status": status, "machine": m, "start": s, "obj": obj})
    return out


def load_pool(I, log):
    """Pool schedules per instance: machine [S, N], start [S, N], source names [S]."""
    pool = [{"machine": [], "start": [], "src": []} for _ in range(I)]
    fx = np.load(C.POOL / "fixed.npz")
    for k in fx.files:
        if not k.endswith("__machine"):
            continue
        name = k[: -len("__machine")]
        m, s = fx[k], fx[name + "__start"]
        for i in range(I):
            pool[i]["machine"].append(m[i][None]); pool[i]["start"].append(s[i][None]); pool[i]["src"].append(name)
    mo = json.loads((C.POOL / "mo_sources.json").read_text())
    repro = {}
    for f in mo["found"]:
        z = np.load(C.POOL / f"mo__{f['name']}.npz")
        repro[f["name"]] = {"greedy": [int(z["greedy_reproduced"].sum()), int(z["greedy_reproduced"].size)]}
        for mode in ("greedy", "s16"):
            if f"{mode}_machine" not in z.files:
                continue
            if mode == "s16":
                repro[f["name"]]["s16"] = [int(z["s16_reproduced"].sum()), int(z["s16_reproduced"].size)]
            for i in range(I):
                pool[i]["machine"].append(z[f"{mode}_machine"][i]); pool[i]["start"].append(z[f"{mode}_start"][i])
                pool[i]["src"] += [f"{f['name']} {mode}"] * z[f"{mode}_machine"].shape[1]
    for i in range(I):
        pool[i]["machine"] = np.concatenate(pool[i]["machine"]).astype(int)
        pool[i]["start"] = np.concatenate(pool[i]["start"]).astype(float)
        pool[i]["src"] = np.asarray(pool[i]["src"])
    return pool, mo, repro


def pref_group(w):
    k = int(np.argmax(w))
    return ["makespan-heavy", "carbon-heavy", "tardiness-heavy"][k] if w[k] >= 0.8 - 1e-9 else "mixed"


GROUPS = ["makespan-heavy", "carbon-heavy", "tardiness-heavy", "mixed"]


# ------------------------------------------------------------------------------------------------ Part 4 helpers

def trace_divergence(inst, tr, w_idx, mD, sD, mT, sT):
    """First step at which DANIEL's greedy trajectory leaves the target schedule, and the target's action there.
    tr: trace arrays of this (instance, preference): action [N], prob [N, J*M], allowed [N, J*M]."""
    N, M = inst.n, inst.m
    J = len(inst.jl)
    nxt = inst.first.copy()
    free = np.zeros(M); jr = np.zeros(J)
    for k in range(N):
        a = int(tr["action"][k]); j, m = a // M, a % M
        o = nxt[j]
        t = sD[o]
        if not (mT[o] == m and abs(sT[o] - t) < EPS):
            allowed = tr["allowed"][k]; prob = tr["prob"][k].astype(float)
            cons = []
            for aa in np.flatnonzero(allowed):
                jj, mm = aa // M, aa % M
                oo = nxt[jj]
                if mT[oo] == mm and abs(sT[oo] - t) < EPS:
                    cons.append(aa)
            order = np.argsort(-np.where(allowed, prob, -1.0), kind="stable")
            rank = {int(x): r + 1 for r, x in enumerate(order[: int(allowed.sum())])}
            best = min(cons, key=lambda x: rank[int(x)]) if cons else None
            # state at the divergence step (raw times)
            unfinished = np.flatnonzero(nxt <= inst.last)
            rem_min = np.array([sum(inst.pt[q][inst.elig[q]].min() for q in range(nxt[jj], inst.last[jj] + 1))
                                for jj in unfinished])
            slack = inst.due[unfinished] - t - rem_min
            p_allowed = prob[allowed]
            ent = float(-(p_allowed[p_allowed > 0] * np.log(p_allowed[p_allowed > 0])).sum())
            return {"step": k, "phase": phase(k, N), "kind": "assignment" if mT[o] != m else "sequencing",
                    "daniel_prob": float(prob[a]), "target_rank": rank[int(best)] if best is not None else None,
                    "target_prob": float(prob[best]) if best is not None else None,
                    "target_prob_all": float(prob[cons].sum()) if cons else 0.0, "n_target_actions": len(cons),
                    "n_allowed": int(allowed.sum()), "n_ready_jobs": int(len({x // M for x in np.flatnonzero(allowed)})),
                    "busy_machines": int((free > t + EPS).sum()), "unfinished_jobs": int(len(unfinished)),
                    "behind_jobs": int((slack < 0).sum()), "min_slack": float(slack.min()), "entropy": ent,
                    "top_prob": float(p_allowed.max())}
        free[m] = sD[o] + inst.pt[o, m]; jr[j] = free[m]; nxt[j] += 1
    return None


def closest_better(inst, tr, sD, pm, ps, better):
    """Among the pool schedules better than DANIEL's (mask `better`), the one DANIEL's greedy path follows longest:
    (index, number of steps followed)."""
    M = inst.m
    nxt = inst.first.copy(); ops = np.zeros(inst.n, int)
    for k in range(inst.n):
        j = int(tr["action"][k]) // M
        ops[k] = nxt[j]; nxt[j] += 1
    mk = tr["action"] % M
    mism = (pm[:, ops] != mk[None]) | (np.abs(ps[:, ops] - sD[ops][None]) > EPS)
    first = np.where(mism.any(1), mism.argmax(1), inst.n)
    cand = np.flatnonzero(better)
    k = cand[np.argmax(first[cand])]
    return int(k), int(first[k])


def phase(k, N):
    return "early" if k < N / 3 else ("middle" if k < 2 * N / 3 else "late")


def step_states(inst, tr, mD, sD):
    """State descriptors at every step of a DANIEL greedy trajectory (base rates for Part 4)."""
    N, M = inst.n, inst.m
    nxt = inst.first.copy(); free = np.zeros(M)
    rows = []
    for k in range(N):
        a = int(tr["action"][k]); j, m = a // M, a % M
        o = nxt[j]; t = sD[o]
        allowed = tr["allowed"][k]
        unfinished = np.flatnonzero(nxt <= inst.last)
        rem_min = np.array([sum(inst.pt[q][inst.elig[q]].min() for q in range(nxt[jj], inst.last[jj] + 1))
                            for jj in unfinished])
        slack = inst.due[unfinished] - t - rem_min
        p = tr["prob"][k].astype(float)[allowed]
        rows.append({"step": k, "phase": phase(k, N), "n_allowed": int(allowed.sum()),
                     "n_ready_jobs": int(len({x // M for x in np.flatnonzero(allowed)})),
                     "busy_machines": int((free > t + EPS).sum()), "behind_jobs": int((slack < 0).sum()),
                     "top_prob": float(p.max())})
        free[m] = sD[o] + inst.pt[o, m]; nxt[j] += 1
    return rows


# ------------------------------------------------------------------------------------------------ the run

def run(args, log):
    t0 = time.time()
    data = B.load_split(C.SPLIT, C.SIZE)
    I = C.N_INST
    insts = [Inst(data, i) for i in range(I)]
    W = C.PREFS[:C.G]
    norm = json.loads((C.REPO / "results" / "baselines_v3" / "normalisation_points.json").read_text())["instances"]
    R = {"n_instances": I, "keys": [x.key for x in insts], "preferences": W.tolist(),
         "groups": [pref_group(w) for w in W]}

    # ---------------- Part 1: CP-SAT facts that are read from results (the code facts are in the report)
    fronts = [cpsat_front(x) for x in insts]
    b2_status, front_kind, lim = {}, {}, set()
    for x in insts:
        b2 = json.loads((C.B2 / f"{Path(x.key).stem}.json").read_text())
        lim.add((b2["limits"]["primary"], b2["grid"], b2["workers"]))
        for p in b2["points"]:
            b2_status[p["status"]] = b2_status.get(p["status"], 0) + 1
    b1_status = {}
    for x in insts:
        b1 = json.loads((C.B1 / f"{Path(x.key).stem}.json").read_text())
        for k, v in b1["points"].items():
            key = f"{k} primary {v['primary']['status']}"
            b1_status[key] = b1_status.get(key, 0) + 1
            for s in v.get("lex_stages", []):
                key = f"{k} stage {s['status']}"
                b1_status[key] = b1_status.get(key, 0) + 1
    for fr in fronts:
        for f in fr:
            front_kind[f["kind"]] = front_kind.get(f["kind"], 0) + 1
    b1_limits = json.loads((C.B1 / f"{Path(insts[0].key).stem}.json").read_text())["limits"]
    R["part1"] = {"b2_solve_status": b2_status, "b2_settings": sorted(lim), "b1_status": b1_status,
                  "b1_limits": b1_limits, "front_points": int(sum(len(f) for f in fronts)), "front_kind": front_kind,
                  "front_sizes": [len(f) for f in fronts], "front_rescored": "all front objectives equal the "
                  "independent scorer's re-score of the stored schedules"}
    log(f"part 1: {R['part1']['front_points']} front schedules")

    # ---------------- the pool, checked
    pool, mo, repro = load_pool(I, log)
    R["pool"] = {"mo_found": [f["name"] for f in mo["found"]], "mo_missing": [f["name"] for f in mo["missing_checkpoint"]],
                 "reproduced": repro, "fixed": json.loads((C.POOL / "fixed.json").read_text()),
                 "schedules_per_instance": [int(len(p["src"])) for p in pool]}
    pool_obj = []
    nd_fail = 0
    for x, p in zip(insts, pool):
        u, ui = np.unique(np.concatenate([p["machine"], np.rint(p["start"] * 1e6).astype(np.int64)], 1),
                          axis=0, return_index=True)
        o = np.stack([x.objectives(p["machine"][s], p["start"][s]) for s in range(len(p["src"]))])
        pool_obj.append(o)
        for s in ui:
            nd_fail += int(not is_nondelay(x, p["machine"][s], p["start"][s]))
        p["distinct"] = int(len(ui))
    R["pool"]["distinct_per_instance"] = [p["distinct"] for p in pool]
    R["pool"]["nondelay_violations"] = nd_fail
    assert nd_fail == 0, "a pool schedule is not non-delay"
    log(f"pool: {np.mean(R['pool']['distinct_per_instance']):.0f} distinct schedules per instance, all non-delay")

    # ---------------- per-preference CP-SAT choice and best non-delay
    choice, best = np.zeros((I, C.G), int), np.zeros((I, C.G), int)
    for i, x in enumerate(insts):
        fo = np.stack([f["obj"] for f in fronts[i]])
        for g, w in enumerate(W):
            choice[i, g] = int(np.argmin(x.score(fo, w)))
            best[i, g] = int(np.argmin(x.score(pool_obj[i], w)))

    # ---------------- Parts 2 and 3 on every distinct front schedule
    per_sched = []
    ls_m, ls_s, ls_i = [], [], []
    for i, x in enumerate(insts):
        for f in fronts[i]:
            ia = idle_analysis(x, f["machine"], f["start"])
            s_ls = left_shift(x, f["machine"], f["start"])
            o_ls = x.objectives(f["machine"], s_ls)
            ib = idle_analysis(x, f["machine"], s_ls)
            s_nd = nondelay_fixed(x, f["machine"], f["start"])
            o_nd = x.objectives(f["machine"], s_nd)
            ic = idle_analysis(x, f["machine"], s_nd)
            per_sched.append({"i": i, "source": f["source"], "kind": f["kind"], "status": f["status"], "obj": f["obj"],
                              "idle": ia, "ls_obj": o_ls, "ls_idle": ib, "nd_obj": o_nd, "nd_idle": ic,
                              "ls_start": s_ls, "nd_start": s_nd, "machine": f["machine"], "start": f["start"],
                              "late_shift": float((f["start"] - s_ls).sum())})
            ls_m.append(f["machine"]); ls_s.append(s_ls); ls_i.append(i)
    # DANIEL-rule checks: left-shifted and fixed-assignment schedules that pass the non-delay test are replayed
    for key, st in (("ls", "ls_start"), ("nd", "nd_start")):
        cand = [k for k, r in enumerate(per_sched) if r[f"{key}_idle"]["deliberate"] <= EPS]
        ok = env_reproduces(data, np.asarray([per_sched[k]["i"] for k in cand], int),
                            np.stack([per_sched[k]["machine"] for k in cand]) if cand else np.zeros((0, 50), int),
                            np.stack([per_sched[k][st] for k in cand]) if cand else np.zeros((0, 50)))
        for k in range(len(per_sched)):
            per_sched[k][f"{key}_daniel"] = False
        for k, v in zip(cand, ok):
            per_sched[k][f"{key}_daniel"] = bool(v)
        log(f"{key}: {len(cand)} pass the non-delay test, {int(ok.sum())} reproduced by DANIEL's rule")
    # CP-SAT's schedules through DANIEL's mask
    res = through_mask(data, np.asarray([r["i"] for r in per_sched]), np.stack([r["machine"] for r in per_sched]),
                       np.stack([r["start"] for r in per_sched]))
    for k, r in enumerate(per_sched):
        x = insts[r["i"]]
        r["env_obj"] = x.objectives(res["machine"][k], res["start"][k])
        r["env_machine"], r["env_start"] = res["machine"][k], res["start"][k]
        r["env_same_machine"] = float((res["machine"][k] == r["machine"]).mean())
        r["env_is_cpsat"] = bool(np.all(res["machine"][k] == r["machine"]) and np.allclose(res["start"][k], r["start"]))
    offs = np.cumsum([0] + [len(f) for f in fronts])

    def S(i, g):
        return per_sched[offs[i] + choice[i, g]]

    # ---------------- per (instance, preference)
    traces = {}
    for key, name in C.TRACE.items():
        z = np.load(C.POOL / f"mo__{name}.npz")
        traces[key] = {"obj": z["greedy_obj"], "machine": z["greedy_machine"], "start": z["greedy_start"],
                       "action": z["trace_action"], "prob": z["trace_prob"], "allowed": z["trace_allowed"]}
    rows = []
    for i, x in enumerate(insts):
        for g, w in enumerate(W):
            r = S(i, g)
            b = best[i, g]
            bo = pool_obj[i][b]
            row = {"i": i, "g": g, "group": pref_group(w), "cpsat_source": r["source"], "best_source": str(pool[i]["src"][b]),
                   "cpsat": x.score(r["obj"], w), "ls": x.score(r["ls_obj"], w), "nd": x.score(r["nd_obj"], w),
                   "env": x.score(r["env_obj"], w), "best": x.score(bo, w),
                   "cpsat_obj": r["obj"].tolist(), "ls_obj": r["ls_obj"].tolist(), "nd_obj": r["nd_obj"].tolist(),
                   "env_obj": r["env_obj"].tolist(), "best_obj": bo.tolist(),
                   "cpsat_deliberate": r["idle"]["deliberate"], "ls_deliberate": r["ls_idle"]["deliberate"],
                   "same_machine_best": float((pool[i]["machine"][b] == r["machine"]).mean())}
            for key, tr in traces.items():
                row[f"{key}"] = x.score(tr["obj"][i, g], w)
                row[f"{key}_obj"] = tr["obj"][i, g].tolist()
            rows.append(row)
    R["rows"] = rows

    # ---------------- HV of whole sets (same normalisation and reference point as the evaluations)
    hv = {k: [] for k in ("cpsat", "ls", "nd", "env", "pool", "wm_final", "fs_final")}
    for i, x in enumerate(insts):
        ideal, nadir = MM.ideal_nadir(norm[x.key])
        sel = per_sched[offs[i]:offs[i + 1]]
        hv["cpsat"].append(MM.instance_hv(np.stack([r["obj"] for r in sel]), ideal, nadir))
        hv["ls"].append(MM.instance_hv(np.stack([r["ls_obj"] for r in sel]), ideal, nadir))
        hv["nd"].append(MM.instance_hv(np.stack([r["nd_obj"] for r in sel]), ideal, nadir))
        hv["env"].append(MM.instance_hv(np.stack([r["env_obj"] for r in sel]), ideal, nadir))
        po = pool_obj[i]
        hv["pool"].append(MM.instance_hv(po[MM.nondominated(po)], ideal, nadir))
        for key, tr in traces.items():
            hv[key].append(MM.instance_hv(tr["obj"][i, :C.G], ideal, nadir))
    R["hv"] = {k: float(np.mean(v)) for k, v in hv.items()}
    log("hv: " + ", ".join(f"{k} {v:.3f}" for k, v in R["hv"].items()))

    # ---------------- per distinct front schedule (Parts 2 and 3)
    R["schedules"] = [{k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in r.items()
                       if k not in ("machine", "start", "ls_start", "nd_start", "env_machine", "env_start")}
                      | {"idle": {k: v for k, v in r["idle"].items() if k != "periods"},
                         "ls_idle": {k: v for k, v in r["ls_idle"].items() if k != "periods"},
                         "nd_idle": {k: v for k, v in r["nd_idle"].items() if k != "periods"}}
                      for r in per_sched]

    # ---------------- examples (instance 0: CP-SAT's schedules for the three corners and equal weights)
    ex = {}
    for lab, w in (("makespan", (1, 0, 0)), ("carbon", (0, 1, 0)), ("tardiness", (0, 0, 1)), ("equal", (1 / 3, 1 / 3, 1 / 3))):
        x = insts[0]
        fo = np.stack([f["obj"] for f in fronts[0]])
        k = int(np.argmin(x.score(fo, np.asarray(w))))
        r = per_sched[offs[0] + k]
        comp = r["start"] + x.pt[np.arange(x.n), r["machine"]]
        seq = {m: [int(o) for o in np.flatnonzero(r["machine"] == m)[np.argsort(r["start"][r["machine"] == m])]]
               for m in range(x.m)}
        ex[lab] = {"source": r["source"], "objectives": r["obj"].tolist(), "machine": r["machine"].tolist(),
                   "start": r["start"].tolist(), "completion": comp.tolist(), "machine_sequence": seq,
                   "job_completion": comp[x.last].tolist(), "due": x.due.tolist(),
                   "idle": idle_analysis(x, r["machine"], r["start"], detail=True),
                   "left_shift_start": r["ls_start"].tolist(), "left_shift_objectives": r["ls_obj"].tolist(),
                   "left_shift_idle": idle_analysis(x, r["machine"], r["ls_start"], detail=True),
                   "job_of_op": x.job.tolist(), "pt": x.pt.tolist()}
    R["examples"] = ex

    # ---------------- Part 4: DANIEL greedy against the best known non-delay schedule
    part4 = {}
    for key, tr in traces.items():
        divs, base = [], []
        for i, x in enumerate(insts):
            for g, w in enumerate(W):
                b = best[i, g]
                sd = x.score(tr["obj"][i, g], w); sb = x.score(pool_obj[i][b], w)
                trr = {k: tr[k][i, g] for k in ("action", "prob", "allowed")}
                st = step_states(x, trr, tr["machine"][i, g], tr["start"][i, g])
                if sd <= sb + 1e-12:
                    base += [dict(s, at_risk=True, diverged=False) for s in st]
                    divs.append({"i": i, "g": g, "group": pref_group(w), "gap": 0.0, "div": None, "closest": None,
                                 "n_better": 0})
                    continue
                dv = trace_divergence(x, trr, g, tr["machine"][i, g], tr["start"][i, g],
                                      pool[i]["machine"][b], pool[i]["start"][b])
                better = x.score(pool_obj[i], w) < sd - 1e-12
                cb, _ = closest_better(x, trr, tr["start"][i, g], pool[i]["machine"], pool[i]["start"], better)
                dc = trace_divergence(x, trr, g, tr["machine"][i, g], tr["start"][i, g],
                                      pool[i]["machine"][cb], pool[i]["start"][cb])
                divs.append({"i": i, "g": g, "group": pref_group(w), "gap": float(sd - sb), "div": dv,
                             "target_source": str(pool[i]["src"][b]), "closest": dc,
                             "closest_gap": float(sd - x.score(pool_obj[i][cb], w)), "n_better": int(better.sum())})
                k = dv["step"] if dv else x.n
                base += [dict(s, at_risk=s["step"] <= k, diverged=s["step"] == k) for s in st]
        part4[key] = {"cases": divs, "steps": base}
        nd = sum(d["div"] is not None for d in divs)
        log(f"part 4 {key}: {nd} of {len(divs)} (instance, preference) pairs diverge from the best known schedule")
    R["part4"] = part4
    R["seconds"] = time.time() - t0
    R["inputs_sha256"] = {"manifest": B.file_sha256(B.MANIFEST),
                          "pool": {f.name: C.sha(f) for f in sorted(C.POOL.glob("*.npz"))}}
    import gzip
    with gzip.open(C.OUT / "analysis.json.gz", "wt") as fh:
        fh.write(json.dumps(R, default=_np) + "\n")
    log(f"analysis written ({R['seconds']:.0f} s)")


def _np(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    raise TypeError(type(o))
