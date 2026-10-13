"""Independent schedule builder, dispatching rules, random dispatching and feasibility validator
(results/objective_validation.md, Part 2). Pure numpy; does not use the DANIEL environment.

Active-style dispatching: with r_j = completion of job j's last scheduled operation and F_m = machine free time,
c* = min over candidate pairs (j, eligible m) of max(r_j, F_m) + p; the conflict set is the jobs whose next operation
could start before c* on some eligible machine; the JOB rule picks one (ties: lowest job index), the MACHINE rule
picks one of its eligible machines, and the operation starts at max(r_j, F_m).
"""
from __future__ import annotations
import numpy as np

JOB_RULES = ("SPT", "LPT", "MOR", "MWKR", "EDD", "FIFO")
MACHINE_RULES = ("ECT", "SPT", "MINE", "OLDC")


def build(inst, job_rule, machine_rule, energy=None, oldc=None, due=None, rng=None):
    """Returns (machine [N], start [N]) of one complete schedule. job_rule may be 'RANDOM' (then machine is random
    too; rng required)."""
    pt = inst.pt.astype(float)
    J, M, N = inst.n_j, inst.n_m, inst.n_ops
    elig_all = pt > 0
    last = inst.first + inst.job_length - 1
    nxt = inst.first.copy()
    r = np.zeros(J); F = np.zeros(M)
    machine = np.full(N, -1); start = np.zeros(N)
    mean_pt = np.array([row[row > 0].mean() for row in pt])
    min_pt = np.array([row[row > 0].min() for row in pt])
    rem_ops = inst.job_length.astype(float).copy()
    rem_work = np.array([mean_pt[inst.job_of == j].sum() for j in range(J)])
    for _ in range(N):
        act = np.flatnonzero(nxt <= last)
        ops = nxt[act]
        P = pt[ops]; E = elig_all[ops]
        S = np.maximum(r[act, None], F[None, :])
        cstar = np.where(E, S + P, np.inf).min()
        est = np.where(E, S, np.inf).min(1)
        cand = act[est < cstar]
        if job_rule == "RANDOM":
            j = int(rng.choice(cand))
        else:
            key = {"SPT": min_pt[nxt[cand]], "LPT": -min_pt[nxt[cand]], "MOR": -rem_ops[cand],
                   "MWKR": -rem_work[cand], "EDD": None if due is None else due[cand], "FIFO": r[cand]}[job_rule]
            j = int(cand[np.argmin(key)])                      # argmin returns the first = lowest job index
        o = nxt[j]
        ms = np.flatnonzero(elig_all[o])
        s_m = np.maximum(r[j], F[ms]); c_m = s_m + pt[o, ms]
        if job_rule == "RANDOM":
            m = int(rng.choice(ms))
        else:
            if machine_rule == "ECT":
                order = np.lexsort((ms, c_m))
            elif machine_rule == "SPT":
                order = np.lexsort((ms, c_m, pt[o, ms]))
            elif machine_rule == "MINE":
                order = np.lexsort((ms, c_m, energy[o, ms]))
            elif machine_rule == "OLDC":
                order = np.lexsort((ms, c_m, oldc[o, ms]))
            else:
                raise ValueError(machine_rule)
            m = int(ms[order[0]])
        st = max(r[j], F[m]); ct = st + pt[o, m]
        machine[o], start[o] = m, st
        r[j] = ct; F[m] = ct; nxt[j] += 1
        rem_ops[j] -= 1; rem_work[j] -= mean_pt[o]
    return machine, start


def validate(inst, machine, start):
    """Raises AssertionError on any infeasibility."""
    N = inst.n_ops
    assert machine.shape == (N,) and (machine >= 0).all(), "operation unscheduled"
    p = inst.pt[np.arange(N), machine]
    assert (p > 0).all(), "ineligible machine"
    assert (start >= 0).all(), "negative start"
    comp = start + p
    for j in range(inst.n_j):
        o = inst.first[j]
        for q in range(o, o + inst.job_length[j] - 1):
            assert start[q + 1] >= comp[q] - 1e-9, "precedence"
    for m in range(inst.n_m):
        on = np.flatnonzero(machine == m)
        order = on[np.argsort(start[on], kind="stable")]
        assert (start[order[1:]] >= comp[order[:-1]] - 1e-9).all(), "machine overlap"
    return comp


def metrics(inst, machine, start, carbon_new, carbon_old):
    comp = validate(inst, machine, start)
    idx = np.arange(inst.n_ops)
    Cj = comp[inst.first + inst.job_length - 1]
    return dict(mk=float(comp.max()), workload=float(inst.pt[idx, machine].sum()),
                c_new=float(carbon_new[idx, machine].sum()), c_old=float(carbon_old[idx, machine].sum()), Cj=Cj)
