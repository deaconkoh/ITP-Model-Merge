"""Training-free greedy dispatcher, reporting every candidate objective from one schedule.

Extends the Track B pre-check dispatcher (tools/precheck_structured_headroom.py), which scored
only makespan and carbon. The scheduling loop is unchanged; what is new is that a schedule now
also yields per-job completion times (for tardiness) and machine busy times (for load balance),
so the objectives are measured on the SAME schedules rather than on separate runs.

LIMITATION, carried over verbatim from the Track B pre-check: a heuristic dispatcher's response
is not identical to a trained policy's, so this is a screening indicator, not a guarantee. It is
however the same screen that would have caught operation-priority before six GPU-hours were spent
on a near-collinear axis.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np


def parse(path):
    """SD2 carbon+priority format: per operation, machine_count priority (machine pt carbon)*."""
    lines = Path(path).read_text().splitlines()
    h = lines[0].split()
    n_jobs, n_mch = int(h[0]), int(h[1])
    jobs, job_prios, prios = [], [], []
    for line in lines[1:]:
        t = line.strip().split()
        if not t:
            continue
        i, ops = 1, []
        while i < len(t):
            mc = int(t[i]); q = float(t[i + 1]); i += 2
            opts = []
            for _ in range(mc):
                opts.append((int(t[i]) - 1, float(t[i + 1]), float(t[i + 2]))); i += 3
            ops.append(opts); prios.append(q)
        jobs.append(ops); job_prios.append(prios); prios = []
    return n_jobs, n_mch, jobs, job_prios


def instance_stats(jobs):
    pts = [pt for ops in jobs for opts in ops for _, pt, _ in opts]
    cbs = [cb for ops in jobs for opts in ops for _, _, cb in opts]
    return float(np.mean(pts)), float(np.mean(cbs))


def job_work_content(jobs):
    """P_j = sum over the job's operations of the MEAN processing time across eligible machines.

    The mean (not the min) is the neutral choice: it does not presuppose that the scheduler will
    win the routing decision, which is exactly what the scheduler is being asked to do.
    """
    return np.array([sum(float(np.mean([pt for _, pt, _ in opts])) for opts in ops) for ops in jobs])


def dispatch(jobs, n_mch, score_fn, due=None, job_prios=None):
    """Greedy list scheduling. score_fn(j, m, pt, cb, start, ctx) -> float; lowest wins.

    ctx carries the normalisers and live state a rule may need (machine busy times, due dates).
    Ties break on earliest start, then job, then machine -- deterministic, no RNG anywhere.
    """
    n_jobs = len(jobs)
    nxt = [0] * n_jobs
    job_ready = np.zeros(n_jobs)
    mach_free = np.zeros(n_mch)
    mach_busy = np.zeros(n_mch)
    pt_mean, cb_mean = instance_stats(jobs)
    carbon = 0.0
    ctx = {"pt_mean": pt_mean, "cb_mean": cb_mean, "mach_busy": mach_busy,
           "due": due, "due_mean": float(np.mean(due)) if due is not None else None,
           "prio": job_prios, "nxt": nxt}
    op_ct = [[0.0] * len(o) for o in jobs]
    for _ in range(sum(len(o) for o in jobs)):
        best = None
        for j in range(n_jobs):
            if nxt[j] >= len(jobs[j]):
                continue
            for m, pt, cb in jobs[j][nxt[j]]:
                start = max(job_ready[j], mach_free[m])
                key = (score_fn(j, m, pt, cb, start, ctx), start, j, m)
                if best is None or key < best[0]:
                    best = (key, j, m, pt, cb, start)
        _, j, m, pt, cb, start = best
        end = start + pt
        op_ct[j][nxt[j]] = end
        job_ready[j] = end
        mach_free[m] = end
        mach_busy[m] += pt
        nxt[j] += 1
        carbon += cb
    makespan = float(job_ready.max())
    out = {"makespan": makespan, "carbon": float(carbon),
           "job_completion": job_ready.copy(), "op_ct": op_ct,
           # load balance: spread of machine utilisation, 0 = perfectly even.
           "load_imbalance": float(np.std(mach_busy / makespan)) if makespan > 0 else 0.0}
    if due is not None:
        out["tardiness"] = float(np.maximum(0.0, job_ready - due).sum())
        out["tardy_fraction"] = float((job_ready > due + 1e-9).mean())
    return out


# ---- scoring rules -------------------------------------------------------------------------
def time_rule(j, m, pt, cb, start, ctx):
    return pt / ctx["pt_mean"]


def carbon_rule(j, m, pt, cb, start, ctx):
    return cb / ctx["cb_mean"]


def edd_rule(j, m, pt, cb, start, ctx):
    """Earliest due date: the canonical tardiness heuristic."""
    return ctx["due"][j] / ctx["due_mean"]


def slack_rule(j, m, pt, cb, start, ctx):
    """Minimum slack: due date less the work already committed. Tardiness-aware and dynamic."""
    return (ctx["due"][j] - start - pt) / ctx["due_mean"]


def balance_rule(j, m, pt, cb, start, ctx):
    """Least-loaded machine first: the load-balance specialist. The completion term breaks the
    degeneracy of an otherwise load-only score, which would otherwise idle the shop."""
    mb = ctx["mach_busy"]
    return (mb[m] + pt) / (mb.mean() + pt + 1e-9) + 0.001 * (start + pt) / ctx["pt_mean"]


def blend(rule_a, rule_b, lam):
    """lam=0 -> pure rule_a, lam=1 -> pure rule_b."""
    return lambda j, m, pt, cb, start, ctx: ((1 - lam) * rule_a(j, m, pt, cb, start, ctx)
                                             + lam * rule_b(j, m, pt, cb, start, ctx))


def priority_weighted_completion(op_ct, job_prios):
    """The RETIRED objective, sum(q_o C_o)/sum(q_o), measured in this same framework so the
    screen can be validated against the axis that is known to have failed."""
    num = sum(c * q for cs, qs in zip(op_ct, job_prios) for c, q in zip(cs, qs))
    den = sum(q for qs in job_prios for q in qs)
    return float(num / den)


def wspt_rule(j, m, pt, cb, start, ctx):
    """Weighted earliest completion: the priority-weighted-completion specialist.

    Plain weighted-shortest-processing-time ignores machine availability and loses on its own
    objective, so the completion time -- not the processing time -- is what gets weighted.
    """
    return (start + pt) / (ctx["prio"][j][ctx["nxt"][j]] + 1e-9)


def load_pool(repo, size, split="data_train_vali"):
    pool = Path(repo) / "daniel/data" / split / "SD2" / f"{size}+carbon+priority"
    files = sorted(pool.glob("*.fjs"), key=lambda p: int("".join(filter(str.isdigit, p.stem))))
    return [(f, *parse(f)) for f in files]


def ect_rule(j, m, pt, cb, start, ctx):
    """Earliest completion time. A much stronger time heuristic than shortest-processing-time,
    and therefore a fairer stand-in for what a trained policy achieves when calibrating due
    dates: a weak reference makes due dates look tight that a good scheduler would never miss."""
    return (start + pt) / ctx["pt_mean"]


def atc_rule(kappa=2.0):
    """Apparent Tardiness Cost (Vepsalainen & Morton): the standard tardiness dispatching rule.

    index = (1/pt) * exp(-max(0, d_j - pt - t) / (kappa * pt_mean)); higher index dispatches
    first, so the score is its negation. It interpolates between weighted-shortest-processing-
    time when nothing is urgent and minimum-slack when deadlines bite -- which is exactly why a
    bare EDD rule, which ignores processing time entirely, loses on its own objective.
    """
    def rule(j, m, pt, cb, start, ctx):
        slack = max(0.0, ctx["due"][j] - pt - start)
        # (start + pt), not pt: in a FLEXIBLE shop the machine choice is part of the decision,
        # and an index blind to machine availability loses to plain earliest-completion-time.
        return -np.exp(-slack / (kappa * ctx["pt_mean"])) / max(start + pt, 1e-9)
    return rule


def watc_rule(kappa=2.0):
    """ATC's weighted-completion sibling for the RETIRED priority axis: same rule family, so the
    two candidate third objectives are screened with equally strong heuristics rather than one
    being handicapped."""
    def rule(j, m, pt, cb, start, ctx):
        q = ctx["prio"][j][ctx["nxt"][j]]
        return -q / max(start + pt, 1e-9)
    return rule


def due_dates(P, k, spread=0.0, seed=0):
    """TWK due dates with optional per-job tightness dispersion.

    d_j = k_j * P_j,  k_j ~ U[k(1-spread/2), k(1+spread/2)]

    spread = 0 reproduces uniform TWK, under which every job carries identical relative
    tightness -- so no job is more urgent than another and tardiness has no sequencing
    trade-off left to make. Dispersion is how the scheduling literature generates tardiness
    benchmarks (the tardiness-factor / relative-range-of-due-dates scheme); it is the term
    that creates differential urgency. Deterministic given (instance index, k, spread).
    """
    P = np.asarray(P, dtype=float)
    if spread <= 0:
        return k * P
    rng = np.random.default_rng(int(seed))
    kj = rng.uniform(k * (1 - spread / 2), k * (1 + spread / 2), size=P.shape)
    return kj * P
