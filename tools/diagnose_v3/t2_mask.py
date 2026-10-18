"""Test 2: DANIEL's action mask (hypotheses A1, B5).

Mask (daniel/fjsp_env_same_op_nums.py): a pair (job j's next operation, machine m) is allowed only if m is eligible
(candidate_process_relation, l.268 / l.551-554) AND its earliest start max(r_j, F_m) equals the earliest start over
all eligible candidate pairs, next_schedule_time (l.558-566, l.613-617); daniel/model/main_model.py:174 sets masked
logits to -inf. So every decision starts an operation at the earliest possible time: machines cannot be held idle.

On 20 instances per size (validation, topped up from development) this script reports, along the trajectory of the
greedy "lowest-energy ALLOWED pair" rule:
  * share of steps where some unfinished job's next operation has its lowest-energy eligible machine masked,
  * share of steps where no allowed pair puts any operation on its lowest-energy machine,
  * share of operations that end on a machine other than their lowest-energy one,
  * the carbon reached, against the unrestricted minimum-energy assignment.
For B5 it adds EDD inside the mask, and a replay of unrestricted target schedules through the mask (CP-SAT minimum
tardiness and minimum makespan, 20 s / 10 s; the EDD+MINE carbon-optimal schedule): at each step the allowed target
pair with the earliest target start is taken; when no target pair is allowed the step is a deviation (allowed pair
on the target machine if any, else earliest completion). Zero deviations proves the target is reachable; deviations
are evidence (not proof) that it is not. Exact test: every schedule the environment can produce is NON-DELAY (no
machine is idle at a time t while an operation eligible on it is ready, r_o <= t < s_o): if machine m' were idle at
such a t, the pair (o, m') would have earliest start <= t while o is still unscheduled, so next_schedule_time <= t and
no operation starting after t (o itself, or anything that would fill m' later) could be dispatched before o or m' is
used. A target with a non-delay violation is therefore unreachable. The check is also run on the environment's own
schedules (expected: zero violations).

    python tools/diagnose_v3/t2_mask.py
"""
import sys, time
import numpy as np
import dv3
from dv3 import B, S

sys.path.insert(0, str(dv3.REPO / "tools"))
import cpsat_baselines_v3 as CP                             # noqa: E402

N_INST = 20


def instances(size):
    """20 instances: the validation split, topped up from the development split. Returns (data, idx, split names)."""
    val, dev = B.load_split("validation", size), B.load_split("development", size)
    data = B.BenchmarkData("validation+development", size, t_ref=val.t_ref)
    names = []
    for src, tag in ((val, "validation"), (dev, "development")):
        for i in range(len(src)):
            if len(data) == N_INST:
                break
            for f in ("keys", "job_length", "op_pt", "power", "carbon", "due", "mk_ref", "cb_ref"):
                getattr(data, f).append(getattr(src, f)[i])
            names.append(tag)
    return data, list(range(len(data))), names


def pair_view(env):
    """Per env: carbon [E,J,M], completion (normalised) [E,J,M], allowed [E,J,M], eligible [E,J,M], live jobs [E,J]."""
    e = env.env_idxs[:, None]
    cand = env.candidate
    carbon = env.op_carbon[e, cand]                                             # [E,J,M]
    start = np.maximum(env.candidate_free_time[:, :, None], env.mch_free_time[:, None, :])
    comp = start + env.candidate_pt
    allowed = ~env.dynamic_pair_mask.astype(bool)
    eligible = env.process_relation[e, cand].astype(bool)
    live = ~env.mask.astype(bool)
    return carbon, comp, allowed, eligible, live


def pick(env, key):
    """argmin of key over allowed pairs (ties: lowest flat index)."""
    _, _, allowed, _, _ = pair_view(env)
    k = np.where(allowed, key, np.inf).reshape(env.number_of_envs, -1)
    return k.argmin(1)


def run(env, policy, stats=False):
    st = {"steps": 0, "some_min_masked": 0, "no_min_allowed": 0}
    n_m = env.number_of_machines
    machine = np.full((env.number_of_envs, env.number_of_ops), -1)
    while True:
        carbon, comp, allowed, eligible, live = pair_view(env)
        if stats:
            cmin = np.where(eligible, carbon, np.inf).min(2, keepdims=True)
            is_min = eligible & (carbon <= cmin + 1e-12)                       # lowest-energy machine(s) of each op
            min_masked = (is_min & ~allowed).any(2) & ~(is_min & allowed).any(2) & live
            st["steps"] += env.number_of_envs
            st["some_min_masked"] += int(min_masked.any(1).sum())
            st["no_min_allowed"] += int((~(is_min & allowed).any((1, 2))).sum())
        a = policy(env)
        op = env.candidate[env.env_idxs, a // n_m]
        machine[env.env_idxs, op] = a % n_m
        _, _, done = env.step(a)
        if np.all(done):
            break
    return machine, st


def min_energy_allowed(env):
    carbon, comp, _, _, _ = pair_view(env)
    return pick(env, carbon + 1e-6 * comp)


def edd_allowed(env):
    _, comp, _, _, _ = pair_view(env)
    return pick(env, env.due_dates[:, :, None] * 1e3 + comp)


def replayer(targets):
    """targets: list of (machine [N], start [N]) per env. Returns a policy and its deviation counter."""
    tm = np.array([t[0] for t in targets]); ts = np.array([t[1] for t in targets], dtype=float)
    dev = np.zeros(len(targets), dtype=int)

    def policy(env):
        _, comp, allowed, _, _ = pair_view(env)
        e = env.env_idxs[:, None]
        cand = env.candidate
        tmach = tm[e, cand]                                                     # [E,J]
        tstart = ts[e, cand]
        on_target = np.zeros_like(allowed)
        on_target[e, np.arange(env.number_of_jobs)[None, :], tmach] = True
        ok = (allowed & on_target).any((1, 2))
        dev[:] += ~ok
        key = np.where(on_target, 0.0, 1e9) + tstart[:, :, None] * 1e3 + comp
        return pick(env, key)
    return policy, dev


def nondelay_violations(job_length, op_pt, machine, start):
    """Number of operations o that wait (s_o > r_o) while some machine eligible for o is idle somewhere in [r_o, s_o)."""
    op_pt = np.asarray(op_pt, dtype=float); machine = np.asarray(machine); start = np.asarray(start, dtype=float)
    n = len(machine)
    end = start + op_pt[np.arange(n), machine]
    first = np.concatenate([[0], np.cumsum(job_length)[:-1]]).astype(int)
    ready = np.zeros(n)
    for f, l in zip(first, first + np.asarray(job_length) - 1):
        ready[f + 1:l + 1] = end[f:l]
    busy = {m: sorted(zip(start[machine == m], end[machine == m])) for m in range(op_pt.shape[1])}
    bad = 0
    for o in range(n):
        a, b = ready[o], start[o]
        if b <= a + 1e-9:
            continue
        for m in np.flatnonzero(op_pt[o] > 0):
            t = a                                                               # sweep: is [a, b) fully covered?
            for s0, e0 in busy[m]:
                if e0 <= t + 1e-9:
                    continue
                if s0 > t + 1e-9:
                    break
                t = e0
                if t >= b - 1e-9:
                    break
            if t < b - 1e-9:
                bad += 1
                break
    return bad


def realised(env, machine):
    pt = env.true_op_pt[env.env_idxs[:, None], np.arange(env.number_of_ops)[None, :], machine]
    return env.true_op_ct - pt


def main():
    out = {}
    for size in dv3.SIZES:
        data, idx, names = instances(size)
        rec = {"instances": {"validation": names.count("validation"), "development": names.count("development")}}
        lb = np.array([np.where(data.op_pt[i] > 0, data.carbon[i], np.inf).min(1).sum() for i in idx])
        # lowest-energy allowed rule
        env, _ = dv3.make_env(data, idx)
        machine, st = run(env, min_energy_allowed, stats=True)
        mk, cb, td = dv3.env_objectives(env)
        own_start = realised(env, machine)
        own_viol = [nondelay_violations(data.job_length[i], data.op_pt[i], machine[r], own_start[r])
                    for r, i in enumerate(idx)]
        on_min = [np.isclose(data.carbon[i][np.arange(len(machine[r])), machine[r]],
                             np.where(data.op_pt[i] > 0, data.carbon[i], np.inf).min(1)) for r, i in enumerate(idx)]
        rec["min_energy_allowed"] = {
            **dv3.ratios(data, idx, mk, cb, td),
            "share_steps_some_min_machine_masked": st["some_min_masked"] / st["steps"],
            "share_steps_no_min_pair_allowed": st["no_min_allowed"] / st["steps"],
            "share_ops_not_on_min_machine": float(1 - np.mean(np.concatenate(on_min))),
            "carbon_unrestricted_min": float(lb.mean()),
            "carbon_ratio_unrestricted_min": float(np.mean(lb / np.asarray(data.cb_ref))),
            "carbon_excess_over_min_pct": float(100 * np.mean(cb / lb - 1)),
            "env_schedules_with_nondelay_violation": int(np.sum(np.array(own_viol) > 0))}
        # EDD inside the mask vs unrestricted EDD+ECT
        env, _ = dv3.make_env(data, idx)
        run(env, edd_allowed)
        rec["EDD_allowed"] = dv3.ratios(data, idx, *dv3.env_objectives(env))
        tgt = {"EDD+ECT": [], "EDD+MINE": [], "CPSAT_tard": [], "CPSAT_mk": []}
        cp_status = {"tard": [], "mk": []}
        for i in idx:
            inst = dv3.ov_instance(data, i)
            for jr, mr in (("EDD", "ECT"), ("EDD", "MINE")):
                tgt[f"{jr}+{mr}"].append(S.build(inst, jr, mr, energy=data.carbon[i], due=data.due[i]))
            ci = CP.Inst(data, i)
            for obj, lim in (("tard", 20.0), ("mk", 10.0)):
                r = CP.solve(ci, obj, lim, 8, hint=CP.rule_hint(ci, obj))
                cp_status[obj].append(r["status"])
                tgt[f"CPSAT_{obj}"].append((np.array(r["machine"]), np.array(r["start"])))
        for name, targets in tgt.items():
            t_obj = [B.schedule_objectives(data.job_length[i], data.op_pt[i], data.carbon[i], data.due[i], *targets[r])
                     for r, i in enumerate(idx)]
            env, _ = dv3.make_env(data, idx)
            pol, dev = replayer(targets)
            machine, _ = run(env, pol)
            mk, cb, td = dv3.env_objectives(env)
            viol = np.array([nondelay_violations(data.job_length[i], data.op_pt[i], *targets[r]) for r, i in enumerate(idx)])
            rs = realised(env, machine)
            env_viol = [nondelay_violations(data.job_length[i], data.op_pt[i], machine[r], rs[r]) for r, i in enumerate(idx)]
            rec[f"replay_{name}"] = {
                "target": dv3.ratios(data, idx, [t["makespan"] for t in t_obj], [t["carbon"] for t in t_obj],
                                     [t["tardiness"] for t in t_obj]),
                "through_mask": dv3.ratios(data, idx, mk, cb, td),
                "instances_zero_deviation": int((dev == 0).sum()),
                "targets_non_delay": int((viol == 0).sum()),
                "mean_waiting_ops_with_idle_eligible_machine": float(viol.mean()),
                "replayed_schedules_with_nondelay_violation": int(np.sum(np.array(env_viol) > 0)),
                "mean_deviation_steps": float(dev.mean())}
        rec["cpsat_status"] = {k: {s: v.count(s) for s in set(v)} for k, v in cp_status.items()}
        out[size] = rec
        print(size, dv3.json.dumps(rec, indent=1, default=float), flush=True)
    dv3.save_json("t2_mask.json", out)


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"done in {time.time() - t0:.0f} s")
