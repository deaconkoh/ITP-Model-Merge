"""Shared helpers for the implementation audit tests (results/code_audit.md). Evaluation-only; changes no repo code.

Importing fixes the process config to the canonical v1 config (via tools/v2_common -> complementarity); v2 networks
and environments are requested explicitly. Final-test data is never loaded.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[2]
_argv = sys.argv[:]
sys.path.insert(0, str(REPO / "tools"))
import v2_common as V                                   # noqa: E402
sys.argv = _argv
import torch                                            # noqa: E402
import torch.nn.functional as F                         # noqa: E402
from fjsp_env_same_op_nums import FJSPEnvForSameOpNums   # noqa: E402
import ceiling_scorer as SC                              # noqa: E402

DAN = REPO / "daniel"
HELD = SC.HELD
POOL = "data_train_vali"


def pool(split="data_train_vali"):
    """(data tuple, due list) for a 10x5 carbon+priority pool (never data_final_test)."""
    assert "final_test" not in split
    from data_utils import load_priority_carbon_data_from_files
    from due_dates import due_dates_for_directory, load_manifest
    d = DAN / f"data/{split}/SD2/10x5+carbon+priority"
    return load_priority_carbon_data_from_files(str(d)), due_dates_for_directory(d, load_manifest())


@torch.no_grad()
def rollout(pol, data, due, inst, obs="v1", mode="greedy", gen=None, reward_refs=None, reward_version="v1"):
    """One batched episode over instances `inst` (pool indices, as ONE batch, whatever their maxima).
    mode: greedy | sample (policy softmax) | random (uniform over valid actions; pol may be None).
    Returns dict with per-row machine [E,N], start [E,N], actions [T,E], rewards [T,E], probs_max_diff
    placeholders, and the environment's mk, cb, tard."""
    sub = tuple([d[i] for i in inst] for d in data); dues = [due[i] for i in inst]
    env = FJSPEnvForSameOpNums(n_j=sub[0][0].shape[0], n_m=sub[1][0].shape[1], obs_version=obs,
                               reward_version=reward_version)
    state = env.set_initial_data(*[list(x) for x in sub], due_date_list=list(dues), reward_refs=reward_refs)
    E, M, rows = env.number_of_envs, env.number_of_machines, np.arange(env.number_of_envs)
    machine = np.full((E, env.number_of_ops), -1)
    acts, rews, steps = [], [], 0
    while True:
        if mode == "random":
            valid = ~state.dynamic_pair_mask_tensor.reshape(E, -1)
            logits = torch.where(valid, torch.zeros_like(valid, dtype=torch.float32), torch.full_like(valid, float("-inf"), dtype=torch.float32))
            a = torch.multinomial(F.softmax(logits, 1), 1, generator=gen).squeeze(1)
        else:
            s = pol.forward_logits(**V.P.args_of(state))[0]
            a = s.argmax(1) if mode == "greedy" else torch.multinomial(F.softmax(s, 1), 1, generator=gen).squeeze(1)
        a = a.cpu().numpy()
        op = env.candidate[rows, a // M]
        assert (machine[rows, op] == -1).all(), "operation scheduled twice"
        machine[rows, op] = a % M
        state, r, done = env.step(actions=a)
        acts.append(a); rews.append(np.asarray(r, dtype=np.float64)); steps += 1
        if np.all(done):
            break
        assert not np.any(done), "some environments finished before others"
    comp = env.true_op_ct.copy()
    start = comp - env.true_op_pt[rows[:, None], np.arange(env.number_of_ops)[None, :], machine]
    tard = np.maximum(0.0, env.true_op_ct[env.env_job_idx, env.job_last_op_id] - np.asarray(dues)).sum(1)
    return dict(machine=machine, start=start, actions=np.array(acts), rewards=np.array(rews), steps=steps,
                n_ops=env.number_of_ops, mk=env.current_makespan.copy(), cb=env.total_carbon.copy(), tard=tard,
                ub=env.pt_upper_bound, env=env)


def validate_schedule(ins: SC.Instance, machine, start):
    """Independent validator. Returns a list of violation strings (empty = feasible). Also checks the append
    start rule: each operation starts exactly at max(job predecessor completion, previous operation on its machine
    in start order completion)."""
    v = []
    machine = [int(m) for m in machine]; start = [float(s) for s in start]
    if len(machine) != ins.n_ops:
        return ["wrong number of operations"]
    comp = []
    for o in range(ins.n_ops):
        if machine[o] not in ins.pt[o]:
            v.append(f"op {o} on ineligible machine {machine[o]}"); comp.append(np.nan); continue
        if start[o] < 0:
            v.append(f"op {o} negative start")
        comp.append(start[o] + ins.pt[o][machine[o]])
    for ops in ins.job_ops:
        for a, b in zip(ops, ops[1:]):
            if start[b] < comp[a] - 1e-9:
                v.append(f"precedence {a}->{b}")
    prev_on_m = {}
    for m in range(ins.n_m):
        on = sorted((start[o], comp[o], o) for o in range(ins.n_ops) if machine[o] == m)
        for (s0, c0, o0), (s1, _, o1) in zip(on, on[1:]):
            if s1 < c0 - 1e-9:
                v.append(f"overlap on machine {m}: ops {o0},{o1}")
            prev_on_m[o1] = c0
    job_pred = {b: a for ops in ins.job_ops for a, b in zip(ops, ops[1:])}
    for o in range(ins.n_ops):
        expect = max(comp[job_pred[o]] if o in job_pred else 0.0, prev_on_m.get(o, 0.0))
        if abs(start[o] - expect) > 1e-9:
            v.append(f"op {o} start {start[o]} != append rule {expect}")
    return v


def write_result(name, obj):
    out = REPO / "results/code_audit/tests"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{name}.json").write_text(json.dumps(obj, indent=1, default=float))
