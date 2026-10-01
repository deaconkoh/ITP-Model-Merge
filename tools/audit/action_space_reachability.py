"""Audit diagnostic: can DANIEL's MASKED action space produce the CP-SAT schedules? (results/code_audit.md)

The environment masks every (operation, machine) pair whose start time max(job ready, machine free) exceeds the
earliest start over all compatible candidate pairs (fjsp_env_same_op_nums.py:588-592, identical in the original
DANIEL code). Every action therefore starts an operation at the earliest possible time: the reachable schedules are
the NON-DELAY schedules, a subset of the active schedules that need not contain an optimum. ceiling_check's
reachability check replayed CP-SAT schedules with the append rule only, ignoring this mask.

For the seed-111 CP-SAT 60 s solutions on the 49 HELD-OUT instances (results/ceiling_check/raw/cpsat_t60):
  exact     replay in the real environment, at each step taking the unscheduled operation with the earliest CP-SAT
            start on its CP-SAT machine; reachable iff that pair is never masked;
  guided    the best non-delay imitation: at each step, among UNMASKED pairs, take the one whose operation has the
            earliest CP-SAT start (CP-SAT's machine preferred, then the shortest processing time); scored with the
            Step 4 score. The gap guided - CP-SAT is a lower bound on what the mask costs this solution.
EVALUATION ONLY; no repo code is changed.

    python tools/audit/action_space_reachability.py
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tests/audit"))
import audit_common as A                                # noqa: E402

SEED = 111


def replay(data, due, i, sol, guided):
    sub = tuple([d[i]] for d in data)
    env = A.FJSPEnvForSameOpNums(10, 5, obs_version="v1", reward_version="v1")
    state = env.set_initial_data(*[list(x) for x in sub], due_date_list=[due[i]])
    M = env.number_of_machines
    cs, cm = np.array(sol["start"], float), np.array(sol["machine"])
    blocked = 0
    while True:
        mask = state.dynamic_pair_mask_tensor.cpu().numpy()[0]              # [J, M], True = masked
        cand = env.candidate[0]
        live = [j for j in range(env.number_of_jobs) if not env.mask[0, j]]
        j_want = min(live, key=lambda j: (cs[cand[j]], cand[j]))
        a = j_want * M + cm[cand[j_want]]
        if mask[j_want, cm[cand[j_want]]]:
            blocked += 1
            if not guided:
                return None, blocked
            ok = [(cs[cand[j]], int(cm[cand[j]] != m), float(env.true_op_pt[0, cand[j], m]), j, m)
                  for j in live for m in range(M) if not mask[j, m]]
            _, _, _, j, m = min(ok)
            a = j * M + m
        state, _, done = env.step(actions=np.array([a]))
        if done[0]:
            break
    td = float(np.maximum(0.0, env.true_op_ct[0, env.job_last_op_id[0]] - np.asarray(due[i])).sum())
    return A.SC.score(SEED, i, float(env.current_makespan[0]), float(env.total_carbon[0]), td), blocked


def main():
    data, due = A.pool()
    rows = []
    for i in A.HELD:
        sol = json.loads((REPO / f"results/ceiling_check/raw/cpsat_t60/s{SEED}_i{int(i)}.json").read_text())
        exact, _ = replay(data, due, i, sol, guided=False)
        g, blocked = replay(data, due, i, sol, guided=True)
        rows.append(dict(inst=int(i), cpsat=sol["score"], exact_reachable=exact is not None, guided=g, blocked_steps=blocked))
    cp = np.array([r["cpsat"] for r in rows]); gd = np.array([r["guided"] for r in rows])
    out = dict(n=len(rows), exactly_reachable=int(sum(r["exact_reachable"] for r in rows)),
               mean_blocked_steps=float(np.mean([r["blocked_steps"] for r in rows])),
               cpsat_mean=float(cp.mean()), guided_nondelay_mean=float(gd.mean()),
               guided_gap_mean=float(np.mean((gd - cp) / cp)), rows=rows)
    A.write_result("action_space_reachability", out)
    print(f"CP-SAT schedules exactly reachable under the mask: {out['exactly_reachable']}/{out['n']}; "
          f"mean blocked decisions per schedule {out['mean_blocked_steps']:.1f}/50")
    print(f"CP-SAT mean {out['cpsat_mean']:.4f}; best non-delay imitation {out['guided_nondelay_mean']:.4f} "
          f"(gap {100*out['guided_gap_mean']:+.1f}%)")


if __name__ == "__main__":
    main()
