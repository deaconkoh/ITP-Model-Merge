"""Test 3: what DANIEL-C (and, for comparison, DANIEL-M and DANIEL-T) choose (hypothesis A4).

Greedy rollouts of the selected checkpoints on the whole validation split. For every decision, the chosen machine is
compared with the chosen operation's (a) lowest-energy, (b) fastest, (c) earliest-completing ELIGIBLE machine, and with
the lowest-energy machine among the pairs the mask allows for that operation; plus how often the chosen pair is the
lowest-carbon allowed pair overall. The base rate "fastest eligible = lowest-energy eligible" is reported per size.

    python tools/diagnose_v3/t3_choices.py
"""
import numpy as np
import torch
import dv3
from dv3 import B
from common_utils import greedy_select_action


@torch.no_grad()
def choices(net, data, idx):
    env, state = dv3.make_env(data, idx)
    n_m = env.number_of_machines
    e = env.env_idxs
    c = {k: 0 for k in ("n", "lowest_energy", "fastest", "earliest_completing", "lowest_energy_allowed_for_op",
                        "lowest_carbon_allowed_overall", "fastest_is_lowest_energy", "allowed_machines_for_op")}
    while True:
        pi, _ = dv3.forward(net, state)
        a = greedy_select_action(pi).cpu().numpy()
        j, m = a // n_m, a % n_m
        op = env.candidate[e, j]
        pt = env.true_op_pt[e, op]                                              # [E,M], 0 = ineligible
        elig = pt > 0
        carbon = env.op_carbon[e, op]
        start = np.maximum(env.true_candidate_free_time[e, j][:, None], env.true_mch_free_time)
        comp = np.where(elig, start + pt, np.inf)
        allowed = ~env.dynamic_pair_mask.astype(bool)                           # [E,J,M]
        allowed_op = allowed[e, j]                                              # [E,M]
        cm = np.where(elig, carbon, np.inf)
        ce = cm[e, m]
        c["n"] += len(a)
        c["lowest_energy"] += int(np.sum(ce <= cm.min(1) + 1e-12))
        c["fastest"] += int(np.sum(pt[e, m] <= np.where(elig, pt, np.inf).min(1)))
        c["earliest_completing"] += int(np.sum(comp[e, m] <= comp.min(1)))
        c["lowest_energy_allowed_for_op"] += int(np.sum(ce <= np.where(allowed_op, carbon, np.inf).min(1) + 1e-12))
        all_c = np.where(allowed, env.op_carbon[e[:, None], env.candidate], np.inf).reshape(len(a), -1).min(1)
        c["lowest_carbon_allowed_overall"] += int(np.sum(ce <= all_c + 1e-12))
        fast = np.where(elig, pt, np.inf).argmin(1)
        c["fastest_is_lowest_energy"] += int(np.sum(cm[e, fast] <= cm.min(1) + 1e-12))
        c["allowed_machines_for_op"] += int(allowed_op.sum())
        state, _, done = env.step(a)
        if np.all(done):
            break
    n = c.pop("n")
    rec = {k: v / n for k, v in c.items()}
    rec["decisions"] = n
    rec.update(dv3.ratios(data, idx, *dv3.env_objectives(env)))
    return rec


def main():
    out = {}
    for size in dv3.SIZES:
        data = B.load_split("validation", size)
        idx = list(range(len(data)))
        for goal in "cmt":
            for seed in dv3.SEEDS:
                name = dv3.run_name(size, goal, seed)
                out[name] = choices(dv3.load_net(name), data, idx)
                print(name, {k: round(v, 3) for k, v in out[name].items()}, flush=True)
    dv3.save_json("t3_choices.json", out)


if __name__ == "__main__":
    main()
