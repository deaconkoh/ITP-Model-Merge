"""Correctness gate: a random admissible permutation must leave the policy function unchanged.

If this fails, every downstream alignment number is meaningless, so it runs first.
Run from daniel/.
"""
import sys, os
sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.abspath("../tools"))
sys.argv = ["verify", "--device", "cpu"]

import numpy as np
import torch
from data_utils import load_priority_carbon_data_from_files
from fjsp_env_same_op_nums import FJSPEnvForSameOpNums
from model.PPO import PPO_initialize
from common_utils import load_checkpoint_state_dict
from params import configs
import permutation_symmetry as PS


def get_state(pool, n=3):
    ds = load_priority_carbon_data_from_files(pool)
    n_j = ds[0][0].shape[0]; n_op, n_m = ds[1][0].shape
    env = FJSPEnvForSameOpNums(n_j=n_j, n_m=n_m)
    return env.set_initial_data(ds[0][:n], ds[1][:n], ds[2][:n], ds[3][:n])


def policy_out(ppo, state):
    with torch.no_grad():
        pi, v = ppo.policy(
            fea_j=state.fea_j_tensor, op_mask=state.op_mask_tensor,
            candidate=state.candidate_tensor, fea_m=state.fea_m_tensor,
            mch_mask=state.mch_mask_tensor, comp_idx=state.comp_idx_tensor,
            dynamic_pair_mask=state.dynamic_pair_mask_tensor,
            fea_pairs=state.fea_pairs_tensor)
    return pi.clone(), v.clone()


def main():
    pool = "./data/data_development/SD2/10x5+carbon+priority"
    state = get_state(pool)
    ppo = PPO_initialize()
    sd = load_checkpoint_state_dict("./trained_network/SD2/10x5+carbon+priority+c_s111.pth",
                                    map_location="cpu", expected_schema=configs.feature_schema)

    ppo.policy.load_state_dict(sd); pi0, v0 = policy_out(ppo, state)

    rng = np.random.default_rng(0)
    print(f"{'trial':>5} {'max|dpi|':>12} {'max|dv|':>12} {'param dist':>12}  verdict")
    ok = True
    for t in range(5):
        P = PS.random_perm(rng)
        sd_p = PS.apply_perm(sd, P)
        ppo.policy.load_state_dict(sd_p)
        pi1, v1 = policy_out(ppo, state)
        dpi = float((pi1 - pi0).abs().max()); dv = float((v1 - v0).abs().max())
        pdist = float(np.sqrt(PS.sq_dist(sd, sd_p, list(sd.keys()))))
        good = dpi < 1e-5 and dv < 1e-5 and pdist > 1e-3
        ok &= good
        print(f"{t:>5} {dpi:12.3e} {dv:12.3e} {pdist:12.4f}  "
              f"{'OK (function preserved, weights moved)' if good else 'FAIL'}")

    # identity must be a no-op
    sd_i = PS.apply_perm(sd, PS.identity_perm())
    idd = max(float((sd[k] - sd_i[k]).abs().max()) for k in sd)
    print(f"\nidentity permutation is a no-op: {idd == 0.0}")
    print(f"\nGATE: {'PASS' if ok and idd == 0.0 else 'FAIL'}")
    return 0 if (ok and idd == 0.0) else 1


if __name__ == "__main__":
    sys.exit(main())
