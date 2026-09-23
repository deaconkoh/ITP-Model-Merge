"""Record each job's completion time under a trained policy (greedy decoding), per instance.

Needed to choose the due-date tightness k against what a TRAINED policy actually achieves:
the heuristic dispatcher is 24-38% worse than the trained makespan specialists, so it makes
every k look tighter than a trained policy will experience it.

Output: <out>/<model>__<pool>.npy, float array [n_instances, n_jobs] of raw completion times,
instances in the same order as data_utils.load_priority_carbon_data_from_files.
Run from daniel/.
"""
import sys, os, time, argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "daniel"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool-root", default="./data/data_train_vali")
    ap.add_argument("--pool", required=True)
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed-test", type=int, default=50)
    args = ap.parse_args()

    sys.argv = ["eval_job_completion", "--device", "cuda"]
    import numpy as np
    import torch
    from data_utils import load_priority_carbon_data_from_files
    import test_trained_model as T
    from fjsp_env_same_op_nums import FJSPEnvForSameOpNums

    data = load_priority_carbon_data_from_files(f"{args.pool_root}/SD2/{args.pool}")
    os.makedirs(args.out, exist_ok=True)
    for name in args.models:
        out = f"{args.out}/{name}__{args.pool}.npy"
        if os.path.exists(out):
            print(f"skip (exists) {name}", flush=True); continue
        t0 = time.time()
        T.setup_seed(args.seed_test)
        T.ppo.policy.load_state_dict(T.load_checkpoint_state_dict(
            f"./trained_network/SD2/{name}.pth", map_location=T.configs.device,
            expected_schema=T.configs.feature_schema))
        T.ppo.policy.eval()
        n_j = data[0][0].shape[0]; n_m = data[1][0].shape[1]
        env = FJSPEnvForSameOpNums(n_j=n_j, n_m=n_m)
        rows, mks = [], []
        for i in range(len(data[0])):
            state = env.set_initial_data([data[0][i]], [data[1][i]], [data[2][i]], [data[3][i]])
            while True:
                with torch.no_grad():
                    pi, _ = T.ppo.policy(fea_j=state.fea_j_tensor, op_mask=state.op_mask_tensor,
                                         candidate=state.candidate_tensor, fea_m=state.fea_m_tensor,
                                         mch_mask=state.mch_mask_tensor, comp_idx=state.comp_idx_tensor,
                                         dynamic_pair_mask=state.dynamic_pair_mask_tensor,
                                         fea_pairs=state.fea_pairs_tensor)
                state, _, done = env.step(actions=T.greedy_select_action(pi).cpu().numpy())
                if done:
                    break
            rows.append(env.true_op_ct[0, env.job_last_op_id[0]].astype(float))
            mks.append(float(env.current_makespan[0]))
        np.save(out, np.array(rows))
        print(f"{name}: {time.time()-t0:.1f}s  mean makespan={np.mean(mks):.1f}", flush=True)
    print("EVAL_JOB_COMPLETION_DONE", flush=True)


if __name__ == "__main__":
    main()
