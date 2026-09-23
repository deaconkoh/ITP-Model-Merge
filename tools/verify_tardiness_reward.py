"""STEP 1 check: does the per-step tardiness reward sum to exactly -T over an episode?

Reward decomposition (potential-based shaping, same construction as the makespan term):

    Phi_t   = sum_j max(0, LB_j(t) - d_j)      LB_j(t) = job j's completion-time LOWER BOUND
    r_t     = Phi_{t-1} - Phi_t                 (normalised time units)
    Phi_{-1} := 0

Once every operation is scheduled the lower bounds ARE the true completion times, so the sum
telescopes to  sum_t r_t = Phi_{-1} - Phi_final = -T / time_scale.  Starting from Phi = 0 rather
than Phi_0 charges, on the first step, any tardiness already unavoidable at t = 0; without that
the return would be -T/time_scale + Phi_0, exact only up to a per-instance constant.

The check runs BOTH environment classes, in batch (the time scale is a batch-level maximum),
with random feasible actions and with a fixed greedy policy, at several due-date tightnesses --
including a deliberately tight one so that Phi_0 > 0 and the first-step charge is exercised.

The due dates here are PLACEHOLDERS for testing the reward algebra only: they are not the
Step 2 generation, which is selected separately and then frozen.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "daniel"))
sys.argv = [sys.argv[0], "--config", str(REPO / "configs/canonical/t.json"), "--device", "cpu"]
from data_utils import load_priority_carbon_data_from_files   # noqa: E402
from fjsp_env_same_op_nums import FJSPEnvForSameOpNums        # noqa: E402
from fjsp_env_various_op_nums import FJSPEnvForVariousOpNums  # noqa: E402


def work_content(job_length, op_pt):
    """P_j = sum over j's operations of the mean processing time over ELIGIBLE machines (raw)."""
    op_mean = np.array([row[row > 0].mean() for row in op_pt])
    bounds = np.concatenate([[0], np.cumsum(job_length)])
    return np.array([op_mean[bounds[j]:bounds[j + 1]].sum() for j in range(len(job_length))])


def placeholder_due(job_length, op_pt, k, s, seed):
    P = work_content(job_length, op_pt)
    kj = np.random.default_rng(seed).uniform(k * (1 - s / 2), k * (1 + s / 2), size=P.shape)
    return kj * P


def run(env_cls, data, due, policy, seed):
    jl, pt, pr, cb = data
    n_j, n_m = jl[0].shape[0], pt[0].shape[1]
    env = env_cls(n_j, n_m)
    state = env.set_initial_data(jl, pt, pr, cb, due_date_list=due)
    rng = np.random.default_rng(seed)
    E = env.number_of_envs
    ret = np.zeros(E)
    first_step = None
    while True:
        feasible = ~np.asarray(env.dynamic_pair_mask)            # [E(active), J, M]
        acts = []
        for e in range(feasible.shape[0]):
            flat = np.flatnonzero(feasible[e].reshape(-1))
            acts.append(rng.choice(flat) if policy == "random" else flat[0])
        _, r, done = env.step(np.array(acts))
        r = np.asarray(r, dtype=np.float64)
        if first_step is None:
            first_step = r.copy()
        ret += r
        if np.all(done):
            break
    # independent recomputation of T from the schedule the env actually produced
    C = env.true_op_ct[env.env_job_idx, env.job_last_op_id]
    T = np.array([sum(max(0.0, c - d) for c, d in zip(C[e], due[e])) for e in range(E)])
    return ret * env.pt_upper_bound, T


def main():
    worst = 0.0
    for size in ("10x5", "20x10"):
        data = load_priority_carbon_data_from_files(
            str(REPO / f"daniel/data/data_train_vali/SD2/{size}+carbon+priority"))
        n = 20
        data = tuple(d[:n] for d in data)
        print(f"=== {size}: batch of {n} instances ===")
        for k in (0.5, 1.0, 1.5):
            due = np.stack([placeholder_due(data[0][i], data[1][i], k, 0.8, seed=i)
                            for i in range(n)])
            # how often is tardiness already unavoidable at t=0 (Phi_0 > 0)?
            lb0 = []
            for i in range(n):
                jl, pt = data[0][i], data[1][i]
                op_min = np.array([row[row > 0].min() for row in pt])
                b = np.concatenate([[0], np.cumsum(jl)])
                lb0.append(sum(max(0.0, op_min[b[j]:b[j + 1]].sum() - due[i][j])
                               for j in range(len(jl))))
            frac_phi0 = float(np.mean(np.array(lb0) > 0))
            for env_cls in (FJSPEnvForSameOpNums, FJSPEnvForVariousOpNums):
                for policy in ("random", "greedy"):
                    ret, T = run(env_cls, data, due, policy, seed=7)
                    err = float(np.max(np.abs(ret + T)))
                    rel = float(np.max(np.abs(ret + T) / np.maximum(T, 1.0)))
                    worst = max(worst, err)
                    print(f"  k={k:<4} {env_cls.__name__:<24} {policy:<7} "
                          f"mean T={T.mean():9.1f}  max|return*scale + T| = {err:.2e}"
                          f"  (rel {rel:.1e})   instances with Phi_0>0: {frac_phi0:.0%}")
    print(f"\nWORST absolute discrepancy across all runs: {worst:.3e} time units")
    print("PASS: episode return x time_scale == -T to floating-point precision"
          if worst < 1e-6 else "FAIL: episode return does not equal -T")
    return 0 if worst < 1e-6 else 1


if __name__ == "__main__":
    raise SystemExit(main())
