"""Audit diagnostic: how good can schedules INSIDE DANIEL's masked (non-delay) action space get? (code_audit.md)

Empirical search within the real environment (every schedule produced obeys the mask), seed-111 Step 4 score on the
49 HELD-OUT instances, best per instance over:
  C1@1000 (v1) sampled at temperature 1 and 2, 500 rollouts each;  V2-M@1000 (v2) at temperature 1 and 2, 500 each;
  a uniform-random policy over unmasked actions, 1,000 rollouts.
The best found is an UPPER bound on the best non-delay score (the true non-delay optimum can only be lower).
Rollouts batch only instances sharing a maximum processing time (canonical scaling). EVALUATION ONLY.

    python tools/audit/nondelay_search.py
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tests/audit"))
import audit_common as A                                # noqa: E402
import torch                                            # noqa: E402
import torch.nn.functional as F                         # noqa: E402

SEED = 111


@torch.no_grad()
def batch_scores(pol, obs, data, due, inst, temp, gen):
    """inst: pool indices (repeats). Returns Step 4 scores in input order."""
    ub = np.array([float(data[1][i][data[1][i] > 0].max()) for i in inst])
    sc = np.zeros(len(inst))
    for u in np.unique(ub):
        rows = np.flatnonzero(ub == u)
        for c0 in range(0, len(rows), 4000):
            r = rows[c0:c0 + 4000]
            sub = tuple([d[i] for i in inst[r]] for d in data); dues = [due[i] for i in inst[r]]
            env = A.FJSPEnvForSameOpNums(10, 5, obs_version=obs, reward_version="v1")
            state = env.set_initial_data(*[list(x) for x in sub], due_date_list=dues)
            E = env.number_of_envs
            while True:
                if pol is None:
                    valid = ~state.dynamic_pair_mask_tensor.reshape(E, -1)
                    logits = torch.where(valid, 0.0, float("-inf"))
                else:
                    logits = pol.forward_logits(**A.V.P.args_of(state))[0] / temp
                a = torch.multinomial(F.softmax(logits, 1), 1, generator=gen).squeeze(1).cpu().numpy()
                state, _, done = env.step(actions=a)
                if np.all(done):
                    break
            td = np.maximum(0.0, env.true_op_ct[env.env_job_idx, env.job_last_op_id] - np.asarray(dues)).sum(1)
            for k, i in enumerate(inst[r]):
                sc[r[k]] = A.SC.score(SEED, i, env.current_makespan[k], env.total_carbon[k], td[k])
    return sc


def main():
    data, due = A.pool()
    gen = torch.Generator(device=A.V.P.DEV); gen.manual_seed(20261006)
    held = np.array(A.HELD); n = len(held)
    best = {}
    sources = [("C1@1000 T=1", "10x5+carbon+priority+s4t_spec_m_k125s060_s111@u1000", "v1", 1.0, 500),
               ("C1@1000 T=2", "10x5+carbon+priority+s4t_spec_m_k125s060_s111@u1000", "v1", 2.0, 500),
               ("V2-M@1000 T=1", "v2rr/v2rr_V2-M_s111@u1000", "v2", 1.0, 500),
               ("V2-M@1000 T=2", "v2rr/v2rr_V2-M_s111@u1000", "v2", 2.0, 500),
               ("uniform random", None, "v1", 1.0, 1000)]
    for label, name, obs, temp, N in sources:
        pol = A.V.load_policy(name, obs) if name else None
        s = batch_scores(pol, obs, data, due, np.repeat(held, N), temp, gen).reshape(n, N)
        best[label] = s.min(1)
        print(f"{label:<16} mean sample {s.mean():.4f}   best-of-{N} {s.min(1).mean():.4f}", flush=True)
    allbest = np.min(np.stack(list(best.values())), 0)
    cp = np.array([json.loads((REPO / f"results/ceiling_check/raw/cpsat_t60/s{SEED}_i{int(i)}.json").read_text())["score"] for i in held])
    out = {k: float(v.mean()) for k, v in best.items()}
    out.update(best_of_all_3000=float(allbest.mean()), cpsat=float(cp.mean()),
               gap_best_nondelay_to_cpsat=float(np.mean((allbest - cp) / cp)))
    A.write_result("nondelay_search", out)
    print(f"best of all 3,000 non-delay schedules per instance: {allbest.mean():.4f}  vs CP-SAT {cp.mean():.4f} "
          f"(gap {100*out['gap_best_nondelay_to_cpsat']:+.1f}%)")


if __name__ == "__main__":
    main()
