"""Parts A3/A4 of results/v2_fixes.md (EVALUATION ONLY).

C1@1000 converted to v2 (s4t_spec_m_k125s060_s<seed>@u1000_obsv2), v2 observations + v2 reward, goal mct, on the
audit's training-generator episodes (10x5 TRAIN pool, 5 batches of 20 drawn with default_rng(20260929), batch-max
normalisation), sampled actions (generator 20260929 + seed) and greedy, 4 seeds:
  * A3: |sum of per-step rewards + Step 4 score of the episode| < 1e-9 for every episode;
  * A4: per-term mean |return| and between-episode sd, as input_reward_audit measured them for v1;
  * feature check: the v2 features recomputed per job with plain loops from raw instance data and the
    environment's schedule state (before the per-feature standardisation) must match the environment's to 1e-9.
Writes results/v2_fixes/reward_balance.json.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))
import v2_common as V                                  # noqa: E402
import torch                                           # noqa: E402
import torch.nn.functional as F                        # noqa: E402
from fjsp_env_same_op_nums import FJSPEnvForSameOpNums           # noqa: E402
from data_utils import load_priority_carbon_data_from_files      # noqa: E402
from due_dates import due_dates_for_directory, load_manifest     # noqa: E402
import reward_refs as RR                                          # noqa: E402

OUT = REPO / "results/v2_fixes"
TRAIN = REPO / "daniel/data/data_train/SD2/10x5+carbon+priority"
SEEDS, N_BATCH, BATCH, RNG_SEED = [111, 222, 333, 444], 5, 20, 20260929


class Recorder:
    def __init__(self, inner):
        self.inner, self.rec = inner, []
        self.uses_tardiness, self.weights, self.tbar = inner.uses_tardiness, inner.weights, inner.tbar

    def step_reward(self, mk_d, carbon, tard_d, u, mk_ref, cb_ref):
        w_m, w_c, w_t = self.inner.weights
        self.rec.append((w_m * u * np.asarray(mk_d) / mk_ref, -w_c * np.asarray(carbon) / cb_ref,
                         w_t * u * np.asarray(tard_d) / self.inner.tbar))
        return self.inner.step_reward(mk_d, carbon, tard_d, u, mk_ref, cb_ref)

    def score(self, *a):
        return self.inner.score(*a)


def naive_features(env, raw_pt, due_true):
    """Independent per-job recomputation of (slack, time-to-due) per op and projected lateness per pair, scaled."""
    E, J, M, U = env.number_of_envs, env.number_of_jobs, env.number_of_machines, env.pt_upper_bound
    slack_op = np.zeros((E, env.number_of_ops)); ttd_op = np.zeros_like(slack_op); late = np.zeros((E, J, M))
    for e in range(E):
        pt = np.asarray(raw_pt[e], dtype=float) / U
        for j in range(J):
            first, last = env.job_first_op_id[e, j], env.job_last_op_id[e, j]
            ops = range(first, last + 1)
            d = due_true[e][j] / U
            ready = env.candidate_free_time[e, j]
            unsched = [o for o in ops if not env.op_scheduled_flag[e, o]]
            minpt = lambda o: pt[o][pt[o] > 0].min()
            if not unsched:
                est, work = ready, 0.0
                late[e, j, :] = max(0.0, ready - d)
            else:
                o = unsched[0]
                est = min(max(ready, env.mch_free_time[e, m]) for m in range(M) if pt[o][m] > 0)
                work = sum(minpt(q) for q in unsched)
                for m in range(M):
                    late[e, j, m] = max(0.0, max(ready, env.mch_free_time[e, m]) + pt[o][m] + work - minpt(o) - d)
            for o in ops:
                slack_op[e, o], ttd_op[e, o] = d - est - work, d - est
    return slack_op, ttd_op, late


@torch.no_grad()
def episode(pol, data, dues, refs, gen, check):
    env = FJSPEnvForSameOpNums(n_j=10, n_m=5, obs_version="v2", reward_version="v2")
    raw = {}
    orig = env.norm_op_features

    def capture():
        raw["fea_j"] = env.fea_j.copy(); orig()
    env.norm_op_features = capture
    state = env.set_initial_data(*[list(d) for d in data], due_date_list=list(dues), reward_refs=refs)
    env.reward_v2 = Recorder(env.reward_v2)
    fdiff, t = 0.0, 0
    while True:
        if check and t % 7 == 0:
            s_op, t_op, late = naive_features(env, data[1], dues)
            fdiff = max(fdiff, np.abs(raw["fea_j"][:, :, 11] - s_op).max(), np.abs(raw["fea_j"][:, :, 12] - t_op).max(),
                        np.abs(env.fea_pairs[:, :, :, 9] - late).max(),
                        np.abs(env.fea_pairs[:, :, :, 8] * 100 - np.array([env.op_carbon[k][env.candidate[k]]
                                                                          for k in range(env.number_of_envs)])).max())
        s = pol.forward_logits(**V.P.args_of(state))[0]
        a = (s.argmax(1) if gen is None else torch.multinomial(F.softmax(s, 1), 1, generator=gen).squeeze(1)).cpu().numpy()
        state, r, done = env.step(actions=a)
        raw.setdefault("R", np.zeros(env.number_of_envs)); raw["R"] += r
        t += 1
        if np.all(done):
            break
    rec = env.objective_terms = env.reward_v2.rec
    R_m, R_c, R_t = (np.sum([x[i] for x in rec], 0) for i in range(3))
    T = np.maximum(0.0, env.true_op_ct[env.env_job_idx, env.job_last_op_id] - np.asarray(dues)).sum(1)
    score = env.reward_v2.score(env.current_makespan, env.total_carbon, T, env.v2_mk_ref, env.v2_cb_ref)
    return dict(R_m=R_m, R_c=R_c, R_t=R_t, R=raw["R"], score=score, err=np.abs(raw["R"] + score), mk=env.current_makespan.copy(),
                cb=env.total_carbon.copy(), T=T), fdiff


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    data = load_priority_carbon_data_from_files(str(TRAIN))
    due = due_dates_for_directory(TRAIN, load_manifest())
    rr = RR.load_manifest()
    rng = np.random.default_rng(RNG_SEED)
    draws = [rng.integers(0, len(data[0]), BATCH) for _ in range(N_BATCH)]
    res, worst_err, worst_feat = {}, 0.0, 0.0
    for s in SEEDS:
        pairs, tbar = RR.refs_for_directory(TRAIN, rr, s)
        pol = V.load_policy(f"10x5+carbon+priority+s4t_spec_m_k125s060_s{s}@u1000_obsv2", "v2")
        gen = torch.Generator(device=V.P.DEV); gen.manual_seed(RNG_SEED + s)
        for mode in ("sampled", "greedy"):
            acc = {}
            for b, idx in enumerate(draws):
                refs = {"goal": "mct", "tbar": tbar, "mk": [pairs[i][0] for i in idx], "cb": [pairs[i][1] for i in idx]}
                o, fd = episode(pol, tuple([d[i] for i in idx] for d in data), [due[i] for i in idx], refs,
                                gen if mode == "sampled" else None, check=(mode == "greedy" and b < 2))
                worst_feat = max(worst_feat, fd)
                for k, v in o.items():
                    acc.setdefault(k, []).append(v)
            res[f"{s}_{mode}"] = {k: np.concatenate(v).tolist() for k, v in acc.items()}
            worst_err = max(worst_err, max(res[f"{s}_{mode}"]["err"]))
        print(f"seed {s}: max |return + score| so far {worst_err:.2e}; feature check max diff {worst_feat:.2e}", flush=True)
    (OUT / "reward_balance.json").write_text(json.dumps({"draws": [d.tolist() for d in draws], "runs": res,
                                                        "max_abs_return_plus_score": worst_err,
                                                        "feature_check_max_abs_diff": worst_feat}))
    print(f"A3 {'PASS' if worst_err < 1e-9 else 'FAIL'}: max |return + score| = {worst_err:.3g}; "
          f"features {'PASS' if worst_feat < 1e-9 else 'FAIL'} ({worst_feat:.3g})")


if __name__ == "__main__":
    main()
