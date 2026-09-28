"""Input and reward audit (results/input_reward_audit.md). READ-ONLY: changes no environment, feature, reward or model
code; evaluation rollouts only. Reserved final-test sets are never loaded.

Part 4 (reward composition): C1@1000 (s4t_spec_m_k125s060_s<seed>@u1000) rolled out on the 10x5 TRAIN pool the way
training does it -- 5 batches of 20 environments, instances drawn with replacement (numpy default_rng(20260929)),
batch-max processing-time normalisation, actions SAMPLED from the policy (as the PPO rollouts), plus a greedy pass on
the same batches. The environment's reward is computed by its own code; the objective object is wrapped only to
record the three terms it is handed (makespan delta, chosen carbon, tardiness delta) with Step 4's weights
(goal mct, carbon_reward_weight 0.002384, tardiness_reward_weight 5.815362). Also records raw pair-feature ranges.

Part 5 (behaviour): per job on the 49 HELD-OUT instances, due date, completion and tardiness under C1@1000 greedy
(results/ceiling_check/raw/c1_greedy_s*.npz) and CP-SAT 60 s best (results/ceiling_check/raw/cpsat_t60), tightness
d_j / P_j, Spearman(tightness, tardiness).

    python tools/audit_input_reward.py      # writes results/input_reward_audit/{reward_terms.json, jobs.csv, ...}
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))
import complementarity as P                                     # noqa: E402  canonical config + policy loader
import torch                                                    # noqa: E402
import torch.nn.functional as F                                 # noqa: E402
from scipy import stats                                         # noqa: E402
from fjsp_env_same_op_nums import FJSPEnvForSameOpNums          # noqa: E402
from objectives import ObjectiveSpec                            # noqa: E402
from data_utils import load_priority_carbon_data_from_files     # noqa: E402
from due_dates import due_dates_for_directory, load_manifest, work_content   # noqa: E402
import ceiling_scorer as SC                                     # noqa: E402

OUT = REPO / "results/input_reward_audit"
TRAIN = REPO / "daniel/data/data_train/SD2/10x5+carbon+priority"
WC, WT = 0.002384, 5.815362
SEEDS = [111, 222, 333, 444]
N_BATCH, BATCH = 5, 20
RNG_SEED = 20260929


class Recorder:
    """Delegates to Step 4's ObjectiveSpec; records the per-step terms it is handed."""

    def __init__(self):
        self.spec = ObjectiveSpec("mct", carbon_weight=WC, tardiness_weight=WT)
        self.goal, self.uses_tardiness, self.rec = self.spec.goal, True, []

    def step_reward(self, mk_d, carbon, prio_d, tard_d):
        r = self.spec.step_reward(mk_d, carbon, prio_d, tard_d)
        self.rec.append((np.array(mk_d, float), np.array(carbon, float), np.array(tard_d, float), np.array(r, float)))
        return r


@torch.no_grad()
def episode(pol, data, dues, sample, gen):
    env = FJSPEnvForSameOpNums(n_j=10, n_m=5)
    env.objective = Recorder()
    state = env.set_initial_data(*[list(d) for d in data], due_date_list=list(dues))
    pair = []
    while True:
        pair.append(state.fea_pairs_tensor.cpu().numpy()[~state.dynamic_pair_mask_tensor.cpu().numpy()])
        s = pol.forward_logits(**P.args_of(state))[0]
        a = (torch.multinomial(F.softmax(s, 1), 1, generator=gen).squeeze(1) if sample else s.argmax(1)).cpu().numpy()
        state, _, done = env.step(actions=a)
        if np.all(done):
            break
    R = env.objective.rec
    mk_d, cb, td, r = (np.stack([x[i] for x in R]) for i in range(4))      # [T, E]
    T_true = np.maximum(0.0, env.true_op_ct[env.env_job_idx, env.job_last_op_id] - np.asarray(dues)).sum(1)
    ub = env.pt_upper_bound
    return dict(R_m=mk_d.sum(0), R_c=-WC * cb.sum(0), R_t=WT * td.sum(0), R=r.sum(0), ub=np.full(len(T_true), ub),
                mk=env.current_makespan.copy(), mk_over_ub=env.current_makespan / ub,
                lb0_over_ub=env.old_init_quality.copy(), cb=env.total_carbon.copy(), T=T_true,
                shaping_err=WT * td.sum(0) - (-WT * T_true / ub)), np.concatenate(pair)


def part4():
    data = load_priority_carbon_data_from_files(str(TRAIN))
    due = due_dates_for_directory(TRAIN, load_manifest())                  # hash-checked
    rng = np.random.default_rng(RNG_SEED)
    draws = [rng.integers(0, len(data[0]), BATCH) for _ in range(N_BATCH)]
    res, pairs = {}, []
    for s in SEEDS:
        pol = P.load_policy(f"10x5+carbon+priority+s4t_spec_m_k125s060_s{s}@u1000")
        gen = torch.Generator(device=P.DEV); gen.manual_seed(RNG_SEED + s)
        for mode in ("sampled", "greedy"):
            acc = {}
            for idx in draws:
                o, pf = episode(pol, tuple([d[i] for i in idx] for d in data), [due[i] for i in idx], mode == "sampled", gen)
                pairs.append(pf)
                for k, v in o.items():
                    acc.setdefault(k, []).append(v)
            res[f"{s}_{mode}"] = {k: np.concatenate(v).tolist() for k, v in acc.items()}
    pf = np.concatenate(pairs)
    feat = {"names": ["pt", "pt/op_max_pt", "pt/mch_max_candidate_pt", "pt/max_remain_pt", "pt/mch_max_remain_pt",
                      "pt/pair_max_pt", "pt/job_remain_work", "pair_wait_time", "carbon (raw)"],
            "min": pf.min(0).tolist(), "max": pf.max(0).tolist(), "mean": pf.mean(0).tolist(), "std": pf.std(0).tolist()}
    return res, feat, [d.tolist() for d in draws]


def part5():
    insts, dues = SC.instances(), SC.due_dates()
    rows = []
    for s in SEEDS:
        g = np.load(SC.OUT / "raw" / f"c1_greedy_s{s}.npz")
        for k, i in enumerate(SC.HELD):
            i = int(i); ins = insts[i]
            r = json.loads((SC.OUT / "raw" / "cpsat_t60" / f"s{s}_i{i}.json").read_text())
            pt_mat = np.zeros((ins.n_ops, ins.n_m))
            for o, d in enumerate(ins.pt):
                for m, p in d.items():
                    pt_mat[o, m] = p
            Pj = work_content(np.array([len(o) for o in ins.job_ops]), pt_mat)
            for lab, mach, st in (("policy", g["machine"][k], g["start"][k]), ("cpsat", r["machine"], r["start"])):
                for j, ops in enumerate(ins.job_ops):
                    o = ops[-1]; C = float(st[o]) + ins.pt[o][int(mach[o])]
                    rows.append((s, i, j, lab, float(dues[i][j]), float(Pj[j]), C, max(0.0, C - float(dues[i][j]))))
    return rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    res, feat, draws = part4()
    (OUT / "reward_terms.json").write_text(json.dumps({"weights": {"carbon": WC, "tardiness": WT}, "draws": draws,
                                                       "runs": res, "pair_features": feat}))
    rows = part5()
    with open(OUT / "jobs.csv", "w") as f:
        f.write("seed,pool_idx,job,schedule,due,P_j,completion,tardiness\n")
        for r in rows:
            f.write(",".join(str(x) for x in r) + "\n")
    print("AUDIT_RAW_DONE")


if __name__ == "__main__":
    main()
