"""Audit tests 2 (log-prob consistency, rollout vs update path), 3 (GAE vs a reference) and 5 (reward sums).

    python -m unittest tests.audit.test_ppo_reward -v
Results are also written to results/code_audit/tests/*.json.
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import audit_common as A                               # noqa: E402
import numpy as np                                     # noqa: E402
import torch                                           # noqa: E402
from model.PPO import Memory                           # noqa: E402
from model.main_model import DANIEL                    # noqa: E402
from common_utils import sample_action, eval_actions   # noqa: E402
from objectives import ObjectiveSpec                   # noqa: E402
import reward_refs as RR                               # noqa: E402

WC, WT = 0.002384, 5.815362


def train_batches(n_batches=5, size=20, seed=20261004):
    data, due = A.pool("data_train")
    rng = np.random.default_rng(seed)
    return data, due, [rng.integers(0, len(data[0]), size) for _ in range(n_batches)]


class LogProbConsistency(unittest.TestCase):
    def test_rollout_vs_update_path(self):
        data, due, batches = train_batches(2)
        out = {}
        for label, name, obs in (("v1 C1@1000", "10x5+carbon+priority+s4t_spec_m_k125s060_s111@u1000", "v1"),
                                 ("v2 V2-M@1000", "v2rr/v2rr_V2-M_s111@u1000", "v2")):
            pol = A.V.load_policy(name, obs)               # rollout policy (= policy_old before any update)
            mem = Memory(gamma=1.0, gae_lambda=0.98)
            idx = batches[0]
            sub = tuple([d[i] for i in idx] for d in data)
            env = A.FJSPEnvForSameOpNums(10, 5, obs_version=obs, reward_version="v1")
            state = env.set_initial_data(*[list(x) for x in sub], due_date_list=[due[i] for i in idx])
            torch.manual_seed(0)
            while True:                                    # exactly the rollout code of train.Trainer.train
                mem.push(state)
                with torch.no_grad():
                    pi, v = pol(**A.V.P.args_of(state))
                a, lp = sample_action(pi)
                state, r, done = env.step(actions=a.cpu().numpy())
                mem.done_seq.append(torch.from_numpy(done)); mem.reward_seq.append(torch.from_numpy(r))
                mem.action_seq.append(a); mem.log_probs.append(lp); mem.val_seq.append(v.squeeze(1))
                if done.all():
                    break
            t = mem.transpose_data()                       # exactly the update code path of PPO.update
            with torch.no_grad():
                pis, vals = pol(fea_j=t[0], op_mask=t[1], candidate=t[6], fea_m=t[2], mch_mask=t[3], comp_idx=t[5],
                                dynamic_pair_mask=t[4], fea_pairs=t[7])
                lp2, ent = eval_actions(pis, t[8])
            d = float((lp2 - t[12]).abs().max())
            ratio = float((torch.exp(lp2 - t[12]) - 1).abs().max())
            # entropy over valid actions only: masked actions have probability exactly 0
            masked_p = float(pis[t[4].reshape(pis.shape[0], -1)].abs().max())
            out[label] = {"transitions": int(t[12].numel()), "max_abs_logprob_diff": d, "max_abs_ratio_minus_1": ratio,
                          "max_prob_on_masked_actions": masked_p, "mean_entropy_nats": float(ent)}
        A.write_result("t2_logprob_consistency", out)
        for label, r in out.items():
            self.assertLess(r["max_abs_logprob_diff"], 1e-4, label)
            self.assertEqual(r["max_prob_on_masked_actions"], 0.0, label)


def reference_gae(r, v, done, gamma, lam):
    """Standard GAE with episode ends: r, v, done [E, T]; bootstrap 0 after a done step and at the buffer end."""
    E, T = r.shape
    adv = np.zeros((E, T)); last = np.zeros(E)
    for t in reversed(range(T)):
        nonterm = 1.0 - done[:, t]
        nv = v[:, t + 1] if t + 1 < T else np.zeros(E)
        delta = r[:, t] + gamma * nv * nonterm - v[:, t]
        last = delta + gamma * lam * nonterm * last
        adv[:, t] = last
    return adv


class GAE(unittest.TestCase):
    def test_against_reference(self):
        rng = np.random.default_rng(3)
        E, T = 4, 12
        r = rng.normal(size=(E, T)); v = rng.normal(size=(E, T))
        out = {}
        for case, done in (("episodes end at the last step (the only case training produces)",
                            np.eye(1, T, T - 1).repeat(E, 0)),
                           ("an episode ends mid-buffer (never produced: all environments end together)",
                            np.where(np.arange(T)[None] == 5, 1.0, 0.0).repeat(E, 0) + np.eye(1, T, T - 1).repeat(E, 0))):
            m = Memory(gamma=1.0, gae_lambda=0.98)
            m.reward_seq = [torch.tensor(r[:, t]) for t in range(T)]
            m.t_old_val_seq = torch.tensor(v, dtype=torch.float32)
            _, vt = m.get_gae_advantages()
            adv_impl = vt.reshape(E, T).numpy() - v                       # v_target = advantage + value
            ref = reference_gae(r, v, done, 1.0, 0.98)
            out[case] = float(np.abs(adv_impl - ref).max())
        A.write_result("t3_gae", out)
        vals = list(out.values())
        self.assertLess(vals[0], 1e-5)                                    # the case that occurs in training
        self.assertGreater(vals[1], 1e-3)                                 # documents: done flags are ignored


class RewardSums(unittest.TestCase):
    def test_per_step_rewards_sum_to_objective(self):
        data, due, batches = train_batches(5)
        insts = {i: A.SC.Instance(f) for i, f in enumerate(sorted((A.DAN / "data/data_train/SD2/10x5+carbon+priority").glob("*.fjs")))}
        man = RR.load_manifest()
        pairs, tbar = RR.refs_for_directory(A.DAN / "data/data_train/SD2/10x5+carbon+priority", man, 111)
        pol1 = A.V.load_policy("10x5+carbon+priority+s4t_spec_m_k125s060_s111@u1000", "v1")
        pol2 = A.V.load_policy("v2rr/v2rr_V2-M_s111@u1000", "v2")
        worst = {"v1 mct (Step 4 weights)": 0.0, "v1 m": 0.0, "v1 t": 0.0, "v2 mct": 0.0, "v2 t": 0.0}
        n = 0
        gen = torch.Generator(device=A.V.P.DEV); gen.manual_seed(5)
        for idx in batches:
            for key in worst:
                v2 = key.startswith("v2"); goal = key.split()[1]
                refs = ({"goal": goal, "tbar": tbar, "mk": [pairs[i][0] for i in idx], "cb": [pairs[i][1] for i in idx]}
                        if v2 else None)
                if not v2:
                    spec = ObjectiveSpec(goal, carbon_weight=WC, tardiness_weight=WT if goal == "mct" else 1.0)
                    orig = A.FJSPEnvForSameOpNums.__init__

                    def patched(self, *a, **k):
                        orig(self, *a, **k); self.objective = spec
                    A.FJSPEnvForSameOpNums.__init__ = patched
                try:
                    o = A.rollout(pol2 if v2 else pol1, data, due, idx, obs="v2" if v2 else "v1", mode="sample",
                                  gen=gen, reward_refs=refs, reward_version="v2" if v2 else "v1")
                finally:
                    if not v2:
                        A.FJSPEnvForSameOpNums.__init__ = orig
                R = o["rewards"].sum(0)
                for k, i in enumerate(idx):
                    ins = insts[i]
                    mk, cb, td = A.SC.raw_objectives(ins, due[i], o["machine"][k], o["start"][k])   # independent
                    U = o["ub"]
                    lb0 = max(sum(min(ins.pt[q].values()) for q in ops) for ops in ins.job_ops) / U
                    if v2:
                        w = {"mct": (1 / 3, 1 / 3, 1 / 3), "t": (0, 0, 1)}[goal]
                        target = -(w[0] * mk / pairs[i][0] + w[1] * cb / pairs[i][1] + w[2] * td / tbar)
                    else:
                        target = {"mct": (lb0 - mk / U) - WC * cb - WT * td / U, "m": lb0 - mk / U, "t": -td / U}[goal]
                    worst[key] = max(worst[key], abs(R[k] - target))
                    n += 1
        A.write_result("t5_reward_sums", {"episodes_per_variant": n // len(worst), "max_abs_error": worst})
        for key, e in worst.items():
            self.assertLess(e, 1e-9, key)


if __name__ == "__main__":
    unittest.main()
