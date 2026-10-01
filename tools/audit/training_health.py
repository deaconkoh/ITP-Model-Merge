"""Audit Part 4: training health reconstructed from saved snapshots (results/code_audit.md). NO optimiser steps.

Entropy, KL, clip fraction, explained variance and gradient norms were never logged by any run (train.py logs only
the mean training return per update and validation metrics; the v2 re-run logs only chunk timings). For each snapshot
this script rolls out 100 training-generator episodes (5 fixed batches of 20 from the 63-file train pool, sampled from
the policy, as PPO's rollouts are) under the arm's own reward and measures:
  entropy        mean per-decision policy entropy (nats);
  return         mean undiscounted episode return (the 'training return');
  EV             critic explained variance, 1 - Var(G_t - V(s_t)) / Var(G_t), G_t = reward-to-go;
  |r|, sd(G)     mean absolute per-step reward and the spread of the value targets;
  grad norms     of the first PPO minibatch loss terms (ratio = 1 at the snapshot) w.r.t. the SHARED feature
                 extractor, the actor and the critic: 0.5 * value loss, policy loss, 0.01 * entropy bonus;
  KL             mean KL(pi_prev || pi_this) between consecutive snapshots on the previous snapshot's states.
Arms (seed 111): V2-M, V2-S, V2-T (v2 reward), Step 4 makespan arm and scratch (v1 mct reward, Step 4 weights), and
the v1 makespan specialist m_s111 under DANIEL's own makespan-only reward (the original method's regime).

    python tools/audit/training_health.py
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
from torch.distributions import Categorical             # noqa: E402
from model.PPO import Memory                            # noqa: E402
from common_utils import sample_action, eval_actions    # noqa: E402
from objectives import ObjectiveSpec                    # noqa: E402
import reward_refs as RR                                # noqa: E402

SEED = 111
V2U = [0, 50, 100, 250, 500, 1000, 1500, 2000, 2500, 3000]
S4U = [0, 50, 100, 250, 500, 1000]
ARMS = {"V2-M": ("v2", "mct", [f"v2rr/v2rr_V2-M_s111@u{u}" for u in V2U], V2U),
        "V2-S": ("v2", "mct", [f"v2rr/v2rr_V2-S_s111@u{u}" for u in V2U], V2U),
        "V2-T": ("v2", "mct", [f"v2rr/v2rr_V2-T_s111@u{u}" for u in V2U], V2U),
        "Step4 spec_m (v1)": ("v1", "mct", [f"10x5+carbon+priority+s4t_spec_m_k125s060_s111@u{u}" for u in S4U], S4U),
        "Step4 scratch (v1)": ("v1", "mct", [f"10x5+carbon+priority+s4t_scratch_k125s060_s111@u{u}" for u in S4U], S4U),
        "m_s111, makespan-only reward (DANIEL regime)": ("v1", "m", ["10x5+carbon+priority+m_s111"], [1000])}


def episodes(pol, obs, goal, data, due, batches, refs):
    """Sampled rollouts storing PPO memory; returns per-batch Memory objects and envs."""
    out = []
    for b, idx in enumerate(batches):
        sub = tuple([d[i] for i in idx] for d in data)
        rr = None
        if obs == "v2":
            rr = {"goal": goal, "tbar": refs[1], "mk": [refs[0][i][0] for i in idx], "cb": [refs[0][i][1] for i in idx]}
        env = A.FJSPEnvForSameOpNums(10, 5, obs_version=obs, reward_version=obs)
        if obs == "v1":
            env.objective = ObjectiveSpec(goal, carbon_weight=0.002384, tardiness_weight=5.815362)
        state = env.set_initial_data(*[list(x) for x in sub], due_date_list=[due[i] for i in idx], reward_refs=rr)
        mem = Memory(gamma=1.0, gae_lambda=0.98)
        torch.manual_seed(1000 + b)
        while True:
            mem.push(state)
            with torch.no_grad():
                pi, v = pol(**A.V.P.args_of(state))
            a, lp = sample_action(pi)
            state, r, done = env.step(actions=a.cpu().numpy())
            mem.done_seq.append(torch.from_numpy(done)); mem.reward_seq.append(torch.from_numpy(r).to(v.device))
            mem.action_seq.append(a); mem.log_probs.append(lp); mem.val_seq.append(v.squeeze(1))
            if done.all():
                break
        out.append(mem)
    return out


def measure(pol, mems):
    ent, ret, ev_num, ev_den, absr, sdG, grads = [], [], [], [], [], [], []
    for mem in mems:
        t = mem.transpose_data()
        R = torch.stack(mem.reward_seq, 0).double()                         # [T, E]
        G = torch.flip(torch.cumsum(torch.flip(R, [0]), 0), [0])            # reward-to-go
        Vv = mem.t_old_val_seq.transpose(0, 1).double()
        ret.append(float(R.sum(0).mean())); absr.append(float(R.abs().mean())); sdG.append(float(G.std()))
        ev_num.append(float(((G - Vv) ** 2).mean())); ev_den.append(float(G.var()))
        adv, vt = mem.get_gae_advantages()
        pol.zero_grad()
        pis, vals = pol(fea_j=t[0], op_mask=t[1], candidate=t[6], fea_m=t[2], mch_mask=t[3], comp_idx=t[5],
                        dynamic_pair_mask=t[4], fea_pairs=t[7])
        lp, e = eval_actions(pis, t[8])
        ent.append(float(e))
        ratio = torch.exp(lp - t[12].detach())
        terms = {"value(0.5*MSE)": 0.5 * F.mse_loss(vals.squeeze(1), vt),
                 "policy": (-torch.min(ratio * adv, torch.clamp(ratio, 0.8, 1.2) * adv)).mean(),
                 "entropy(0.01*H)": -0.01 * e}
        g = {}
        for name, loss in terms.items():
            pol.zero_grad()
            loss.backward(retain_graph=True)
            for part in ("feature_exact", "actor", "critic"):
                ps = [p.grad for n, p in pol.named_parameters() if n.startswith(part) and p.grad is not None]
                g[f"{name}|{part}"] = float(torch.sqrt(sum((x ** 2).sum() for x in ps))) if ps else 0.0
        grads.append(g)
    G = {k: float(np.mean([g[k] for g in grads])) for k in grads[0]}
    return dict(entropy=float(np.mean(ent)), train_return=float(np.mean(ret)), mean_abs_step_reward=float(np.mean(absr)),
                sd_value_target=float(np.mean(sdG)), explained_variance=float(1 - np.sum(ev_num) / np.sum(ev_den)), grad=G)


@torch.no_grad()
def kl(prev, cur, mems):
    out = []
    for mem in mems:
        t = mem.transpose_data()
        kw = dict(fea_j=t[0], op_mask=t[1], candidate=t[6], fea_m=t[2], mch_mask=t[3], comp_idx=t[5],
                  dynamic_pair_mask=t[4], fea_pairs=t[7])
        lp = F.log_softmax(prev.forward_logits(**kw)[0], 1); lq = F.log_softmax(cur.forward_logits(**kw)[0], 1)
        p = lp.exp()
        out.append(float(torch.where(p > 0, p * (lp - lq), torch.zeros_like(p)).sum(1).mean()))
    return float(np.mean(out))


def main():
    data, due = A.pool("data_train")
    rng = np.random.default_rng(20261005)
    batches = [rng.integers(0, len(data[0]), 20) for _ in range(5)]
    refs = RR.refs_for_directory(A.DAN / "data/data_train/SD2/10x5+carbon+priority", RR.load_manifest(), SEED)
    res = {}
    for arm, (obs, goal, names, ups) in ARMS.items():
        prev, prev_mems = None, None
        for name, u in zip(names, ups):
            pol = A.V.load_policy(name, obs)
            for p in pol.parameters():
                p.requires_grad_(True)
            mems = episodes(pol, obs, goal, data, due, batches, refs)
            m = measure(pol, mems)
            m["kl_from_previous_snapshot"] = kl(prev, pol, prev_mems) if prev is not None else None
            res.setdefault(arm, {})[u] = m
            g = m["grad"]
            print(f"{arm:<44} u{u:<5} H {m['entropy']:.3f}  ret {m['train_return']:+.3f}  EV {m['explained_variance']:+.2f}"
                  f"  |r| {m['mean_abs_step_reward']:.4f} sd(G) {m['sd_value_target']:.3f}  "
                  f"grad(feature extractor) value {g['value(0.5*MSE)|feature_exact']:.3g} policy {g['policy|feature_exact']:.3g}"
                  + (f"  KL {m['kl_from_previous_snapshot']:.3f}" if m['kl_from_previous_snapshot'] is not None else ""), flush=True)
            prev, prev_mems = pol, mems
    A.write_result("training_health", res)


if __name__ == "__main__":
    main()
