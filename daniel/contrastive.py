"""Contrastive (pairwise-ranking, DPO-style) fine-tuning for the preference-conditioned DANIEL
(docs/contrastive_protocol_v3.md). Replaces only the PPO update; train.py --finetune_objective contrastive.

Per update: each training instance gets one preference w and K schedules sampled from the current policy (one group).
Pairs (better, worse) inside a group come from
  (a) Pareto dominance on the raw objectives (valid for every w), and
  (b) the Tchebycheff score g(x|w) = max_k w_k f_k(x) on the training-normalised objectives (ideal point 0), only when
      g(worse) - g(better) > margin.
Loss, averaged over all pairs of the update: -log sigmoid(beta * [(l_i - l_i^ref) - (l_j - l_j^ref)]), l = sum_t
log pi(a_t | s_t) over the full schedule, with the starting network frozen as the reference. The gradient is computed
in two passes (trajectory log-probabilities without gradient, then the per-state log-probabilities in minibatches,
weighted by dL/dl of their trajectory), which equals the one-pass gradient because the network has no dropout.
"""
from __future__ import annotations
import numpy as np
import torch

from common_utils import sample_action

KEYS = ("fea_j", "op_mask", "candidate", "fea_m", "mch_mask", "comp_idx", "dynamic_pair_mask", "fea_pairs", "pref")
DOM_TOL = 1e-9


def group_pairs(obj, f_hat, w, margin):
    """Pairs (better, worse) of one group. obj: raw objectives [K, 3]; f_hat: normalised objectives [K, 3]; w: [3].
    Returns (pairs, n_dominance, n_tchebycheff); a pair found by both sources counts once (as dominance)."""
    obj, f_hat, w = np.asarray(obj, float), np.asarray(f_hat, float), np.asarray(w, float)
    g = (w[None, :] * f_hat).max(1)
    pairs, n_dom, n_tch = [], 0, 0
    for i in range(len(obj)):
        for j in range(len(obj)):
            if i == j:
                continue
            if np.all(obj[i] <= obj[j] + DOM_TOL) and np.any(obj[i] < obj[j] - DOM_TOL):
                pairs.append((i, j)); n_dom += 1
            elif g[j] - g[i] > margin:
                pairs.append((i, j)); n_tch += 1
    return pairs, n_dom, n_tch


def dpo_loss(ell, ell_ref, pairs, beta):
    """Mean pairwise logistic loss over pairs [(better, worse)] (global trajectory indices)."""
    b = torch.as_tensor([p[0] for p in pairs], device=ell.device)
    w = torch.as_tensor([p[1] for p in pairs], device=ell.device)
    z = beta * ((ell[b] - ell_ref[b]) - (ell[w] - ell_ref[w]))
    return -torch.nn.functional.logsigmoid(z).mean()


def policy_inputs(state):
    return {k: getattr(state, f"{k}_tensor") for k in KEYS}


@torch.no_grad()
def rollout(policy, ref, env, state):
    """Sample one schedule per environment from `policy`; per step the sampled action, its log-probability under the
    policy and under the frozen reference, and the policy entropy. States are kept for the gradient pass.
    ref None (no anchor, docs/merge_failure_test_protocol_v3.md): reference log-probabilities are 0, so the loss becomes
    -log sigmoid(beta * (l_better - l_worse))."""
    steps = []
    while True:
        x = policy_inputs(state)
        pi, _ = policy(**x)
        a, lp = sample_action(pi)
        if ref is None:
            lr = torch.zeros_like(lp)
        else:
            pr, _ = ref(**x)
            lr = torch.log(pr.gather(1, a[:, None]).squeeze(1).clamp_min(1e-30))
        ent = -(pi * torch.log(pi.clamp_min(1e-30))).sum(1)
        steps.append({"x": x, "a": a, "lp": lp, "lr": lr, "ent": ent})
        state, _, done = env.step(a.cpu().numpy())
        if done.all():
            return steps


def stack(steps):
    """Flatten [T steps][N envs] into trajectory-major [N * T] tensors; traj[k] = environment index of row k."""
    T, N = len(steps), len(steps[0]["a"])
    flat = {k: torch.stack([s["x"][k] for s in steps], 1).flatten(0, 1) for k in KEYS}
    acts = torch.stack([s["a"] for s in steps], 1)                                   # [N, T]
    traj = torch.arange(N, device=acts.device).repeat_interleave(T)
    return flat, acts.flatten(), traj, acts


def dpo_update(policy, optimizer, steps, pairs, beta, clip, minibatch):
    """One gradient step on the mean DPO loss over `pairs` (global trajectory indices). Returns (loss, grad norm
    before clipping); (None, None) when there are no pairs (no step)."""
    if not pairs:
        return None, None
    ell = torch.stack([s["lp"] for s in steps], 1).sum(1)                          # [N] sampling-time log-prob
    ell_ref = torch.stack([s["lr"] for s in steps], 1).sum(1)
    ell_v = ell.detach().clone().requires_grad_(True)
    loss = dpo_loss(ell_v, ell_ref, pairs, beta)
    loss.backward()
    coef = ell_v.grad.detach()                                                       # dL/dl per trajectory
    flat, a, traj, _ = stack(steps)
    optimizer.zero_grad()
    for s in range(0, len(a), minibatch):
        e = min(s + minibatch, len(a))
        pi, _ = policy(**{k: v[s:e] for k, v in flat.items()})
        lp = torch.log(pi.gather(1, a[s:e, None]).squeeze(1).clamp_min(1e-30))
        (coef[traj[s:e]] * lp).sum().backward()
    norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), clip)
    optimizer.step()
    return float(loss.detach()), float(norm)


def group_stats(acts, obj, k):
    """Per group of k consecutive environments: distinct action sequences and distinct objective vectors."""
    seqs = acts.cpu().numpy()
    n_groups = len(seqs) // k
    sched = [len({seqs[g * k + r].tobytes() for r in range(k)}) for g in range(n_groups)]
    objs = [len({tuple(np.round(obj[g * k + r], 9)) for r in range(k)}) for g in range(n_groups)]
    return float(np.mean(sched)), float(np.mean(objs))
