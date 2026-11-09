"""World-model trunk (docs/world_model_m1_protocol_v3.md): action-conditioned outcome prediction on top of DANIEL's
dual-attention encoder.

The TRUNK is exactly DANIEL's encoder (`DualAttentionNetwork`, the specialists' `feature_exact`), so a trained trunk
loads into any v3 DANIEL. It sees the state only; the candidate action (operation of job j_a, machine m_a) is read
out of its embeddings the same way DANIEL's actor reads an action:

    a = [h_op(candidate operation of j_a), h_mch(m_a), g_op, g_mch]            (4 x 8 = 32 values)

The continuation rule c (one-hot) enters ONLY the heads:
    job head      [h_op(candidate operation of job j), a, c] -> completion time of job j        (every job)
    machine head  [h_mch(m), a, c]                           -> busy time, energy of machine m  (every machine)
"""
from __future__ import annotations
import torch
import torch.nn as nn

from model.main_model import DualAttentionNetwork

N_RULES = 3


def mlp(i, h, o):
    return nn.Sequential(nn.Linear(i, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU(), nn.Linear(h, o))


def trunk_forward(trunk, s):
    """s: dict of batched state tensors. Returns per-operation [B, N, d], per-machine [B, M, d], globals [B, d] x 2."""
    return trunk(s["fea_j"], s["op_mask"], s["candidate"], s["fea_m"], s["mch_mask"], s["comp_idx"])


def readout(h_op, h_mch, g_op, g_mch, candidate, job, mch):
    """State-action embedding a [B, A, 4d] for actions (job [B, A], mch [B, A]); also the per-job candidate embeddings
    [B, J, d]."""
    d = h_op.shape[-1]
    h_job = torch.gather(h_op, 1, candidate.long().unsqueeze(-1).expand(-1, -1, d))           # [B, J, d]
    A = job.shape[1]
    a_op = torch.gather(h_job, 1, job.long().clamp(min=0).unsqueeze(-1).expand(-1, -1, d))     # [B, A, d]
    a_m = torch.gather(h_mch, 1, mch.long().clamp(min=0).unsqueeze(-1).expand(-1, -1, d))
    g = torch.cat([g_op, g_mch], -1).unsqueeze(1).expand(-1, A, -1)
    return torch.cat([a_op, a_m, g], -1), h_job


class WorldModel(nn.Module):
    def __init__(self, config, hidden=64):
        super().__init__()
        self.trunk = DualAttentionNetwork(config)
        d = config.layer_fea_output_dim[-1]
        self.d = d
        self.job_head = mlp(d + 4 * d + N_RULES, hidden, 1)
        self.mch_head = mlp(d + 4 * d + N_RULES, hidden, 2)

    def embed(self, s, job, mch):
        h_op, h_mch, g_op, g_mch = trunk_forward(self.trunk, s)
        a, h_job = readout(h_op, h_mch, g_op, g_mch, s["candidate"], job, mch)
        return a, h_job, h_mch

    def forward(self, s, job, mch):
        """Predictions for every (action, rule): jobs [B, A, R, J], machines [B, A, R, M, 2] (normalised units)."""
        a, h_job, h_mch = self.embed(s, job, mch)
        B, A, _ = a.shape
        J, M = h_job.shape[1], h_mch.shape[1]
        c = torch.eye(N_RULES, device=a.device)                                              # [R, R]
        aa = a[:, :, None, None, :]                                                          # [B, A, 1, 1, 4d]
        cc = c[None, None, :, None, :]                                                       # [1, 1, R, 1, R]
        xj = torch.cat([h_job[:, None, None].expand(B, A, N_RULES, J, self.d), aa.expand(B, A, N_RULES, J, 4 * self.d),
                        cc.expand(B, A, N_RULES, J, N_RULES)], -1)
        xm = torch.cat([h_mch[:, None, None].expand(B, A, N_RULES, M, self.d), aa.expand(B, A, N_RULES, M, 4 * self.d),
                        cc.expand(B, A, N_RULES, M, N_RULES)], -1)
        return self.job_head(xj).squeeze(-1), self.mch_head(xm)


def loss(pred, target, valid):
    """Huber (delta 1) on normalised targets, averaged over the 20 outputs of every valid (state, action, rule).
    target: jobs [B, A, R, J], machines [B, A, R, M, 2]; valid: [B, A] (padding actions are -1)."""
    pj, pm = pred
    tj, tm = target
    hj = nn.functional.huber_loss(pj, tj, reduction="none", delta=1.0).sum(-1)               # [B, A, R]
    hm = nn.functional.huber_loss(pm, tm, reduction="none", delta=1.0).sum((-1, -2))
    n_out = tj.shape[-1] + tm.shape[-2] * tm.shape[-1]
    per = (hj + hm) / n_out
    w = valid[:, :, None].float().expand_as(per)
    return (per * w).sum() / w.sum().clamp(min=1)
