"""Permutation symmetry group of the DANIEL architecture: apply, verify, and align.

Symmetry variables (all function-preserving when applied jointly):
  op0_head  in S_4    op layer-0 head order            (heads are CONCATENATED)
  op0_chan  4 x S_32  per-head output channel perm     (independent per head)
  mch0_head in S_4    mch layer-0 head order
  mch0_chan 4 x S_32  per-head output channel perm
  op1_head  in S_4    op layer-1 head order            (heads are MEAN-averaged -> free)
  op1_chan  in S_8    SHARED across the 4 heads        (mean aggregation forces sharing)
  mch1_head in S_4
  mch1_chan in S_8    SHARED across the 4 heads
  actor_h1/h2, critic_h1/h2 in S_64  MLP hidden units

Constraints honoured (from the Step 0 audit):
  * every `a` vector is co-permuted blockwise with its head's channel permutation
    (op: 2 blocks, mch: 3 blocks) -- they carry no independent freedom
  * layer-1 channel perms are SHARED across heads, because mean aggregation is only
    permutation-equivariant if all heads agree
  * the op layer-0 output feeds TWO consumers, op_blocks.1.W and mch_blocks.1.W_edge,
    which therefore take the SAME 128-dim row permutation
  * the actor input is grouped [op_emb(8) | mch_emb(8) | op_global(8) | mch_global(8) |
    pair(9)]; the two op groups share op1_chan, the two mch groups share mch1_chan, and
    the 9 raw pair features are pinned
  * pinned interfaces are never permuted: op L0 W rows (11 raw op features), mch L0 W rows
    (8 machine features), mch L0 W_edge rows (11 raw op features), actor pair columns,
    both output units

Convention throughout: new[i] = src[perm[i]].
"""
from __future__ import annotations
import numpy as np
import torch

NH = 4          # heads per block
D0 = 32         # layer-0 output dim per head
D1 = 8          # layer-1 output dim per head
HID = 64        # actor/critic hidden width

OP0 = "feature_exact.op_attention_blocks.0.attention_{h}"
OP1 = "feature_exact.op_attention_blocks.1.attention_{h}"
MCH0 = "feature_exact.mch_attention_blocks.0.attention_{h}"
MCH1 = "feature_exact.mch_attention_blocks.1.attention_{h}"


def identity_perm():
    return {
        "op0_head": np.arange(NH), "op0_chan": [np.arange(D0) for _ in range(NH)],
        "mch0_head": np.arange(NH), "mch0_chan": [np.arange(D0) for _ in range(NH)],
        "op1_head": np.arange(NH), "op1_chan": np.arange(D1),
        "mch1_head": np.arange(NH), "mch1_chan": np.arange(D1),
        "actor_h1": np.arange(HID), "actor_h2": np.arange(HID),
        "critic_h1": np.arange(HID), "critic_h2": np.arange(HID),
    }


def random_perm(rng):
    return {
        "op0_head": rng.permutation(NH), "op0_chan": [rng.permutation(D0) for _ in range(NH)],
        "mch0_head": rng.permutation(NH), "mch0_chan": [rng.permutation(D0) for _ in range(NH)],
        "op1_head": rng.permutation(NH), "op1_chan": rng.permutation(D1),
        "mch1_head": rng.permutation(NH), "mch1_chan": rng.permutation(D1),
        "actor_h1": rng.permutation(HID), "actor_h2": rng.permutation(HID),
        "critic_h1": rng.permutation(HID), "critic_h2": rng.permutation(HID),
    }


def idx128(head_perm, chan_perms):
    """Row index map for a 128-dim concatenated layer-0 output."""
    return np.concatenate([head_perm[b] * D0 + chan_perms[b] for b in range(NH)])


def apply_perm(sd, P):
    """Return a new state_dict with the permutation applied. Function-preserving."""
    out = {k: v.clone() for k, v in sd.items()}
    i_op = idx128(P["op0_head"], P["op0_chan"])
    i_mch = idx128(P["mch0_head"], P["mch0_chan"])

    for h in range(NH):
        # ---- op layer 0: pick source head, permute its output channels
        s = P["op0_head"][h]; c = P["op0_chan"][h]
        out[f"{OP0.format(h=h)}.W"] = sd[f"{OP0.format(h=s)}.W"][:, c]
        a = sd[f"{OP0.format(h=s)}.a"]
        out[f"{OP0.format(h=h)}.a"] = torch.cat([a[:D0][c], a[D0:][c]], 0)

        # ---- mch layer 0: W and W_edge share the same output-channel perm
        s = P["mch0_head"][h]; c = P["mch0_chan"][h]
        out[f"{MCH0.format(h=h)}.W"] = sd[f"{MCH0.format(h=s)}.W"][:, c]
        out[f"{MCH0.format(h=h)}.W_edge"] = sd[f"{MCH0.format(h=s)}.W_edge"][:, c]
        a = sd[f"{MCH0.format(h=s)}.a"]
        out[f"{MCH0.format(h=h)}.a"] = torch.cat([a[:D0][c], a[D0:2 * D0][c], a[2 * D0:][c]], 0)

        # ---- op layer 1: rows follow op L0 output; columns use the SHARED op1_chan
        s = P["op1_head"][h]; c1 = P["op1_chan"]
        out[f"{OP1.format(h=h)}.W"] = sd[f"{OP1.format(h=s)}.W"][i_op][:, c1]
        a = sd[f"{OP1.format(h=s)}.a"]
        out[f"{OP1.format(h=h)}.a"] = torch.cat([a[:D1][c1], a[D1:][c1]], 0)

        # ---- mch layer 1: W rows follow mch L0, W_edge rows follow OP L0 (the coupling)
        s = P["mch1_head"][h]; c1 = P["mch1_chan"]
        out[f"{MCH1.format(h=h)}.W"] = sd[f"{MCH1.format(h=s)}.W"][i_mch][:, c1]
        out[f"{MCH1.format(h=h)}.W_edge"] = sd[f"{MCH1.format(h=s)}.W_edge"][i_op][:, c1]
        a = sd[f"{MCH1.format(h=s)}.a"]
        out[f"{MCH1.format(h=h)}.a"] = torch.cat([a[:D1][c1], a[D1:2 * D1][c1], a[2 * D1:][c1]], 0)

    # ---- actor: input groups [op(8) | mch(8) | op_global(8) | mch_global(8) | pair(9 pinned)]
    op1, mch1 = P["op1_chan"], P["mch1_chan"]
    i41 = np.concatenate([op1, D1 + mch1, 2 * D1 + op1, 3 * D1 + mch1, np.arange(4 * D1, 4 * D1 + 9)])
    r1, r2 = P["actor_h1"], P["actor_h2"]
    out["actor.linears.0.weight"] = sd["actor.linears.0.weight"][r1][:, i41]
    out["actor.linears.0.bias"] = sd["actor.linears.0.bias"][r1]
    out["actor.linears.1.weight"] = sd["actor.linears.1.weight"][r2][:, r1]
    out["actor.linears.1.bias"] = sd["actor.linears.1.bias"][r2]
    out["actor.linears.2.weight"] = sd["actor.linears.2.weight"][:, r2]

    # ---- critic: input groups [op_global(8) | mch_global(8)]
    i16 = np.concatenate([op1, D1 + mch1])
    r1, r2 = P["critic_h1"], P["critic_h2"]
    out["critic.linears.0.weight"] = sd["critic.linears.0.weight"][r1][:, i16]
    out["critic.linears.0.bias"] = sd["critic.linears.0.bias"][r1]
    out["critic.linears.1.weight"] = sd["critic.linears.1.weight"][r2][:, r1]
    out["critic.linears.1.bias"] = sd["critic.linears.1.bias"][r2]
    out["critic.linears.2.weight"] = sd["critic.linears.2.weight"][:, r2]
    return out


# ------------------------------------------------------------------ layer groups
def layer_groups(sd):
    """Group parameter names into reportable blocks."""
    g = {"op_attn_L0": [], "op_attn_L1": [], "mch_attn_L0": [], "mch_attn_L1": [],
         "actor_mlp": [], "critic_mlp": []}
    for k in sd:
        if "op_attention_blocks.0" in k: g["op_attn_L0"].append(k)
        elif "op_attention_blocks.1" in k: g["op_attn_L1"].append(k)
        elif "mch_attention_blocks.0" in k: g["mch_attn_L0"].append(k)
        elif "mch_attention_blocks.1" in k: g["mch_attn_L1"].append(k)
        elif k.startswith("actor."): g["actor_mlp"].append(k)
        elif k.startswith("critic."): g["critic_mlp"].append(k)
    return g


def sq_dist(a, b, keys):
    return float(sum(((a[k] - b[k]) ** 2).sum() for k in keys))
