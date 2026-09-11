"""Weight-matching alignment over DANIEL's admissible permutation group.

Coordinate descent: hold every permutation fixed but one, solve that one exactly as a
linear assignment problem, iterate to a fixed point. The op/mch layer-0 variables live in a
wreath product S_32 wr S_4, solved in two stages (best within-head assignment for each head
pair -> 4x4 head assignment).

Objective: maximise <theta_A, Pi(theta_B)>, equivalently minimise ||theta_A - Pi(theta_B)||^2.
"""
from __future__ import annotations
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment as lap

from permutation_symmetry import (NH, D0, D1, HID, OP0, OP1, MCH0, MCH1,
                                  identity_perm, apply_perm, idx128)


def _np(x):
    return x.detach().cpu().numpy().astype(np.float64)


def align(A, B, iters=30, verbose=False):
    """Find P maximising <A, apply_perm(B, P)>. Returns (P, history)."""
    P = identity_perm()
    a = {k: _np(v) for k, v in A.items()}
    b = {k: _np(v) for k, v in B.items()}
    hist = []

    def opW(sd, blk, h, name="W"):
        return sd[f"{blk.format(h=h)}.{name}"]

    for it in range(iters):
        changed = False

        # ---------- actor / critic MLP hidden units ----------
        for tag, l0w, l0b, l1w, l1b, l2w, k1, k2, in_idx in [
            ("actor", "actor.linears.0.weight", "actor.linears.0.bias", "actor.linears.1.weight",
             "actor.linears.1.bias", "actor.linears.2.weight", "actor_h1", "actor_h2", None),
            ("critic", "critic.linears.0.weight", "critic.linears.0.bias", "critic.linears.1.weight",
             "critic.linears.1.bias", "critic.linears.2.weight", "critic_h1", "critic_h2", None)]:
            if tag == "actor":
                ii = np.concatenate([P["op1_chan"], D1 + P["mch1_chan"], 2 * D1 + P["op1_chan"],
                                     3 * D1 + P["mch1_chan"], np.arange(4 * D1, 4 * D1 + 9)])
            else:
                ii = np.concatenate([P["op1_chan"], D1 + P["mch1_chan"]])
            # h1: rows of L0, bias, columns of L1
            C = a[l0w] @ b[l0w][:, ii].T          # rows of layer 0
            C += np.outer(a[l0b], b[l0b])          # layer-0 bias
            C += a[l1w].T @ b[l1w][P[k2]]          # columns of layer 1
            r, c = lap(C, maximize=True)
            if not np.array_equal(P[k1], c): changed = True
            P[k1] = c
            # h2: rows of L1, bias, columns of L2
            C = a[l1w] @ b[l1w][:, P[k1]].T
            C += np.outer(a[l1b], b[l1b])
            C = C + a[l2w].T @ b[l2w]
            r, c = lap(C, maximize=True)
            if not np.array_equal(P[k2], c): changed = True
            P[k2] = c

        # ---------- layer-1 shared channel perms (op / mch) ----------
        i_op = idx128(P["op0_head"], P["op0_chan"])
        i_mch = idx128(P["mch0_head"], P["mch0_chan"])

        # op1_chan
        C = np.zeros((D1, D1))
        for h in range(NH):
            s = P["op1_head"][h]
            C += opW(a, OP1, h).T @ opW(b, OP1, s)[i_op]
            aa, bb = a[f"{OP1.format(h=h)}.a"][:, 0], b[f"{OP1.format(h=s)}.a"][:, 0]
            C += np.outer(aa[:D1], bb[:D1]) + np.outer(aa[D1:], bb[D1:])
        ii_a = P["actor_h1"]; ii_c = P["critic_h1"]
        C += a["actor.linears.0.weight"][:, :D1].T @ b["actor.linears.0.weight"][ii_a][:, :D1]
        C += a["actor.linears.0.weight"][:, 2 * D1:3 * D1].T @ b["actor.linears.0.weight"][ii_a][:, 2 * D1:3 * D1]
        C += a["critic.linears.0.weight"][:, :D1].T @ b["critic.linears.0.weight"][ii_c][:, :D1]
        r, c = lap(C, maximize=True)
        if not np.array_equal(P["op1_chan"], c): changed = True
        P["op1_chan"] = c

        # mch1_chan
        C = np.zeros((D1, D1))
        for h in range(NH):
            s = P["mch1_head"][h]
            C += opW(a, MCH1, h).T @ opW(b, MCH1, s)[i_mch]
            C += opW(a, MCH1, h, "W_edge").T @ opW(b, MCH1, s, "W_edge")[i_op]
            aa, bb = a[f"{MCH1.format(h=h)}.a"][:, 0], b[f"{MCH1.format(h=s)}.a"][:, 0]
            for t in range(3):
                C += np.outer(aa[t * D1:(t + 1) * D1], bb[t * D1:(t + 1) * D1])
        C += a["actor.linears.0.weight"][:, D1:2 * D1].T @ b["actor.linears.0.weight"][ii_a][:, D1:2 * D1]
        C += a["actor.linears.0.weight"][:, 3 * D1:4 * D1].T @ b["actor.linears.0.weight"][ii_a][:, 3 * D1:4 * D1]
        C += a["critic.linears.0.weight"][:, D1:].T @ b["critic.linears.0.weight"][ii_c][:, D1:]
        r, c = lap(C, maximize=True)
        if not np.array_equal(P["mch1_chan"], c): changed = True
        P["mch1_chan"] = c

        # ---------- layer-1 head order (free: mean aggregation) ----------
        for blk, key, chan, extra in [(OP1, "op1_head", "op1_chan", None),
                                      (MCH1, "mch1_head", "mch1_chan", "W_edge")]:
            c1 = P[chan]
            C = np.zeros((NH, NH))
            for h in range(NH):
                for s in range(NH):
                    rows = i_op if blk is OP1 else i_mch
                    v = float((opW(a, blk, h) * opW(b, blk, s)[rows][:, c1]).sum())
                    if extra:
                        v += float((opW(a, blk, h, "W_edge") * opW(b, blk, s, "W_edge")[i_op][:, c1]).sum())
                    aa, bb = a[f"{blk.format(h=h)}.a"][:, 0], b[f"{blk.format(h=s)}.a"][:, 0]
                    nb = 2 if blk is OP1 else 3
                    for t in range(nb):
                        v += float(aa[t * D1:(t + 1) * D1] @ bb[t * D1:(t + 1) * D1][c1])
                    C[h, s] = v
            r, c = lap(C, maximize=True)
            if not np.array_equal(P[key], c): changed = True
            P[key] = c

        # ---------- layer-0 wreath products ----------
        # op layer 0: upstream W,a ; downstream op L1 W rows and mch L1 W_edge rows
        cost = np.zeros((NH, NH)); assign = {}
        for hA in range(NH):
            for sB in range(NH):
                M = opW(a, OP0, hA).T @ opW(b, OP0, sB)
                aa, bb = a[f"{OP0.format(h=hA)}.a"][:, 0], b[f"{OP0.format(h=sB)}.a"][:, 0]
                M = M + np.outer(aa[:D0], bb[:D0]) + np.outer(aa[D0:], bb[D0:])
                for h in range(NH):
                    s = P["op1_head"][h]
                    M += opW(a, OP1, h)[hA * D0:(hA + 1) * D0] @ \
                         opW(b, OP1, s)[sB * D0:(sB + 1) * D0][:, P["op1_chan"]].T
                    sm = P["mch1_head"][h]
                    M += opW(a, MCH1, h, "W_edge")[hA * D0:(hA + 1) * D0] @ \
                         opW(b, MCH1, sm, "W_edge")[sB * D0:(sB + 1) * D0][:, P["mch1_chan"]].T
                rr, cc = lap(M, maximize=True)
                cost[hA, sB] = M[rr, cc].sum(); assign[(hA, sB)] = cc
        r, c = lap(cost, maximize=True)
        if not np.array_equal(P["op0_head"], c): changed = True
        P["op0_head"] = c
        P["op0_chan"] = [assign[(h, c[h])] for h in range(NH)]

        # mch layer 0: upstream W, W_edge, a ; downstream mch L1 W rows
        cost = np.zeros((NH, NH)); assign = {}
        for hA in range(NH):
            for sB in range(NH):
                M = opW(a, MCH0, hA).T @ opW(b, MCH0, sB)
                M = M + opW(a, MCH0, hA, "W_edge").T @ opW(b, MCH0, sB, "W_edge")
                aa, bb = a[f"{MCH0.format(h=hA)}.a"][:, 0], b[f"{MCH0.format(h=sB)}.a"][:, 0]
                for t in range(3):
                    M = M + np.outer(aa[t * D0:(t + 1) * D0], bb[t * D0:(t + 1) * D0])
                for h in range(NH):
                    s = P["mch1_head"][h]
                    M += opW(a, MCH1, h)[hA * D0:(hA + 1) * D0] @ \
                         opW(b, MCH1, s)[sB * D0:(sB + 1) * D0][:, P["mch1_chan"]].T
                rr, cc = lap(M, maximize=True)
                cost[hA, sB] = M[rr, cc].sum(); assign[(hA, sB)] = cc
        r, c = lap(cost, maximize=True)
        if not np.array_equal(P["mch0_head"], c): changed = True
        P["mch0_head"] = c
        P["mch0_chan"] = [assign[(h, c[h])] for h in range(NH)]

        d = float(sum(((A[k] - apply_perm(B, P)[k]) ** 2).sum() for k in A))
        hist.append(d)
        if verbose:
            print(f"    iter {it}: sq-dist {d:.4f}")
        if not changed:
            break
    return P, hist
