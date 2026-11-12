"""Decoding strategies (tools/decoding_v3.py, docs/decoding_ceiling_protocol_v3.md):
- a copied simulator row is independent and behaves exactly like the original row (beam and lookahead branches);
- beam search of width 1 reproduces greedy decoding exactly;
- lookahead with greedy continuations is never worse than greedy for any preference (rollout improvement), and its
  executed schedule is its final incumbent;
- the merge key of a partial schedule does not depend on the order in which its operations were placed.
CPU, validation split, World-model base mix final checkpoint; never the final-test sets.

    python -m unittest tests.test_decoding_ceiling_v3
"""
import sys
import unittest
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))
import decoding_v3 as D                    # noqa: E402
import benchmark_v3 as B                   # noqa: E402


def setUpModule():
    global data, net
    data = B.load_split("validation", "10x5")
    net = D.load_net(REPO / D.CKPTS["wm"][1])


def per_env(env):
    out = {}
    for k, v in vars(env).items():
        if isinstance(v, np.ndarray) and v.ndim > 0 and v.shape[0] == env.number_of_envs and k != "op_idx":
            out[k] = np.ma.getdata(v)
    for sk in ("state",):
        for k, v in vars(getattr(env, sk)).items():
            if isinstance(v, torch.Tensor) and v.ndim > 0:
                out[f"{sk}.{k}"] = v.cpu().numpy()
    return out


def legal_random(env, rng):
    m = ~env.dynamic_pair_mask.reshape(env.number_of_envs, -1)
    return np.array([rng.choice(np.flatnonzero(r)) for r in m])


class Copies(unittest.TestCase):
    def test_select_is_exact_and_independent(self):
        rng = np.random.default_rng(0)
        prefs = D.PREFS[[0, 10, 30, 65]]
        a = D.make_env(data, [0, 1, 2, 3], prefs)
        acts = []
        for _ in range(12):
            x = legal_random(a, rng); acts.append(x); D.step(a, x)
        rows = [2, 0, 2]
        b = D.select(a, rows)
        before = {k: v.copy() for k, v in per_env(a).items()}
        # continue the copies with different actions per duplicate; the original must not change
        cont = []
        for _ in range(20):
            x = legal_random(b, rng); cont.append(x); D.step(b, x)
        for k, v in per_env(a).items():
            self.assertTrue(np.array_equal(v, before[k]), k)
        # a fresh environment replaying the same full action sequence matches the copies exactly
        c = D.make_env(data, [[0, 1, 2, 3][r] for r in rows], prefs[rows])
        for x in acts:
            D.step(c, x[rows])
        for x in cont:
            D.step(c, x)
        pb, pc = per_env(b), per_env(c)
        for k in pc:
            self.assertTrue(np.array_equal(pb[k], pc[k]), k)


class Decoders(unittest.TestCase):
    def test_beam1_is_greedy(self):
        idx = [0] * 6 + [1] * 6
        prefs = np.tile(D.PREFS[[0, 5, 20, 40, 60, 66]], (2, 1))
        g = D.greedy(net, data, idx, prefs, D.Cost(len(idx)))
        b = D.beam(net, data, idx, prefs, 1, D.Cost(len(idx)))
        self.assertTrue(np.array_equal(g[0], b[0]))
        self.assertTrue(np.array_equal(g[1], b[1]))

    def test_lookahead_not_worse_than_greedy(self):
        idx = [0] * 4 + [3] * 4
        prefs = np.tile(D.PREFS[[0, 9, 33, 65]], (2, 1))
        g = D.greedy(net, data, idx, prefs, D.Cost(len(idx)))
        la = D.lookahead(net, data, idx, prefs, 2, 0, 52, D.Cost(len(idx)))   # asserts executed == incumbent
        self.assertTrue(np.all(D.score(data, idx, la[0], prefs) <= D.score(data, idx, g[0], prefs) + 1e-12))

    def test_merge_key_order_free(self):
        op, m, st = np.array([3, 7, 11]), np.array([0, 2, 1]), np.array([0.0, 5.0, 12.0])
        z = D.zobrist(op, m, st)
        self.assertEqual(z[0] ^ z[1] ^ z[2], z[2] ^ z[0] ^ z[1])
        self.assertEqual(len(set(D.zobrist(np.arange(50), np.zeros(50, int), np.zeros(50)).tolist())), 50)


if __name__ == "__main__":
    unittest.main()
