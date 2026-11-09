"""Tardiness diagnosis (docs/tardiness_diagnosis_protocol_v3.md): the lean no-waiting decoder allows exactly the pairs
the environment allows, key decoding replays schedules exactly, and the rules pick what they say. Evaluation split,
first instances only; never the final-test sets.

    python -m unittest tests.test_tardiness_diagnosis_v3
"""
import sys
import unittest
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))
import tardiness_diagnosis_v3 as T          # noqa: E402

B = T.B


def lockstep(size, policy, n=6):
    """Runs the decoder and the real environment side by side; returns the number of mask disagreements."""
    data = B.load_split("evaluation", size)
    idx = np.arange(n)
    d = T.NonDelay(data, idx); d.reset()
    env = T.FJSPEnvForSameOpNums(*B.SIZES[size]); env.set_initial_data(**data.env_kwargs(idx, "t"))
    bad = 0
    for _ in range(d.N):
        ok, nst = d.allowed()
        bad += int((ok != ~env.dynamic_pair_mask.astype(bool)).sum())
        a = policy(d, ok, nst)
        d.step(a); env.step(a)
    return bad, d, env


class Mask(unittest.TestCase):
    def test_same_mask_as_environment(self):
        for size in ("10x5", "20x10"):
            keys = np.random.default_rng(1).random((6, *B.load_split("evaluation", size).op_pt[0].shape))
            for name, pol in [("keys", T.key_policy(keys))] + [(r, T.rule_policy(r)) for r in T.RULES]:
                bad, d, env = lockstep(size, pol)
                self.assertEqual(bad, 0, f"{size} {name}")
                comp = env.true_op_ct[env.env_job_idx, env.job_last_op_id]
                self.assertTrue(np.allclose(comp, d.rr, atol=0, rtol=0), f"{size} {name}")    # bit-identical times


class Keys(unittest.TestCase):
    def test_action_keys_replay_exactly(self):
        data = B.load_split("evaluation", "10x5"); idx = np.arange(5)
        d = T.NonDelay(data, idx)
        a = d.run(T.key_policy(np.random.default_rng(3).random((5, d.N, d.M))))
        k = T.keys_from_actions(a["actions"], T.op_sequence(d, a), d.M)
        b = T.NonDelay(data, idx).run(T.key_policy(k))
        self.assertTrue(np.array_equal(a["actions"], b["actions"]))

    def test_committed_daniel_schedules_are_reproduced(self):
        for size in ("10x5", "20x10"):
            data = B.load_split("evaluation", size); idx = np.arange(5)
            spec, _ = T.specialist(size, 111, "greedy")
            d = T.NonDelay(data, idx)
            r = d.run(T.key_policy(T.keys_from_schedule(spec["machine"][idx], spec["start"][idx], d.M)))
            self.assertTrue(np.allclose(r["tardiness"], spec["tardiness"][idx], atol=1e-9), size)
            self.assertTrue(np.array_equal(r["machine"], spec["machine"][idx]), size)
            T.check(data, idx, r, "DANIEL-T")                     # environment and independent scorer agree


class Rules(unittest.TestCase):
    def test_rules_pick_what_they_say(self):
        data = B.load_split("evaluation", "10x5"); idx = np.arange(6)
        d = T.NonDelay(data, idx); d.reset()
        ok, nst = d.allowed()
        a = T.rule_policy("EDD+ECT")(d, ok, nst)
        j, m = a // d.M, a % d.M
        for e in range(6):
            can = np.flatnonzero(ok[e].any(1))
            self.assertEqual(j[e], can[np.argmin(d.due[e, can])])
            ms = np.flatnonzero(ok[e, j[e]])
            self.assertEqual(m[e], ms[np.argmin(d.pt[e, d.cand[e, j[e]], ms])])
        # least slack at the first step: due - 0 - remaining minimum work
        a = T.rule_policy("LST+ECT")(d, ok, nst)
        work = np.stack([np.add.reduceat(d.min_pt[e], d.first[e]) for e in range(6)])
        for e in range(6):
            can = np.flatnonzero(ok[e].any(1))
            self.assertEqual(a[e] // d.M, can[np.argmin((d.due[e] / B.PT_SCALE - work[e])[can])])


if __name__ == "__main__":
    unittest.main()
