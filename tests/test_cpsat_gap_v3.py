"""CP-SAT gap analysis (tools/cpsat_gap_v3_analysis.py): the definitions behave as stated.
- schedules made by DANIEL's decision rule have no deliberate waiting, and the rule reproduces them from their keys;
- delaying one ready operation creates deliberate waiting;
- a left shift keeps every machine and machine order, stays feasible and is never worse;
- the fixed-assignment non-delay construction keeps every machine and never idles a machine while an operation
  assigned to it is ready.
Evaluation split, first 3 instances, CPU; never the final-test sets.

    python -m unittest tests.test_cpsat_gap_v3
"""
import json
import sys
import unittest
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools")); sys.path.insert(0, str(REPO / "daniel"))
import tardiness_diagnosis_v3 as TD        # noqa: E402  (sets up params for the CPU)
import benchmark_v3 as B                   # noqa: E402
import cpsat_gap_v3_analysis as AN         # noqa: E402

B1 = REPO / "results" / "baselines_v3" / "cpsat" / "b1" / "evaluation_10x5"


def setUpModule():
    global data, insts, rule, cp
    data = B.load_split("evaluation", "10x5")
    insts = [AN.Inst(data, i) for i in range(3)]
    rule = TD.NonDelay(data, np.arange(3)).run(TD.rule_policy("EDD+ECT"))
    cp = []
    for x in insts:
        t = json.loads((B1 / f"{Path(x.key).stem}.json").read_text())["points"]["tard"]["final"]
        cp.append((np.asarray(t["machine"]), np.asarray(t["start"], float)))


class Definitions(unittest.TestCase):
    def test_union(self):
        self.assertEqual(AN.union_len([(0, 2), (1, 3), (5, 6)]), 4)
        self.assertEqual(AN.union_len([]), 0)

    def test_rule_schedules_are_nondelay_and_reproduced(self):
        for r, x in enumerate(insts):
            self.assertTrue(AN.is_nondelay(x, rule["machine"][r], rule["start"][r]))
        ok = AN.env_reproduces(data, np.arange(3), rule["machine"], rule["start"])
        self.assertTrue(ok.all())

    def test_delay_creates_deliberate_waiting(self):
        x = insts[0]
        m, s = rule["machine"][0].copy(), rule["start"][0].copy()
        last = int(np.argmax(s))                       # the last operation to start: nothing depends on it
        s[last] += 7.0
        B.schedule_objectives(x.jl, x.pt, x.carbon, x.due, m, s)          # still feasible
        ia = AN.idle_analysis(x, m, s)
        self.assertGreater(ia["deliberate"], 6.9)
        self.assertGreater(ia["deliberate_same"], 6.9)
        self.assertFalse(AN.env_reproduces(data, np.array([0]), m[None], s[None]).any())

    def test_left_shift(self):
        for x, (m, s) in zip(insts, cp):
            s2 = AN.left_shift(x, m, s)
            o, o2 = x.objectives(m, s), x.objectives(m, s2)                 # feasible (raises otherwise)
            self.assertTrue(np.all(o2 <= o + 1e-9))
            self.assertEqual(o2[1], o[1])                                   # carbon: assignment only
            self.assertTrue(np.all(s2 <= s + 1e-9))
            for mm in range(x.m):
                on = np.flatnonzero(m == mm)
                self.assertEqual(list(on[np.argsort(s[on], kind="stable")]), list(on[np.argsort(s2[on], kind="stable")]))
            self.assertEqual(AN.left_shift(x, m, s2).tolist(), s2.tolist())          # idempotent

    def test_nondelay_fixed(self):
        for x, (m, s) in zip(insts, cp):
            s3 = AN.nondelay_fixed(x, m, s)
            o3 = x.objectives(m, s3)
            self.assertAlmostEqual(o3[1], x.objectives(m, s)[1])
            ia = AN.idle_analysis(x, m, s3)
            self.assertLessEqual(ia["deliberate_same"], 1e-6)


if __name__ == "__main__":
    unittest.main()
