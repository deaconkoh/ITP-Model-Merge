"""CP-SAT restricted to DANIEL's rule (tools/nondelay_cpsat_v3.py, docs/decoding_ceiling_protocol_v3.md, Part B):
- a schedule made by DANIEL's rule is feasible when fixed in the restricted model;
- a non-delay violation (one operation delayed while its machine idles) makes a fixed schedule infeasible, while the
  unrestricted model still accepts it;
- a short restricted solve returns a schedule that DANIEL's rule reproduces exactly;
- the integer objective and the proven lower bound are consistent with the exact score.
Evaluation split, first instances, CPU; never the final-test sets.

    python -m unittest tests.test_nondelay_cpsat_v3
"""
import sys
import unittest
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))
import nondelay_cpsat_v3 as N              # noqa: E402
import tardiness_diagnosis_v3 as TD        # noqa: E402
import benchmark_v3 as B                   # noqa: E402
from ortools.sat.python import cp_model    # noqa: E402


def setUpModule():
    global data, rule
    data = B.load_split("evaluation", "10x5")
    rule = TD.NonDelay(data, np.arange(2)).run(TD.rule_policy("EDD+ECT"))


def status(inst, m, s, nondelay=True):
    md, _ = N.build(inst, fix=(m, s), nondelay=nondelay)
    sv = cp_model.CpSolver(); sv.parameters.max_time_in_seconds = 30; sv.parameters.num_workers = 1
    return sv.StatusName(sv.Solve(md))


class Model(unittest.TestCase):
    def test_daniel_schedule_feasible(self):
        for r in range(2):
            self.assertEqual(status(N.Inst(data, r), rule["machine"][r], rule["start"][r]), "OPTIMAL")

    def test_delay_infeasible(self):
        inst = N.Inst(data, 0)
        m, s = rule["machine"][0].copy(), rule["start"][0].copy()
        last = int(np.argmax(s))
        s[last] += 5
        self.assertEqual(status(inst, m, s), "INFEASIBLE")
        self.assertEqual(status(inst, m, s, nondelay=False), "OPTIMAL")

    def test_solve_reproduced_and_bound(self):
        inst = N.Inst(data, 1)
        w = N.PREFS[41]
        r = N.solve(inst, w, 10, 4, hint=(rule["machine"][1], rule["start"][1]))
        self.assertIn(r["status"], ("OPTIMAL", "FEASIBLE"))
        m, s = np.asarray(r["machine"])[None], np.asarray(r["start"], float)[None]
        res = TD.NonDelay(data, np.array([1])).run(TD.key_policy(TD.keys_from_schedule(m, s, 5)))
        self.assertTrue(np.array_equal(res["machine"], m) and np.allclose(res["start"], s))
        self.assertLessEqual(r["lower_bound"], r["score"] + 1e-9)
        hint_score = inst.score(inst.exact(rule["machine"][1], rule["start"][1]), w)
        self.assertLessEqual(r["score"], hint_score + 1e-6)


if __name__ == "__main__":
    unittest.main()
