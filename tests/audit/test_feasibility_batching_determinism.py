"""Audit tests 1 (feasibility), 4 (batch invariance, eval and training batching) and 7 (determinism).

    python -m unittest tests.audit.test_feasibility_batching_determinism -v
Results are also written to results/code_audit/tests/*.json.
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import audit_common as A                               # noqa: E402
import numpy as np                                     # noqa: E402
import torch                                           # noqa: E402

CKPTS = [("m_s111 (v1 makespan specialist)", "10x5+carbon+priority+m_s111", "v1"),
         ("C1@1000 s111 (v1)", "10x5+carbon+priority+s4t_spec_m_k125s060_s111@u1000", "v1"),
         ("V2-M@3000 s111 (v2)", "v2rr/v2rr_V2-M_s111@u3000", "v2"),
         ("V2-S@3000 s111 (v2)", "v2rr/v2rr_V2-S_s111@u3000", "v2")]


def ub_of(data, i):
    return float(data[1][i][data[1][i] > 0].max())


class Feasibility(unittest.TestCase):
    def test_random_and_greedy_schedules_are_feasible(self):
        insts = A.SC.instances()
        data, due = A.pool()
        rng = np.random.default_rng(20261003)
        gen = torch.Generator(device=A.V.P.DEV); gen.manual_seed(20261003)
        n, bad, append_ok = 0, [], True
        for b in range(10):                                         # 10 batches x 100 = 1,000 random-policy episodes
            inst = rng.integers(0, len(data[0]), 100)
            for obs in (("v1",) if b % 2 == 0 else ("v2",)):
                o = A.rollout(None, data, due, inst, obs=obs, mode="random", gen=gen)
                self.assertEqual(o["steps"], o["n_ops"])
                for k, i in enumerate(inst):
                    viol = A.validate_schedule(insts[i], o["machine"][k], o["start"][k])
                    mk, cb, td = A.SC.raw_objectives(insts[i], due[i], o["machine"][k], o["start"][k])
                    if viol or abs(mk - o["mk"][k]) > 1e-9 or abs(cb - o["cb"][k]) > 1e-9 or abs(td - o["tard"][k]) > 1e-6:
                        bad.append((int(i), viol[:3], mk, o["mk"][k]))
                    n += 1
        greedy = {}
        for label, name, obs in CKPTS:
            pol = A.V.load_policy(name, obs)
            cnt = 0
            for u in sorted(set(ub_of(data, i) for i in A.HELD)):          # canonical grouping
                rows = np.array([i for i in A.HELD if ub_of(data, i) == u])
                o = A.rollout(pol, data, due, rows, obs=obs)
                for k, i in enumerate(rows):
                    viol = A.validate_schedule(insts[i], o["machine"][k], o["start"][k])
                    if viol:
                        bad.append((label, int(i), viol[:3]))
                    cnt += 1
            greedy[label] = cnt
        A.write_result("t1_feasibility", {"random_policy_episodes": n, "greedy_schedules": greedy,
                                          "violations": len(bad), "examples": bad[:5]})
        self.assertEqual(bad, [])


class BatchInvariance(unittest.TestCase):
    def test_eval_and_training_batching(self):
        data, due = A.pool()
        held = list(A.HELD)
        low = [i for i in held if ub_of(data, i) < 99][:5]                # instances whose own max pt < 99
        high = [i for i in held if ub_of(data, i) == 99][:5]
        targets = low + high
        rep = {"targets": [(int(i), ub_of(data, i)) for i in targets], "cases": {}}
        for label, name, obs in (CKPTS[1], CKPTS[2]):
            pol = A.V.load_policy(name, obs)
            res = []
            for i in targets:
                alone = A.rollout(pol, data, due, np.array([i]), obs=obs)
                same = [j for j in held if j != i and ub_of(data, j) == ub_of(data, i)][:9]
                with_same = A.rollout(pol, data, due, np.array([i] + same), obs=obs)
                c99 = [j for j in held if j != i and ub_of(data, j) == 99][:9]
                with_99 = A.rollout(pol, data, due, np.array([i] + c99), obs=obs)
                score = lambda o, k=0: A.SC.score(111, i, o["mk"][k], o["cb"][k], o["tard"][k])
                res.append(dict(inst=int(i), own_max=ub_of(data, i),
                                same_max_identical=bool((alone["actions"][:, 0] == with_same["actions"][:, 0]).all()),
                                max99_identical=bool((alone["actions"][:, 0] == with_99["actions"][:, 0]).all()),
                                score_alone=score(alone), score_with_99=score(with_99)))
            rep["cases"][label] = res
        # TRAINING batching path: Trainer draws 20 instances from the 63-file train pool and builds ONE env.
        tr, tr_due = A.pool("data_train")
        ubs = np.array([ub_of(tr, i) for i in range(len(tr[0]))])
        rng = np.random.default_rng(1)
        batch_ub = [max(ubs[rng.integers(0, len(ubs), 20)]) for _ in range(10000)]
        rep["training_pool"] = {"n": len(ubs), "own_max_counts": {str(u): int((ubs == u).sum()) for u in np.unique(ubs)},
                                "share_of_20_instance_batches_with_max_99": float(np.mean(np.array(batch_ub) == 99))}
        A.write_result("t4_batch_invariance", rep)
        for label, res in rep["cases"].items():
            for r in res:
                self.assertTrue(r["same_max_identical"], f"{label}: inst {r['inst']} changes with same-max companions")


class Determinism(unittest.TestCase):
    def test_same_checkpoint_same_scores(self):
        data, due = A.pool()
        out = {}
        for label, name, obs in CKPTS:
            pol = A.V.load_policy(name, obs)
            a = A.V.run(pol, data, due, A.HELD, obs=obs); b = A.V.run(A.V.load_policy(name, obs), data, due, A.HELD, obs=obs)
            out[label] = {k: float(np.abs(a[k] - b[k]).max()) for k in ("mk", "cb", "tard")}
        A.write_result("t7_determinism", out)
        for label, d in out.items():
            self.assertEqual(max(d.values()), 0.0, label)


if __name__ == "__main__":
    unittest.main()
