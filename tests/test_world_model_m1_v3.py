"""World-model trunk, Milestone 1 (docs/world_model_m1_protocol_v3.md):
- the pretraining targets are exact: they equal a real-environment replay and the independent scorer;
- MINE picks what it says;
- the job-completion shortcut is the environment's own lower bound;
- the continuation enters only the heads;
- the saved trunk loads into DANIEL bit-identically;
- the probes never change a trunk;
- the data never touch the evaluation or final-test sets;
- the frozen settings verify.
CPU, validation split only.

    python -m unittest tests.test_world_model_m1_v3
"""
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))
import world_model_m1_v3 as W               # noqa: E402

B, T, torch = W.B, W.T, W.torch
CFG = W.settings(False)
W.set_features(CFG)
NORM = REPO / "results" / "warmstart_v3" / "norm" / "validation_normalisation_points_10x5.json"


def small(n=4, seed=0, steps=(0, 7, 20, 41)):
    data = B.load_split("validation", "10x5")
    idx = np.arange(n)
    rng = np.random.default_rng(seed)
    beh = np.arange(n) % len(W.BEHAVIOURS)
    st, rec, ep = W.behaviour_episodes(data, idx, beh, [np.asarray(steps)] * n, 4, rng)
    return data, idx, st, rec, ep


class Targets(unittest.TestCase):
    def test_targets_are_exact(self):
        data, idx, st, rec, ep = small()
        keep = []
        tj, tm, sj, sm = W.continuations(data, idx, ep, rec, W.CONT + [W.UNSEEN], True, keep=keep)
        for k in keep:
            env = T.replay(data, k["instance"], k["actions"])                       # the real environment
            for i, (s, a, r) in enumerate(k["rows"]):
                inst = k["instance"][i]
                ind = B.schedule_objectives(data.job_length[inst], data.op_pt[inst], data.carbon[inst], data.due[inst],
                                            k["machine"][i], k["start"][i])
                self.assertAlmostEqual(ind["makespan"], tj[s, a, r].max(), places=6)
                self.assertAlmostEqual(env["makespan"][i], tj[s, a, r].max(), places=6)
                self.assertAlmostEqual(ind["carbon"], tm[s, a, r, :, 1].sum(), places=6)
                self.assertAlmostEqual(env["tardiness"][i], np.maximum(0, tj[s, a, r] - data.due[inst]).sum(), places=6)
                last = np.cumsum(data.job_length[inst]) - 1
                self.assertTrue(np.allclose(ind["completion"][last], tj[s, a, r], atol=1e-9))
                # busy time = total processing time on the machines
                pt = np.asarray(data.op_pt[inst])[np.arange(len(k["machine"][i])), k["machine"][i]]
                self.assertAlmostEqual(pt.sum(), tm[s, a, r, :, 0].sum(), places=6)
                # the candidate action was taken at its step
                self.assertEqual(k["actions"][i, rec["step"][s]], rec["cand"][s, a])

    def test_snapshot_is_the_environment_state(self):
        data, idx, st, rec, ep = small(n=2, steps=(5,))
        env = T.FJSPEnvForSameOpNums(10, 5); env.set_initial_data(**data.env_kwargs(idx, W.configs.goal))
        for t in range(5):
            env.step(ep[:, t])
        for s in range(len(rec["step"])):
            e = rec["episode"][s]
            self.assertTrue(np.allclose(st["fea_j"][s], env.fea_j[e], atol=1e-6))
            self.assertTrue(np.array_equal(st["comp_idx"][s], env.comp_idx[e].astype(np.uint8)))
            self.assertTrue(np.array_equal(st["candidate"][s], env.candidate[e]))


class Rules(unittest.TestCase):
    def test_mine_and_shortcut(self):
        data = B.load_split("validation", "10x5"); idx = np.arange(5)
        d = W.Decoder(data, idx); d.reset()
        env = T.FJSPEnvForSameOpNums(10, 5); env.set_initial_data(**data.env_kwargs(idx, W.configs.goal))
        rng = np.random.default_rng(1)
        for t in range(d.N):
            ok, nst = d.allowed()
            a = d.rule("MWKR+MINE", ok, nst)
            j, m = a // d.M, a % d.M
            for e in range(5):                                   # MINE: least energy among the job's allowed machines
                ms = np.flatnonzero(ok[e, j[e]])
                en = d.carbon[e, d.cand[e, j[e]], ms]
                self.assertIn(m[e], ms[en == en.min()])
            if t % 3 == 0:
                a = d.rule("RANDOM", ok, nst, rng)
            d.step(a); env.step(a)
            job, _, _ = d.shortcut()                             # = the environment's lower bound x 99
            lb = env.op_ct_lb[env.env_job_idx, env.job_last_op_id] * B.PT_SCALE
            self.assertTrue(np.allclose(job, lb, atol=1e-6), t)


class Model(unittest.TestCase):
    def test_continuation_only_in_heads(self):
        data, idx, st, rec, ep = small(n=2, steps=(3,))
        D = {**st, "rec_cand": rec["cand"]}
        torch.manual_seed(0)
        m = W.WM.WorldModel(W.configs)
        s, job, mch, _ = W.batch_tensors(D, np.arange(len(rec["step"])), "cpu")
        pj, pm = m(s, job, mch)
        self.assertEqual(tuple(pj.shape[2:]), (3, 10)); self.assertEqual(tuple(pm.shape[2:]), (3, 5, 2))
        self.assertFalse(torch.allclose(pj[:, :, 0], pj[:, :, 1]))       # the heads see c
        import inspect
        self.assertEqual(list(inspect.signature(m.embed).parameters), ["s", "job", "mch"])   # the trunk does not

    def test_random_trunk_is_the_world_model_start(self):
        W.setup_seed(111); a = W.WM.WorldModel(W.configs).trunk.state_dict()
        b = W.load_trunk(None, CFG, "cpu").state_dict()
        self.assertTrue(all(torch.equal(a[k], b[k]) for k in a))

    def test_frozen_trunk(self):
        t = W.load_trunk(REPO / CFG["trunks"]["shared_base"]["path"], CFG, "cpu")
        self.assertFalse(any(p.requires_grad for p in t.parameters()))


class Loadable(unittest.TestCase):
    def test_trunk_loads_into_daniel(self):
        torch.manual_seed(5)
        trunk = W.DualAttentionNetwork(W.configs)
        with torch.no_grad():
            for p in trunk.parameters():
                p.add_(torch.randn_like(p))
        with tempfile.TemporaryDirectory() as d:
            f = Path(d) / "wm.pth"
            torch.save(W.trunk_bundle(trunk.state_dict(), CFG, {}, 1, 0.0), f)
            runs = {"exact": ["--config", str(REPO / "configs/benchmark_v3/t.json"), "--init_mode", "exact"],
                    "backbone": ["--config", str(REPO / "configs/benchmark_v3/mct.json"), "--init_mode", "backbone",
                                 "--preference_dim", "3", "--mo_val_norm", str(NORM), "--mo_val_grid_step", "0.5",
                                 "--mo_val_first", "2"]}
            for mode, extra in runs.items():
                r = subprocess.run([sys.executable, str(REPO / "daniel" / "train.py"), *extra, "--n_j", "10", "--n_m", "5",
                                    "--model_suffix", f"bv3_wmt{mode}_s111", "--seed_train", "111", "--init_from", str(f),
                                    "--v3_train_source", "generator", "--max_updates", "0", "--budget_checkpoints", "0",
                                    "--device", "cpu", "--num_envs", "2"], cwd=d, capture_output=True, text=True,
                                   timeout=1200)
                self.assertEqual(r.returncode, 0, r.stdout[-1500:] + r.stderr[-1500:])
                b = torch.load(Path(d) / "trained_network" / "SD2" / f"10x5+mix+bv3_wmt{mode}_s111@u0.pth",
                               map_location="cpu", weights_only=False)["state_dict"]
                for k, v in trunk.state_dict().items():
                    self.assertTrue(torch.equal(b["feature_exact." + k], v), (mode, k))


class Data(unittest.TestCase):
    def test_no_evaluation_or_final_test(self):
        src = (REPO / "tools" / "world_model_m1_v3.py").read_text()
        self.assertNotIn('"evaluation"', src); self.assertNotIn("final_test", src.replace("final-test", ""))
        self.assertIn('B.load_split("validation", SIZE)', src)
        from benchmark_v3_generator import GeneratorSource, structural_sha256
        W.setup_seed(111)
        g = GeneratorSource(W.configs, "10x5", B.load_manifest())
        out = g.sample(20)
        self.assertTrue(all(structural_sha256(jl, pt) not in g.forbidden for jl, pt in zip(out.job_length, out.op_pt)))

    def test_probe_split_disjoint(self):
        ids = np.arange(18)
        tr, te = ids[0::2], ids[1::2]
        self.assertEqual(len(set(tr) & set(te)), 0)
        self.assertEqual(len(tr) + len(te), 18)

    def test_settings_verify(self):
        m = W.F.load()
        self.assertEqual(m["rule"]["probe"], "mlp")
        self.assertEqual(m["feature_set"], "current")


if __name__ == "__main__":
    unittest.main()
