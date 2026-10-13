"""Benchmark v3 (docs/benchmark_v3.md): frozen parameters reproduce, objectives / observations / rewards are correct,
guards hold. Runs on CPU; reads only the benchmark v3 tree, the DEV instances and data_train (never final test).

    python -m unittest tests.test_benchmark_v3
"""
import copy
import importlib.util
import json
import sys
import unittest
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
DANIEL_DIR = REPO / "daniel"
GOALS = ("m", "c", "t", "mc", "mt", "ct", "mct")


def setUpModule():
    sys.path.insert(0, str(DANIEL_DIR))
    sys.path.insert(0, str(REPO / "tools" / "objective_validation"))
    if "params" not in sys.modules:                  # params parses argv on import; keep the v1 defaults
        argv, sys.argv = sys.argv, ["benchmark-v3-test", "--device", "cpu"]
        import params                                # noqa: F401
        sys.argv = argv
    global B
    import benchmark_v3 as B


def random_rollout(env, rng):
    """Uniformly random valid actions; returns (return per env, machine per op [E, N])."""
    ret = np.zeros(env.number_of_envs)
    machine = np.full((env.number_of_envs, env.number_of_ops), -1)
    while True:
        # the mask the policy sees (reset() restores the state object, not env.dynamic_pair_mask)
        mask = env.state.dynamic_pair_mask_tensor.cpu().numpy().reshape(env.number_of_envs, -1)
        a = np.array([rng.choice(np.flatnonzero(~m)) for m in mask])
        op = env.candidate[env.env_idxs, a // env.number_of_machines]
        machine[env.env_idxs, op] = a % env.number_of_machines
        _, r, done = env.step(a)
        ret += r
        if done.all():
            return ret, machine


class Definitions(unittest.TestCase):
    def test_carbon_formula(self):
        pt = np.array([[10, 0], [0, 60]]); power = np.array([6.0, 12.0])
        np.testing.assert_allclose(B.carbon_matrix(pt, power), [[0.54 * 6 * 10 / 60, 0], [0, 0.54 * 12 * 60 / 60]])
        self.assertAlmostEqual(B.CARBON_SCALE, 0.54 * 15 * 99 / 60)

    def test_due_dates_twk(self):
        jl = np.array([2, 1]); pt = np.array([[10, 20], [30, 0], [0, 8]])     # op means 15, 30, 8 -> W = 45, 8
        v = np.array([0.0, 1.0])
        np.testing.assert_allclose(B.work_content(jl, pt), [45, 8])
        k, s = B.DUE["10x5"]
        np.testing.assert_allclose(B.due_dates(jl, pt, v, "10x5"), [k * (1 - s / 2) * 45, k * (1 + s / 2) * 8])

    def test_keyed_draws(self):
        np.testing.assert_array_equal(B.machine_power("abc", 5), B.machine_power("abc", 5))
        self.assertFalse(np.array_equal(B.keyed_rng("abc", "power").uniform(size=5),
                                        B.keyed_rng("abc", "kdraw").uniform(size=5)))
        p = B.machine_power("abc", 1000)
        self.assertTrue((p >= 4).all() and (p < 15).all())

    def test_goal_weights(self):
        self.assertEqual(B.goal_weights("mct"), (1 / 3, 1 / 3, 1 / 3))
        self.assertEqual(B.goal_weights("ct"), (0.0, 0.5, 0.5))
        for bad in ("p", "mcp", "mm", ""):
            with self.assertRaises(ValueError):
                B.goal_weights(bad)

    def test_independent_scorer_rejects_infeasible(self):
        jl = np.array([2]); pt = np.array([[5, 0], [3, 4]]); c = B.carbon_matrix(pt, [5, 5]); d = np.array([10.0])
        ok = B.schedule_objectives(jl, pt, c, d, [0, 1], [0, 5])
        self.assertEqual((ok["makespan"], ok["tardiness"]), (9.0, 0.0))
        for machine, start in (([1, 1], [0, 5]), ([0, 1], [0, 4]), ([0, 0], [0, 4]), ([0, 1], [-1, 5])):
            with self.assertRaises(ValueError):
                B.schedule_objectives(jl, pt, c, d, machine, start)
        jl2 = np.array([1, 1]); pt2 = np.array([[5], [5]])
        with self.assertRaisesRegex(ValueError, "overlap"):
            B.schedule_objectives(jl2, pt2, B.carbon_matrix(pt2, [5]), np.zeros(2), [0, 0], [0, 3])


class FrozenManifest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.man = B.load_manifest()

    def test_every_parameter_regenerates_exactly(self):
        for key, rec in self.man["instances"].items():
            path = B.DATA_ROOT / key
            self.assertEqual(B.file_sha256(path), rec["sha256"], key)
            p = B.generate_parameters(path, rec["size"])
            self.assertEqual(p["structural_sha256"], rec["structural_sha256"])
            for f in ("power_kw", "kdraw_v", "k_j", "work_content", "due_dates"):
                np.testing.assert_array_equal(p[f], np.asarray(rec[f]), err_msg=f"{key} {f}")

    def test_reference_heuristic_and_tref_regenerate(self):
        import common as OV
        import schedules as S
        for key, rec in self.man["instances"].items():
            jl, pt = B.read_plain_instance(B.DATA_ROOT / key)
            inst = OV.Instance(key, jl, pt, np.repeat(np.arange(len(jl)), jl), np.concatenate([[0], np.cumsum(jl)[:-1]]))
            m, s = S.build(inst, *B.REFERENCE_RULE)
            r = B.schedule_objectives(jl, pt, B.carbon_matrix(pt, rec["power_kw"]), rec["due_dates"], m, s)
            self.assertEqual((r["makespan"], r["carbon"], r["tardiness"]),
                             tuple(rec["reference"][x] for x in ("makespan", "carbon", "tardiness")))
        for size, v in self.man["sizes"].items():
            tr = [r["reference"]["tardiness"] for r in self.man["instances"].values()
                  if r["size"] == size and r["split"] == "train"]
            self.assertEqual(v["t_ref"], float(np.mean(tr)))
            self.assertEqual((v["k"], v["s"]), B.DUE[size])

    def test_ranges(self):
        for rec in self.man["instances"].values():
            k, s = B.DUE[rec["size"]]
            p, kj = np.asarray(rec["power_kw"]), np.asarray(rec["k_j"])
            self.assertTrue((p >= 4).all() and (p <= 15).all())
            self.assertTrue((kj >= k * (1 - s / 2)).all() and (kj <= k * (1 + s / 2)).all())
            self.assertTrue((np.asarray(rec["due_dates"]) > 0).all())

    def test_splits_disjoint_and_no_final_test(self):
        for size in B.SIZES:
            hs = [r["structural_sha256"] for r in self.man["instances"].values() if r["size"] == size]
            self.assertEqual(len(hs), len(set(hs)))
        self.assertFalse(any("final_test" in k or "final_test" in r["source"]
                             for k, r in self.man["instances"].items()))
        counts = {(r["split"], r["size"]) for r in self.man["instances"].values()}
        self.assertEqual(counts, {(s, z) for s in B.SPLITS for z in B.SIZES})

    def test_spec_records_this_manifest(self):
        spec = (REPO / "configs" / "problem_spec_v3.yaml").read_text()
        self.assertIn(f"sha256: {B.file_sha256(B.MANIFEST)}", spec)
        self.assertRegex(spec, r"status: FROZEN")
        for size, v in self.man["sizes"].items():
            self.assertIn(f'"{size}": {{k: {v["k"]}, s: {v["s"]}}}', spec)
            self.assertIn(repr(v["t_ref"]), spec)

    def test_loader_verifies_hashes(self):
        directory = B.ROOT / "development" / "SD2" / "10x5+mix"
        data = B.load_directory(directory, self.man)
        self.assertEqual(len(data), 19)
        np.testing.assert_array_equal(data.carbon[0], B.carbon_matrix(data.op_pt[0], data.power[0]))
        bad = copy.deepcopy(self.man)
        bad["instances"][data.keys[3]]["sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "HASH MISMATCH"):
            B.load_directory(directory, bad)
        del bad["instances"][data.keys[3]]
        with self.assertRaises(KeyError):
            B.load_directory(directory, bad)

    def test_heldout_subset_maps_to_the_original_pool(self):
        sub = self.man["subsets"]["evaluation/10x5/heldout_49"]
        data = B.load_split("evaluation", "10x5", self.man)
        self.assertEqual([Path(data.keys[i]).name for i in sub["loader_idx"]], sub["files"])
        self.assertEqual(len(sub["files"]), 49)

    def test_retired_format_rejected(self):
        legacy = sorted((DANIEL_DIR / "data/data_train/SD2/10x5+carbon+priority").glob("*.fjs"))[0]
        with self.assertRaises((ValueError, IndexError)):
            B.read_plain_instance(legacy)


@unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch is required for environment tests")
class EnvironmentV3(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from fjsp_env_same_op_nums import FJSPEnvForSameOpNums
        cls.Env = FJSPEnvForSameOpNums
        cls.data = {size: B.load_split("development", size) for size in B.SIZES}

    def env(self, size, idx, goal="mct"):
        n_j, n_m = B.SIZES[size]
        e = self.Env(n_j, n_m, obs_version="v3", reward_version="v3")
        state = e.set_initial_data(**self.data[size].env_kwargs(idx, goal))
        return e, state

    def test_dimensions_and_fixed_scales(self):
        for size, (n_j, n_m) in B.SIZES.items():
            e, st = self.env(size, range(4))
            self.assertEqual(tuple(st.fea_j_tensor.shape), (4, n_j * n_m, 12))
            self.assertEqual(tuple(st.fea_m_tensor.shape), (4, n_m, 9))
            self.assertEqual(tuple(st.fea_pairs_tensor.shape), (4, n_j, n_m, 10))
            self.assertEqual(e.pt_upper_bound, 99.0)
            np.testing.assert_allclose(e.due_dates * 99.0, np.asarray(self.data[size].due[:4]))

    def test_carbon_and_power_observations(self):
        d = self.data["10x5"]; idx = list(range(len(d)))
        e, st = self.env("10x5", idx)
        cand_carbon = np.array([d.carbon[i][e.candidate[k]] for k, i in enumerate(idx)])
        np.testing.assert_allclose(st.fea_pairs_tensor[..., 8].cpu().numpy(), cand_carbon / B.CARBON_SCALE,
                                   rtol=1e-6, atol=1e-7)
        full = ~e.delete_mask_fea_m[:, :, 0].any(1)                  # no machine deleted at t = 0
        p = np.asarray(d.power)[full] / 15.0
        z = (p - p.mean(1, keepdims=True)) / (p.std(1, keepdims=True) + 1e-8)
        np.testing.assert_allclose(st.fea_m_tensor[full][..., 8].cpu().numpy(), z, rtol=1e-5, atol=1e-5)

    def test_objectives_match_independent_scorer_and_return_is_minus_score(self):
        rng = np.random.default_rng(20261013)
        for size in B.SIZES:
            d = self.data[size]; idx = list(range(len(d)))
            for goal in GOALS:
                e, _ = self.env(size, idx, goal)
                ret, machine = random_rollout(e, rng)
                start = e.true_op_ct - np.take_along_axis(e.true_op_pt, machine[:, :, None], 2)[:, :, 0]
                ind = [B.schedule_objectives(d.job_length[i], d.op_pt[i], d.carbon[i], d.due[i], machine[k], start[k])
                       for k, i in enumerate(idx)]
                mk = np.array([r["makespan"] for r in ind]); cb = np.array([r["carbon"] for r in ind])
                td = np.array([r["tardiness"] for r in ind])
                np.testing.assert_array_equal(mk, e.current_makespan)
                np.testing.assert_allclose(cb, e.total_carbon, rtol=0, atol=1e-9)
                comp = e.true_op_ct[e.env_job_idx, e.job_last_op_id]
                np.testing.assert_allclose(td, np.maximum(0, comp - e.true_due_dates).sum(1), rtol=0, atol=1e-9)
                sc = B.score(B.goal_weights(goal), mk, cb, td, np.array(d.mk_ref), np.array(d.cb_ref), d.t_ref)
                np.testing.assert_allclose(ret, -sc, rtol=0, atol=1e-9, err_msg=f"{size} {goal}")
                from objectives import environment_metrics
                self.assertAlmostEqual(environment_metrics(e)["score_v3"], float(sc.mean()), places=10)

    def test_batch_invariance(self):
        """Fixed scales: an instance's observations do not depend on the rest of its batch."""
        rng = np.random.default_rng(7)
        alone, st1 = self.env("10x5", [0])
        batch, st2 = self.env("10x5", [0, 5, 9, 11])
        while True:
            for a, b in ((st1.fea_j_tensor, st2.fea_j_tensor), (st1.fea_m_tensor, st2.fea_m_tensor),
                         (st1.fea_pairs_tensor, st2.fea_pairs_tensor),
                         (st1.dynamic_pair_mask_tensor, st2.dynamic_pair_mask_tensor)):
                np.testing.assert_array_equal(a[0].cpu().numpy(), b[0].cpu().numpy())
            mask = batch.dynamic_pair_mask.reshape(4, -1)
            act = np.array([rng.choice(np.flatnonzero(~m)) for m in mask])
            st1, r1, d1 = alone.step(act[:1]); st2, r2, _ = batch.step(act)
            self.assertEqual(r1[0], r2[0])
            if d1.all():
                break

    def test_reset_reproduces_episode(self):
        e, s0 = self.env("20x10", range(3))
        f0 = s0.fea_j_tensor.cpu().numpy().copy()          # the state object is updated in place by step()
        r1, _ = random_rollout(e, np.random.default_rng(3)); mk1 = e.current_makespan.copy()
        s1 = e.reset()
        np.testing.assert_array_equal(f0, s1.fea_j_tensor.cpu().numpy())
        r2, _ = random_rollout(e, np.random.default_rng(3))
        np.testing.assert_array_equal(r1, r2); np.testing.assert_array_equal(mk1, e.current_makespan)

    def test_guards(self):
        kw = self.data["10x5"].env_kwargs([0], "mct")
        with self.assertRaises(ValueError):
            self.Env(10, 5, obs_version="v3", reward_version="v1")
        e = self.Env(10, 5, obs_version="v3", reward_version="v3")
        for drop in ("machine_power_list", "carbon_list", "due_date_list", "reward_refs"):
            with self.assertRaises(ValueError):
                e.set_initial_data(**{k: v for k, v in kw.items() if k != drop})
        with self.assertRaises(ValueError):
            e.set_initial_data(priority_list=[np.ones(50)], **kw)
        bad = dict(kw, op_pt_list=[kw["op_pt_list"][0] * 2])               # 198 > fixed time scale 99
        with self.assertRaisesRegex(ValueError, "fixed time scale"):
            e.set_initial_data(**bad)

    def test_model_forward(self):
        import torch
        from model.main_model import DANIEL
        from params import configs
        cfg = copy.copy(configs)
        cfg.fea_j_input_dim, cfg.fea_m_input_dim, cfg.fea_pair_input_dim = 12, 9, 10
        cfg.device = "cpu"
        torch.set_default_tensor_type("torch.FloatTensor")
        net = DANIEL(cfg)
        e, st = self.env("10x5", range(3))
        with torch.no_grad():
            pi, v = net(fea_j=st.fea_j_tensor.cpu(), op_mask=st.op_mask_tensor.cpu(),
                        candidate=st.candidate_tensor.cpu(), fea_m=st.fea_m_tensor.cpu(),
                        mch_mask=st.mch_mask_tensor.cpu(), comp_idx=st.comp_idx_tensor.cpu(),
                        dynamic_pair_mask=st.dynamic_pair_mask_tensor.cpu(), fea_pairs=st.fea_pairs_tensor.cpu())
        self.assertTrue(torch.allclose(pi.sum(1), torch.ones(3)))
        self.assertTrue(bool((pi[st.dynamic_pair_mask_tensor.cpu().reshape(3, -1)] == 0).all()))
        self.assertTrue(bool(torch.isfinite(v).all()))


class ConfigsAndSchema(unittest.TestCase):
    def test_schema(self):
        from argparse import Namespace
        from feature_schemas import SCHEMA_V3, validate_config_against_schema
        cfg = Namespace(feature_schema=SCHEMA_V3, fea_j_input_dim=12, fea_m_input_dim=9, fea_pair_input_dim=10,
                        enable_priority=False, enable_carbon=True, carbon_feature=True, obs_version="v3")
        self.assertEqual(validate_config_against_schema(cfg).obs_version, "v3")
        with self.assertRaisesRegex(ValueError, "obs_version"):
            validate_config_against_schema(Namespace(**{**vars(cfg), "obs_version": "v2"}))
        with self.assertRaisesRegex(ValueError, "do not match"):
            validate_config_against_schema(Namespace(**{**vars(cfg), "fea_m_input_dim": 8}))

    def test_specialist_configs(self):
        for goal in ("m", "c", "t", "mct"):
            cfg = json.loads((REPO / "configs" / "benchmark_v3" / f"{goal}.json").read_text())
            self.assertEqual((cfg["obs_version"], cfg["reward_version"], cfg["feature_schema"], cfg["goal"]),
                             ("v3", "v3", "benchmark_v3_f12_m9_p10", goal))
            self.assertEqual((cfg["fea_j_input_dim"], cfg["fea_m_input_dim"], cfg["fea_pair_input_dim"]), (12, 9, 10))


if __name__ == "__main__":
    unittest.main()
