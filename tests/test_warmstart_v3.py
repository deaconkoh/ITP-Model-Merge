"""Warm-start arms (docs/warmstart_protocol_v3.md): weight transfer, merging, distillation and the analysis metrics.
Uses the committed specialists in checkpoints/specialists_v3 and the v3 validation split; never the final-test sets.

    python -m unittest tests.test_warmstart_v3
"""
import copy
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
SPEC = REPO / "checkpoints" / "specialists_v3"


def setUpModule():
    sys.path.insert(0, str(REPO / "daniel"))
    if "params" not in sys.modules:
        argv, sys.argv = sys.argv, ["warmstart-v3-test", "--device", "cpu"]
        import params                                # noqa: F401
        sys.argv = argv
    global B, MM, WS, torch, configs, DANIEL, Env, DEV, load_sd
    import torch
    import benchmark_v3 as B
    import mo_metrics as MM
    import warmstart as WS
    from params import configs
    from model.main_model import DANIEL
    from common_utils import load_checkpoint_state_dict as load_sd
    from fjsp_env_same_op_nums import FJSPEnvForSameOpNums as Env, EnvState
    DEV = str(EnvState.device)


def cfg(pref_dim):
    c = copy.copy(configs)
    c.device = DEV
    c.fea_j_input_dim, c.fea_m_input_dim, c.fea_pair_input_dim = 12, 9, 10
    c.obs_version = c.reward_version = "v3"
    c.feature_schema = "benchmark_v3_f12_m9_p10_w3" if pref_dim else "benchmark_v3_f12_m9_p10"
    c.enable_priority, c.enable_carbon, c.carbon_feature = False, True, True
    c.preference_dim, c.goal = pref_dim, "mct"
    return c


def spec_sd(o, seed=111, size="10x5"):
    return load_sd(SPEC / f"{size}+mix+bv3_spec_{o}_s{seed}.pth", map_location=DEV,
                   expected_schema="benchmark_v3_f12_m9_p10")


def fwd(net, st, pref=None):
    return net(fea_j=st.fea_j_tensor, op_mask=st.op_mask_tensor, candidate=st.candidate_tensor,
               fea_m=st.fea_m_tensor, mch_mask=st.mch_mask_tensor, comp_idx=st.comp_idx_tensor,
               dynamic_pair_mask=st.dynamic_pair_mask_tensor, fea_pairs=st.fea_pairs_tensor, pref=pref)


class Transfer(unittest.TestCase):
    def test_full_copy_equals_specialist_at_update_0(self):
        data = B.load_split("validation", "10x5")
        idx = list(range(6))
        for o in "mct":
            torch.manual_seed(0)
            spec = DANIEL(cfg(0)).to(DEV); spec.load_state_dict(spec_sd(o)); spec.eval()
            mo = DANIEL(cfg(3)).to(DEV)
            rep = WS.transfer(spec_sd(o), mo, "full"); mo.eval()
            self.assertEqual(sorted(rep["padded"]), sorted(WS.PREF_LAYERS)); self.assertEqual(rep["kept"], [])
            for k in WS.PREF_LAYERS:
                self.assertTrue(torch.all(mo.state_dict()[k][:, -3:] == 0))
            w = np.random.default_rng(1).dirichlet([1, 1, 1], size=len(idx))
            env = Env(10, 5, obs_version="v3", reward_version="v3")
            st = env.set_initial_data(**data.env_kwargs(idx, "mct"))
            pref = torch.as_tensor(w, dtype=torch.float32, device=DEV)
            with torch.no_grad():
                while True:
                    p1, v1 = fwd(spec, st); p2, v2 = fwd(mo, st, pref)
                    self.assertLessEqual(float((p1 - p2).abs().max()), 1e-6)
                    self.assertLessEqual(float((v1 - v2).abs().max()), 1e-5)
                    self.assertTrue(torch.equal(p1.argmax(1), p2.argmax(1)))
                    st, _, done = env.step(p1.argmax(1).cpu().numpy())
                    if done.all():
                        break

    def test_backbone_copies_only_the_extractor(self):
        torch.manual_seed(3)
        mo = DANIEL(cfg(3)).to(DEV)
        before = {k: v.clone() for k, v in mo.state_dict().items()}
        src = spec_sd("c")
        rep = WS.transfer(src, mo, "backbone")
        after = mo.state_dict()
        for k, v in after.items():
            if k.startswith("feature_exact."):
                self.assertTrue(torch.equal(v, src[k].to(v.device)), k)
            else:
                self.assertTrue(torch.equal(v, before[k]), k)
        self.assertEqual(len(rep["copied"]), sum(k.startswith("feature_exact.") for k in after))

    def test_no_critic_keeps_the_fresh_critic(self):
        torch.manual_seed(4); student = DANIEL(cfg(3)).to(DEV)
        torch.manual_seed(5); arm = DANIEL(cfg(3)).to(DEV)
        fresh_critic = {k: v.clone() for k, v in arm.state_dict().items() if k.startswith("critic.")}
        WS.transfer(student.state_dict(), arm, "no_critic")
        for k, v in arm.state_dict().items():
            ref = fresh_critic[k] if k.startswith("critic.") else student.state_dict()[k]
            self.assertTrue(torch.equal(v, ref), k)

    def test_shape_mismatch_is_rejected(self):
        bad = dict(spec_sd("m")); bad["actor.linears.1.weight"] = torch.zeros(3, 3)
        with self.assertRaises(ValueError):
            WS.transfer(bad, DANIEL(cfg(3)).to(DEV), "full")


class Merging(unittest.TestCase):
    def test_uniform_merge_and_corner_interpolation(self):
        sds = [spec_sd(o) for o in "mct"]
        m = WS.uniform_merge(sds)
        for k in m:
            np.testing.assert_allclose(m[k].cpu().numpy(), sum(s[k].cpu().double().numpy() for s in sds) / 3,
                                       rtol=0, atol=1e-6)
        for x in range(3):
            lam = np.eye(3)[x]
            corner = WS.interpolate(sds, lam)
            for k in corner:
                self.assertTrue(torch.equal(corner[k], sds[x][k]), k)


class Metrics(unittest.TestCase):
    def test_regret_auc_time_to(self):
        self.assertAlmostEqual(MM.preference_regret([0.5, 0.5], [[0.2, 0.9], [0.5, 0.5]], [1, 0]), 0.3)
        self.assertEqual(MM.preference_regret([0.2, 0.9], [[0.2, 0.9], [0.5, 0.5]], [1, 0]), 0.0)
        self.assertAlmostEqual(MM.auc([0, 50, 100], [0.0, 1.0, 1.0], 100), 0.5)
        self.assertAlmostEqual(MM.auc([0, 50], [0.2, 0.4], 200), (0.2 * 50 + 0.4 * 150) / 200)
        self.assertEqual(MM.time_to([0, 50, 100], [0.0, 0.5, 1.0], [0, 10, 20], 0.9), (100, 20.0))
        self.assertEqual(MM.time_to([0, 50], [0.0, 0.5], [0, 10], 0.9), (None, None))


class Inputs(unittest.TestCase):
    def test_committed_inputs_verify(self):
        sys.path.insert(0, str(REPO / "tools"))
        import ec2_inputs
        self.assertEqual(ec2_inputs.verify(), 0)


class Distillation(unittest.TestCase):
    def test_tiny_distillation_lowers_kl(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "student.pth"
            r = subprocess.run([sys.executable, "distill_v3.py", "--size", "10x5", "--seed", "111", "--instances", "6",
                                "--epochs", "3", "--batch", "64", "--val_first", "2", "--out", str(out),
                                "--device", "cpu"], cwd=REPO / "daniel", capture_output=True, text=True, timeout=900)
            self.assertEqual(r.returncode, 0, r.stderr[-2000:])
            rec = json.loads(out.with_suffix(".distill.json").read_text())
            kl = [h["train_kl"] for h in rec["history"]]
            self.assertLess(kl[-1], kl[0])
            self.assertTrue(out.exists())


if __name__ == "__main__":
    unittest.main()
