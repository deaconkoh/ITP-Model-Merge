"""Contrastive fine-tuning (docs/contrastive_protocol_v3.md): pair construction, the DPO loss, the two-pass gradient,
a 2-update train.py run and the frozen manifest. Uses the v3 validation split only; never the final-test sets.

    python -m unittest tests.test_contrastive_v3
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


def setUpModule():
    sys.path.insert(0, str(REPO / "daniel"))
    if "params" not in sys.modules:
        argv, sys.argv = sys.argv, ["contrastive-v3-test", "--device", "cpu"]
        import params                                # noqa: F401
        sys.argv = argv
    global B, CL, torch, configs, DANIEL, Env, DEV
    import torch
    import benchmark_v3 as B
    import contrastive as CL
    from params import configs
    from model.main_model import DANIEL
    from fjsp_env_same_op_nums import FJSPEnvForSameOpNums as Env, EnvState
    DEV = str(EnvState.device)


def mo_cfg():
    c = copy.copy(configs)
    c.device = DEV
    c.fea_j_input_dim, c.fea_m_input_dim, c.fea_pair_input_dim = 12, 9, 10
    c.obs_version = c.reward_version = "v3"
    c.feature_schema = "benchmark_v3_f12_m9_p10_w3"
    c.enable_priority, c.enable_carbon, c.carbon_feature = False, True, True
    c.preference_dim, c.goal = 3, "mct"
    return c


class _NoStep:
    """Optimizer stand-in: keeps the accumulated gradients for inspection."""
    def __init__(self, params):
        self.params = list(params)

    def zero_grad(self):
        for p in self.params:
            p.grad = None

    def step(self):
        pass


class Pairs(unittest.TestCase):
    def test_dominance_tchebycheff_margin_and_ties(self):
        obj = np.array([[10, 10, 10], [11, 10, 10], [10, 10, 10], [5, 30, 10]], float)
        f = obj / 10.0
        w = np.array([0.5, 0.25, 0.25])
        pairs, n_dom, n_tch = CL.group_pairs(obj, f, w, margin=0.01)
        self.assertIn((0, 1), pairs); self.assertIn((2, 1), pairs)            # dominance
        self.assertNotIn((0, 2), pairs); self.assertNotIn((2, 0), pairs)      # identical objectives: no pair
        # g = max_k w_k f_k: [0.5, 0.55, 0.5, 0.75] -> 0, 1 and 2 beat 3 by more than the margin
        self.assertIn((1, 3), pairs); self.assertNotIn((3, 1), pairs)
        self.assertEqual(n_dom, 2); self.assertEqual(n_dom + n_tch, len(pairs)); self.assertEqual(len(set(pairs)), len(pairs))
        _, _, n_tch_big = CL.group_pairs(obj, f, w, margin=1.0)
        self.assertEqual(n_tch_big, 0)                                         # a large margin removes score pairs

    def test_loss_at_reference_is_log2(self):
        ell = torch.tensor([-3.0, -5.0, -4.0])
        loss = CL.dpo_loss(ell, ell.clone(), [(0, 1), (2, 1)], beta=0.1)
        self.assertAlmostEqual(float(loss), float(np.log(2)), places=6)


class Gradient(unittest.TestCase):
    def test_two_pass_gradient_equals_one_pass(self):
        torch.manual_seed(7)
        net = DANIEL(mo_cfg()).to(DEV)
        ref = copy.deepcopy(net).eval()
        torch.manual_seed(8)
        with torch.no_grad():                                                  # make the policy differ from the reference
            for p in net.parameters():
                p.add_(0.05 * torch.randn_like(p))
        data = B.load_split("validation", "10x5")
        k, idx = 3, [0, 0, 0, 1, 1, 1]
        env = Env(10, 5, obs_version="v3", reward_version="v3")
        env.set_initial_data(**data.env_kwargs(idx, "mct"))
        w = np.repeat(np.array([[0.2, 0.3, 0.5], [0.6, 0.2, 0.2]]), k, axis=0)
        steps = CL.rollout(net, ref, env, env.set_preference(w))
        pairs = [(0, 1), (2, 1), (3, 5)]
        opt = _NoStep(net.parameters())
        loss2, _ = CL.dpo_update(net, opt, steps, pairs, beta=0.1, clip=1e9, minibatch=64)
        g2 = {n: p.grad.clone() for n, p in net.named_parameters() if p.grad is not None}
        opt.zero_grad()
        lps = []
        for s in steps:                                                        # one pass: autograd through every state
            pi, _ = net(**s["x"])
            lps.append(torch.log(pi.gather(1, s["a"][:, None]).squeeze(1)))
        ell = torch.stack(lps, 1).sum(1)
        ell_ref = torch.stack([s["lr"] for s in steps], 1).sum(1)
        loss1 = CL.dpo_loss(ell, ell_ref, pairs, 0.1)
        loss1.backward()
        self.assertAlmostEqual(loss2, float(loss1.detach()), places=5)
        for n, p in net.named_parameters():
            if p.grad is None:
                self.assertNotIn(n, g2); continue
            self.assertTrue(torch.allclose(p.grad, g2[n], rtol=1e-3, atol=1e-6), n)
        self.assertFalse(any(n.startswith("critic.") for n in g2))             # the critic gets no gradient

    def test_no_pairs_no_step(self):
        self.assertEqual(CL.dpo_update(None, None, [], [], 0.1, 1.0, 64), (None, None))


class Training(unittest.TestCase):
    def test_two_update_run_logs_everything(self):
        norm = REPO / "results" / "warmstart_v3" / "norm" / "validation_normalisation_points_10x5.json"
        with tempfile.TemporaryDirectory() as d:
            r = subprocess.run(
                [sys.executable, str(REPO / "daniel" / "train.py"), "--config", str(REPO / "configs" / "benchmark_v3" / "mct.json"),
                 "--n_j", "10", "--n_m", "5", "--preference_dim", "3", "--mo_val_norm", str(norm), "--mo_val_grid_step", "0.5",
                 "--mo_val_first", "2", "--model_suffix", "bv3_cltest_s111", "--seed_train", "111", "--v3_train_source",
                 "generator", "--validate_timestep", "2", "--max_updates", "2", "--num_envs", "3", "--finetune_objective",
                 "contrastive", "--cl_k", "3", "--device", "cpu"],
                cwd=d, capture_output=True, text=True, timeout=1200)
            self.assertEqual(r.returncode, 0, r.stdout[-2000:] + r.stderr[-2000:])
            name = "10x5+mix+bv3_cltest_s111"
            log = json.loads((Path(d) / "train_log" / "SD2" / f"reward_{name}.json").read_text())
            self.assertEqual([e["update"] for e in log], [0, 1])
            for key in ("loss", "pairs", "pairs_dominance", "pairs_tchebycheff", "groups_with_pair", "distinct_schedules",
                        "distinct_objective_vectors", "entropy", "grad_norm", "skipped", "env_steps", "env_steps_total",
                        "seconds"):
                self.assertIn(key, log[0]["cl"])
            self.assertEqual(log[0]["cl"]["env_steps"] % 9, 0)                  # 3 instances x K = 3 environments
            self.assertEqual(log[1]["cl"]["env_steps_total"], 2 * log[0]["cl"]["env_steps"])
            val = json.loads((Path(d) / "train_log" / "SD2" / f"valiquality_{name}.json").read_text())
            self.assertEqual([e["update"] for e in val], [0, 2])                # validation at update 0 included
            st = json.loads((Path(d) / "train_log" / "SD2" / f"{name}.status.json").read_text())
            self.assertEqual(st["status"], "FINISHED")

    def test_default_objective_is_ppo(self):
        self.assertEqual(configs.finetune_objective, "ppo")


class Manifest(unittest.TestCase):
    def test_frozen_manifest_verifies(self):
        sys.path.insert(0, str(REPO / "tools"))
        import freeze_contrastive_v3 as F
        m = F.load()
        self.assertEqual({k: v["updates"] for k, v in m["arms"].items()}, {"CLB": 2600, "CLD": 1000, "CLE": 3350})
        self.assertEqual(m["seed"], 111)


if __name__ == "__main__":
    unittest.main()
