"""MO-DANIEL on benchmark v3 (docs/evaluation_protocol_v3.md, Addendum A): metrics, preference-weighted reward,
preference conditioning, and PPO statistics logging that leaves the update unchanged. CPU only; reads the v3
validation/development trees, never the reserved final-test sets.

    python -m unittest tests.test_mo_daniel_v3
"""
import copy
import sys
import unittest
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]


def setUpModule():
    sys.path.insert(0, str(REPO / "daniel"))
    if "params" not in sys.modules:
        argv, sys.argv = sys.argv, ["mo-daniel-v3-test", "--device", "cpu"]
        import params                                # noqa: F401
        sys.argv = argv
    global B, MM, torch, configs, DANIEL, Env, PPO, Memory, DEV
    import torch
    import benchmark_v3 as B
    import mo_metrics as MM
    from params import configs
    from model.main_model import DANIEL
    from model.PPO import PPO, Memory
    from fjsp_env_same_op_nums import FJSPEnvForSameOpNums as Env, EnvState
    DEV = str(EnvState.device)                       # the state tensors' device (set by whichever suite ran first)


def v3_config(pref_dim, log_stats=False):
    c = copy.copy(configs)
    c.device = DEV
    c.fea_j_input_dim, c.fea_m_input_dim, c.fea_pair_input_dim = 12, 9, 10
    c.obs_version = c.reward_version = "v3"
    c.feature_schema = "benchmark_v3_f12_m9_p10_w3" if pref_dim else "benchmark_v3_f12_m9_p10"
    c.enable_priority, c.enable_carbon, c.carbon_feature = False, True, True
    c.preference_dim, c.log_ppo_stats, c.goal = pref_dim, log_stats, "mct"
    return c


def forward(net, state):
    return net(fea_j=state.fea_j_tensor, op_mask=state.op_mask_tensor, candidate=state.candidate_tensor,
               fea_m=state.fea_m_tensor, mch_mask=state.mch_mask_tensor, comp_idx=state.comp_idx_tensor,
               dynamic_pair_mask=state.dynamic_pair_mask_tensor, fea_pairs=state.fea_pairs_tensor,
               pref=state.pref_tensor)


class Metrics(unittest.TestCase):
    def test_grids(self):
        for step, n in ((0.25, 15), (0.1, 66), (0.5, 6)):
            g = MM.simplex_grid(step)
            self.assertEqual(len(g), n)
            np.testing.assert_allclose(g.sum(1), 1)
            self.assertTrue((g >= 0).all())

    def test_hypervolume_exact(self):
        self.assertAlmostEqual(MM.hypervolume(np.array([[0.1, 0.2, 0.3]])), 1.0 * 0.9 * 0.8)
        two = np.array([[0.0, 0.5, 0.5], [0.5, 0.0, 0.5]])                      # union of two boxes
        self.assertAlmostEqual(MM.hypervolume(two), 2 * 1.1 * 0.6 * 0.6 - 0.6 * 0.6 * 0.6)
        self.assertEqual(MM.hypervolume(np.array([[1.2, 0.0, 0.0]])), 0.0)      # outside the reference box
        rng = np.random.default_rng(3)
        p = rng.random((10, 3))
        s = rng.random((200_000, 3)) * 1.1
        mc = np.any(np.all(p[None] <= s[:, None], 2), 1).mean() * 1.1 ** 3
        self.assertAlmostEqual(MM.hypervolume(p), mc, delta=0.01)
        self.assertAlmostEqual(MM.hypervolume(p), MM.hypervolume(p[MM.nondominated(p)]))

    def test_normalisation_drops_degenerate_objective(self):
        p, keep = MM.normalise(np.array([[2.0, 5.0, 0.0]]), [1.0, 5.0, 0.0], [3.0, 5.0, 4.0])
        self.assertEqual(keep.tolist(), [True, False, True])
        np.testing.assert_allclose(p, [[0.5, 0.0]])
        self.assertAlmostEqual(MM.instance_hv(np.array([[2.0, 5.0, 0.0]]), [1.0, 5.0, 0.0], [3.0, 5.0, 4.0]),
                               (1.1 - 0.5) * 1.1)

    def test_igd_plus(self):
        z = np.array([[0.0, 0.0, 0.0]])
        self.assertEqual(MM.igd_plus(np.array([[0.0, 0.0, 0.0]]), z), 0.0)
        self.assertAlmostEqual(MM.igd_plus(np.array([[0.3, 0.4, 0.0]]), z), 0.5)
        self.assertEqual(MM.igd_plus(np.array([[0.0, 0.0, 0.0]]), np.array([[0.2, 0.2, 0.2]])), 0.0)  # dominates


class PreferenceEnvironment(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = B.load_split("validation", "10x5")

    def _rollout(self, env, rng):
        ret = np.zeros(env.number_of_envs)
        while True:
            mask = env.state.dynamic_pair_mask_tensor.cpu().numpy().reshape(env.number_of_envs, -1)
            a = np.array([rng.choice(np.flatnonzero(~m)) for m in mask])
            _, r, done = env.step(a)
            ret += r
            if done.all():
                return ret

    def test_return_is_minus_preference_weighted_score(self):
        rng = np.random.default_rng(0)
        idx = list(range(6))
        w = np.vstack([np.eye(3), rng.dirichlet([1, 1, 1], size=3)])
        env = Env(10, 5, obs_version="v3", reward_version="v3")
        env.set_initial_data(**self.data.env_kwargs(idx, "mct"))
        env.set_preference(w)
        for rep in range(2):                                     # the preference survives reset()
            if rep:
                env.reset()
                self.assertIsNotNone(env.state.pref_tensor)
            ret = self._rollout(env, rng)
            comp = env.true_op_ct[env.env_job_idx, env.job_last_op_id]
            tard = np.maximum(0.0, comp - env.true_due_dates).sum(1)
            score = (w[:, 0] * env.current_makespan / np.asarray(self.data.mk_ref)[idx]
                     + w[:, 1] * env.total_carbon / np.asarray(self.data.cb_ref)[idx] + w[:, 2] * tard / self.data.t_ref)
            np.testing.assert_allclose(ret, -score, atol=1e-9)

    def test_corner_preference_equals_specialist_reward(self):
        idx = [0, 1]
        for k, goal in enumerate("mct"):
            rets = []
            for pref in (True, False):
                env = Env(10, 5, obs_version="v3", reward_version="v3")
                env.set_initial_data(**self.data.env_kwargs(idx, "mct" if pref else goal))
                if pref:
                    env.set_preference(np.tile(np.eye(3)[k], (2, 1)))
                rets.append(self._rollout(env, np.random.default_rng(1)))
            np.testing.assert_allclose(rets[0], rets[1], atol=1e-12)

    def test_guards_and_fresh_instances(self):
        env = Env(10, 5, obs_version="v3", reward_version="v3")
        env.set_initial_data(**self.data.env_kwargs([0, 1], "mct"))
        with self.assertRaises(ValueError):
            env.set_preference(np.array([[0.5, 0.5, 0.5], [1, 0, 0]]))
        env.set_preference(np.array([[1.0, 0, 0], [0, 1.0, 0]]))
        env.step(np.flatnonzero(~env.dynamic_pair_mask.reshape(2, -1)[0])[:1].repeat(2))
        with self.assertRaises(ValueError):
            env.set_preference(np.array([[1.0, 0, 0], [0, 1.0, 0]]))     # mid-episode
        env.set_initial_data(**self.data.env_kwargs([0, 1], "mct"))
        self.assertIsNone(env.pref_weights)                               # new instances: no stale preference
        self.assertIsNone(env.state.pref_tensor)


class Conditioning(unittest.TestCase):
    def setUp(self):
        self.data = B.load_split("validation", "10x5")
        self.env = Env(10, 5, obs_version="v3", reward_version="v3")
        self.state = self.env.set_initial_data(**self.data.env_kwargs([0, 1], "mct"))

    def test_published_architecture_unchanged_without_preference(self):
        torch.manual_seed(0)
        net = DANIEL(v3_config(0))
        self.assertEqual(net.actor.linears[0].weight.shape[1], 4 * 8 + 10)
        self.assertEqual(net.critic.linears[0].weight.shape[1], 2 * 8)
        with self.assertRaises(ValueError):
            net(fea_j=self.state.fea_j_tensor, op_mask=self.state.op_mask_tensor, candidate=self.state.candidate_tensor,
                fea_m=self.state.fea_m_tensor, mch_mask=self.state.mch_mask_tensor, comp_idx=self.state.comp_idx_tensor,
                dynamic_pair_mask=self.state.dynamic_pair_mask_tensor, fea_pairs=self.state.fea_pairs_tensor,
                pref=torch.ones(2, 3, device=DEV) / 3)

    def test_preference_reaches_actor_and_critic_only(self):
        torch.manual_seed(0)
        net = DANIEL(v3_config(3))
        self.assertEqual(net.actor.linears[0].weight.shape[1], 4 * 8 + 10 + 3)
        self.assertEqual(net.critic.linears[0].weight.shape[1], 2 * 8 + 3)
        ref = DANIEL(v3_config(0))
        self.assertEqual({k: v.shape for k, v in net.feature_exact.state_dict().items()},
                         {k: v.shape for k, v in ref.feature_exact.state_dict().items()})
        with self.assertRaises(ValueError):
            forward(net, self.state)                                     # no preference set
        st = self.env.set_preference(np.array([[1.0, 0, 0], [1.0, 0, 0]]))
        p1, v1 = forward(net, st)
        st = self.env.set_preference(np.array([[0, 0, 1.0], [0, 0, 1.0]]))
        p2, v2 = forward(net, st)
        self.assertFalse(torch.allclose(p1, p2)); self.assertFalse(torch.allclose(v1, v2))
        self.assertTrue(torch.all(p1[self.state.dynamic_pair_mask_tensor.reshape(2, -1)] == 0))

    def test_schema(self):
        from feature_schemas import validate_config_against_schema
        self.assertEqual(validate_config_against_schema(v3_config(3)).preference_dim, 3)
        bad = v3_config(3); bad.preference_dim = 0
        with self.assertRaises(ValueError):
            validate_config_against_schema(bad)


class PPOStatistics(unittest.TestCase):
    def _memory(self, pref_dim):
        data = B.load_split("validation", "10x5")
        env = Env(10, 5, obs_version="v3", reward_version="v3")
        state = env.set_initial_data(**data.env_kwargs([0, 1, 2], "mct"))
        if pref_dim:
            state = env.set_preference(np.random.default_rng(0).dirichlet([1, 1, 1], size=3))
        torch.manual_seed(1)
        net = DANIEL(v3_config(pref_dim))
        mem = Memory(gamma=1.0, gae_lambda=0.98)
        g = torch.Generator(device=DEV).manual_seed(2)
        with torch.no_grad():
            while True:
                mem.push(state)
                pi, v = forward(net, state)
                a = torch.multinomial(pi, 1, generator=g).squeeze(1)
                state, r, done = env.step(a.cpu().numpy())
                mem.done_seq.append(torch.from_numpy(done).to(DEV)); mem.reward_seq.append(torch.from_numpy(r).to(DEV))
                mem.action_seq.append(a); mem.log_probs.append(torch.log(pi.gather(1, a[:, None]).squeeze(1)))
                mem.val_seq.append(v.squeeze(1))
                if done.all():
                    return mem

    def test_logging_leaves_the_update_bit_identical(self):
        for pref_dim in (0, 3):
            mem = self._memory(pref_dim)
            out = []
            for log in (False, True):
                torch.manual_seed(5)
                c = v3_config(pref_dim, log_stats=log); c.minibatch_size = 64
                ppo = PPO(c)
                losses = ppo.update(copy.deepcopy(mem))
                out.append((losses, [p.detach().clone() for p in ppo.policy.parameters()], ppo.last_stats))
            self.assertEqual(out[0][0], out[1][0])
            for a, b in zip(out[0][1], out[1][1]):
                self.assertTrue(torch.equal(a, b))
            self.assertIsNone(out[0][2])
            st = out[1][2]
            for k in ("entropy", "approx_kl", "clip_fraction", "policy_loss", "value_loss", "grad_norm",
                      "explained_variance"):
                self.assertIn(k, st); self.assertTrue(np.isfinite(st[k]), k)
            self.assertGreater(st["grad_norm"], 0)


if __name__ == "__main__":
    unittest.main()
