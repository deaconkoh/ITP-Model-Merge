import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np


DANIEL_DIR = Path(__file__).resolve().parents[1] / "daniel"


@unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch is required for environment integration tests")
class EnvironmentSemanticsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(DANIEL_DIR))
        cls.original_argv = sys.argv[:]
        sys.argv = [
            "environment-test",
            "--device", "cpu",
            "--goal", "m",
            "--enable_priority", "True",
            "--enable_carbon", "True",
            "--carbon_feature", "True",
            "--priority_scope", "operation",
        ]
        from fjsp_env_same_op_nums import FJSPEnvForSameOpNums
        from fjsp_env_various_op_nums import FJSPEnvForVariousOpNums
        cls.same_class = FJSPEnvForSameOpNums
        cls.various_class = FJSPEnvForVariousOpNums

    @classmethod
    def tearDownClass(cls):
        sys.argv = cls.original_argv

    @staticmethod
    def fixture():
        jobs = [np.array([1])]
        processing = [np.array([[3.0, 5.0]])]
        priority = [np.array([2.0])]
        carbon = [np.array([[7.0, 1.0]])]
        return jobs, processing, priority, carbon

    def test_raw_minimum_edge_remains_feasible_and_machine_min_is_minimum(self):
        env = self.same_class(1, 2)
        env.set_initial_data(*self.fixture())
        self.assertTrue(env.process_relation[0, 0, 0])
        self.assertTrue(env.process_relation[0, 0, 1])
        np.testing.assert_allclose(env.mch_min_pt[0], [0.6, 1.0])

    def test_goal_m_receives_only_makespan_delta(self):
        env = self.same_class(1, 2)
        env.set_initial_data(*self.fixture())
        _, reward, done = env.step(np.array([1]))  # slower but lower-carbon machine
        self.assertTrue(done[0])
        self.assertAlmostEqual(float(reward[0]), -0.4)
        self.assertAlmostEqual(float(env.current_makespan[0]), 5.0)
        self.assertAlmostEqual(float(env.total_carbon[0]), 1.0)

    def test_uniform_and_varying_environments_agree(self):
        same = self.same_class(1, 2)
        various = self.various_class(1, 2)
        same.set_initial_data(*self.fixture())
        various.set_initial_data(*self.fixture())
        _, same_reward, _ = same.step(np.array([1]))
        _, various_reward, _ = various.step(np.array([1]))
        np.testing.assert_allclose(same_reward, various_reward)
        np.testing.assert_allclose(same.current_makespan, various.current_makespan)
        np.testing.assert_allclose(same.total_carbon, various.total_carbon)
        np.testing.assert_allclose(same.true_op_ct, various.true_op_ct)

    def test_seed_setup_reproduces_torch_and_numpy_streams(self):
        import torch
        from common_utils import setup_seed

        setup_seed(1234)
        first_torch = torch.rand(4)
        first_numpy = np.random.random(4)
        setup_seed(1234)
        self.assertTrue(torch.equal(first_torch, torch.rand(4)))
        np.testing.assert_array_equal(first_numpy, np.random.random(4))


if __name__ == "__main__":
    unittest.main()
