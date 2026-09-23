import sys
import unittest
from pathlib import Path

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "daniel"))

from objectives import (ObjectiveSpec, operation_priority_weighted_completion, total_tardiness,
                        tardiness_potential)
from environment_utils import machine_min_processing_time, normalize_processing_times


class ObjectiveTests(unittest.TestCase):
    def test_each_goal_contains_exactly_declared_terms(self):
        makespan = np.array([2.0])
        carbon = np.array([5.0])
        priority = np.array([3.0])
        expected = {
            "m": 2.0,
            "c": -0.5,
            "p": 6.0,
            "mc": 1.5,
            "mp": 8.0,
            "mcp": 7.5,
        }
        for goal, value in expected.items():
            with self.subTest(goal=goal):
                spec = ObjectiveSpec(goal, carbon_weight=0.1, priority_weight=2.0)
                self.assertAlmostEqual(float(spec.step_reward(makespan, carbon, priority)[0]), value)

    def test_tardiness_goals_contain_exactly_declared_terms(self):
        makespan, carbon, priority, tardy = (np.array([2.0]), np.array([5.0]),
                                             np.array([3.0]), np.array([-4.0]))
        expected = {"t": -12.0, "mt": -10.0, "ct": -12.5, "mct": -10.5}
        for goal, value in expected.items():
            with self.subTest(goal=goal):
                spec = ObjectiveSpec(goal, carbon_weight=0.1, priority_weight=2.0,
                                     tardiness_weight=3.0)
                self.assertAlmostEqual(
                    float(spec.step_reward(makespan, carbon, priority, tardy)[0]), value)

    def test_old_goals_ignore_a_tardiness_term(self):
        # passing a tardiness delta must not leak into any pre-existing goal
        for goal in ("m", "c", "p", "mc", "mp", "mcp"):
            with self.subTest(goal=goal):
                spec = ObjectiveSpec(goal, carbon_weight=0.1, priority_weight=2.0)
                a = spec.step_reward(np.array([2.0]), np.array([5.0]), np.array([3.0]))
                b = spec.step_reward(np.array([2.0]), np.array([5.0]), np.array([3.0]),
                                     np.array([-99.0]))
                self.assertEqual(float(a[0]), float(b[0]))

    def test_tardiness_goal_without_due_dates_fails_loudly(self):
        with self.assertRaises(ValueError):
            ObjectiveSpec("t").step_reward(np.array([1.0]), np.array([1.0]), np.array([1.0]))

    def test_total_tardiness_is_unweighted_and_ignores_early_jobs(self):
        completion = np.array([[10.0, 4.0, 9.0]])
        due = np.array([[7.0, 6.0, 9.0]])
        self.assertEqual(float(total_tardiness(completion, due)[0]), 3.0)

    def test_tardiness_selection_metric_is_time_scaled(self):
        metrics = {"makespan": 20.0, "carbon": 100.0, "tardiness": 30.0, "time_scale": 10.0}
        self.assertEqual(ObjectiveSpec("mct", 0.01, 0.5, 2.0).selection_metric(metrics),
                         2.0 + 1.0 + 6.0)

    def test_tardiness_potential_reads_each_jobs_last_operation(self):
        op_ct_lb = np.array([[1.0, 5.0, 2.0, 8.0]])          # jobs = ops {0,1} and {2,3}
        last = np.array([[1, 3]]); env_job = np.array([[0, 0]])
        due = np.array([[4.0, 9.0]])
        self.assertEqual(float(tardiness_potential(op_ct_lb, last, env_job, due)[0]), 1.0)

    def test_goal_m_ignores_priority_even_when_nonzero(self):
        spec = ObjectiveSpec("m", carbon_weight=99.0, priority_weight=99.0)
        reward = spec.step_reward(np.array([1.25]), np.array([100.0]), np.array([100.0]))
        self.assertEqual(float(reward[0]), 1.25)

    def test_checkpoint_selection_matches_inherited_scalarization(self):
        metrics = {"makespan": 20.0, "carbon": 100.0, "priority": 7.0}
        self.assertEqual(ObjectiveSpec("mcp", 0.01, 0.5).selection_metric(metrics), 24.5)
        self.assertEqual(ObjectiveSpec("p", 0.01, 0.5).selection_metric(metrics), 3.5)

    def test_carbon_combination_preserves_inherited_time_normalization(self):
        metrics = {"makespan": 20.0, "carbon": 100.0, "priority": 10.0, "time_scale": 10.0}
        self.assertEqual(ObjectiveSpec("mcp", 0.01, 0.5).selection_metric(metrics), 3.5)

    def test_operation_priority_metric(self):
        completion = np.array([2.0, 5.0, 7.0])
        priority = np.array([1.0, 3.0, 2.0])
        self.assertAlmostEqual(
            float(operation_priority_weighted_completion(completion, priority)),
            (2.0 + 15.0 + 14.0) / 6.0,
        )

    def test_processing_normalization_preserves_feasibility_and_machine_minimum(self):
        raw = np.array([[[3.0, 0.0], [5.0, 7.0]]])
        normalized, relation, reverse, lower, upper = normalize_processing_times(raw)
        np.testing.assert_array_equal(relation, raw > 0)
        np.testing.assert_array_equal(reverse, raw == 0)
        self.assertEqual((lower, upper), (3.0, 7.0))
        np.testing.assert_allclose(machine_min_processing_time(normalized, relation), [[3 / 7, 1.0]])

    def test_dummy_operations_are_excluded(self):
        completion = np.array([[2.0, 5.0, 999.0]])
        priority = np.array([[1.0, 3.0, 100.0]])
        mask = np.array([[True, True, False]])
        value = operation_priority_weighted_completion(completion, priority, mask)
        self.assertAlmostEqual(float(value[0]), 17.0 / 4.0)


if __name__ == "__main__":
    unittest.main()
