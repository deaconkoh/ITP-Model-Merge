"""v3 baseline machinery (docs/evaluation_protocol_v3.md): the fresh-instance training source and the convergence rule.
CPU only; never opens the reserved final-test files.

    python -m unittest tests.test_baselines_v3
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
        argv, sys.argv = sys.argv, ["baselines-v3-test", "--device", "cpu"]
        import params                                # noqa: F401
        sys.argv = argv
    global B, G, C, configs
    import benchmark_v3 as B
    import benchmark_v3_generator as G
    import convergence as C
    from params import configs


def cfg(size):
    c = copy.copy(configs)
    c.n_j, c.n_m = B.SIZES[size]
    c.data_suffix, c.low, c.high, c.op_per_job = "mix", 1, 99, 0
    return c


class Generator(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.man = B.load_manifest()

    def test_matrix_hash_equals_file_hash(self):
        for key, rec in self.man["instances"].items():
            jl, pt = B.read_plain_instance(B.DATA_ROOT / key)
            self.assertEqual(G.structural_sha256(jl, pt), rec["structural_sha256"], key)

    def test_fresh_instances_use_the_frozen_definitions(self):
        for size in B.SIZES:
            np.random.seed(111)
            src = G.GeneratorSource(cfg(size), size, self.man)
            d = src.sample(5)
            self.assertEqual(len(d), 5); self.assertEqual(d.t_ref, self.man["sizes"][size]["t_ref"])
            for i in range(5):
                key = d.keys[i]; jl, pt = d.job_length[i], d.op_pt[i]
                self.assertEqual(key, G.structural_sha256(jl, pt))
                self.assertEqual((len(jl), pt.shape[1]), B.SIZES[size])
                self.assertTrue((jl == B.SIZES[size][1]).all())
                np.testing.assert_array_equal(d.power[i], B.machine_power(key, pt.shape[1]))
                np.testing.assert_array_equal(d.due[i], B.due_dates(jl, pt, B.kdraw(key, len(jl)), size))
                np.testing.assert_array_equal(d.carbon[i], B.carbon_matrix(pt, d.power[i]))
                self.assertEqual((d.mk_ref[i], d.cb_ref[i]), G.reference(jl, pt, d.carbon[i], d.due[i]))
                self.assertNotIn(key, src.forbidden)

    def test_seeded_and_rng_untouched_by_final_test_regeneration(self):
        np.random.seed(5); before = np.random.get_state()[1].copy()
        src = G.GeneratorSource(cfg("10x5"), "10x5", self.man)
        np.testing.assert_array_equal(before, np.random.get_state()[1])
        self.assertEqual(len(src.forbidden), src.n_pool_hashes + G.FINAL_TEST_COUNT)
        np.random.seed(5); a = src.sample(3).keys
        np.random.seed(5); b = G.GeneratorSource(cfg("10x5"), "10x5", self.man).sample(3).keys
        self.assertEqual(a, b)

    def test_forbidden_instances_are_redrawn(self):
        src = G.GeneratorSource(cfg("10x5"), "10x5", self.man)
        np.random.seed(9); first = src.sample(1).keys[0]
        src.forbidden.add(first)
        np.random.seed(9); again = src.sample(1).keys[0]
        self.assertNotEqual(first, again); self.assertEqual(src.collisions, 1)

    def test_final_test_stream_is_the_generator_with_its_seed(self):
        c = cfg("10x5")
        h = G.final_test_stream_hashes(c, 400, count=3)
        np.random.seed(400)
        from data_utils import SD2_instance_generator
        exp = {G.structural_sha256(*SD2_instance_generator(config=c)[:2]) for _ in range(3)}
        self.assertEqual(h, exp)


class Convergence(unittest.TestCase):
    def test_rule(self):
        log = [(u, 1.0) for u in range(0, 1001, 50)]
        self.assertEqual(C.is_converged(log, 1000, 0.005, 1000), (True, 0.0))
        self.assertFalse(C.is_converged(log[:-1], 1000, 0.005, 1000)[0])            # before 1,000 updates
        log2 = [(0, 2.0)] + [(u, 1.0) for u in range(50, 1001, 50)]
        conv, rel = C.is_converged(log2, 1000, 0.005, 1000)
        self.assertFalse(conv); self.assertAlmostEqual(rel, 0.5)
        log3 = [(u, 1.0 - 0.004 * (u >= 1500)) for u in range(0, 2001, 50)]       # 0.4% gain within the window
        self.assertTrue(C.is_converged(log3, 1000, 0.005, 1000)[0])
        log4 = [(u, 1.0 - 0.006 * (u >= 1500)) for u in range(0, 2001, 50)]       # 0.6% gain
        self.assertFalse(C.is_converged(log4, 1000, 0.005, 1000)[0])
        self.assertEqual(C.is_converged(log4, 0, 0.005, 1000), (False, None))     # disabled


if __name__ == "__main__":
    unittest.main()
