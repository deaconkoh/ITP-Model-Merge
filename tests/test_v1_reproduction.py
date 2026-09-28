"""v1 must stay the default and reproduce earlier results (results/v2_fixes.md).

Re-evaluates C1@1000 (s4t_spec_m_k125s060_s<seed>@u1000, 4 seeds) greedily on the 49 HELD-OUT instances with the
default (v1) observations and requires makespan, carbon and tardiness identical to the Step 4 result files, and the
Step 4 score identical to the one computed from those files. Needs CUDA and the checkpoints; skipped otherwise.

    python -m unittest tests.test_v1_reproduction
"""
import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
CK = REPO / "daniel/trained_network/SD2"
S4 = REPO / "daniel/test_results/SD2/trainvali-k125s060_10x5+carbon+priority"
SEEDS = [111, 222, 333, 444]


def _available():
    if importlib.util.find_spec("torch") is None:
        return False
    import torch
    return torch.cuda.is_available() and all(
        (CK / f"10x5+carbon+priority+s4t_spec_m_k125s060_s{s}@u1000.pth").exists() for s in SEEDS)


@unittest.skipUnless(_available(), "needs CUDA and the Step 4 checkpoints")
class V1ReproducesStep4(unittest.TestCase):
    def test_c1_at_1000_held_out(self):
        argv = sys.argv[:]
        sys.path.insert(0, str(REPO / "tools"))
        import ceiling_sampling as CS                          # v1 defaults (configs/canonical/mc.json)
        import ceiling_scorer as SC
        sys.argv = argv
        self.assertEqual(CS.P.configs.obs_version, "v1")
        self.assertEqual(CS.P.configs.reward_version, "v1")
        for s in SEEDS:
            with self.subTest(seed=s):
                g = CS.greedy_c1(s)
                ref = np.load(S4 / f"Result_DANIELG+10x5+carbon+priority+s4t_spec_m_k125s060_s{s}@u1000_trainvali-"
                                   f"k125s060_10x5+carbon+priority.npy")[:, [0, 1, 3]][SC.HELD]
                mine = np.stack([g["mk"], g["cb"], g["tard"]], 1)
                np.testing.assert_array_equal(mine, ref)
                r = SC.reference(s)
                score = lambda x: np.array([SC.score(s, i, *x[k], ref=r) for k, i in enumerate(SC.HELD)])
                np.testing.assert_array_equal(score(mine), score(ref))


if __name__ == "__main__":
    unittest.main()
