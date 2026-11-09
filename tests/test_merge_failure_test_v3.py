"""Merge-failure test (docs/merge_failure_test_protocol_v3.md): the three one-change arms build exactly the intended
starting network, the no-anchor loss drops the reference, the defaults change nothing, and the frozen manifest verifies.
CPU, tiny runs on the v3 validation split only; never the final-test sets.

    python -m unittest tests.test_merge_failure_test_v3
"""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))
NORM = REPO / "results" / "warmstart_v3" / "norm" / "validation_normalisation_points_10x5.json"
SOUP = REPO / "checkpoints" / "merge_categories_v3" / "10x5+mix+bv3_mcSBAVG_start_s111.pth"
INIT = ["--init_from", str(SOUP), "--init_mode", "full"]


def train(cwd, suffix, *extra):
    cmd = [sys.executable, str(REPO / "daniel" / "train.py"), "--config", str(REPO / "configs" / "benchmark_v3" / "mct.json"),
           "--n_j", "10", "--n_m", "5", "--preference_dim", "3", "--mo_val_norm", str(NORM), "--mo_val_grid_step", "0.5",
           "--mo_val_first", "2", "--model_suffix", suffix, "--seed_train", "111", "--v3_train_source", "generator",
           "--validate_timestep", "1", "--reset_env_timestep", "2", "--num_envs", "3", "--finetune_objective",
           "contrastive", "--cl_k", "3", "--resume", "True", "--device", "cpu", *extra]
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=1200)


def setUpModule():
    global torch, D
    import torch
    D = tempfile.TemporaryDirectory()
    runs = {"soup": INIT, "rand": [], "sp": INIT + ["--init_shrink", "0.3", "--init_perturb", "0.0001"],
            "pr": INIT + ["--init_pref_columns", "fresh"],
            "dflt": INIT + ["--init_shrink", "1.0", "--init_perturb", "0.0", "--init_pref_columns", "zero"]}
    for k, extra in runs.items():
        r = train(D.name, f"bv3_mft{k}_s111", "--max_updates", "0", "--budget_checkpoints", "0", *extra)
        assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    for k, extra in {"soup1": INIT, "na1": INIT + ["--cl_anchor", "none"],
                     "ref1": INIT + ["--cl_anchor", "reference"]}.items():
        r = train(D.name, f"bv3_mft{k}_s111", "--max_updates", "2", *extra)
        assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]


def tearDownModule():
    D.cleanup()


def net(k):
    b = torch.load(Path(D.name) / "trained_network" / "SD2" / f"10x5+mix+bv3_mft{k}_s111@u0.pth", map_location="cpu",
                   weights_only=False)
    return b["state_dict"] if "state_dict" in b else b


def log(k, kind):
    tl = Path(D.name) / "train_log" / "SD2"
    return json.loads((tl / (f"valiquality_10x5+mix+bv3_mft{k}_s111.json" if kind == "val" else
                             f"reward_10x5+mix+bv3_mft{k}_s111.json")).read_text())


def identity(k):
    """The run identity stored in the resume state (unpickled where train.py's modules resolve, as in
    tests/test_contrastive_converged_v3.py)."""
    f = Path(D.name) / "trained_network" / "SD2" / f"10x5+mix+bv3_mft{k}_s111.resume.pth"
    r = subprocess.run([sys.executable, "-c", "import sys, json; sys.argv = ['i', '--device', 'cpu']; import params, torch, "
                        f"benchmark_v3; print(json.dumps(torch.load({str(f)!r}, map_location='cpu', "
                        "weights_only=False)['identity']))"], cwd=REPO / "daniel", capture_output=True, text=True,
                       timeout=600)
    assert r.returncode == 0, r.stderr[-2000:]
    return json.loads(r.stdout.strip().splitlines()[-1])


class Starts(unittest.TestCase):
    PREF = ("actor.linears.0.weight", "critic.linears.0.weight")

    def test_random_preference_input(self):
        soup, rand, pr = net("soup"), net("rand"), net("pr")
        for k in soup:
            if k in self.PREF:
                self.assertTrue(torch.equal(pr[k][:, -3:], rand[k][:, -3:]), k)       # fresh, as From scratch
                self.assertTrue(torch.equal(pr[k][:, :-3], soup[k][:, :-3]), k)
                self.assertEqual(float(soup[k][:, -3:].abs().sum()), 0.0)            # Model Soup: zero
                self.assertGreater(float(pr[k][:, -3:].abs().sum()), 0.0)
            else:
                self.assertTrue(torch.equal(pr[k], soup[k]), k)                      # everything else: the merge

    def test_shrink_and_perturb(self):
        soup, rand, sp = net("soup"), net("rand"), net("sp")
        for k in soup:
            self.assertTrue(torch.allclose(sp[k], 0.3 * soup[k] + 1e-4 * rand[k], atol=1e-7, rtol=0), k)

    def test_defaults_change_nothing(self):
        soup, d = net("soup"), net("dflt")
        for k in soup:
            self.assertTrue(torch.equal(soup[k], d[k]), k)
        a, b = identity("soup"), identity("dflt")
        self.assertEqual({k: v for k, v in a.items() if k != "model_name"}, {k: v for k, v in b.items() if k != "model_name"})
        for key in ("cl_anchor", "init_shrink", "init_perturb", "init_pref_columns"):
            self.assertNotIn(key, a)                                                  # earlier identities unchanged


class Anchor(unittest.TestCase):
    def test_no_anchor_drops_the_reference(self):
        soup, na = log("soup1", "reward"), log("na1", "reward")
        self.assertAlmostEqual(soup[0]["cl"]["loss"], 0.6931471805599453, places=6)   # policy = reference: log 2
        self.assertNotAlmostEqual(na[0]["cl"]["loss"], 0.6931471805599453, places=4)  # -log sigmoid(beta(l_b - l_w))
        vs, vn = log("soup1", "val"), log("na1", "val")
        self.assertEqual(vs[0]["hv"], vn[0]["hv"])                                    # same start network
        self.assertEqual(identity("na1")["cl_anchor"], "none")

    def test_explicit_reference_is_the_default(self):
        a, b = log("soup1", "val"), log("ref1", "val")
        self.assertEqual([e["hv"] for e in a], [e["hv"] for e in b])                  # bit-for-bit
        self.assertEqual([e["cl"]["loss"] for e in log("soup1", "reward")], [e["cl"]["loss"] for e in log("ref1", "reward")])


class Manifest(unittest.TestCase):
    def test_manifest(self):
        import freeze_merge_failure_test_v3 as F
        m = F.load()
        self.assertEqual(m["rule"]["cap"], 8000)
        self.assertEqual(m["arms"]["MFSP"]["flags"], ["--init_shrink", "0.3", "--init_perturb", "0.0001"])
        self.assertEqual(m["arms"]["MFNA"]["flags"], ["--cl_anchor", "none"])
        self.assertEqual(m["arms"]["MFPR"]["flags"], ["--init_pref_columns", "fresh"])
        self.assertEqual(set(m["diagnostics"]["subjects"]), {"SOUP", "DIST", "RAND", "MFSP", "MFNA", "MFPR"})

    def test_option_checks(self):
        r = subprocess.run([sys.executable, "-c", "import sys; sys.argv = ['p', '--cl_anchor', 'none']; import params"],
                           cwd=REPO / "daniel", capture_output=True, text=True, timeout=300)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("--cl_anchor applies to --finetune_objective contrastive only", r.stderr)


if __name__ == "__main__":
    unittest.main()
