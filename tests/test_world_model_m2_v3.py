"""World-model M2 (docs/world_model_m2_protocol_v3.md):
- single-objective contrastive pairs are exactly "better on the goal by more than the margin";
- the preference-conditioned (MO) contrastive path is unchanged (pinned weights, checked against the pre-change
  commit 61214dd);
- a single-objective contrastive run works without a preference input, records its mode and logs trunk drift;
- the seed-111 bases equal M1's start and trunk bundles and differ only in the trunk;
- both bases load bit-identically with --init_mode exact;
- the frozen settings verify.
CPU, tiny runs on the v3 validation split; never the final-test sets.

    python -m unittest tests.test_world_model_m2_v3
"""
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))
NORM = REPO / "results" / "warmstart_v3" / "norm" / "validation_normalisation_points_10x5.json"
MO_PIN = "129c3de910a72eed"          # 2-update MO contrastive run below, identical before (61214dd) and after the change
COMMON = ["--n_j", "10", "--n_m", "5", "--seed_train", "111", "--v3_train_source", "generator", "--resume", "True",
          "--device", "cpu", "--finetune_objective", "contrastive"]


def train(cwd, *args):
    r = subprocess.run([sys.executable, str(REPO / "daniel" / "train.py"), *COMMON, *args], cwd=cwd, capture_output=True,
                       text=True, timeout=1800, env={"OMP_NUM_THREADS": "4", "PATH": "/usr/bin:/bin"})
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]


def torch_load(path):
    """Loads a checkpoint where train.py's modules resolve (as the other v3 tests do)."""
    r = subprocess.run([sys.executable, "-c", "import sys, json, hashlib; sys.argv = ['i', '--device', 'cpu']; import params, "
                        f"torch; b = torch.load({str(path)!r}, map_location='cpu', weights_only=False); "
                        "sd = {k: v for k, v in b.get('state_dict', {}).items() if hasattr(v, 'numpy')}; "
                        "print(json.dumps({'weights': hashlib.sha256(b''.join(k.encode() + v.numpy().tobytes() for k, v in "
                        "sorted(sd.items()))).hexdigest()[:16], 'identity': b.get('identity'), "
                        "'parts': {k: hashlib.sha256(v.numpy().tobytes()).hexdigest() for k, v in sd.items()}}))"],
                       cwd=REPO / "daniel", capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stderr[-2000:]
    return json.loads(r.stdout.strip().splitlines()[-1])


def setUpModule():
    global D
    D = tempfile.TemporaryDirectory()
    train(D.name, "--config", str(REPO / "configs/benchmark_v3/mct.json"), "--preference_dim", "3", "--mo_val_norm", str(NORM),
          "--mo_val_grid_step", "0.5", "--mo_val_first", "2", "--model_suffix", "bv3_m2pin_s111", "--validate_timestep", "1",
          "--reset_env_timestep", "2", "--num_envs", "3", "--cl_k", "3", "--max_updates", "2")
    train(D.name, "--config", str(REPO / "configs/benchmark_v3/t.json"), "--model_suffix", "bv3_m2so_s111",
          "--validate_timestep", "1", "--reset_env_timestep", "2", "--num_envs", "3", "--cl_k", "3", "--max_updates", "3")
    subprocess.run([sys.executable, str(REPO / "tools" / "world_model_m2_bases.py"), "--seeds", "111", "--out",
                    str(Path(D.name) / "bases")], check=True, capture_output=True, timeout=600)
    for arm in ("shared_random_base", "world_model_base"):
        train(D.name, "--config", str(REPO / "configs/benchmark_v3/m.json"), "--model_suffix", f"bv3_m2{arm[:5]}_s111",
              "--init_from", str(Path(D.name) / "bases" / f"{arm}_s111.pth"), "--init_mode", "exact", "--max_updates", "0",
              "--budget_checkpoints", "0", "--num_envs", "2", "--cl_k", "2")


def tearDownModule():
    D.cleanup()


def tn(name):
    return Path(D.name) / "trained_network" / "SD2" / f"10x5+mix+{name}"


class Pairs(unittest.TestCase):
    def test_single_objective_pairs(self):
        sys.path.insert(0, str(REPO / "daniel"))
        import contrastive as CL
        f = [0.50, 0.505, 0.60, 0.40]
        self.assertEqual(sorted(CL.single_objective_pairs(f, 0.01)),
                         [(0, 2), (1, 2), (3, 0), (3, 1), (3, 2)])       # 0 vs 1 differ by 0.005 < margin: no pair
        self.assertEqual(CL.single_objective_pairs([0.3, 0.3], 0.01), [])


class Runs(unittest.TestCase):
    def test_mo_contrastive_unchanged(self):
        self.assertEqual(torch_load(tn("bv3_m2pin_s111.pth"))["weights"], MO_PIN)
        self.assertNotIn("cl_pairs", torch_load(tn("bv3_m2pin_s111.resume.pth"))["identity"])

    def test_single_objective_run(self):
        self.assertEqual(torch_load(tn("bv3_m2so_s111.resume.pth"))["identity"]["cl_pairs"], "single_objective")
        val = json.loads((Path(D.name) / "train_log/SD2/valiquality_10x5+mix+bv3_m2so_s111.json").read_text())
        self.assertEqual(val[0]["trunk_drift"], 0.0)
        self.assertGreater(val[-1]["trunk_drift"], 0.0); self.assertGreater(val[-1]["actor_drift"], 0.0)
        log = json.loads((Path(D.name) / "train_log/SD2/reward_10x5+mix+bv3_m2so_s111.json").read_text())
        self.assertAlmostEqual(log[0]["cl"]["loss"], 0.6931471805599453, places=5)      # policy = reference: log 2
        self.assertNotIn("pref", log[0]); self.assertGreater(log[0]["cl"]["pairs"], 0)

    def test_option_checks(self):
        r = subprocess.run([sys.executable, "-c", "import sys; sys.argv = ['p', '--obs_version', 'v3', '--reward_version', "
                            "'v3', '--goal', 'mct', '--finetune_objective', 'contrastive', '--v3_train_source', "
                            "'generator']; import params"], cwd=REPO / "daniel", capture_output=True, text=True, timeout=300)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("single-objective --goal m, c or t", r.stderr)


class Bases(unittest.TestCase):
    def test_bases_match_m1_and_differ_only_in_trunk(self):
        c = torch_load(Path(D.name) / "bases" / "shared_random_base_s111.pth")
        w = torch_load(Path(D.name) / "bases" / "world_model_base_s111.pth")
        m1s = torch_load(REPO / "checkpoints/world_model_m1_v3/world_model_trunk_start.pth")
        m1t = torch_load(REPO / "checkpoints/world_model_m1_v3/world_model_trunk.pth")
        self.assertEqual(c["weights"], m1s["weights"]); self.assertEqual(w["weights"], m1t["weights"])
        diff = {k.split(".")[0] for k in c["parts"] if c["parts"][k] != w["parts"][k]}
        self.assertEqual(diff, {"feature_exact"})

    def test_bases_load_exactly(self):
        for arm, short in (("shared_random_base", "share"), ("world_model_base", "world")):
            b = torch_load(Path(D.name) / "bases" / f"{arm}_s111.pth")
            u0 = torch_load(tn(f"bv3_m2{short}_s111@u0.pth"))
            self.assertEqual(b["weights"], u0["weights"], arm)

    def test_settings_verify(self):
        import freeze_world_model_m2_v3 as F
        m = F.load()
        self.assertEqual(m["rule"]["cap"], 8000)
        self.assertEqual(m["criteria"]["faster"]["ratio"], 0.70)
        self.assertEqual(m["contrastive"]["cl_k"], 8)


if __name__ == "__main__":
    unittest.main()
