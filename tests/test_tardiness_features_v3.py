"""Tardiness features (docs/tardiness_features_protocol_v3.md): --lateness_inputs on appends exactly the five fixed-scale
lateness columns and leaves the 12 existing ones untouched; off (the default) changes nothing, bit for bit; the run
identity records the option only when on; evaluate_v3 loads the new schema only when asked. CPU, tiny runs on the v3
validation/development splits; never the final-test sets.

    python -m unittest tests.test_tardiness_features_v3
"""
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))
import tardiness_diagnosis_v3 as T          # noqa: E402  (sets up params for benchmark v3, goal t)

B, configs = T.B, T.configs
# weights after the 5-update default run below, identical before and after the change (checked against the pre-change commit f5e2920)
DEFAULT_WEIGHTS_SHA = "912a40096a47300e"


def env_pair(size="10x5", n=4):
    data = B.load_split("validation", size)
    idx = np.arange(n)
    envs = []
    for flag in ("off", "on"):
        configs.lateness_inputs = flag
        e = T.FJSPEnvForSameOpNums(*B.SIZES[size]); e.set_initial_data(**data.env_kwargs(idx, "t"))
        envs.append(e)
    configs.lateness_inputs = "off"
    return envs


def train(cwd, suffix, *extra):
    cmd = [sys.executable, str(REPO / "daniel" / "train.py"), "--config", str(REPO / "configs" / "benchmark_v3" / "t.json"),
           "--n_j", "10", "--n_m", "5", "--model_suffix", suffix, "--seed_train", "111", "--v3_train_source", "generator",
           "--validate_timestep", "5", "--reset_env_timestep", "5", "--converge_window", "1000", "--min_updates", "1000",
           "--resume", "True", "--device", "cpu", *extra]
    r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=1800,
                       env={"OMP_NUM_THREADS": "4", "PATH": "/usr/bin:/bin"})
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]


def weights_sha(path):
    import torch
    b = torch.load(path, map_location="cpu", weights_only=False)
    sd = b["state_dict"] if "state_dict" in b else b
    return hashlib.sha256(b"".join(k.encode() + v.numpy().tobytes() for k, v in sorted(sd.items()))).hexdigest()[:16]


def identity(path):
    r = subprocess.run([sys.executable, "-c", "import sys, json; sys.argv = ['i', '--device', 'cpu']; import params, torch, "
                        f"benchmark_v3; print(json.dumps(torch.load({str(path)!r}, map_location='cpu', "
                        "weights_only=False)['identity']))"], cwd=REPO / "daniel", capture_output=True, text=True,
                       timeout=600)
    assert r.returncode == 0, r.stderr[-2000:]
    return json.loads(r.stdout.strip().splitlines()[-1])


def setUpModule():
    global D
    D = tempfile.TemporaryDirectory()
    train(D.name, "bv3_tft_dflt_s111", "--max_updates", "5")
    train(D.name, "bv3_tft_late_s111", "--max_updates", "5", "--lateness_inputs", "on", "--budget_checkpoints", "0")


def tearDownModule():
    D.cleanup()


def tn(name):
    return Path(D.name) / "trained_network" / "SD2" / f"10x5+mix+{name}"


class Features(unittest.TestCase):
    def test_existing_columns_unchanged_and_new_ones_exact(self):
        off, on = env_pair()
        rng = np.random.default_rng(0)
        for step in range(off.number_of_ops):
            self.assertEqual(off.fea_j.shape[2], 12); self.assertEqual(on.fea_j.shape[2], 17)
            self.assertTrue(np.array_equal(off.fea_j, on.fea_j[:, :, :12]), step)
            est, work, _, fin = on.v2_job_state()
            t = on.next_schedule_time[:, None]
            d = on.due_dates
            want = [np.broadcast_to(t, d.shape), d - t, d - est - work, ((t > d) & ~fin).astype(float),
                    (est + work > d).astype(float)]
            for c, w in enumerate(want):
                self.assertTrue(np.array_equal(on.fea_j[:, :, 12 + c], np.take_along_axis(w, on.v2_op_job, 1)),
                                (step, c))
            if step == off.number_of_ops - 1:
                break
            ok = ~off.dynamic_pair_mask.reshape(off.number_of_envs, -1).astype(bool)
            a = np.array([rng.choice(np.flatnonzero(r)) for r in ok])
            off.step(a); on.step(a)
        self.assertGreater(on.fea_j[:, :, 15].sum() + on.fea_j[:, :, 16].sum(), 0)   # some job ends up late

    def test_fixed_scale_not_batch_dependent(self):
        """The new columns of an instance do not depend on the other instances in the batch (no rescaling)."""
        data = B.load_split("validation", "10x5")
        configs.lateness_inputs = "on"
        try:
            a = T.FJSPEnvForSameOpNums(10, 5); a.set_initial_data(**data.env_kwargs([0], "t"))
            b = T.FJSPEnvForSameOpNums(10, 5); b.set_initial_data(**data.env_kwargs([0, 1, 2], "t"))
        finally:
            configs.lateness_inputs = "off"
        self.assertTrue(np.array_equal(a.fea_j[0, :, 12:], b.fea_j[0, :, 12:]))
        due = np.asarray(data.due[0]) / 99.0                                   # time to due at t = 0, raw / 99
        self.assertTrue(np.allclose(a.fea_j[0, ::5, 13], due))


class Runs(unittest.TestCase):
    def test_default_is_bit_identical(self):
        self.assertEqual(weights_sha(tn("bv3_tft_dflt_s111.pth")), DEFAULT_WEIGHTS_SHA)
        self.assertNotIn("lateness_inputs", identity(tn("bv3_tft_dflt_s111.resume.pth")))

    def test_on_run(self):
        self.assertEqual(identity(tn("bv3_tft_late_s111.resume.pth"))["lateness_inputs"], "on")
        import torch
        b = torch.load(tn("bv3_tft_late_s111@u0.pth"), map_location="cpu", weights_only=False)
        self.assertEqual(b["config"]["feature_schema"], "benchmark_v3_f17_m9_p10_late")
        shapes = {k: tuple(v.shape) for k, v in b["state_dict"].items() if v.shape[:1] == (17,)}
        self.assertEqual(len(shapes), 8, shapes)

    def test_evaluator_needs_the_option(self):
        ck = str(tn("bv3_tft_late_s111.pth"))
        base = [sys.executable, "evaluate_v3.py", "--models", ck, "--split", "development", "--size", "10x5",
                "--first", "2", "--out", D.name + "/ev", "--device", "cpu"]
        ok = subprocess.run(base + ["--lateness_inputs", "on"], cwd=REPO / "daniel", capture_output=True, text=True,
                            timeout=900)
        self.assertEqual(ok.returncode, 0, ok.stderr[-2000:])
        bad = subprocess.run(base, cwd=REPO / "daniel", capture_output=True, text=True, timeout=900)
        self.assertNotEqual(bad.returncode, 0)
        self.assertIn("does not match requested", bad.stderr)


class Options(unittest.TestCase):
    def test_option_checks(self):
        for argv in ("['p', '--lateness_inputs', 'on']",
                     "['p', '--lateness_inputs', 'on', '--obs_version', 'v3', '--reward_version', 'v3', "
                     "'--goal', 'mct', '--preference_dim', '3']"):
            r = subprocess.run([sys.executable, "-c", f"import sys; sys.argv = {argv}; import params"],
                               cwd=REPO / "daniel", capture_output=True, text=True, timeout=300)
            self.assertNotEqual(r.returncode, 0, argv)
            self.assertIn("--lateness_inputs on needs benchmark v3", r.stderr)


if __name__ == "__main__":
    unittest.main()
