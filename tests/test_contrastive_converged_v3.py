"""Converged contrastive arms (docs/contrastive_converged_protocol_v3.md): the replay of the stopping rule, the frozen
manifest, and run extension (--resume_from): a run cut between two instance resamplings, from a resume state without the
saved batch (as the fixed-budget runs), continues bit-for-bit like the uninterrupted run; an extension that changes
anything but the model name and the stopping rule is refused. Uses the v3 validation split only; never the final-test
sets.

    python -m unittest tests.test_contrastive_converged_v3
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


def train(cwd, suffix, *extra):
    cmd = [sys.executable, str(REPO / "daniel" / "train.py"), "--config", str(REPO / "configs" / "benchmark_v3" / "mct.json"),
           "--n_j", "10", "--n_m", "5", "--preference_dim", "3", "--mo_val_norm", str(NORM), "--mo_val_grid_step", "0.5",
           "--mo_val_first", "2", "--model_suffix", suffix, "--seed_train", "111", "--v3_train_source", "generator",
           "--validate_timestep", "1", "--reset_env_timestep", "2", "--num_envs", "3", "--finetune_objective",
           "contrastive", "--cl_k", "3", "--resume", "True", "--device", "cpu", *extra]
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=1200)


class Replay(unittest.TestCase):
    def test_rule_replay(self):
        import freeze_contrastive_converged_v3 as F
        flat = [{"update": u, "selection_metric": -0.5} for u in range(0, 1501, 50)]
        self.assertEqual(F.replay(flat), 1000)                             # no improvement: stops at min_updates
        rising = [{"update": u, "selection_metric": -0.1 - u / 1e4} for u in range(0, 3001, 50)]
        self.assertIsNone(F.replay(rising))

    def test_manifest_and_modes(self):
        import freeze_contrastive_converged_v3 as F
        m = F.load()
        self.assertEqual(m["rule"]["cap"], 8000)
        modes = {k: (v["mode"], v["replay_stop_update"]) for k, v in m["arms"].items()}
        self.assertEqual(modes, {"CCB": ("continue", None), "CCD": ("continue", None), "CCE": ("continue", None),
                                 "CCTA": ("continue", None), "CCSBAVG": ("rerun", 1800), "CCTIES": ("rerun", 2500)})


class Extension(unittest.TestCase):
    def test_mid_cycle_extension_is_bit_exact_and_refuses_other_changes(self):
        with tempfile.TemporaryDirectory() as d:
            ref = train(d, "bv3_ccref_s111", "--max_updates", "5")           # uninterrupted
            self.assertEqual(ref.returncode, 0, ref.stdout[-2000:] + ref.stderr[-2000:])
            src = train(d, "bv3_ccsrc_s111", "--max_updates", "3")           # stops between two resamplings (3 % 2)
            self.assertEqual(src.returncode, 0, src.stdout[-2000:] + src.stderr[-2000:])
            old = Path(d) / "old.resume.pth"
            strip = subprocess.run(                                          # as the fixed-budget runs' resume states:
                [sys.executable, "-c",                                       # no saved instance batch
                 "import sys; sys.argv = ['strip', '--device', 'cpu']; import params, torch, benchmark_v3; "
                 f"s = torch.load({str(Path(d) / 'trained_network' / 'SD2' / '10x5+mix+bv3_ccsrc_s111.resume.pth')!r}, "
                 "map_location='cpu', weights_only=False); assert s.get('cl_batch') is not None; del s['cl_batch']; "
                 f"torch.save(s, {str(old)!r})"], cwd=REPO / "daniel", capture_output=True, text=True, timeout=600)
            self.assertEqual(strip.returncode, 0, strip.stderr[-2000:])
            ext = train(d, "bv3_ccext_s111", "--max_updates", "5", "--converge_window", "1000", "--min_updates", "1000",
                        "--resume_from", str(old))
            self.assertEqual(ext.returncode, 0, ext.stdout[-2000:] + ext.stderr[-2000:])
            self.assertIn("rebuilt the instance batch of update 3", ext.stdout)
            tl = Path(d) / "train_log" / "SD2"
            rec = json.loads((tl / "10x5+mix+bv3_ccext_s111.extension.json").read_text())
            self.assertTrue(rec["exact"]); self.assertEqual(rec["logged_update"], 3)
            hv = lambda n: [(e["update"], e["hv"]) for e in json.loads((tl / f"valiquality_10x5+mix+{n}.json").read_text())]
            self.assertEqual(hv("bv3_ccext_s111"), hv("bv3_ccref_s111"))      # bit-for-bit the uninterrupted run
            st = json.loads((tl / "10x5+mix+bv3_ccext_s111.status.json").read_text())
            self.assertEqual((st["status"], st["update"]), ("NOT_CONVERGED", 5))   # hit the cap: reported, not converged
            bad = train(d, "bv3_ccbad_s111", "--max_updates", "5", "--lr", "0.001", "--resume_from", str(old))
            self.assertNotEqual(bad.returncode, 0)
            self.assertIn("differs in ['lr']", bad.stdout + bad.stderr)


if __name__ == "__main__":
    unittest.main()
