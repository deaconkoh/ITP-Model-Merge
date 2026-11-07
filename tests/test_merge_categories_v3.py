"""Merge categories (docs/merge_categories_protocol_v3.md): Task Arithmetic and TIES-Merging on toy state dicts, the
preference columns after --init_mode full, the 0-update contrastive probe and the frozen manifest. Uses the v3
validation split only; never the final-test sets.

    python -m unittest tests.test_merge_categories_v3
"""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def setUpModule():
    sys.path.insert(0, str(REPO / "daniel"))
    global torch, W
    import torch
    import warmstart as W


def sd(*vals):
    """A two-tensor toy state dict: 'a' [2] and 'b' [2, 1]."""
    return {"a": torch.tensor(vals[:2], dtype=torch.float32), "b": torch.tensor(vals[2:], dtype=torch.float32)[:, None]}


class TaskArithmetic(unittest.TestCase):
    def test_third_equals_uniform_average(self):
        base, sds = sd(1, 2, 3, 4), [sd(2, 0, 3, 7), sd(-1, 5, 0, 4), sd(1, 2, 9, -2)]
        ta = W.task_arithmetic(base, sds, 1 / 3)
        avg = W.uniform_merge(sds)
        for k in base:
            self.assertTrue(torch.allclose(ta[k], avg[k], atol=1e-6), k)

    def test_definition(self):
        base, sds = sd(0, 0, 0, 0), [sd(1, 2, 3, 4), sd(1, 0, 0, 0), sd(0, 0, 0, 1)]
        ta = W.task_arithmetic(base, sds, 0.5)
        self.assertEqual(ta["a"].tolist(), [1.0, 1.0]); self.assertEqual(ta["b"].flatten().tolist(), [1.5, 2.5])


class Ties(unittest.TestCase):
    def test_hand_worked_example(self):
        # task vectors over the 4 entries (a0, a1, b0, b1); keep 50% = the 2 largest magnitudes per task
        base = sd(0, 0, 0, 0)
        t1 = sd(4, -1, 3, 0.5)          # keeps a0 = 4, b0 = 3
        t2 = sd(-2, 6, 1, -5)           # keeps a1 = 6, b1 = -5
        t3 = sd(-3, -0.1, 0, 2)         # keeps a0 = -3, b1 = 2
        m = W.ties_merge(base, [t1, t2, t3], lam=1.0, keep=0.5)
        # a0: trimmed {4, -3} -> elected + -> mean(4) = 4
        # a1: trimmed {6} -> 6;  b0: trimmed {3} -> 3
        # b1: trimmed {-5, 2} -> sum -3 -> elected - -> mean(-5) = -5
        self.assertEqual(m["a"].tolist(), [4.0, 6.0]); self.assertEqual(m["b"].flatten().tolist(), [3.0, -5.0])
        m2 = W.ties_merge(base, [t1, t2, t3], lam=0.5, keep=0.5)
        self.assertEqual(m2["a"].tolist(), [2.0, 3.0])

    def test_no_kept_value_gives_base(self):
        base = sd(1, 1, 1, 1)
        m = W.ties_merge(base, [sd(5, 1, 1, 1), sd(1, 1, 1, 6)], lam=1.0, keep=0.25)   # b0 and a1: nothing kept
        self.assertEqual(m["a"][1].item(), 1.0); self.assertEqual(m["b"][0, 0].item(), 1.0)
        self.assertEqual(m["a"][0].item(), 5.0); self.assertEqual(m["b"][1, 0].item(), 6.0)

    def test_keep_all_same_signs_is_mean(self):
        base, sds = sd(1, 1, 1, 1), [sd(2, 3, 4, 5), sd(3, 2, 2, 2), sd(4, 4, 3, 1.5)]
        m = W.ties_merge(base, sds, lam=0.7, keep=1.0)
        avg = W.uniform_merge(sds)
        for k in base:
            self.assertTrue(torch.allclose(m[k], base[k] + 0.7 * (avg[k] - base[k]), atol=1e-6), k)


class Probe(unittest.TestCase):
    def test_zero_update_contrastive_probe(self):
        norm = REPO / "results" / "warmstart_v3" / "norm" / "validation_normalisation_points_10x5.json"
        with tempfile.TemporaryDirectory() as d:
            src = REPO / "checkpoints" / "specialists_v3" / "10x5+mix+bv3_spec_m_s111.pth"    # stands in for a merge
            from checkpointing import unwrap_checkpoint
            sd0, _ = unwrap_checkpoint(torch.load(src, map_location="cpu", weights_only=False))
            r = subprocess.run(
                [sys.executable, str(REPO / "daniel" / "train.py"), "--config", str(REPO / "configs" / "benchmark_v3" / "mct.json"),
                 "--n_j", "10", "--n_m", "5", "--preference_dim", "3", "--mo_val_norm", str(norm), "--mo_val_grid_step", "0.5",
                 "--mo_val_first", "2", "--model_suffix", "bv3_mctest_s111", "--seed_train", "111", "--v3_train_source",
                 "generator", "--validate_timestep", "50", "--converge_window", "0", "--finetune_objective", "contrastive",
                 "--init_from", str(src), "--init_mode", "full", "--max_updates", "0", "--device", "cpu"],
                cwd=d, capture_output=True, text=True, timeout=1200)
            self.assertEqual(r.returncode, 0, r.stdout[-2000:] + r.stderr[-2000:])
            name = "10x5+mix+bv3_mctest_s111"
            val = json.loads((Path(d) / "train_log" / "SD2" / f"valiquality_{name}.json").read_text())
            self.assertEqual([e["update"] for e in val], [0])                    # validation at update 0 only
            st = json.loads((Path(d) / "train_log" / "SD2" / f"{name}.status.json").read_text())
            self.assertEqual((st["status"], st["update"]), ("FINISHED", 0))
            out, _ = unwrap_checkpoint(torch.load(Path(d) / "trained_network" / "SD2" / f"{name}.pth",
                                                  map_location="cpu", weights_only=False))
            for k in W.PREF_LAYERS:                                              # preference columns zero, rest copied
                self.assertTrue(torch.equal(out[k][:, -3:], torch.zeros_like(out[k][:, -3:])), k)
                self.assertTrue(torch.equal(out[k][:, :-3], sd0[k]), k)
            for k, v in sd0.items():
                if k not in W.PREF_LAYERS:
                    self.assertTrue(torch.equal(out[k], v), k)


class Manifest(unittest.TestCase):
    def test_frozen_manifest_verifies(self):
        sys.path.insert(0, str(REPO / "tools"))
        import freeze_merge_categories_v3 as F
        m = F.load()
        self.assertEqual(m["budget"]["updates"], 3350)
        self.assertEqual(m["method"]["lambda_grid"], [0.3, 0.5, 0.7, 1.0])
        self.assertEqual(m["contrastive"]["constants"]["cl_k"], 8)

    def test_committed_checkpoints_match_their_hashes(self):
        f = REPO / "checkpoints" / "merge_categories_v3" / "SHA256SUMS.json"
        if not f.exists():
            return                                                              # before the run: nothing committed yet
        import hashlib
        for name, h in json.loads(f.read_text())["files"].items():
            self.assertEqual(hashlib.sha256((f.parent / name).read_bytes()).hexdigest(), h, name)


if __name__ == "__main__":
    unittest.main()
