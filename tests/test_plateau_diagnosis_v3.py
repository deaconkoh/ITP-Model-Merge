"""Plateau diagnosis (docs/plateau_diagnosis_protocol_v3.md):
- snapshots in contrastive mode (--budget_checkpoints) and per-update drift logging (--cl_log_drift) leave training
  unchanged: final weights, validation and loss are identical to a run without them; the snapshots hold the policy at
  those updates; the update-0 snapshot is not overwritten on a resume; the logged drift equals the drift recomputed from
  the snapshots (merge mix: over its three networks);
- the start-network rebuild (--max_updates 0, on the GPU as the run was) gives the network whose hash From scratch
  logged;
- a saved final state unwrapped as a bundle loads in evaluate_mo_v3 and holds the final weights;
- each Part B arm differs from the M3 World-model base mix run in exactly its one change (plus fixed length, no
  stopping rule, snapshots, drift logging and the run name);
- the frozen settings verify.
CPU, tiny runs on the v3 validation split; never the final-test sets.

    python -m unittest tests.test_plateau_diagnosis_v3
"""
import json
import re
import subprocess
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools")); sys.path.insert(0, str(REPO / "daniel"))
PY = sys.executable
NORM = REPO / "results" / "warmstart_v3" / "norm" / "validation_normalisation_points_10x5.json"
SPEC = [REPO / "checkpoints" / "specialists_v3" / f"10x5+mix+bv3_spec_{o}_s111.pth" for o in "mct"]
ENV = {"OMP_NUM_THREADS": "4", "PATH": "/usr/bin:/bin"}
BASE = ["--config", str(REPO / "configs/benchmark_v3/mct.json"), "--n_j", "10", "--n_m", "5", "--seed_train", "111",
        "--v3_train_source", "generator", "--preference_dim", "3", "--mo_val_norm", str(NORM), "--mo_val_first", "2",
        "--finetune_objective", "contrastive", "--merge_mix", ",".join(map(str, SPEC)), "--validate_timestep", "2",
        "--reset_env_timestep", "2", "--num_envs", "2", "--cl_k", "2", "--max_updates", "4", "--chunk_updates", "2",
        "--resume", "True", "--device", "cpu"]


def train(cwd, *args):
    r = subprocess.run([PY, str(REPO / "daniel" / "train.py"), *args], cwd=cwd, capture_output=True, text=True,
                       timeout=1800, env=ENV)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]


def load(p):
    import freeze_plateau_diagnosis_v3 as F
    return F.torch_load(p)


def setUpModule():
    global D
    D = tempfile.TemporaryDirectory()
    plain, inst = Path(D.name) / "plain", Path(D.name) / "inst"
    plain.mkdir(); inst.mkdir()
    for _ in range(2):                                                    # two chunks of 2 updates (resume)
        train(plain, *BASE, "--model_suffix", "bv3_pdt_s111")
        train(inst, *BASE, "--model_suffix", "bv3_pdt_s111", "--budget_checkpoints", "0,2,4", "--cl_log_drift", "True")


def tearDownModule():
    D.cleanup()


class Instrumentation(unittest.TestCase):
    def p(self, which, f):
        return Path(D.name) / which / f

    def test_training_unchanged(self):
        a = load(self.p("plain", "trained_network/SD2/10x5+mix+bv3_pdt_s111.resume.pth"))["policy"]
        b = load(self.p("inst", "trained_network/SD2/10x5+mix+bv3_pdt_s111.resume.pth"))["policy"]
        self.assertEqual(a.keys(), b.keys())
        for k in a:
            self.assertTrue(torch.equal(a[k], b[k]), k)
        va = json.loads(self.p("plain", "train_log/SD2/valiquality_10x5+mix+bv3_pdt_s111.json").read_text())
        vb = json.loads(self.p("inst", "train_log/SD2/valiquality_10x5+mix+bv3_pdt_s111.json").read_text())
        self.assertEqual([e["hv"] for e in va], [e["hv"] for e in vb])
        la = json.loads(self.p("plain", "train_log/SD2/reward_10x5+mix+bv3_pdt_s111.json").read_text())
        lb = json.loads(self.p("inst", "train_log/SD2/reward_10x5+mix+bv3_pdt_s111.json").read_text())
        self.assertEqual([e["cl"]["loss"] for e in la], [e["cl"]["loss"] for e in lb])
        self.assertNotIn("trunk_drift", la[-1]["cl"])

    def test_snapshots_and_drift(self):
        tn = self.p("inst", "trained_network/SD2")
        fin = load(tn / "10x5+mix+bv3_pdt_s111.resume.pth")
        u0, u4 = load(tn / "10x5+mix+bv3_pdt_s111@u0.pth"), load(tn / "10x5+mix+bv3_pdt_s111@u4.pth")
        self.assertTrue((tn / "10x5+mix+bv3_pdt_s111@u2.pth").exists())
        self.assertEqual((u0["update"], u4["update"]), (0, 4))
        for k in fin["policy"]:
            self.assertTrue(torch.equal(u4["state_dict"][k], fin["policy"][k]), k)
        import plateau_diagnosis_v3_runner as R
        self.assertEqual(R.state_hash(u0["state_dict"]), fin["identity"]["cl_reference_sha256"])   # not overwritten
        log = json.loads(self.p("inst", "train_log/SD2/reward_10x5+mix+bv3_pdt_s111.json").read_text())
        self.assertEqual(len(log), 4)
        for name, pre in (("trunk", "feature_exact."), ("actor", "actor.")):
            ks = [k for k in u0["state_dict"] if re.sub(r"^nets\.\d+\.", "", k).startswith(pre)]
            self.assertTrue(all(k.startswith("nets.") for k in ks) and len({k.split(".")[1] for k in ks}) == 3)
            d = sum(float(((u4["state_dict"][k].double() - u0["state_dict"][k].double()) ** 2).sum()) for k in ks) ** .5
            n0 = sum(float((u0["state_dict"][k].double() ** 2).sum()) for k in ks) ** .5
            self.assertAlmostEqual(log[-1]["cl"][f"{name}_drift"], d / n0, places=9)
            self.assertGreater(d, 0.0)

    def test_final_state_bundle_evaluates(self):
        from checkpointing import make_checkpoint_bundle
        fin = load(self.p("inst", "trained_network/SD2/10x5+mix+bv3_pdt_s111.resume.pth"))
        run = json.loads(self.p("inst", "train_log/SD2/10x5+mix+bv3_pdt_s111.run.json").read_text())
        b = make_checkpoint_bundle(fin["policy"], config=run["config"], data_fingerprints={}, update=4,
                                   validation_metrics={}, selection_metric=0.0, repo_root=REPO)
        out = Path(D.name) / "final.pth"; torch.save(b, out)
        u4 = Path(D.name) / "inst/trained_network/SD2/10x5+mix+bv3_pdt_s111@u4.pth"
        hv = []
        for m, o in ((out, "ev1"), (u4, "ev2")):
            r = subprocess.run([PY, "evaluate_mo_v3.py", "--model", str(m), "--split", "validation", "--first", "2",
                                "--grid_step", "0.5", "--out", str(Path(D.name) / o), "--device", "cpu", "--norm", str(NORM)],
                               cwd=REPO / "daniel", capture_output=True, text=True, timeout=1800, env=ENV)
            self.assertEqual(r.returncode, 0, r.stderr[-2000:])
            hv.append(json.loads((Path(D.name) / o / "validation_10x5" / f"{m.stem}.json").read_text())["hv_mean"])
        self.assertEqual(hv[0], hv[1])


class Rebuild(unittest.TestCase):
    def test_from_scratch_start_hash(self):
        import freeze_plateau_diagnosis_v3 as F
        import plateau_diagnosis_v3_runner as R
        r = R.Runner.__new__(R.Runner)
        out = Path(D.name) / "fs_u0.pth"
        # the run was initialised on the GPU (a CPU initialisation gives other weights), so the rebuild needs it too
        r.rebuild_start("results/contrastive_v3/runs/10x5+mix+bv3_cl_CLB_s111/run.json", out, device="cuda")
        b = load(out)
        self.assertEqual(R.state_hash(b["state_dict"]), F.load()["start_network_sha256"]["fs"])


class Settings(unittest.TestCase):
    def test_part_b_one_change_each(self):
        import freeze_plateau_diagnosis_v3 as F
        import plateau_diagnosis_v3_runner as R
        import world_model_m3_v3_runner as M
        m = M.Runner.__new__(M.Runner)
        m.cfg = M.F.load(); m.tagB = "wm3"; m.val_first = 0; m.validate = 50; m.cap = 8000; m.chunk = 500
        base = m.args("wm")

        def as_dict(a):
            return {a[i]: a[i + 1] for i in range(1, len(a), 2)}
        b = as_dict(base)
        r = R.Runner.__new__(R.Runner); r.cfg = F.load(); r.smoke = False
        fixed = {"--model_suffix", "--converge_window", "--converge_tol", "--min_updates", "--max_updates",
                 "--budget_checkpoints", "--cl_log_drift"}
        flag = {"cl_beta": "--cl_beta", "lr": "--lr", "cl_k": "--cl_k"}
        for arm, spec in r.cfg["part_b"]["arms"].items():
            d = as_dict(r.args(arm))
            diff = {k for k in set(b) | set(d) if b.get(k) != d.get(k)} - fixed
            self.assertEqual(diff, {flag[k] for k in spec["change"]}, arm)
            for k, v in spec["change"].items():
                self.assertEqual(float(d[flag[k]]), float(v))
            self.assertEqual((d["--max_updates"], d["--converge_window"], d["--chunk_updates"]), ("1500", "0", "500"))
            self.assertEqual(d["--merge_mix"], b["--merge_mix"])
        self.assertEqual(r.cfg["part_b"]["arms"]["ctrl"]["change"], {})

    def test_settings_verify(self):
        import freeze_plateau_diagnosis_v3 as F
        m = F.load()
        self.assertEqual(m["part_a"]["evaluation"]["sample"], 16)
        self.assertEqual(m["part_b"]["updates"], 1500)
        self.assertEqual(len(m["part_a"]["checkpoints"]), 18)


if __name__ == "__main__":
    unittest.main()
