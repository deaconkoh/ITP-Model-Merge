"""World-model M3 (docs/world_model_m3_protocol_v3.md):
- Part A on the Original specialists reproduces the committed E0 seed-111 front exactly;
- the per-sample mixture (batched path) equals the grouped reference and the plain interpolated DANIEL;
- gradients reach each specialist in proportion to its weight (w = (1,0,0): exactly zero for C and T); the preference
  columns start at zero;
- a --merge_mix contrastive run works and resumes, its update-0 validation HV equals Part A's on the same instances, and
  evaluate_mo_v3 loads its bundle;
- corner split and ridge barrier on hand examples; the frozen settings verify.
The unchanged MO contrastive path (pinned weights) and DANIEL defaults are covered by tests.test_world_model_m2_v3 and
tests.test_contrastive_v3, which run in the same pre-flight. CPU, tiny runs on the v3 validation/evaluation splits;
never the final-test sets.

    python -m unittest tests.test_world_model_m3_v3
"""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))
PY = sys.executable
NORM = REPO / "results" / "warmstart_v3" / "norm" / "validation_normalisation_points_10x5.json"
SPEC = [REPO / "checkpoints" / "specialists_v3" / f"10x5+mix+bv3_spec_{o}_s111.pth" for o in "mct"]
ENV = {"OMP_NUM_THREADS": "4", "PATH": "/usr/bin:/bin"}

MIX_CHECK = r"""
import sys, json
sys.argv = ["m", "--device", "cpu", "--obs_version", "v3", "--reward_version", "v3", "--goal", "mct", "--preference_dim", "3"]
import numpy as np, torch
from params import configs
import benchmark_v3 as B
from fjsp_env_same_op_nums import FJSPEnvForSameOpNums
from model.main_model import DANIEL
from mixture import MixturePolicy, load_specialists, mix_state_dicts
from warmstart import PREF_LAYERS
paths = sys.stdin.read().split()
nets, parents = load_specialists(paths, configs, "cpu")
mix, ref = MixturePolicy(nets), MixturePolicy(nets, grouped=True)
data = B.load_split("validation", "10x5")
w = np.array([[1, 0, 0], [0.2, 0.5, 0.3], [0.2, 0.5, 0.3], [0, 0, 1], [1 / 3, 1 / 3, 1 / 3], [0.6, 0.1, 0.3]])
env = FJSPEnvForSameOpNums(10, 5)
env.set_initial_data(**data.env_kwargs(list(range(len(w))), "mct"))
st = env.set_preference(w)
for _ in range(5):
    m = st.dynamic_pair_mask_tensor.reshape(len(w), -1).numpy()
    st, _, _ = env.step(np.array([int(np.flatnonzero(~m[b])[0]) for b in range(len(w))]))
K = ("fea_j", "op_mask", "candidate", "fea_m", "mch_mask", "comp_idx", "dynamic_pair_mask", "fea_pairs", "pref")
x = {k: getattr(st, f"{k}_tensor") for k in K}
out = {}
with torch.no_grad():
    pb, vb = mix(**x); pg, vg = ref(**x)
    out["batched_vs_grouped"] = float(max((pb - pg).abs().max(), (vb - vg).abs().max()))
    sds = [n.state_dict() for n in nets]
    d = []
    for r in range(len(w)):
        plain = DANIEL(configs); plain.load_state_dict(mix_state_dicts(sds, w[r])); plain.eval()
        pp, _ = plain(**{k: v[r:r + 1] for k, v in x.items()})
        d.append(float((pb[r:r + 1] - pp).abs().max()))
    out["batched_vs_interpolated"] = max(d)
out["pref_columns_zero"] = all(float(dict(n.named_parameters())[k][:, -3:].abs().max()) == 0.0 for n in nets for k in PREF_LAYERS)
xa = {k: v[[0]] for k, v in x.items()}                            # w = (1, 0, 0)
pi, _ = mix(**xa); torch.log(pi.clamp_min(1e-30)).max().backward()
g = [sum(float(p.grad.abs().sum()) for p in n.parameters() if p.grad is not None) for n in nets]
out["grad_corner"] = g
for n in nets: n.zero_grad()
pi, _ = mix(**{k: v[[1]] for k, v in x.items()}); torch.log(pi.clamp_min(1e-30)).max().backward()
out["grad_mixed_nonzero"] = [sum(float(p.grad.abs().sum()) for p in n.parameters() if p.grad is not None) > 0 for n in nets]
print(json.dumps(out))
"""


def train(cwd, *args):
    r = subprocess.run([PY, str(REPO / "daniel" / "train.py"), *args], cwd=cwd, capture_output=True, text=True,
                       timeout=1800, env=ENV)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    return r


def setUpModule():
    global D, MIX, MM_ARGS
    D = tempfile.TemporaryDirectory()
    r = subprocess.run([PY, "-c", MIX_CHECK], cwd=REPO / "daniel", input=" ".join(map(str, SPEC)), capture_output=True,
                       text=True, timeout=900, env=ENV)
    assert r.returncode == 0, r.stderr[-3000:]
    MIX = json.loads(r.stdout.strip().splitlines()[-1])
    # Part A rollouts: Original specialists, 5 evaluation instances and 2 validation instances
    for split, n in (("evaluation", 5), ("validation", 2)):
        subprocess.run([PY, str(REPO / "tools" / "world_model_m3_part_a.py"), "--sets", "orig", "--splits", split,
                        "--first", str(n), "--out", str(Path(D.name) / "part_a"), "--device", "cpu"],
                       check=True, capture_output=True, timeout=1800, env=ENV)
    MM_ARGS = ["--config", str(REPO / "configs/benchmark_v3/mct.json"), "--n_j", "10", "--n_m", "5", "--seed_train", "111",
               "--v3_train_source", "generator", "--preference_dim", "3", "--mo_val_norm", str(NORM),
               "--mo_val_first", "2", "--finetune_objective", "contrastive", "--merge_mix", ",".join(map(str, SPEC)),
               "--model_suffix", "bv3_m3t_s111", "--validate_timestep", "1", "--reset_env_timestep", "2",
               "--num_envs", "2", "--cl_k", "2", "--max_updates", "4", "--chunk_updates", "2", "--resume", "True",
               "--device", "cpu"]
    train(D.name, *MM_ARGS, "--budget_checkpoints", "0")          # updates 0-2
    train(D.name, *MM_ARGS)                                        # resumes: updates 2-4


def tearDownModule():
    D.cleanup()


class Mixture(unittest.TestCase):
    def test_batched_equals_grouped_and_interpolated(self):
        self.assertLess(MIX["batched_vs_grouped"], 1e-5)
        self.assertLess(MIX["batched_vs_interpolated"], 1e-5)

    def test_gradients_and_pref_columns(self):
        self.assertTrue(MIX["pref_columns_zero"])
        self.assertGreater(MIX["grad_corner"][0], 0.0)
        self.assertEqual(MIX["grad_corner"][1:], [0.0, 0.0])          # w = (1,0,0): C and T get exactly zero
        self.assertEqual(MIX["grad_mixed_nonzero"], [True, True, True])


class PartA(unittest.TestCase):
    def test_reproduces_e0_seed_111(self):
        a = np.load(Path(D.name) / "part_a" / "evaluation_10x5" / "10x5+mix+bv3_wm3A_orig_s111.npz")
        e = np.load(REPO / "results/warmstart_v3/eval/evaluation_10x5/10x5+mix+bv3_ws_E0_s111.npz")
        self.assertTrue(np.array_equal(a["objectives"], e["objectives"][:5]))
        self.assertTrue(np.array_equal(a["hv"], e["hv"][:5]))
        self.assertTrue(np.array_equal(a["prefs"], e["prefs"]))


class PartB(unittest.TestCase):
    def tl(self, f):
        return json.loads((Path(D.name) / "train_log" / "SD2" / f).read_text())

    def test_run_resumes_and_records_mode(self):
        st = self.tl("10x5+mix+bv3_m3t_s111.status.json")
        self.assertEqual(st["update"], 4)
        val = self.tl("valiquality_10x5+mix+bv3_m3t_s111.json")
        self.assertEqual([e["update"] for e in val], [0, 1, 2, 3, 4])
        log = self.tl("reward_10x5+mix+bv3_m3t_s111.json")
        self.assertAlmostEqual(log[0]["cl"]["loss"], 0.6931471805599453, places=5)   # policy = reference: log 2
        run = self.tl("10x5+mix+bv3_m3t_s111.run.json")
        self.assertEqual(len(run["parents"]), 3)
        self.assertEqual([p["init_mode"] for p in run["parents"]], ["merge_mix"] * 3)

    def test_update0_matches_part_a(self):
        val = self.tl("valiquality_10x5+mix+bv3_m3t_s111.json")
        pa = json.loads((Path(D.name) / "part_a" / "validation_10x5" / "10x5+mix+bv3_wm3A_orig_s111_g025.json").read_text())
        self.assertAlmostEqual(val[0]["hv"], pa["hv_mean"], places=9)

    def test_evaluate_loads_bundle(self):
        tn = Path(D.name) / "trained_network" / "SD2"
        r = subprocess.run([PY, "evaluate_mo_v3.py", "--model", str(tn / "10x5+mix+bv3_m3t_s111@u0.pth"), "--split",
                            "validation", "--first", "2", "--grid_step", "0.25", "--out", str(Path(D.name) / "ev"),
                            "--device", "cpu", "--norm", str(NORM)], cwd=REPO / "daniel", capture_output=True, text=True,
                           timeout=1800, env=ENV)
        self.assertEqual(r.returncode, 0, r.stderr[-2000:])
        rec = json.loads(r.stdout.strip().splitlines()[-1]) if r.stdout.strip().startswith("{") else None
        out = json.loads((Path(D.name) / "ev" / "validation_10x5" / "10x5+mix+bv3_m3t_s111@u0.json").read_text())
        pa = json.loads((Path(D.name) / "part_a" / "validation_10x5" / "10x5+mix+bv3_wm3A_orig_s111_g025.json").read_text())
        self.assertAlmostEqual(out["hv_mean"], pa["hv_mean"], places=9)   # start mix = the plain weight mix
        del rec


class Metrics(unittest.TestCase):
    def test_corner_split(self):
        import analyze_world_model_m3_v3 as A
        import mo_metrics as MM
        c = A.corner_mask(MM.simplex_grid(0.1), 0.8)
        self.assertEqual((int(c.sum()), int((~c).sum())), (18, 48))

    def test_ridge_barrier(self):
        import analyze_world_model_m3_v3 as A
        import mo_metrics as MM
        grid = MM.simplex_grid(0.1)
        norm = {"k": {"ideal": {"makespan": 0, "carbon": 0, "tardiness": 0},
                      "nadir": {"makespan": 1, "carbon": 1, "tardiness": 1}}}
        pts = np.zeros((1, len(grid), 3))
        for g, w in enumerate(grid):                 # makespan = 1 - w_M, carbon = 1 - w_C: s(t) = 2t(1-t) on M-C
            pts[0, g] = [1 - w[0], 1 - w[1], 0.5]
        r = A.ridge(pts, ["k"], norm, grid, "mc")
        self.assertAlmostEqual(r["barrier"], 0.5, places=9)         # max of 2t(1-t) at t = 0.5; ends both 0
        self.assertAlmostEqual(r["dip_a"], 0.0, places=9)           # each objective alone is linear in t
        pts[0, :, :2] = 0.3
        self.assertAlmostEqual(A.ridge(pts, ["k"], norm, grid, "mc")["barrier"], 0.0, places=9)

    def test_settings_verify(self):
        import freeze_world_model_m3_v3 as F
        m = F.load()
        self.assertEqual(m["criteria"]["mergeable"]["regret_ratio"], 0.90)
        self.assertEqual(m["criteria"]["faster"]["target_hv"], 0.542)
        self.assertEqual(m["part_b"]["rule"]["cap"], 8000)
        self.assertEqual(m["part_b"]["contrastive"]["pref_dirichlet"], 1.0)


if __name__ == "__main__":
    unittest.main()
