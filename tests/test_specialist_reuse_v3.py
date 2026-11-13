"""Specialist reuse (docs/specialist_reuse_protocol_v3.md):
- the preference-weighted TIES rule on a hand-computed example (weighting before election, sum of agreeing entries,
  zero-weight vectors, all-zero entries, exact corners);
- Rewarded Soups and TIES: a preference change changes the merged weights; each exact corner reproduces its
  specialist's weights and outputs (tools/specialist_reuse_v3_checks.py on committed specialists and a random base);
- TIES gradients reach only the kept, agreeing entries of the specialists and never the base; batched equals grouped;
- 2-update TIES and Distillation-stage-2 runs work and resume with the waiting rule, and the evaluator loads them;
- every stage's command carries --dispatch_rule semi_active, and the environment default is unchanged;
- the frozen settings verify, with the same trial count for every method; time to target.
CPU, validation split, tiny runs; never the final-test sets.

    python -m unittest tests.test_specialist_reuse_v3
"""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools")); sys.path.insert(0, str(REPO / "daniel"))
PY = sys.executable
NORM = REPO / "results" / "warmstart_v3" / "norm" / "validation_normalisation_points_10x5.json"
SPEC = [REPO / "checkpoints" / "specialists_v3" / f"10x5+mix+bv3_spec_{o}_s111.pth" for o in "mct"]
ENV = {"OMP_NUM_THREADS": "4", "PATH": "/usr/bin:/bin"}

RULE = r"""
import sys, json, torch
sys.argv = ["m"]
from mixture import pref_ties_flat
base = torch.tensor([0., 0., 0., 0., 1.])
th = torch.tensor([[2., -1., 0.5, 0., 1.], [1., 3., -0.1, 0., 1.], [-4., 1., 0.2, 0., 1.]])
out = {}
# keep 1.0: weights (0.5, 0.25, 0.25): v = [[1,-.5,.25,0,0],[.25,.75,-.025,0,0],[-1,.25,.05,0,0]]
# entry 0: sum 0.25 > 0 -> agreeing 1 + .25 = 1.25; entry 1: sum .5 > 0 -> .75 + .25 = 1.0;
# entry 2: sum .275 > 0 -> .25 + .05 = .3; entry 3: all zero -> 0; entry 4: tau 0 -> base 1
w = torch.tensor([[0.5, 0.25, 0.25]])
out["full"] = pref_ties_flat(base, th, w, 1.0, 1.0)[0].tolist()
out["lam2"] = pref_ties_flat(base, th, w, 1.0, 2.0)[0].tolist()
# zero weight on the third vector: v3 = 0 neither votes nor contributes
out["zero"] = pref_ties_flat(base, th, torch.tensor([[0.5, 0.5, 0.]]), 1.0, 1.0)[0].tolist()
# keep 0.4 (2 of 5 entries per vector): tau1 keeps entries 0,1; tau2 entries 1,0; tau3 entries 0,1
out["trim"] = pref_ties_flat(base, th, torch.tensor([[1/3, 1/3, 1/3]]), 0.4, 1.0)[0].tolist()
out["corner"] = bool(torch.equal(pref_ties_flat(base, th, torch.eye(3), 0.2, 1.5), th))
print(json.dumps(out))
"""

GRAD = r"""
import sys, json, copy, torch, numpy as np
sys.argv = ["m", "--device", "cpu", "--obs_version", "v3", "--reward_version", "v3", "--goal", "mct", "--preference_dim", "3",
            "--dispatch_rule", "semi_active"]
from params import configs
import benchmark_v3 as B
from fjsp_env_same_op_nums import FJSPEnvForSameOpNums
from mixture import TiesMixturePolicy, load_specialists, load_base
paths = sys.stdin.read().split()
nets, _ = load_specialists(paths[:3], configs, "cpu")
base = load_base(paths[3], configs, "cpu")
pol = TiesMixturePolicy(nets, base, 0.2, 1.0)
ref = TiesMixturePolicy(copy.deepcopy(nets), base, 0.2, 1.0, grouped=True)
data = B.load_split("validation", "10x5")
w = np.array([[0.6, 0.2, 0.2], [0.2, 0.5, 0.3], [0.2, 0.5, 0.3], [1, 0, 0]])
env = FJSPEnvForSameOpNums(10, 5); env.set_initial_data(**data.env_kwargs(list(range(4)), "mct")); st = env.set_preference(w)
K = ("fea_j", "op_mask", "candidate", "fea_m", "mch_mask", "comp_idx", "dynamic_pair_mask", "fea_pairs", "pref")
x = {k: getattr(st, f"{k}_tensor") for k in K}
out = {"dispatch_rule": env.dispatch_rule}
pb, _ = pol(**x); pg, _ = ref(**x)
out["batched_vs_grouped"] = float((pb - pg).abs().max())
# gradient of one interior preference only: zero outside the kept, agreeing entries
wi = torch.tensor([[0.6, 0.2, 0.2]])
flat = pol.merged_flat(wi)
flat.sum().backward()
g = torch.cat([torch.stack([dict(n.named_parameters())[k].grad.reshape(-1) if dict(n.named_parameters())[k].grad is not None
                            else torch.zeros(dict(n.named_parameters())[k].numel()) for n in pol.nets]) for k in pol.names], 1)
with torch.no_grad():
    st_ = pol.stacked(); th = torch.cat([st_[k].reshape(3, -1) for k in pol.names], 1)
    tau = th - pol.base_flat()[None]
    n_keep = int(round(0.2 * tau.shape[1]))
    mask = torch.zeros_like(tau, dtype=torch.bool); mask.scatter_(1, tau.abs().topk(n_keep, dim=1).indices, True)
    v = wi[0][:, None] * tau * mask; s = torch.sign(v.sum(0)); agree = (torch.sign(v) == s[None]) & (v != 0)
    expect = wi[0][:, None] * agree.float()
out["grad_matches_kept_agreeing"] = float((g - expect).abs().max())
out["grad_nonzero_entries"] = int((g != 0).sum())
out["base_requires_grad"] = any(b.requires_grad for b in pol.buffers())
print(json.dumps(out))
"""


def run_py(code, stdin=""):
    r = subprocess.run([PY, "-c", code], cwd=REPO / "daniel", input=stdin, capture_output=True, text=True, env=ENV,
                       timeout=1800)
    assert r.returncode == 0, r.stderr[-3000:]
    return json.loads(r.stdout.strip().splitlines()[-1])


def train(cwd, *args):
    r = subprocess.run([PY, str(REPO / "daniel" / "train.py"), *args], cwd=cwd, capture_output=True, text=True,
                       timeout=1800, env=ENV)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]


MO = ["--config", str(REPO / "configs/benchmark_v3/mct.json"), "--n_j", "10", "--n_m", "5", "--seed_train", "111",
      "--v3_train_source", "generator", "--preference_dim", "3", "--mo_val_norm", str(NORM), "--mo_val_first", "2",
      "--finetune_objective", "contrastive", "--validate_timestep", "2", "--reset_env_timestep", "2", "--num_envs", "2",
      "--cl_k", "2", "--max_updates", "4", "--chunk_updates", "2", "--resume", "True", "--device", "cpu",
      "--dispatch_rule", "semi_active"]


def setUpModule():
    global D, BASE
    D = tempfile.TemporaryDirectory()
    BASE = Path(D.name) / "base.pth"
    r = subprocess.run([PY, "-c", "import sys; sys.path.insert(0, 'tools'); import world_model_m2_bases as W; "
                        f"W.save_bundle(W.random_base_sd(111), '{BASE}')"], cwd=REPO, capture_output=True, text=True, env=ENV)
    assert r.returncode == 0, r.stderr[-2000:]


def tearDownModule():
    D.cleanup()


class Rule(unittest.TestCase):
    def test_hand_example(self):
        o = run_py(RULE)
        self.assertEqual([round(x, 6) for x in o["full"]], [1.25, 1.0, 0.3, 0.0, 1.0])
        self.assertEqual([round(x, 6) for x in o["lam2"]], [2.5, 2.0, 0.6, 0.0, 1.0])
        # zero weight: v = [[1,-.5,.25,0,0],[.5,1.5,-.05,0,0],0] -> [1.5, 1.5, .25, 0, base 1]
        self.assertEqual([round(x, 6) for x in o["zero"]], [1.5, 1.5, 0.25, 0.0, 1.0])
        # trim to 2 entries each: v1 = [2,-1,0,0,0]/3, v2 = [1,3,0,0,0]/3, v3 = [-4,1,0,0,0]/3
        # entry 0: sum -1/3 < 0 -> -4/3; entry 1: sum 1 > 0 -> (3 + 1)/3
        self.assertEqual([round(x, 6) for x in o["trim"]], [round(-4 / 3, 6), round(4 / 3, 6), 0.0, 0.0, 1.0])
        self.assertTrue(o["corner"])                      # each exact corner is the specialist, bit for bit


class Merges(unittest.TestCase):
    def test_checks_pass(self):
        out = Path(D.name) / "checks.json"
        r = subprocess.run([PY, str(REPO / "tools" / "specialist_reuse_v3_checks.py"), "--specs", ",".join(map(str, SPEC)),
                            "--base", str(BASE), "--keep", "0.2,0.8", "--lam", "1.0,1.5", "--out", str(out), "--states", "3"],
                           cwd=REPO, capture_output=True, text=True, env=ENV, timeout=1800)
        self.assertEqual(r.returncode, 0, r.stdout[-2000:] + r.stderr[-2000:])
        rec = json.loads(out.read_text())
        self.assertTrue(rec["all_pass"])
        self.assertEqual(len(rec["checks"]), 5)
        for v in rec["checks"].values():
            self.assertTrue(v["preference_changes_weights"])
            self.assertTrue(all(c["weights_equal"] for c in v["corners"].values()))

    def test_ties_gradients_and_batching(self):
        o = run_py(GRAD, " ".join(map(str, SPEC + [BASE])))
        self.assertEqual(o["dispatch_rule"], "semi_active")
        self.assertLess(o["batched_vs_grouped"], 1e-5)
        self.assertEqual(o["grad_matches_kept_agreeing"], 0.0)
        self.assertGreater(o["grad_nonzero_entries"], 0)
        self.assertFalse(o["base_requires_grad"])


class Runs(unittest.TestCase):
    def test_ties_run_resumes_and_loads(self):
        cwd = Path(D.name) / "ties"; cwd.mkdir()
        args = [*MO, "--model_suffix", "bv3_srtest_s111", "--merge_mix", ",".join(map(str, SPEC)), "--merge_rule", "ties",
                "--merge_base", str(BASE), "--ties_keep", "0.5", "--ties_lambda", "1.0"]
        train(cwd, *args); train(cwd, *args)
        st = json.loads((cwd / "train_log/SD2/10x5+mix+bv3_srtest_s111.status.json").read_text())
        self.assertEqual(st["update"], 4)
        run = json.loads((cwd / "train_log/SD2/10x5+mix+bv3_srtest_s111.run.json").read_text())
        self.assertEqual((run["config"]["merge_rule"], run["config"]["dispatch_rule"]), ("ties", "semi_active"))
        code = ("import os, sys; os.environ['DECODING_DEVICE'] = 'cpu'; sys.argv = ['x']; sys.path.insert(0, '../tools'); "
                "import decoding_v3 as D, numpy as np, benchmark_v3 as B, json; D.configs.dispatch_rule = 'semi_active'; "
                f"net = D.load_net('{cwd}/trained_network/SD2/10x5+mix+bv3_srtest_s111.pth'); "
                "data = B.load_split('validation', '10x5'); "
                "o, m, s = D.greedy(net, data, [0, 0], D.PREFS[[0, 30]], D.Cost(2)); "
                "print(json.dumps({'kind': type(net).__name__, 'base_buffers': len(list(net.buffers())), 'obj': o.tolist()}))")
        o = run_py(code)
        self.assertEqual(o["kind"], "TiesMixturePolicy")
        self.assertGreater(o["base_buffers"], 0)

    def test_distill_stage2_run(self):
        cwd = Path(D.name) / "dist"; cwd.mkdir()
        student = cwd / "student.pth"
        r = subprocess.run([PY, "distill_v3.py", "--size", "10x5", "--seed", "111", "--specs", ",".join(map(str, SPEC)),
                            "--dispatch_rule", "semi_active", "--out", str(student), "--instances", "2", "--epochs", "1",
                            "--wall_cap", "120", "--val_first", "2", "--device", "cpu"],
                           cwd=REPO / "daniel", capture_output=True, text=True, env=ENV, timeout=1800)
        self.assertEqual(r.returncode, 0, r.stderr[-2000:])
        train(cwd, *MO, "--model_suffix", "bv3_srdist_s111", "--init_from", str(student), "--init_mode", "no_critic")
        st = json.loads((cwd / "train_log/SD2/10x5+mix+bv3_srdist_s111.status.json").read_text())
        self.assertEqual(st["update"], 2)


class Settings(unittest.TestCase):
    def test_every_stage_uses_waiting(self):
        import freeze_specialist_reuse_v3 as F
        import specialist_reuse_v3_runner as R
        r = R.Runner.__new__(R.Runner)
        r.cfg, r.smoke, r.sm, r.tag = F.load(), False, {}, "sr"
        r.rule, r.validate, r.updates_trial = r.cfg["stopping"], 50, 1000
        r.trials, r.base, r.ckpt = r.cfg["tuning"]["trials"], Path("base.pth"), Path("ck")
        r.state = {"ties_selected": [0.2, 1.0]}
        cmds = [r.specialist_args(o) for o in "mct"] + [r.method_args(m, m, 3e-4, 0.1, full=True) for m in F.METHODS]
        cmds += [r.method_args("ties", "s", ties=(0.5, 1.0), max_updates=0), r.distill_cmd(Path("s.pth"))]
        for c in cmds:
            i = c.index("--dispatch_rule")
            self.assertEqual(c[i + 1], "semi_active")
        full = r.method_args("rs", "rs", full=True)
        for k, v in (("--converge_window", "1000"), ("--converge_tol", "0.005"), ("--min_updates", "1000"),
                     ("--max_updates", "8000")):
            self.assertEqual(full[full.index(k) + 1], v)
        trial = r.method_args("scratch", "t", 1e-3, 0.5)
        self.assertEqual((trial[trial.index("--max_updates") + 1], trial[trial.index("--lr") + 1],
                          trial[trial.index("--cl_beta") + 1]), ("1000", "0.001", "0.5"))
        code = ("import sys; sys.argv=['x','--obs_version','v3','--reward_version','v3']; "
                "from fjsp_env_same_op_nums import FJSPEnvForSameOpNums; print('\"' + FJSPEnvForSameOpNums(10, 5).dispatch_rule + '\"')")
        self.assertEqual(run_py(code), "nondelay")

    def test_settings_verify(self):
        import freeze_specialist_reuse_v3 as F
        m = F.load()
        self.assertEqual(m["tuning"]["trials_per_method"], 6)
        self.assertEqual(len(m["tuning"]["trials"]), 6)
        self.assertEqual(set(m["methods"]), {"scratch", "distill", "rs", "ties"})
        self.assertEqual(m["evaluation"]["decoders"]["sample128"], {"K": 128, "seed": 51})

    def test_time_to_target(self):
        import analyze_specialist_reuse_v3 as A
        v = [{"update": 0, "hv": 0.1, "train_seconds": 1}, {"update": 50, "hv": 0.5, "train_seconds": 90},
             {"update": 100, "hv": 0.7, "train_seconds": 180}]
        self.assertEqual(A.time_to_target(v, 0.5), (50, 90))
        self.assertEqual(A.time_to_target(v, 0.8), (None, None))


if __name__ == "__main__":
    unittest.main()
