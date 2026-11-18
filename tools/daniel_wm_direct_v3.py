"""DANIEL trained directly on the multi-objective task from a world-model encoder start, vs From-scratch DANIEL
(results/specialist_reuse_v3, 10x5, seed 111, waiting allowed). Single new arm; no specialists, merging or tuning.

1. Scratch start: the baseline's own From-scratch command (built by tools/specialist_reuse_v3_runner.Runner with the
   baseline's SELECTED lr / beta) run for 0 updates with --budget_checkpoints 0, which saves the freshly initialised
   network. Its hash (train.py's contrastive-reference hash) must equal the cl_reference_sha256 recorded by the
   baseline From-scratch run.
2. Start: that network with feature_exact replaced by the world-model encoder of checkpoints/specialist_reuse_wm_v3/
   base_s111.pth (the outcome-prediction pilot trunk, checked against its source file); every other tensor unchanged.
3. Training: the From-scratch command + --init_from <start> --init_mode exact (same trainer, contrastive objective,
   lr, beta, stopping rule, validation, checkpoint selection). 4. Evaluation: specialist_reuse_v3_eval.py.

    python tools/daniel_wm_direct_v3.py [--check_only | --summarise]   # -> results|checkpoints/daniel_wm_direct_v3
"""
from __future__ import annotations
import argparse, hashlib, json, shutil, subprocess, sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))
import freeze_specialist_reuse_v3 as F                              # noqa: E402
import specialist_reuse_v3_runner as R                              # noqa: E402

NAME = "daniel_wm_direct_v3"
BASE_RES, BASE_CK = REPO / "results" / "specialist_reuse_v3", REPO / "checkpoints" / "specialist_reuse_v3"
WM_BASE = REPO / "checkpoints" / "specialist_reuse_wm_v3" / "base_s111.pth"
PILOT = REPO / "checkpoints" / "world_model_m1_v3_action_pilot" / "pilot_pairs_abs+diff.pth"
SCRATCH = "10x5+mix+bv3_sr_scratch_s111"
_load = F.load
sha = R.sha


def net_sha(sd):
    """train.py's contrastive-reference hash of a state dict (sorted keys, raw tensor bytes)."""
    h = hashlib.sha256()
    for k, v in sorted(sd.items()):
        h.update(k.encode()); h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()


class D(R.Runner):
    def __init__(self, a):
        F.load = lambda check=True: {**_load(check), "name": NAME}
        super().__init__(a)
        self.tag = "srwmd"
        self.bstate = json.loads((BASE_RES / "state.json").read_text())
        self.sel = self.bstate["selected"]["scratch"]
        self.start = self.ckpt / "start_s111.pth"

    def commit(self, msg, *paths):
        pass

    def scratch_cmd(self, suffix, **kw):
        return self.method_args("scratch", suffix, self.sel["lr"], self.sel["cl_beta"], **kw)

    def new_cmd(self):
        return [*self.scratch_cmd("wmdirect", full=True), "--init_from", str(self.start), "--init_mode", "exact"]

    def command_check(self):
        new = self.new_cmd()
        tag, self.tag = self.tag, "sr"
        old = self.scratch_cmd("scratch", full=True)
        self.tag = tag
        assert new[:len(old)][0] == old[0] and len(new) == len(old) + 4
        diff = [[o, n] for o, n in zip(old, new) if o != n]
        flags = {old[i - 1] for i, (o, n) in enumerate(zip(old, new)) if o != n}
        assert flags == {"--model_suffix"}, flags
        return {"selected": self.sel, "differing_flags": sorted(flags), "diff": diff, "added": new[len(old):]}

    def build_start(self):
        import torch
        rec = self.res / "start_check.json"
        if self.start.exists() and rec.exists():
            return json.loads(rec.read_text())
        name = self.model("scratchinit")
        out = self.res / "logs" / f"{name}.stdout.log"; out.parent.mkdir(parents=True, exist_ok=True)
        rc = self.run([R.PY, *self.scratch_cmd("scratchinit", max_updates=0), "--budget_checkpoints", "0"], 0, 3600, out)
        u0 = R.TN / f"{name}@u0.pth"
        if rc != 0 or not u0.exists():
            raise SystemExit("scratch initialisation run failed")
        shutil.copy2(u0, self.ckpt / "scratch_init_s111.pth")
        b = torch.load(u0, map_location="cpu", weights_only=False)
        sd = b["state_dict"]
        code = ("import json, os, sys, torch; p = sys.argv.pop(); print(json.dumps(torch.load(p, map_location='cpu', "
                "weights_only=False)['identity']))")                 # the resume file pickles daniel/ modules
        ident = json.loads(subprocess.run([R.PY, "-c", code, str(R.TN / f"{SCRATCH}.resume.pth")], cwd=R.DAN,
                                          capture_output=True, text=True, check=True).stdout.strip().splitlines()[-1])
        wm = torch.load(WM_BASE, map_location="cpu", weights_only=False)["state_dict"]
        pilot = torch.load(PILOT, map_location="cpu", weights_only=False)
        trunk = {"feature_exact." + k[len("trunk."):]: v for k, v in pilot.items() if k.startswith("trunk.")}
        fe = {k for k in sd if k.startswith("feature_exact.")}
        assert set(trunk) == fe and all(sd[k].shape == trunk[k].shape for k in fe)
        new = dict(b); new["state_dict"] = {k: (wm[k].clone() if k in fe else v.clone()) for k, v in sd.items()}
        new["update"] = 0
        torch.save(new, self.start)
        a = new["state_dict"]
        chk = {"scratch_init_hash": net_sha(sd), "baseline_scratch_reference_hash": ident["cl_reference_sha256"],
               "scratch_init_matches_baseline_start": net_sha(sd) == ident["cl_reference_sha256"],
               "wm_base_sha_matches_recorded": sha(WM_BASE) == json.loads(
                   (REPO / "results/specialist_reuse_wm_v3/base_check.json").read_text())["new_base_sha256"],
               "encoder_equals_wm_base_encoder": all(torch.equal(a[k], wm[k]) for k in fe),
               "encoder_equals_pilot_trunk": all(torch.equal(a[k], trunk[k]) for k in fe),
               "encoder_differs_from_scratch": any(not torch.equal(a[k], sd[k]) for k in fe),
               "non_encoder_identical_to_scratch_init": all(torch.equal(a[k], sd[k]) for k in sd if k not in fe),
               "encoder_tensors": len(fe), "non_encoder_tensors": len(sd) - len(fe),
               "start_sha256": sha(self.start), "pilot_sha256": sha(PILOT)}
        assert all(v for v in chk.values() if isinstance(v, bool)), chk
        rec.write_text(json.dumps(chk, indent=1) + "\n")
        self.write_sums()
        self.log(f"start built: {chk}")
        return chk

    def record_check(self):
        n = json.loads((self.res / "runs" / self.model("wmdirect") / "run.json").read_text())
        o = json.loads((BASE_RES / "runs" / SCRATCH / "run.json").read_text())
        keys = set(n["config"]) | set(o["config"])
        diff = sorted(k for k in keys if n["config"].get(k) != o["config"].get(k))
        fp = n["data_fingerprints"] == o["data_fingerprints"]
        return {"differing_config_keys": diff, "data_fingerprints_equal": fp,
                "matches": set(diff) <= {"model_suffix", "init_from"} and fp,
                "effective_settings_equal": {k: (n["config"][k], o["config"][k]) for k in
                                             ("finetune_objective", "cl_beta", "cl_k", "cl_margin", "cl_grad_clip",
                                              "cl_anchor", "lr", "minibatch_size", "num_envs", "init_mode",
                                              "pref_dirichlet", "max_updates", "converge_window", "converge_tol",
                                              "min_updates", "validate_timestep", "dispatch_rule", "seed_train")}}

    def main(self):
        t0 = time.time()
        self.log("start")
        self.state["command_check"] = self.command_check()
        self.state["start_check"] = self.build_start(); self.save_state()

        def keep(name):
            d = self.ckpt / "finals"; d.mkdir(exist_ok=True)
            shutil.copy2(R.TN / f"{name}.pth", d / f"{name}.pth"); self.write_sums()
        if not self.pool([("final:wmdirect", self.new_cmd(), self.model("wmdirect"), keep)]):
            raise SystemExit("training failed")
        out = self.res / "eval" / "wmdirect.npz"
        if not out.exists():
            out.parent.mkdir(parents=True, exist_ok=True)
            t1 = time.time()
            cmd = [R.PY, str(REPO / "tools" / "specialist_reuse_v3_eval.py"), "--model",
                   str(self.ckpt / "finals" / f"{self.model('wmdirect')}.pth"), "--out", str(out)]
            if subprocess.run(["taskset", "-c", "0-11", *cmd], cwd=REPO, env=R.ENV, stdout=self.logf,
                              stderr=subprocess.STDOUT).returncode != 0:
                raise SystemExit("evaluation failed")
            self.state["eval_seconds"] = time.time() - t1
        self.state["record_check"] = self.record_check()
        self.state["wall_hours"] = (time.time() - t0) / 3600; self.save_state()
        assert self.state["record_check"]["matches"], self.state["record_check"]
        summarise(self)
        self.log(f"done ({(time.time() - t0) / 3600:.2f} h)")
        return 0


def summarise(r):
    import numpy as np
    import torch
    import analyze_specialist_reuse_v3 as AN
    cfg = _load()
    norm = json.loads((REPO / cfg["evaluation"]["norm"]).read_text())["instances"]
    b2 = REPO / cfg["evaluation"]["igd_front"]
    bu, nu = r.bstate["units"]["final:scratch"], r.state["units"]["final:wmdirect"]
    target = -bu["best"]
    out = {"target_scratch_best_validation_hv": target}
    curves = {}
    for arm, res, name, ck, u, ev in (("scratch", BASE_RES, SCRATCH, BASE_CK, bu, BASE_RES / "eval" / "scratch.npz"),
                                      ("world_model_encoder", r.res, r.model("wmdirect"), r.ckpt, nu,
                                       r.res / "eval" / "wmdirect.npz")):
        v = json.loads((res / "runs" / name / "valiquality.json").read_text())
        curves[arm] = v
        best = max(v, key=lambda e: e["hv"])
        hit = next((e for e in v if e["hv"] >= target - 1e-12), None)
        e = np.load(ev); keys = list(e["keys"])
        g = AN.metrics(e["greedy_obj"], keys, norm, b2); s = AN.metrics(e["sample128_obj"], keys, norm, b2)
        fpath = ck / "finals" / f"{name}.pth"
        bundle = torch.load(fpath, map_location="cpu", weights_only=False)
        out[arm] = {"update0_hv": v[0]["hv"], "best_hv": best["hv"], "best_update": best["update"],
                    "stop_update": u["update"], "status": u["status"],
                    "reaches_target_at": hit["update"] if hit else None,
                    "reaches_target_hours": hit["train_seconds"] / 3600 if hit else None,
                    "greedy_hv": g[0], "best_of_128_hv": s[0], "greedy_igd_plus": g[1], "best_of_128_igd_plus": s[1],
                    "igd_plus_instances": s[2], "eval_instances": len(keys), "eval_preferences": int(e["greedy_obj"].shape[1]),
                    "train_hours": u["train_seconds"] / 3600,
                    "artefact_checks": {"eval_model_sha_equals_selected_checkpoint": str(e["model_sha256"]) == sha(fpath),
                                        "checkpoint_update_equals_best_validation_update": bundle["update"] == best["update"],
                                        "checkpoint_selection_metric_equals_minus_best_hv":
                                            abs(bundle["selection_metric"] + best["hv"]) < 1e-9,
                                        "state_best_equals_curve_best": abs(-u["best"] - best["hv"]) < 1e-9}}
    out["delta"] = {k: out["world_model_encoder"][k] - out["scratch"][k] for k in
                    ("update0_hv", "best_hv", "greedy_hv", "best_of_128_hv", "greedy_igd_plus", "best_of_128_igd_plus")}
    cs = {e["update"]: e["hv"] for e in curves["scratch"]}; cw = {e["update"]: e["hv"] for e in curves["world_model_encoder"]}
    out["matched_updates"] = [{"update": k, "scratch": cs[k], "world_model_encoder": cw[k]} for k in sorted(set(cs) & set(cw))]
    ap = json.loads((REPO / "results/world_model_m1_v3/action_pilot.json").read_text())
    out["costs"] = {"world_model_pretraining_hours": {"pilot_data_regeneration": ap["pilot_data"]["train_regeneration_seconds"] / 3600,
                                                      "pilot_encoder_training": ap["pilot"]["pairs|abs+diff"]["seconds"] / 3600},
                    "direct_policy_training_hours": {"world_model_encoder": nu["train_seconds"] / 3600,
                                                     "scratch": bu["train_seconds"] / 3600},
                    "evaluation_seconds": r.state.get("eval_seconds"),
                    "hardware": "new arm trained alone (one training process); the baseline From-scratch final ran "
                                "alongside TIES, Rewarded Soups and Distillation on 3 CPU slots and one GPU"}
    for k in ("command_check", "start_check", "record_check"):
        out[k] = r.state.get(k)
    (r.res / "summary.json").write_text(json.dumps(out, indent=1) + "\n")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9.5, 4.8), facecolor="white")
    for arm, col, lab in (("scratch", "#7f7f7f", "From-scratch DANIEL"),
                          ("world_model_encoder", "#1f77b4", "DANIEL, world-model encoder start")):
        v = curves[arm]
        ax.plot([e["update"] for e in v], [e["hv"] for e in v], color=col, lw=1.5, marker="o", ms=2,
                label=f"{lab} (best {out[arm]['best_hv']:.4f} @ {out[arm]['best_update']}, stopped {out[arm]['stop_update']})")
    ax.axhline(target, color="#555555", ls="--", lw=.9, label=f"From-scratch best {target:.4f}")
    ax.set_xlabel("updates"); ax.set_ylabel("validation HV (18 instances x 15 preferences)")
    ax.set_title("Direct multi-objective training: world-model encoder start vs From scratch (10x5, seed 111, waiting allowed)",
                 fontsize=10)
    ax.grid(alpha=.3); ax.legend(frameon=False, fontsize=8.5, loc="lower right")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout(); fig.savefig(r.res / "curves.png", dpi=140, facecolor="white"); plt.close(fig)
    print(json.dumps({k: v for k, v in out.items() if k not in ("matched_updates",)}, indent=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--summarise", action="store_true")
    ap.add_argument("--check_only", action="store_true")
    a = ap.parse_args()
    a.smoke, a.jobs = False, 1
    r = D(a)
    if a.summarise:
        summarise(r); return
    if a.check_only:
        print(json.dumps({"command": r.command_check(), "start": r.build_start()}, indent=1)); return
    sys.exit(r.main())


if __name__ == "__main__":
    main()
