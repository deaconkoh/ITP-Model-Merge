"""World-model shared-base Preference-weighted TIES and Distillation (10x5, seed 111, waiting allowed).

Reuses, read-only, the world-model shared base and the three validation-selected specialists of
results|checkpoints/specialist_reuse_wm_v3 (commit bbe1959). No pretraining, no specialist training, no sweep.
Every command is built by the baseline's runner (tools/specialist_reuse_v3_runner.Runner) with the baseline's
SELECTED settings (results/specialist_reuse_v3/state.json):
  TIES          --merge_rule ties --merge_base <world-model base> --ties_keep 0.5 --ties_lambda 1.5, lr 1e-4, beta 0.1
  Distillation  stage 1 distill_v3.py with the world-model specialists as teachers (same instances, sampling seeds,
                budget; the student is a fresh DANIEL seeded 111 + 7001, as in the baseline), then train.py
                --init_from student --init_mode no_critic, lr 3e-4, beta 0.1
Same stopping rule, validation cadence, checkpoint selection and evaluation script (specialist_reuse_v3_eval.py).
Run order: TIES, then Distillation. Outputs: results|checkpoints/specialist_reuse_wm_td_v3 (tag srwmtd).

    python tools/specialist_reuse_wm_td_v3.py [--summarise]
"""
from __future__ import annotations
import argparse, json, shutil, subprocess, sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))
import freeze_specialist_reuse_v3 as F                              # noqa: E402
import specialist_reuse_v3_runner as R                              # noqa: E402

NAME = "specialist_reuse_wm_td_v3"
BASE_RES, BASE_CK = REPO / "results" / "specialist_reuse_v3", REPO / "checkpoints" / "specialist_reuse_v3"
WM_RES, WM_CK = REPO / "results" / "specialist_reuse_wm_v3", REPO / "checkpoints" / "specialist_reuse_wm_v3"
_load = F.load
METHODS = ("ties", "distill")
ALLOWED = {"ties": {"model_suffix", "merge_mix", "merge_base"}, "distill": {"model_suffix", "init_from"}}
sha = R.sha


class TD(R.Runner):
    def __init__(self, a):
        F.load = lambda check=True: {**_load(check), "name": NAME}
        super().__init__(a)
        self.tag = "srwmtd"
        self.prefix = "specialist reuse WM TIES/distill v3"
        self.base = WM_CK / "base_s111.pth"                         # read-only: the exact world-model shared base
        self.bstate = json.loads((BASE_RES / "state.json").read_text())

    def commit(self, msg, *paths):
        pass

    def specs(self):
        return [str(WM_CK / "specialists" / f"10x5+mix+bv3_srwm_spec_{o}_s111.pth") for o in "mct"]

    # ------------------------------------------------------------------ identity checks
    def identity_check(self):
        import torch
        sums = json.loads((WM_CK / "SHA256SUMS.json").read_text())
        out = {"base_sha256": sha(self.base), "base_sha_matches_wm_sums": sums["base_s111.pth"] == sha(self.base),
               "base_sha_matches_wm_base_check": json.loads((WM_RES / "base_check.json").read_text())["new_base_sha256"]
               == sha(self.base)}
        unchanged = subprocess.run(["git", "diff", "--quiet", "bbe1959", "--", str(WM_CK)], cwd=REPO).returncode == 0
        out["wm_checkpoints_unchanged_since_bbe1959"] = unchanged
        wm_state = json.loads((WM_RES / "state.json").read_text())["units"]
        for o, p in zip("mct", self.specs()):
            rel = str(Path(p).relative_to(WM_CK))
            b = torch.load(p, map_location="cpu", weights_only=False)
            run = json.loads((WM_RES / "runs" / f"10x5+mix+bv3_srwm_spec_{o}_s111" / "run.json").read_text())
            out[f"spec_{o}"] = {"sha256": sha(p), "sha_matches_wm_sums": sums[rel] == sha(p),
                                "update": b["update"], "is_validation_selected": b["update"] == wm_state[f"spec:{o}"]["best_update"],
                                "initialised_from_wm_base": Path(run["config"]["init_from"]).resolve() == self.base.resolve()}
        ok = all(v for k, v in out.items() if isinstance(v, bool)) and all(
            all(x for x in d.values() if isinstance(x, bool)) for k, d in out.items() if k.startswith("spec_"))
        assert ok, out
        return out

    def command_check(self, method):
        """The new command vs the baseline command rebuilt by the baseline runner (same selected settings)."""
        sel = self.bstate["selected"][method]
        kw = {"ties": tuple(self.bstate["ties_selected"])} if method == "ties" else {}
        new = self.method_args(method, method, sel["lr"], sel["cl_beta"], full=True, **kw)
        old_tag, old_ck, old_base, old_specs = self.tag, self.ckpt, self.base, self.specs
        self.tag, self.ckpt, self.base = "sr", BASE_CK, BASE_CK / "base_s111.pth"
        self.specs = lambda: [str(BASE_CK / "specialists" / f"10x5+mix+bv3_sr_spec_{o}_s111.pth") for o in "mct"]
        old = self.method_args(method, method, sel["lr"], sel["cl_beta"], full=True, **kw)
        self.tag, self.ckpt, self.base, self.specs = old_tag, old_ck, old_base, old_specs
        assert len(old) == len(new)
        diff = [[o, n] for o, n in zip(old, new) if o != n]
        flags = {old[i - 1] for i, (o, n) in enumerate(zip(old, new)) if o != n}
        assert flags <= {"--model_suffix", "--merge_mix", "--merge_base", "--init_from"}, flags
        return {"selected": sel, **kw, "differing_flags": sorted(flags), "diff": diff}

    def record_check(self, method):
        n = json.loads((self.res / "runs" / self.model(method) / "run.json").read_text())
        o = json.loads((BASE_RES / "runs" / f"10x5+mix+bv3_sr_{method}_s111" / "run.json").read_text())
        keys = set(n["config"]) | set(o["config"])
        diff = sorted(k for k in keys if n["config"].get(k) != o["config"].get(k))
        fp = n["data_fingerprints"] == o["data_fingerprints"]
        return {"differing_config_keys": diff, "data_fingerprints_equal": fp,
                "matches": set(diff) <= ALLOWED[method] and fp}

    # ------------------------------------------------------------------ stages
    def keep_final(self, name):
        d = self.ckpt / "finals"; d.mkdir(exist_ok=True)
        shutil.copy2(R.TN / f"{name}.pth", d / f"{name}.pth"); self.write_sums()

    def final(self, method):
        sel = self.bstate["selected"][method]
        kw = {"ties": tuple(self.bstate["ties_selected"])} if method == "ties" else {}
        self.log(f"{method}: baseline's selected setting {sel} {kw}")
        return self.pool([(f"final:{method}", self.method_args(method, method, sel["lr"], sel["cl_beta"], full=True, **kw),
                           self.model(method), self.keep_final)])

    def evaluate(self, method):
        out = self.res / "eval" / f"{method}.npz"
        if out.exists():
            return
        out.parent.mkdir(parents=True, exist_ok=True)
        cmd = [R.PY, str(REPO / "tools" / "specialist_reuse_v3_eval.py"), "--model",
               str(self.ckpt / "finals" / f"{self.model(method)}.pth"), "--out", str(out)]
        t0 = time.time()
        if subprocess.run(["taskset", "-c", "0-11", *cmd], cwd=REPO, env=R.ENV, stdout=self.logf,
                          stderr=subprocess.STDOUT).returncode != 0:
            raise SystemExit(f"evaluation of {method} failed")
        self.state.setdefault("eval_seconds", {})[method] = time.time() - t0; self.save_state()
        self.log(f"evaluated {method}")

    def teacher_inputs(self):
        """What changed in the distillation stage-1 inputs, from the two saved records."""
        n = json.loads((self.ckpt / "distill" / "student_s111.distill.json").read_text())
        o = json.loads((BASE_CK / "distill" / "student_s111.distill.json").read_text())
        keys = set(n) | set(o)
        same = sorted(k for k in keys if n.get(k) == o.get(k))
        diff = sorted(k for k in keys if n.get(k) != o.get(k))
        return {"student_initialisation": "fresh DANIEL (preference-conditioned), torch.manual_seed(111 + 7001), "
                                          "identical procedure and seed to the baseline (distill_v3.py); not from either shared base",
                "instance_stream": "GeneratorSource, numpy seed 111 + 7001, 400 instances per teacher (same instances as the baseline)",
                "teacher_sampling": "torch generator seeded 111 + 7001; sampled teacher rollouts (actions differ with the teachers, "
                                    "so the stored states and target distributions differ)",
                "teachers_new": self.specs(), "teachers_baseline": [p for p in o.get("parents", [])] or "see baseline record",
                "record_keys_equal": same, "record_keys_differing": diff,
                "generator_equal": n.get("generator") == o.get("generator")}

    def main(self):
        t0 = time.time()
        self.log("start")
        self.state["identity"] = self.identity_check(); self.save_state()
        self.state["command_check"] = {m: self.command_check(m) for m in METHODS}; self.save_state()
        self.log(f"identity and command checks passed: {json.dumps(self.state['command_check'])}")
        # 1. TIES
        if not self.final("ties"):
            raise SystemExit("TIES run failed")
        self.evaluate("ties")
        # 2. Distillation
        self.stage_distill()
        self.state["teacher_inputs"] = self.teacher_inputs(); self.save_state()
        if not self.final("distill"):
            raise SystemExit("distillation run failed")
        self.evaluate("distill")
        self.state["record_check"] = {m: self.record_check(m) for m in METHODS}
        self.state["wall_hours"] = (time.time() - t0) / 3600; self.save_state()
        assert all(v["matches"] for v in self.state["record_check"].values()), self.state["record_check"]
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
    bst, st = r.bstate["units"], r.state["units"]
    out = {"methods": {}}
    for m in METHODS:
        target = -bst[f"final:{m}"]["best"]
        row = {"baseline_best_validation_hv": target}
        for arm, res, name, ck, u in (("random", BASE_RES, f"10x5+mix+bv3_sr_{m}_s111", BASE_CK, bst[f"final:{m}"]),
                                      ("world_model", r.res, r.model(m), r.ckpt, st[f"final:{m}"])):
            v = json.loads((res / "runs" / name / "valiquality.json").read_text())
            best = max(v, key=lambda e: e["hv"])
            hit = next((e for e in v if e["hv"] >= target - 1e-12), None)
            e = np.load(res / "eval" / f"{m}.npz"); keys = list(e["keys"])
            g = AN.metrics(e["greedy_obj"], keys, norm, b2); s = AN.metrics(e["sample128_obj"], keys, norm, b2)
            fpath = ck / "finals" / f"{name}.pth"
            bundle = torch.load(fpath, map_location="cpu", weights_only=False)
            row[arm] = {"update0_hv": v[0]["hv"], "best_hv": best["hv"], "best_update": best["update"],
                        "stop_update": u["update"], "status": u["status"],
                        "reaches_baseline_best_at": hit["update"] if hit else None,
                        "greedy_hv": g[0], "best_of_128_hv": s[0], "greedy_igd_plus": g[1], "best_of_128_igd_plus": s[1],
                        "igd_plus_instances": s[2], "eval_instances": len(keys),
                        "train_hours": u["train_seconds"] / 3600,
                        "artefact_checks": {"eval_model_sha_equals_final_checkpoint": str(e["model_sha256"]) == sha(fpath),
                                            "checkpoint_update_equals_best_validation_update": bundle["update"] == best["update"],
                                            "checkpoint_selection_metric_equals_minus_best_hv":
                                                abs(bundle["selection_metric"] + best["hv"]) < 1e-9,
                                            "state_best_equals_curve_best": abs(-u["best"] - best["hv"]) < 1e-9}}
            row[arm]["curve"] = {x["update"]: x["hv"] for x in v}
        common = sorted(set(row["random"]["curve"]) & set(row["world_model"]["curve"]))
        row["matched_updates"] = [{"update": k, "random": row["random"]["curve"][k],
                                   "world_model": row["world_model"]["curve"][k]} for k in common]
        for arm in ("random", "world_model"):
            row[arm].pop("curve")
        row["delta_world_model_minus_random"] = {k: row["world_model"][k] - row["random"][k] for k in
                                                 ("update0_hv", "best_hv", "greedy_hv", "best_of_128_hv",
                                                  "greedy_igd_plus", "best_of_128_igd_plus")}
        out["methods"][m] = row
    d_new = json.loads((r.ckpt / "distill" / "student_s111.distill.json").read_text())
    d_old = r.bstate["distill_stage1"]
    out["costs_hours"] = {
        "shared_upstream_world_model": json.loads((WM_RES / "summary.json").read_text())["costs_hours"]["pretraining"]
        | {"specialist_training_sum": json.loads((WM_RES / "summary.json").read_text())["costs_hours"]["specialist_training"]["world_model_sum"]},
        "shared_upstream_random_specialists_sum": json.loads((WM_RES / "summary.json").read_text())["costs_hours"]["specialist_training"]["random_sum"],
        "ties_fine_tuning": {"world_model": st["final:ties"]["train_seconds"] / 3600, "random": bst["final:ties"]["train_seconds"] / 3600},
        "distill_teacher_data_and_student_imitation": {
            "world_model": {"collection_s": d_new["collection_seconds"], "imitation_training_s": d_new["training_seconds"],
                            "total_s": d_new["total_seconds"]},
            "random": {"collection_s": d_old["collection_seconds"], "imitation_training_s": d_old["training_seconds"],
                       "total_s": d_old["total_seconds"]}},
        "distill_ppo_training": {"world_model": st["final:distill"]["train_seconds"] / 3600,
                                 "random": bst["final:distill"]["train_seconds"] / 3600},
        "evaluation_seconds": r.state.get("eval_seconds")}
    out["hardware"] = {"world_model": "TIES and distillation run one after the other, one training process at a time",
                       "random": "baseline finals ran TIES, Rewarded Soups, Distillation and From-scratch together on 3 CPU "
                                 "slots and one GPU; compare learning speed in updates"}
    out["identity"] = r.state.get("identity"); out["command_check"] = r.state.get("command_check")
    out["record_check"] = r.state.get("record_check"); out["teacher_inputs"] = r.state.get("teacher_inputs")
    (r.res / "summary.json").write_text(json.dumps(out, indent=1) + "\n")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    title = {"ties": "Preference-weighted TIES", "distill": "Distillation"}
    for m in METHODS:
        row = out["methods"][m]
        fig, ax = plt.subplots(figsize=(8.5, 4.6), facecolor="white")
        for arm, res, name, col, lab in (("random", BASE_RES, f"10x5+mix+bv3_sr_{m}_s111", "#7f7f7f", "random shared base"),
                                         ("world_model", r.res, r.model(m), "#1f77b4", "outcome-prediction shared base")):
            v = json.loads((res / "runs" / name / "valiquality.json").read_text())
            ax.plot([e["update"] for e in v], [e["hv"] for e in v], color=col, lw=1.6, marker="o", ms=2.2,
                    label=f"{lab} (best {row[arm]['best_hv']:.4f} @ {row[arm]['best_update']})")
        ax.axhline(row["baseline_best_validation_hv"], color="#555555", ls="--", lw=.9,
                   label=f"random-base best {row['baseline_best_validation_hv']:.4f}")
        ax.set_xlabel("updates"); ax.set_ylabel("validation HV (18 instances x 15 preferences)")
        ax.set_title(f"{title[m]}: validation HV during training (10x5, seed 111, waiting allowed)", fontsize=10.5)
        ax.grid(alpha=.3); ax.legend(frameon=False, fontsize=8.5, loc="lower right")
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        fig.tight_layout(); fig.savefig(r.res / f"curves_{m}.png", dpi=140, facecolor="white"); plt.close(fig)
    print(json.dumps({m: {k: v for k, v in row.items() if k != "matched_updates"} for m, row in out["methods"].items()}, indent=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--summarise", action="store_true")
    ap.add_argument("--check_only", action="store_true")
    a = ap.parse_args()
    a.smoke, a.jobs = False, 1
    r = TD(a)
    if a.summarise:
        summarise(r); return
    if a.check_only:
        print(json.dumps({"identity": r.identity_check(), "commands": {m: r.command_check(m) for m in METHODS},
                          "distill_cmd": r.distill_cmd(r.ckpt / "distill" / "student_s111.pth")}, indent=1))
        return
    sys.exit(r.main())


if __name__ == "__main__":
    main()
