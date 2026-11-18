"""World-model shared-base Rewarded Soups (10x5, seed 111, waiting allowed): the specialist-reuse Rewarded Soups arm
with ONE change, the specialists' starting encoder.

Baseline: results/specialist_reuse_v3 (configs/specialist_reuse_v3.json), Rewarded Soups, best-of-128 HV 1.085.
New base: the baseline's random shared base (checkpoints/specialist_reuse_v3/base_s111.pth) with feature_exact
replaced by the trunk of checkpoints/world_model_m1_v3_action_pilot/pilot_pairs_abs+diff.pth (the outcome-prediction
pilot encoder with pair features and the action-difference loss; the encoder of the tardiness pilot's pretrained arm).

Every command is built by the baseline's own runner (tools/specialist_reuse_v3_runner.Runner): specialist recipe and
stopping rule, the merge checks, the Rewarded Soups full run with the baseline's SELECTED setting (no re-tuning), and
the evaluation script (100 x 66 + equal weights, greedy and best-of-128 seed 51). Only the run tag (srwm), the output
folders and the base change. The baseline's results and checkpoints are only read.

    python tools/specialist_reuse_wm_v3.py        # -> results|checkpoints/specialist_reuse_wm_v3
"""
from __future__ import annotations
import argparse, hashlib, json, subprocess, sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))
import freeze_specialist_reuse_v3 as F                              # noqa: E402
import specialist_reuse_v3_runner as R                              # noqa: E402

NAME = "specialist_reuse_wm_v3"
BASE_RES = REPO / "results" / "specialist_reuse_v3"
BASE_CK = REPO / "checkpoints" / "specialist_reuse_v3"
PILOT = REPO / "checkpoints" / "world_model_m1_v3_action_pilot" / "pilot_pairs_abs+diff.pth"
TP_PRE = REPO / "checkpoints" / "world_model_m1_v3_tardiness_pilot" / "start_pre.pth"
_load = F.load


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


class WM(R.Runner):
    def __init__(self, a):
        F.load = lambda check=True: {**_load(check), "name": NAME}
        super().__init__(a)
        self.tag = "srwm"
        self.prefix = "specialist reuse WM v3"
        self.md = REPO / "results" / f"{NAME}.md"

    def commit(self, msg, *paths):
        pass                                                         # committed once at the end, by hand

    def stage_base(self):
        import torch
        rec = self.res / "base_check.json"
        if self.base.exists() and rec.exists():
            return
        b = torch.load(BASE_CK / "base_s111.pth", map_location="cpu", weights_only=False)
        pilot = torch.load(PILOT, map_location="cpu", weights_only=False)
        trunk = {k[len("trunk."):]: v for k, v in pilot.items() if k.startswith("trunk.")}
        sd = b["state_dict"]
        fe = {k for k in sd if k.startswith("feature_exact.")}
        assert {f"feature_exact.{k}" for k in trunk} == fe
        new = dict(b); new["state_dict"] = {k: (trunk[k[len("feature_exact."):]].clone() if k in fe else v.clone())
                                            for k, v in sd.items()}
        torch.save(new, self.base)
        a = new["state_dict"]
        tp = torch.load(TP_PRE, map_location="cpu", weights_only=False)["state_dict"]
        tp_settings = json.loads((REPO / "results/world_model_m1_v3/tardiness_pilot/settings.json").read_text())
        chk = {"baseline_base_sha_matches_its_sums": json.loads((BASE_CK / "SHA256SUMS.json").read_text())["base_s111.pth"]
               == sha(BASE_CK / "base_s111.pth"),
               "pilot_sha_matches_tardiness_transfer_run": sha(PILOT) == tp_settings["sources"]["pilot_encoder"]["sha256"],
               "encoder_equals_tardiness_transfer_pretrained_start": all(torch.equal(a[k], tp[k]) for k in fe),
               "identical_outside_encoder_to_baseline_base": all(torch.equal(a[k], sd[k]) for k in sd if k not in fe),
               "encoder_differs_from_baseline_base": any(not torch.equal(a[k], sd[k]) for k in fe),
               "new_base_sha256": sha(self.base)}
        assert all(v for k, v in chk.items() if k != "new_base_sha256"), chk
        rec.write_text(json.dumps(chk, indent=1) + "\n")
        self.write_sums()
        self.log(f"world-model base built and checked: {chk}")

    def stage_pool_a(self):
        units = [(f"spec:{o}", self.specialist_args(o), self.model(f"spec_{o}"), self.keep_specialist) for o in "tcm"]
        return self.pool(units)

    def stage_rs(self):
        base_state = json.loads((BASE_RES / "state.json").read_text())
        sel = base_state["selected"]["rs"]
        self.state["selected"] = {"rs": sel, "source": "baseline state.json (not re-tuned)"}; self.save_state()
        self.log(f"Rewarded Soups with the baseline's selected setting: lr {sel['lr']}, beta {sel['cl_beta']}")

        def keep(name):
            d = self.ckpt / "finals"; d.mkdir(exist_ok=True)
            import shutil
            shutil.copy2(R.TN / f"{name}.pth", d / f"{name}.pth"); self.write_sums()
        return self.pool([("final:rs", self.method_args("rs", "rs", sel["lr"], sel["cl_beta"], full=True),
                           self.model("rs"), keep)])

    def stage_eval_rs(self):
        out = self.res / "eval" / "rs.npz"
        if out.exists():
            return
        cmd = [R.PY, str(REPO / "tools" / "specialist_reuse_v3_eval.py"), "--model",
               str(self.ckpt / "finals" / f"{self.model('rs')}.pth"), "--out", str(out)]
        t0 = time.time()
        if subprocess.run(["taskset", "-c", "0-11", *cmd], cwd=REPO, env=R.ENV, stdout=self.logf,
                          stderr=subprocess.STDOUT).returncode != 0:
            raise SystemExit("evaluation failed")
        self.state["eval_seconds"] = time.time() - t0; self.save_state()
        self.log("evaluated rs")

    def protocol_check(self):
        """Saved run records vs the baseline's: every config key equal except run name and start / merge files."""
        out = {}
        pairs = [(f"spec_{o}", f"spec_{o}") for o in "mct"] + [("rs", "rs")]
        allowed = {"model_suffix", "init_from", "merge_mix"}
        for new, old in pairs:
            n = json.loads((self.res / "runs" / self.model(new) / "run.json").read_text())
            o = json.loads((BASE_RES / "runs" / f"10x5+mix+bv3_sr_{old}_s111" / "run.json").read_text())
            keys = set(n["config"]) | set(o["config"])
            diff = sorted(k for k in keys if n["config"].get(k) != o["config"].get(k))
            out[new] = {"differing_config_keys": diff, "data_fingerprints_equal": n["data_fingerprints"] == o["data_fingerprints"],
                        "matches": set(diff) <= allowed and n["data_fingerprints"] == o["data_fingerprints"]}
        return out

    def main(self):
        t0 = time.time()
        self.log(f"start ({self.jobs} slots)")
        self.stage_base()
        if not self.stage_pool_a():
            raise SystemExit("specialist training had failures")
        self.stage_checks()
        if not self.stage_rs():
            raise SystemExit("Rewarded Soups run failed")
        self.stage_eval_rs()
        self.state["protocol_check"] = self.protocol_check(); self.save_state()
        self.state["wall_hours"] = (time.time() - t0) / 3600; self.save_state()
        summarise(self)
        self.log(f"done ({(time.time() - t0) / 3600:.2f} h this invocation)")
        return 0


def summarise(r):
    import numpy as np
    import analyze_specialist_reuse_v3 as AN
    cfg = _load()
    norm = json.loads((REPO / cfg["evaluation"]["norm"]).read_text())["instances"]
    b2 = REPO / cfg["evaluation"]["igd_front"]
    bst = json.loads((BASE_RES / "state.json").read_text())["units"]
    st = r.state["units"]
    out = {"eval": {}, "specialists": {}, "rs": {}}
    for lab, f in (("random shared base", BASE_RES / "eval" / "rs.npz"), ("world-model shared base", r.res / "eval" / "rs.npz")):
        e = np.load(f); keys = list(e["keys"])
        g = AN.metrics(e["greedy_obj"], keys, norm, b2); s = AN.metrics(e["sample128_obj"], keys, norm, b2)
        out["eval"][lab] = {"greedy_hv": g[0], "best_of_128_hv": s[0], "greedy_igd": g[1], "best_of_128_igd": s[1],
                            "instances": len(keys), "model_sha256": str(e["model_sha256"])}
    for o in "mct":
        u, b = st[f"spec:{o}"], bst[f"spec:{o}"]
        out["specialists"][o] = {k: {"status": x["status"], "updates": x["update"], "best": x["best"],
                                     "best_update": x["best_update"], "hours": x["train_seconds"] / 3600}
                                 for k, x in (("world_model", u), ("random", b))}
    target = -bst["final:rs"]["best"]
    vn = json.loads((r.res / "runs" / r.model("rs") / "valiquality.json").read_text())
    vb = json.loads((BASE_RES / "runs" / "10x5+mix+bv3_sr_rs_s111" / "valiquality.json").read_text())
    hit = next((e for e in vn if e["hv"] >= target - 1e-12), None)
    out["rs"] = {"target_baseline_best_validation_hv": target,
                 "world_model": {"start_hv": vn[0]["hv"], "best_hv": max(e["hv"] for e in vn),
                                 "best_update": max(vn, key=lambda e: e["hv"])["update"], "status": st["final:rs"]["status"],
                                 "updates": st["final:rs"]["update"], "hours": st["final:rs"]["train_seconds"] / 3600,
                                 "reached_update": hit["update"] if hit else None,
                                 "reached_hours": hit["train_seconds"] / 3600 if hit else None},
                 "random": {"start_hv": vb[0]["hv"], "best_hv": max(e["hv"] for e in vb),
                            "best_update": max(vb, key=lambda e: e["hv"])["update"], "status": bst["final:rs"]["status"],
                            "updates": bst["final:rs"]["update"], "hours": bst["final:rs"]["train_seconds"] / 3600}}
    ap = json.loads((REPO / "results/world_model_m1_v3/action_pilot.json").read_text())
    out["costs_hours"] = {
        "pretraining": {"pilot_data_regeneration": ap["pilot_data"]["train_regeneration_seconds"] / 3600,
                        "pilot_encoder_training": ap["pilot"]["pairs|abs+diff"]["seconds"] / 3600},
        "specialist_training": {"world_model_sum": sum(st[f"spec:{o}"]["train_seconds"] for o in "mct") / 3600,
                                "random_sum": sum(bst[f"spec:{o}"]["train_seconds"] for o in "mct") / 3600},
        "merge_fine_tuning": {"world_model": st["final:rs"]["train_seconds"] / 3600,
                              "random": bst["final:rs"]["train_seconds"] / 3600},
        "evaluation_seconds": r.state.get("eval_seconds")}
    out["protocol_check"] = r.state.get("protocol_check")
    (r.res / "summary.json").write_text(json.dumps(out, indent=1) + "\n")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 4, figsize=(18, 4.2), facecolor="white")
    for ax, o in zip(axs[:3], "mct"):
        for lab, f, col in (("random shared base", BASE_RES / "runs" / f"10x5+mix+bv3_sr_spec_{o}_s111" / "valiquality.json", "#7f7f7f"),
                            ("world-model shared base", r.res / "runs" / r.model(f"spec_{o}") / "valiquality.json", "#d62728")):
            v = json.loads(f.read_text())
            ax.plot([e["update"] for e in v], [e["selection_metric"] for e in v], color=col, lw=1.4, label=lab)
        ax.set_yscale("log"); ax.set_title({"m": "makespan", "c": "carbon", "t": "tardiness"}[o] + " specialist")
        ax.set_xlabel("updates"); ax.set_ylabel("own-objective validation score (lower is better)")
    ax = axs[3]
    for lab, v, col in (("random shared base", vb, "#7f7f7f"), ("world-model shared base", vn, "#d62728")):
        ax.plot([e["update"] for e in v], [e["hv"] for e in v], color=col, lw=1.4, label=lab)
    ax.axhline(target, color="#555555", ls="--", lw=.9, label=f"baseline best {target:.4f}")
    ax.set_title("Rewarded Soups fine-tuning"); ax.set_xlabel("updates"); ax.set_ylabel("merged validation HV")
    for a_ in axs:
        a_.grid(alpha=.3); a_.legend(frameon=False, fontsize=8)
        for side in ("top", "right"):
            a_.spines[side].set_visible(False)
    fig.tight_layout(); fig.savefig(r.res / "curves.png", dpi=130, facecolor="white"); plt.close(fig)
    print(json.dumps(out, indent=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--jobs", type=int, default=3)
    ap.add_argument("--summarise", action="store_true")
    a = ap.parse_args()
    a.smoke = False
    r = WM(a)
    if a.summarise:
        summarise(r); return
    sys.exit(r.main())


if __name__ == "__main__":
    main()
