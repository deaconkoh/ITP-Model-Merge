"""Best-of-128 evaluation HV over training for three runs (10x5, seed 111, waiting allowed):
  rs_random       Rewarded Soups, random shared base         (results/specialist_reuse_v3, final:rs)
  rs_world_model  Rewarded Soups, world-model shared base    (results/specialist_reuse_wm_v3, final:rs)
  scratch         From-scratch DANIEL                        (results/specialist_reuse_v3, final:scratch)
The original runs kept no intermediate snapshots, so each is RE-RUN with its original command (built by the
baseline runner with the same selected settings, specialists and base) plus --budget_checkpoints, under a new run
name; the originals are only read. Every snapshot on the grid (and at each run's originally selected update) is
evaluated with the unchanged original evaluation script (specialist_reuse_v3_eval.py: 100 evaluation instances x 67
preferences, best-of-128, seed 51); only the best-of-128 HV is reported (frozen normalisation, reference 1.1).
Reproduction checks: re-run validation curve vs the original; the snapshot at the original selected update vs the
original selected checkpoint (weights and evaluation HV).

    python tools/eval_curve_v3.py [--summarise]       # -> results|checkpoints/eval_curve_v3
"""
from __future__ import annotations
import argparse, json, shutil, subprocess, sys, threading, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))
import freeze_specialist_reuse_v3 as F                              # noqa: E402
import specialist_reuse_v3_runner as R                              # noqa: E402

NAME = "eval_curve_v3"
BASE_RES, BASE_CK = REPO / "results" / "specialist_reuse_v3", REPO / "checkpoints" / "specialist_reuse_v3"
WM_RES, WM_CK = REPO / "results" / "specialist_reuse_wm_v3", REPO / "checkpoints" / "specialist_reuse_wm_v3"
GRID = list(range(0, 2001, 250)) + list(range(2500, 8001, 500))
RUNS = {
    "rs_random": {"label": "Rewarded Soups, random shared base", "method": "rs", "orig_tag": "sr", "orig_suffix": "rs",
                  "ck": BASE_CK, "res": BASE_RES, "unit": "final:rs", "spec_tag": "sr"},
    "rs_world_model": {"label": "Rewarded Soups, world-model shared base", "method": "rs", "orig_tag": "srwm",
                       "orig_suffix": "rs", "ck": WM_CK, "res": WM_RES, "unit": "final:rs", "spec_tag": "srwm"},
    "scratch": {"label": "From-scratch DANIEL", "method": "scratch", "orig_tag": "sr", "orig_suffix": "scratch",
                "ck": BASE_CK, "res": BASE_RES, "unit": "final:scratch", "spec_tag": "sr"},
}
_load = F.load
sha = R.sha


def orig(run):
    c = RUNS[run]
    st = json.loads((c["res"] / "state.json").read_text())
    u = st["units"][c["unit"]]
    sel = st["selected"][c["method"]]
    name = f"10x5+mix+bv3_{c['orig_tag']}_{c['orig_suffix']}_s111"
    return {"name": name, "unit": u, "selected": sel, "best_update": u["best_update"],
            "final": c["ck"] / "finals" / f"{name}.pth", "eval": c["res"] / "eval" / f"{c['method']}.npz",
            "valiquality": c["res"] / "runs" / name / "valiquality.json", "run_json": c["res"] / "runs" / name / "run.json"}


class EC(R.Runner):
    def __init__(self, a):
        F.load = lambda check=True: {**_load(check), "name": NAME}
        super().__init__(a)
        self.tag = "ecv"
        self.ev_lock = threading.Lock()

    def commit(self, msg, *paths):
        pass

    def budget(self):
        return sorted(set(GRID) | {orig(r)["best_update"] for r in RUNS})

    def cmd(self, run, tag=None, suffix=None):
        c, o = RUNS[run], orig(run)
        old_tag, old_specs, old_base = self.tag, self.specs, self.base
        self.tag = tag or "ecv"
        self.specs = lambda: [str(c["ck"] / "specialists" / f"10x5+mix+bv3_{c['spec_tag']}_spec_{x}_s111.pth") for x in "mct"]
        self.base = c["ck"] / "base_s111.pth"
        try:
            return self.method_args(c["method"], suffix or f"{run}", o["selected"]["lr"], o["selected"]["cl_beta"], full=True)
        finally:
            self.tag, self.specs, self.base = old_tag, old_specs, old_base

    def new_cmd(self, run):
        return [*self.cmd(run), "--budget_checkpoints", ",".join(map(str, self.budget()))]

    def command_check(self, run):
        c = RUNS[run]
        old = self.cmd(run, tag=c["orig_tag"], suffix=c["orig_suffix"])
        new = self.new_cmd(run)
        diff = [[a, b] for a, b in zip(old, new) if a != b]
        flags = {old[i - 1] for i, (a, b) in enumerate(zip(old, new)) if a != b}
        assert flags == {"--model_suffix"} and new[len(old):][0] == "--budget_checkpoints", (flags, new[len(old):])
        rec = json.loads(orig(run)["run_json"].read_text())["config"]
        mm = old[old.index("--merge_mix") + 1] if "--merge_mix" in old else ""
        assert rec.get("merge_mix", "") == mm, (run, rec.get("merge_mix"), mm)   # the same specialists as the original
        assert rec["lr"] == orig(run)["selected"]["lr"] and rec["cl_beta"] == orig(run)["selected"]["cl_beta"]
        return {"differing_flags": sorted(flags), "diff": diff, "added": new[len(old):]}

    def keep_snapshots(self, name):
        d = self.ckpt / "snapshots"; d.mkdir(parents=True, exist_ok=True)
        for p in R.TN.glob(f"{name}@u*.pth"):
            shutil.copy2(p, d / p.name)
        shutil.copy2(R.TN / f"{name}.pth", d / f"{name}.selected.pth")
        self.write_sums()

    def eval_targets(self):
        out = []
        for run in RUNS:
            name = self.model(run)
            want = set(GRID) | {orig(run)["best_update"]}
            for p in sorted((self.ckpt / "snapshots").glob(f"{name}@u*.pth")):
                u = int(p.stem.split("@u")[1])
                if u in want:
                    out.append((run, u, p))
        return sorted(out, key=lambda t: (t[1], t[0]))

    def evaluate(self, items):
        free, lock = list(range(self.jobs)), threading.Lock()

        def work(it):
            run, u, p = it
            out = self.res / "eval" / f"{run}_u{u}.npz"
            if out.exists():
                return True
            out.parent.mkdir(parents=True, exist_ok=True)
            with lock:
                slot = free.pop(0)
            try:
                t0 = time.time()
                rc = subprocess.run(["taskset", "-c", R.SLOTS[slot], R.PY, str(REPO / "tools" / "specialist_reuse_v3_eval.py"),
                                     "--model", str(p), "--out", str(out)], cwd=REPO, env=R.ENV,
                                    stdout=self.logf, stderr=subprocess.STDOUT).returncode
                with self.ev_lock:
                    self.state.setdefault("eval_seconds", {})[f"{run}_u{u}"] = time.time() - t0; self.save_state()
                self.log(f"evaluated {run} @ {u} ({time.time() - t0:.0f} s, slot {slot}, rc {rc})")
                return rc == 0
            finally:
                with lock:
                    free.append(slot)
        with ThreadPoolExecutor(self.jobs) as ex:
            ok = list(ex.map(work, items))
        return all(ok)

    def main(self):
        t0 = time.time()
        self.log(f"start ({self.jobs} slots); budget checkpoints {self.budget()}")
        self.state["command_check"] = {r: self.command_check(r) for r in RUNS}; self.save_state()
        units = [(f"train:{r}", self.new_cmd(r), self.model(r), self.keep_snapshots) for r in ("scratch", "rs_random", "rs_world_model")]
        if not self.pool(units):
            raise SystemExit("training failed")
        self.state["train_wall_hours"] = (time.time() - t0) / 3600; self.save_state()
        items = self.eval_targets()
        self.log(f"evaluating {len(items)} snapshots")
        if not self.evaluate(items):
            raise SystemExit("some evaluations failed")
        self.state["wall_hours"] = (time.time() - t0) / 3600; self.save_state()
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

    def b128(f):
        e = np.load(f)
        return AN.metrics(e["sample128_obj"], list(e["keys"]), norm, b2)[0], str(e["model_sha256"])
    out = {"grid": GRID, "runs": {}}
    for run, c in RUNS.items():
        o = orig(run)
        name = r.model(run)
        u = r.state["units"][f"train:{run}"]
        pts = []
        for f in sorted((r.res / "eval").glob(f"{run}_u*.npz"), key=lambda p: int(p.stem.split("_u")[1])):
            up = int(f.stem.split("_u")[1])
            hv, msha = b128(f)
            snap = r.ckpt / "snapshots" / f"{name}@u{up}.pth"
            pts.append({"update": up, "best_of_128_hv": hv, "model_sha_matches_snapshot": msha == sha(snap)})
        vn = json.loads((r.res / "runs" / name / "valiquality.json").read_text())
        vo = json.loads(o["valiquality"].read_text())
        common = min(len(vn), len(vo))
        diffs = [abs(vn[i]["hv"] - vo[i]["hv"]) for i in range(common)]
        first_div = next((vn[i]["update"] for i in range(common) if diffs[i] > 0), None)
        bu = o["best_update"]
        snap = r.ckpt / "snapshots" / f"{name}@u{bu}.pth"
        a = torch.load(snap, map_location="cpu", weights_only=False)["state_dict"]
        b = torch.load(o["final"], map_location="cpu", weights_only=False)["state_dict"]
        same_w = set(a) == set(b) and all(torch.equal(a[k], b[k]) for k in a)
        o_hv = b128(o["eval"])[0]
        s_hv = next((p["best_of_128_hv"] for p in pts if p["update"] == bu), None)
        rn = json.loads((r.res / "runs" / name / "run.json").read_text())
        ro = json.loads(o["run_json"].read_text())
        keys = set(rn["config"]) | set(ro["config"])
        cdiff = sorted(k for k in keys if rn["config"].get(k) != ro["config"].get(k))
        out["runs"][run] = {
            "label": c["label"], "points": pts,
            "rerun": {"stop_update": u["update"], "status": u["status"], "best_update": u["best_update"],
                      "best_validation_hv": -u["best"], "train_hours": u["train_seconds"] / 3600},
            "original": {"stop_update": o["unit"]["update"], "best_update": bu, "best_validation_hv": -o["unit"]["best"],
                         "best_of_128_hv_selected": o_hv},
            "reproduction": {"validation_curve_identical": first_div is None and len(vn) == len(vo),
                             "validation_points_compared": common, "max_abs_validation_hv_diff": max(diffs),
                             "first_diverging_update": first_div,
                             "snapshot_at_original_selected_update_equals_original_checkpoint": same_w,
                             "snapshot_best_of_128_hv_at_original_selected_update": s_hv,
                             "original_selected_best_of_128_hv": o_hv},
            "config_check": {"differing_config_keys": cdiff,
                             "matches": set(cdiff) <= {"model_suffix", "budget_checkpoints"}
                             and rn["data_fingerprints"] == ro["data_fingerprints"]}}
    out["command_check"] = r.state.get("command_check")
    out["eval_seconds"] = r.state.get("eval_seconds"); out["train_wall_hours"] = r.state.get("train_wall_hours")
    (r.res / "summary.json").write_text(json.dumps(out, indent=1) + "\n")
    plot(r, out)
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "points"} for k, v in out["runs"].items()}, indent=1))


def plot(r, out):
    """One panel: every evaluated snapshot on its line; the validation-selected checkpoint is the large ringed point."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    INK, MUTED, GRIDC, BG = "#1f1f1e", "#6b6a64", "#e4e3de", "#fcfcfb"
    style = {"rs_world_model": ("#eb6834", "s"), "rs_random": ("#2a78d6", "o"), "scratch": ("#1baf7a", "^")}
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
                         "text.color": INK, "axes.labelcolor": INK})
    fig, ax = plt.subplots(figsize=(11, 6), facecolor=BG)
    ax.set_facecolor(BG)
    for run in style:
        d = out["runs"][run]
        col, mk = style[run]
        pts = sorted(d["points"], key=lambda p: p["update"])
        x = [p["update"] for p in pts]; y = [p["best_of_128_hv"] for p in pts]
        bu = d["original"]["best_update"]; bv = next(p["best_of_128_hv"] for p in pts if p["update"] == bu)
        ax.plot(x, y, color=col, lw=2.2, marker=mk, ms=5, mec=BG, mew=1, zorder=3,
                label=f"{d['label']}: selected checkpoint {bv:.3f} (update {bu:,})")
        ax.plot(bu, bv, marker=mk, ms=13, color=col, mec=INK, mew=1.4, zorder=5)
        ax.annotate(f"{bv:.3f}", (bu, bv), xytext=(0, 11), textcoords="offset points", ha="center",
                    fontsize=9.5, fontweight="bold", color=INK, zorder=6)
    ax.plot([], [], marker="o", ms=11, color="#bdbcb5", mec=INK, mew=1.4, ls="none",
            label="large outlined point = validation-selected checkpoint (the reported final policy)")
    ax.set_ylim(0.70, 1.13)
    ax.set_xlim(-100, 4500)
    ax.set_xlabel("training updates")
    ax.set_ylabel("best-of-128 HV on the evaluation set (higher is better)")
    ax.set_title("Best-of-128 evaluation HV during training", fontsize=13, loc="left", pad=12)
    ax.grid(color=GRIDC, lw=.8); ax.set_axisbelow(True)
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)
    ax.legend(frameon=False, loc="lower right", fontsize=9)
    fig.text(0.01, 0.012, "Rewarded Soups starts from merged specialists, so its updates are merge fine-tuning only "
                          "(specialist training happened before update 0).\nFrom scratch starts at HV 0.003 (below the "
                          "axis). 100 evaluation instances x 67 preferences, best of 128 samples (seed 51). "
                          "10x5, seed 111, waiting allowed. Single seed.", fontsize=8.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(r.res / "best_of_128_curve.png", dpi=150, facecolor=BG); plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--summarise", action="store_true")
    ap.add_argument("--check_only", action="store_true")
    ap.add_argument("--jobs", type=int, default=3)
    a = ap.parse_args()
    a.smoke = False
    r = EC(a)
    if a.summarise:
        summarise(r); return
    if a.check_only:
        print(json.dumps({"budget": r.budget(), **{k: r.command_check(k) for k in RUNS}}, indent=1)); return
    sys.exit(r.main())


if __name__ == "__main__":
    main()
