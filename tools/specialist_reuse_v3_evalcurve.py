"""Evaluation-set version of results/specialist_reuse_v3/figures/curves_10x5.png: best-of-128 HV (seed 51, 100
evaluation instances x 66 preferences, the results table's column) of the four full runs every 250 updates.

The full runs saved only their best and final checkpoints, so each is REPLAYED with exactly the runner's command line
(same selected settings, chunking, slots; only the run name differs, plus --budget_checkpoints 0,250,...), saving a
snapshot every 250 updates. Training is deterministic; each replay must reproduce its run's logged validation curve
(every value) and stop update exactly, or nothing is evaluated. Snapshots go to checkpoints/specialist_reuse_v3/
snapshots/ (SHA256SUMS.json); evaluations to results/specialist_reuse_v3/eval_curve/; the figure to
results/specialist_reuse_v3/figures/curves_10x5_eval.png. Resumable; commits locally; never pushes.

    python tools/specialist_reuse_v3_evalcurve.py all [--jobs 3]
    python tools/specialist_reuse_v3_evalcurve.py plot
"""
from __future__ import annotations
import argparse, hashlib, json, shutil, subprocess, sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))
import freeze_specialist_reuse_v3 as F                              # noqa: E402
import specialist_reuse_v3_runner as R                              # noqa: E402

RES = REPO / "results" / "specialist_reuse_v3"
CK = REPO / "checkpoints" / "specialist_reuse_v3" / "snapshots"
OUT = RES / "eval_curve"
EVERY = 250
METHODS = ("ties", "rs", "distill", "scratch")                      # slowest first


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


class Replay(R.Runner):
    def __init__(self, jobs):
        a = argparse.Namespace(smoke=False, jobs=jobs)
        super().__init__(a)
        self.res = RES / "eval_curve" / "replay"; self.res.mkdir(parents=True, exist_ok=True)
        self.logf = open(RES / "eval_curve" / "replay.log", "a")
        self.state_f = self.res / "state.json"
        orig = json.loads((RES / "state.json").read_text())
        self.orig = orig
        self.state = json.loads(self.state_f.read_text()) if self.state_f.exists() else {"units": {}}
        self.state["ties_selected"] = orig["ties_selected"]

    def cmd(self, m):
        t = self.orig["selected"][m]
        a = self.method_args(m, f"{m}_rp", t["lr"], t["cl_beta"], full=True)
        return a + ["--budget_checkpoints", ",".join(str(k) for k in range(0, 8001, EVERY))]

    def run_all(self):
        def keep(name):
            m = name.split("_sr_")[1].split("_rp")[0]
            CK.mkdir(parents=True, exist_ok=True)
            for f in sorted(R.TN.glob(f"{name}@u*.pth")):
                shutil.copy2(f, CK / f"{m}{f.name[f.name.index('@'):]}")
            check(m, name)
            (CK / "SHA256SUMS.json").write_text(json.dumps({p.name: sha(p) for p in sorted(CK.glob("*.pth"))}, indent=1) + "\n")
        units = [(f"replay:{m}", self.cmd(m), self.model(f"{m}_rp"), keep) for m in METHODS]
        if not self.pool(units):
            raise SystemExit("a replay failed")


def check(m, name):
    """The replay must reproduce the original run's logged validation curve and stop update exactly."""
    o = json.loads((RES / "runs" / f"10x5+mix+bv3_sr_{m}_s111" / "valiquality.json").read_text())
    r = json.loads((R.TL / f"valiquality_{name}.json").read_text())
    so = json.loads((RES / "runs" / f"10x5+mix+bv3_sr_{m}_s111" / "status.json").read_text())
    sr = json.loads((R.TL / f"{name}.status.json").read_text())
    same = [e["hv"] for e in o] == [e["hv"] for e in r] and (so["update"], so["status"]) == (sr["update"], sr["status"])
    rec = {"method": m, "logged_points": len(o), "replayed_points": len(r), "identical": same,
           "stop": [so["update"], so["status"]], "replay_stop": [sr["update"], sr["status"]]}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"replay_check_{m}.json").write_text(json.dumps(rec, indent=1) + "\n")
    if not same:
        raise SystemExit(f"replay of {m} does not reproduce the original run: {rec}")


def evaluate():
    for m in METHODS:
        if not json.loads((OUT / f"replay_check_{m}.json").read_text())["identical"]:
            raise SystemExit(f"{m}: replay not verified; not evaluated")
    for f in sorted(CK.glob("*@u*.pth"), key=lambda p: (METHODS.index(p.name.split("@")[0]), int(p.stem.split("@u")[1]))):
        out = OUT / f"{f.stem.replace('@', '_')}.npz"
        if out.exists():
            continue
        t0 = time.time()
        r = subprocess.run(["taskset", "-c", "0-11", sys.executable, str(REPO / "tools" / "specialist_reuse_v3_eval.py"),
                            "--model", str(f), "--out", str(out)], cwd=REPO, env=R.ENV, capture_output=True, text=True)
        if r.returncode != 0:
            raise SystemExit(f"evaluation of {f.name} failed: {r.stderr[-2000:]}")
        print(f"[{time.strftime('%H:%M:%S')}] evaluated {f.name} ({time.time() - t0:.0f} s)", flush=True)


def hv_of(npz):
    sys.path.insert(0, str(REPO / "daniel"))
    import numpy as np
    import mo_metrics as MM
    cfg = F.load()
    norm = json.loads((REPO / cfg["evaluation"]["norm"]).read_text())["instances"]
    e = np.load(npz)
    return float(np.mean([MM.instance_hv(e["sample128_obj"][i, :66], *MM.ideal_nadir(norm[k]))
                          for i, k in enumerate(e["keys"])]))


def plot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import plot_specialist_reuse_v3_curves as P
    st = json.loads((RES / "state.json").read_text())["units"]
    summary = json.loads((RES / "summary.json").read_text())["rows"]
    cache_f = OUT / "hv.json"
    cache = json.loads(cache_f.read_text()) if cache_f.exists() else {}
    plt.rcParams.update({"font.size": 11})
    fig, ax = plt.subplots(figsize=(10.5, 5.8), facecolor="white")
    ax.set_facecolor("white"); ax.grid(alpha=.3, lw=.6)
    ends = []
    for name, key, col in P.RUNS:
        pts = []
        for f in sorted(OUT.glob(f"{key}_u*.npz"), key=lambda p: int(p.stem.split("_u")[1])):
            if f.name not in cache:
                cache[f.name] = hv_of(f)
            pts.append((int(f.stem.split("_u")[1]), cache[f.name]))
        if not pts:
            continue
        u, hv = zip(*pts)
        ax.plot(u, hv, color=col, lw=2, marker="o", ms=3.5)
        b = st[f"final:{key}"]
        ax.plot([b["best_update"]], [summary[key]["sample_hv"]], marker="*", ms=13, color=col, mec="black", mew=.6,
                zorder=5)
        ends.append([u[-1], hv[-1], hv[-1], f"{name} ({b['status'].lower()}, {b['update']})", col])
    cache_f.write_text(json.dumps(cache, indent=1) + "\n")
    xmax = max(e[0] for e in ends)
    ref = summary["scratch"]["sample_hv"]
    ax.hlines(ref, 0, xmax, color="#555555", lw=.9, ls="--")
    ends.append([xmax, ref, ref, f"From scratch, table value {ref:.3f}", "#555555"])
    ends.sort(key=lambda e: e[1])
    lo_y, hi_y = ax.get_ylim()
    gap = (hi_y - lo_y) * 0.05
    for lo, hi in zip(ends, ends[1:]):
        hi[2] = max(hi[2], lo[2] + gap)
    shift = sum(e[1] - e[2] for e in ends) / len(ends)
    for e in ends:
        e[2] += shift
    for xe, ye, yl, name, col in ends:
        ax.annotate(name, (xe, ye), xytext=(xmax * 1.04, yl), textcoords="data", va="center", color=col,
                    fontsize=9.5 if name.startswith("From scratch, table") else 11,
                    arrowprops=dict(arrowstyle="-", color=col, lw=.6, shrinkA=0, shrinkB=2, alpha=.6))
    ax.set_xlim(0, xmax * 1.5)
    ax.set_xticks(np.arange(0, xmax + 1, 500))
    ax.set_xlabel("updates"); ax.set_ylabel("evaluation hypervolume, best-of-128")
    ax.set_title("Evaluation hypervolume (best-of-128), specialist reuse vs from scratch, waiting allowed (10x5, seed 111)",
                 fontsize=11.5)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.text(0.5, 0.01, "100 evaluation instances x 66 preferences, sampling seed 51; snapshots every 250 updates. "
             "Stars: the selected (best-validation) checkpoint, as in the results table.", ha="center", fontsize=9.5)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    out = RES / "figures" / "curves_10x5_eval.png"
    fig.savefig(out, dpi=130, facecolor="white"); plt.close(fig)
    print(f"wrote {out.relative_to(REPO)}")


def commit(msg, *paths):
    for p in paths:
        subprocess.run(["git", "add", "-A", "--", str(p)], cwd=REPO, check=True)
    if subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=REPO).returncode:
        subprocess.run(["git", "commit", "-q", "-m", msg, "-m", R.CO_AUTHOR], cwd=REPO, check=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("action", choices=["all", "plot"])
    ap.add_argument("--jobs", type=int, default=3)
    a = ap.parse_args()
    if a.action == "all":
        Replay(a.jobs).run_all()
        commit("specialist reuse v3: replays with snapshots every 250 updates (verified identical)", CK, OUT)
        evaluate()
        commit("specialist reuse v3: best-of-128 evaluation of every snapshot", OUT)
    plot()
    commit("specialist reuse v3: evaluation-set convergence figure", OUT, RES / "figures" / "curves_10x5_eval.png",
           Path(__file__))
