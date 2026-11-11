"""Plateau diagnosis report (docs/plateau_diagnosis_protocol_v3.md): results/plateau_diagnosis_v3.md, verdict.json and
figures, from committed logs and the runner's evaluations. Part B sections appear once its runs exist.

    python tools/analyze_plateau_diagnosis_v3.py [--smoke]
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                     # noqa: E402

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools")); sys.path.insert(0, str(REPO / "daniel"))
import freeze_plateau_diagnosis_v3 as F                             # noqa: E402
import mo_metrics as MM                                             # noqa: E402

COL = {"fs": "#ff7f0e", "dist": "#1f77b4", "wm": "#d62728", "ctrl": "#2ca02c", "orig": "#9467bd",
       "beta05": "#1f77b4", "lr1e3": "#ff7f0e", "k16": "#9467bd"}
ORDER = ["fs", "dist", "wm", "ctrl", "orig"]
EARLY, LATE = 250, 500


def rolling(y, w):
    y = np.asarray(y, float); h = w // 2
    return np.array([np.nanmean(y[max(0, i - h):i + h + 1]) for i in range(len(y))])


def series(log, k):
    """Per-update signal from a reward.json log; pairs per group uses the run's number of groups."""
    out = []
    for e in log:
        c = e["cl"]
        if k == "pairs_per_group":
            out.append(c["pairs"] / c["_groups"])
        elif k == "clipped":
            out.append(float(c["grad_norm"] is not None and c["grad_norm"] > c["_clip"]))
        else:
            v = c.get(k)
            out.append(np.nan if v is None else v)
    return np.asarray(out, float)


def load_log(d):
    log = json.loads((Path(d) / "reward.json").read_text())
    cfg = json.loads((Path(d) / "run.json").read_text())["config"]
    for e in log:
        e["cl"]["_groups"] = cfg["num_envs"]; e["cl"]["_clip"] = cfg["cl_grad_clip"]
    return log, cfg


def boot(diff, n, seed):
    rng = np.random.default_rng(seed)
    d = np.asarray(diff, float)
    m = d[rng.integers(0, len(d), (n, len(d)))].mean(1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def ev(edir, name, split):
    j = edir / f"{split}_10x5" / f"{name}.json"
    if not j.exists():
        return None
    r = json.loads(j.read_text())
    g = np.load(edir / f"{split}_10x5" / f"{name}.npz")
    s = edir / f"{split}_10x5" / f"{name}_sample16.npz"
    s = np.load(s) if s.exists() else None
    return {"hv": r["hv_mean"], "s16": r.get("sample", {}).get("hv_mean"), "hv_i": g["hv"], "obj": g["objectives"],
            "keys": [str(k) for k in g["keys"]], "s16_i": None if s is None else s["hv"],
            "s16_obj": None if s is None else s["objectives"], "igd": r.get("igd_plus_mean"),
            "s16_igd": r.get("sample", {}).get("igd_plus_mean")}


def f3(x, d=3):
    return "-" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{d}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    cfg = F.load()
    sfx = "_smoke" if a.smoke else ""
    res = REPO / "results" / f"plateau_diagnosis_v3{sfx}"
    fig = res / "figures"; fig.mkdir(parents=True, exist_ok=True)
    md = REPO / "results" / f"plateau_diagnosis_v3{sfx}.md"
    split = cfg["smoke"]["eval_split"] if a.smoke else cfg["part_a"]["evaluation"]["split"]
    lab = {k: v["label"] for k, v in cfg["runs"].items()}
    L = ["# Plateau diagnosis (10x5, seed 111)", "",
         "Protocol: `docs/plateau_diagnosis_protocol_v3.md`; settings frozen in `configs/plateau_diagnosis_v3.json`. "
         "Single seed: numbers only. No change to the benchmark; the reserved final-test sets are never read.", ""]
    verdict = {}

    # ------------------------------------------------------------------ Part A.1 signals
    logs = {k: load_log(REPO / cfg["runs"][k]["dir"]) for k in ORDER}
    drift = json.loads((res / "part_a" / "drift.json").read_text()) if (res / "part_a" / "drift.json").exists() else []
    vq = {k: json.loads((REPO / cfg["runs"][k]["dir"] / "valiquality.json").read_text()) for k in ORDER}
    sigs = [("loss", "contrastive loss"), ("entropy", "policy entropy (per step)"),
            ("distinct_schedules", "distinct schedules per group"), ("pairs_per_group", "pairs per group"),
            ("grad_norm", "gradient norm (before clipping)"), ("groups_with_pair", "share of groups with a pair")]
    W = cfg["part_a"]["smooth_updates"]
    f, axs = plt.subplots(2, 4, figsize=(18, 8), facecolor="white")
    axs = axs.ravel()
    for i, (k, t) in enumerate(sigs):
        ax = axs[i]
        for r in ORDER:
            y = series(logs[r][0], k)
            ax.plot(np.arange(1, len(y) + 1), rolling(y, W), color=COL[r], lw=1.4, label=lab[r])
        if k == "loss":
            ax.axhline(cfg["part_a"]["log2"], color="#555", ls="--", lw=.8); ax.text(0, cfg["part_a"]["log2"], " log 2",
                                                                                    va="bottom", fontsize=8, color="#555")
        if k == "grad_norm":
            ax.axhline(1.0, color="#555", ls="--", lw=.8); ax.set_yscale("log")
        ax.set_title(t, fontsize=11); ax.grid(alpha=.3)
    ax = axs[6]
    for r in ORDER:
        p = sorted([d for d in drift if d["run"] == r], key=lambda d: d["update"])
        if p:
            ax.plot([d["update"] for d in p], [d["policy_drift"] for d in p], "o-", color=COL[r], lw=1.4)
    ax.set_title("weight distance from start (trunk + actor, relative)", fontsize=11); ax.grid(alpha=.3)
    ax = axs[7]
    for r in ORDER:
        ax.plot([e["update"] for e in vq[r]], rolling([e["hv"] for e in vq[r]], 5), color=COL[r], lw=1.4, label=lab[r])
    ax.set_title("validation HV (smoothed over 5 points)", fontsize=11); ax.grid(alpha=.3)
    ax.legend(fontsize=8, loc="lower right")
    for ax in axs:
        ax.set_xlabel("updates", fontsize=9)
    f.suptitle(f"Learning signals during contrastive fine-tuning (10x5, seed 111); smoothed over {W} updates", fontsize=13)
    f.tight_layout(); f.savefig(fig / "signals.png", dpi=110, facecolor="white"); plt.close(f)

    L += ["## Part A: no training", "", "### 1. Do the learning signals shrink?", "",
          f"Means over the first {EARLY} updates (early) and the last {LATE} updates (late) of each run's log. Pairs per "
          "group = pairs / 20 groups. Loss at log 2 (0.693) means the policy separates better and worse schedules no "
          "more than its frozen start does.", "",
          "| Run | Updates | Loss early → late | Entropy early → late | Distinct schedules (of K) early → late | "
          "Pairs per group early → late | Groups with a pair, late | Gradient norm early → late | Share clipped, late |",
          "|---|---|---|---|---|---|---|---|---|"]
    rules = {}
    for r in ORDER:
        log, c = logs[r]
        def em(k, sl):
            return float(np.nanmean(series(log, k)[sl]))
        e_, l_ = slice(0, EARLY), slice(-LATE, None)
        row = {k: (em(k, e_), em(k, l_)) for k in ("loss", "entropy", "distinct_schedules", "pairs_per_group",
                                                     "groups_with_pair", "grad_norm", "clipped")}
        K = c["cl_k"]
        s1 = row["loss"][1] > 0.68
        s2 = row["entropy"][1] < 0.5 * row["entropy"][0]
        s3 = row["pairs_per_group"][1] < 0.5 * row["pairs_per_group"][0] or row["distinct_schedules"][1] < K / 2
        rules[r] = {"S1_loss_near_log2": s1, "S2_entropy_collapse": s2, "S3_fewer_pairs": s3, "shrinks": s2 or s3,
                    **{f"{k}_early": v[0] for k, v in row.items()}, **{f"{k}_late": v[1] for k, v in row.items()}}
        L.append(f"| {lab[r]} | {len(log)} | {row['loss'][0]:.3f} → {row['loss'][1]:.3f} | "
                 f"{row['entropy'][0]:.3f} → {row['entropy'][1]:.3f} | {row['distinct_schedules'][0]:.2f} → "
                 f"{row['distinct_schedules'][1]:.2f} (K {K}) | {row['pairs_per_group'][0]:.1f} → "
                 f"{row['pairs_per_group'][1]:.1f} | {row['groups_with_pair'][1]:.2f} | {row['grad_norm'][0]:.3f} → "
                 f"{row['grad_norm'][1]:.3f} | {row['clipped'][1]:.2f} |")
    L += ["", "Rules (protocol): S1 loss stuck near log 2 = late loss > 0.68; S2 entropy collapsing = late < 0.5 × early; "
          "S3 fewer valid pairs = late pairs per group < 0.5 × early, or late distinct schedules < K/2.", "",
          "| Run | S1 loss near log 2 | S2 entropy collapsing | S3 fewer pairs | Signal shrinks |", "|---|---|---|---|---|"]
    yn = lambda b: "yes" if b else "no"                                       # noqa: E731
    for r in ORDER:
        x = rules[r]
        L.append(f"| {lab[r]} | {yn(x['S1_loss_near_log2'])} | {yn(x['S2_entropy_collapse'])} | "
                 f"{yn(x['S3_fewer_pairs'])} | {yn(x['shrinks'])} |")
    if drift:
        L += ["", "**Weight distance from the start** (||θ − θ0|| / ||θ0||, trunk + actor; a mix counts all three "
              "networks). Only the saved checkpoints exist, so this is not per update.", "",
              "| Run | Update | Trunk | Actor | Trunk + actor |", "|---|---|---|---|---|"]
        for d in sorted(drift, key=lambda d: (ORDER.index(d["run"]), d["update"])):
            if d["update"]:
                L.append(f"| {lab[d['run']]} | {d['update']} | {d['trunk_drift']:.3f} | {d['actor_drift']:.3f} | "
                         f"{d['policy_drift']:.3f} |")
    L += ["", "![signals](plateau_diagnosis_v3/figures/signals.png)", ""]
    verdict["signals"] = rules

    # ------------------------------------------------------------------ Part A.2 greedy vs best-of-16
    edir = res / "part_a" / "eval"
    man = json.loads((res / "part_a" / "checkpoints.json").read_text()) if (res / "part_a" / "checkpoints.json").exists() else []
    E = {(c["run"], c["update"]): ev(edir, c["name"], split) for c in man}
    best_u = {r: json.loads((REPO / cfg["runs"][r]["dir"] / "status.json").read_text())["best_update"] for r in ORDER}
    final_u = {r: max(u for (rr, u) in E if rr == r) for r in ORDER if any(rr == r for rr, _ in E)}
    L += ["### 2. Greedy vs best-of-16 along training", "",
          f"Evaluation split ({split}), 66 preferences; best-of-16 = per preference the best of 16 sampled schedules by "
          "the preference-weighted score (torch seed 50). Contrastive runs saved only start, best and final states, so "
          "these are the checkpoints that exist (From scratch also has its first segment's best and final, 2450 and "
          "2600; the World-model base mix has only 3 points).", "",
          "| Run | Update | Checkpoint | Greedy HV | Best-of-16 HV | Gap | Greedy IGD+ | Best-of-16 IGD+ |",
          "|---|---|---|---|---|---|---|---|"]
    for r in ORDER:
        for (rr, u) in sorted(k for k in E if k[0] == r):
            e = E[(rr, u)]
            if e is None:
                continue
            kind = "start" if u == 0 else ("best" if u == best_u[r] else ("final" if u == final_u.get(r) else
                                                                           "first segment"))
            L.append(f"| {lab[r]} | {u} | {kind} | {e['hv']:.3f} | {f3(e['s16'])} | {f3(None if e['s16'] is None else e['s16'] - e['hv'])} | "
                     f"{f3(e['igd'])} | {f3(e['s16_igd'])} |")
    rise = {}
    for r in ORDER:
        b, fi = E.get((r, best_u[r])), E.get((r, final_u.get(r)))
        if b and fi and b["s16"] is not None and fi["s16"] is not None:
            rise[r] = {"greedy_change": fi["hv"] - b["hv"], "s16_change": fi["s16"] - b["s16"],
                       "keeps_rising": (fi["s16"] - b["s16"] >= 0.01) and abs(fi["hv"] - b["hv"]) < 0.01,
                       "gap_final": fi["s16"] - fi["hv"]}
    L += ["", "Rule: best-of-16 keeps rising while greedy stays flat = from the best to the final checkpoint, "
          "best-of-16 rises by ≥ 0.01 while greedy changes by < 0.01.", "",
          "| Run | Greedy change best → final | Best-of-16 change best → final | Keeps rising while greedy flat | "
          "Gap at final |", "|---|---|---|---|---|"]
    for r, x in rise.items():
        L.append(f"| {lab[r]} | {x['greedy_change']:+.3f} | {x['s16_change']:+.3f} | {yn(x['keeps_rising'])} | "
                 f"{x['gap_final']:.3f} |")
    f, ax = plt.subplots(1, 2, figsize=(13, 4.6), facecolor="white")
    for i, r in enumerate(["wm", "fs"]):
        pts = sorted((u, e) for (rr, u), e in E.items() if rr == r and e)
        if pts:
            ax[i].plot([u for u, _ in pts], [e["hv"] for _, e in pts], "o-", color=COL[r], label="greedy")
            ax[i].plot([u for u, _ in pts], [e["s16"] for _, e in pts], "s--", color=COL[r], alpha=.6,
                       label="best-of-16")
        ax[i].set_title(f"{lab[r]}: evaluation HV", fontsize=11); ax[i].set_xlabel("updates"); ax[i].grid(alpha=.3)
        ax[i].legend()
    f.tight_layout(); f.savefig(fig / "greedy_vs_sampling.png", dpi=110, facecolor="white"); plt.close(f)
    L += ["", "![greedy vs best-of-16](plateau_diagnosis_v3/figures/greedy_vs_sampling.png)", ""]
    verdict["greedy"] = rise

    # ------------------------------------------------------------------ Part A.3 noise
    nb, ns = cfg["part_a"]["noise"]["bootstrap"], cfg["part_a"]["noise"]["bootstrap_seed"]
    L += ["### 3. Noise: best vs final checkpoint", "",
          "Difference = best − final. Intervals: paired bootstrap 95% over instances (evaluation 100, validation 18). "
          "Validation spread = standard deviation of the raw validation HV over the run's last 1,000 updates.", "",
          "| Run | Best update | Final update | Validation HV best / final | Validation difference [95%] | "
          "Validation spread | Evaluation HV best / final | Evaluation difference [95%] |", "|---|---|---|---|---|---|---|---|"]
    noise = {}
    for r in ORDER:
        b, fi = E.get((r, best_u[r])), E.get((r, final_u.get(r)))
        v = {e["update"]: e for e in vq[r]}
        vb, vf = v.get(best_u[r]), v.get(final_u.get(r))
        if not (b and fi and vb and vf):
            continue
        vd = np.asarray(vb["hv_per_instance"]) - np.asarray(vf["hv_per_instance"])
        ed = b["hv_i"] - fi["hv_i"]
        last = [e["hv"] for e in vq[r] if e["update"] > vq[r][-1]["update"] - 1000]
        noise[r] = {"val_diff": float(vd.mean()), "val_ci": boot(vd, nb, ns), "val_spread": float(np.std(last)),
                    "eval_diff": float(ed.mean()), "eval_ci": boot(ed, nb, ns)}
        x = noise[r]
        L.append(f"| {lab[r]} | {best_u[r]} | {final_u[r]} | {vb['hv']:.3f} / {vf['hv']:.3f} | {x['val_diff']:+.3f} "
                 f"[{x['val_ci'][0]:+.3f}, {x['val_ci'][1]:+.3f}] | {x['val_spread']:.3f} | {b['hv']:.3f} / "
                 f"{fi['hv']:.3f} | {x['eval_diff']:+.3f} [{x['eval_ci'][0]:+.3f}, {x['eval_ci'][1]:+.3f}] |")
    L.append("")
    verdict["noise"] = noise

    # ------------------------------------------------------------------ context: B2 and the union of non-delay schedules
    norm = json.loads((REPO / (cfg["smoke"]["norm"] if a.smoke else cfg["part_a"]["evaluation"]["norm"])).read_text())["instances"]
    b2 = REPO / cfg["part_a"]["b2_root"] / f"{split}_10x5"
    keys = next((e["keys"] for e in E.values() if e), [])
    ctx = {}
    if b2.exists() and keys:
        idx = [i for i, k in enumerate(keys) if (b2 / f"{Path(k).stem}.json").exists()]
        G = len(MM.simplex_grid(0.1))
        rows = {"b2": [], "union": [], "both": []}
        per = {r: {"greedy": [], "s16": []} for r in ORDER}
        for i in idx:
            ideal, nadir = MM.ideal_nadir(norm[keys[i]])
            fr = np.array([[p[o] for o in MM.OBJECTIVES] for p in json.loads((b2 / f"{Path(keys[i]).stem}.json").read_text())["front"]])
            u = np.vstack([np.vstack([e["obj"][i, :G]] + ([e["s16_obj"][i]] if e["s16_obj"] is not None else []))
                           for e in E.values() if e])
            rows["b2"].append(MM.instance_hv(fr, ideal, nadir)); rows["union"].append(MM.instance_hv(u, ideal, nadir))
            rows["both"].append(MM.instance_hv(np.vstack([fr, u]), ideal, nadir))
            for r in ORDER:
                e = E.get((r, final_u.get(r)))
                if e:
                    per[r]["greedy"].append(MM.instance_hv(e["obj"][i, :G], ideal, nadir))
                    if e["s16_obj"] is not None:
                        per[r]["s16"].append(MM.instance_hv(e["s16_obj"][i], ideal, nadir))
        if idx:
            ctx = {k: float(np.mean(v)) for k, v in rows.items()}
            ctx["n"] = len(idx)
            L += ["### Context: how far are these fronts from what is reachable?", "",
                  f"The {len(idx)} evaluation instances with a CP-SAT B2 front (evaluation normalisation, HV per instance, "
                  "mean). The B2 front may delay operations; every DANIEL schedule is non-delay (no waiting). "
                  "'All non-delay schedules' pools every schedule of every Part A checkpoint, greedy and best-of-16.", "",
                  "| Front | HV |", "|---|---|",
                  f"| CP-SAT B2 (waiting allowed) | {ctx['b2']:.3f} |",
                  f"| All non-delay schedules of Part A, pooled | {ctx['union']:.3f} |",
                  f"| B2 and pooled non-delay together | {ctx['both']:.3f} |"]
            for r in ORDER:
                if per[r]["greedy"]:
                    L.append(f"| {lab[r]}, final checkpoint, greedy / best-of-16 | {np.mean(per[r]['greedy']):.3f} / "
                             f"{f3(float(np.mean(per[r]['s16'])) if per[r]['s16'] else None)} |")
                    ctx[r] = {"greedy": float(np.mean(per[r]["greedy"])),
                              "s16": float(np.mean(per[r]["s16"])) if per[r]["s16"] else None}
            L.append("")
    verdict["context"] = ctx

    # ------------------------------------------------------------------ Part B
    pb = cfg["part_b"]; pdir = res / "part_b"
    nm = {k: f"10x5+mix+bv3_pd{'s' if a.smoke else ''}_{k}_s111" for k in pb["arms"]}
    runs = {k: pdir / "runs" / n for k, n in nm.items() if (pdir / "runs" / n / "done.json").exists()}
    L += ["## Part B: short training runs from the World-model base mix start", ""]
    pbv = {}
    if not runs:
        L += ["Not run yet (set up and smoke-tested only). Command: `bash scripts/run_plateau_diagnosis_v3.sh --jobs 3`.",
              ""]
    else:
        last_bc = (cfg["smoke"] if a.smoke else pb)["budget_checkpoints"][-1]
        L += [f"One change per arm; otherwise the M3 World-model base mix settings, {last_bc} updates fixed, validation "
              "every 50 updates. Evaluation split, 66 preferences; best-of-16 as in Part A.", "",
              "| Arm | Best validation HV (update) | Final validation HV | Final: greedy / best-of-16 HV | "
              "Best checkpoint: greedy / best-of-16 HV | Loss, last 250 | Entropy, last 250 | Distance from start at end | "
              "Wall-clock (h) |", "|---|---|---|---|---|---|---|---|---|"]
        ped = pdir / "eval"
        for k in pb["order"]:
            if k not in runs:
                continue
            d = runs[k]
            v = json.loads((d / "valiquality.json").read_text())
            st = json.loads((d / "status.json").read_text())
            lg = json.loads((d / "reward.json").read_text())
            proc = json.loads((d / "process.json").read_text())["process_seconds"] if (d / "process.json").exists() else None
            fe, be = ev(ped, f"{nm[k]}@u{last_bc}", split), ev(ped, f"{nm[k]}_best", split)
            bi = int(np.argmax([e["hv"] for e in v]))
            loss = np.nanmean([np.nan if e["cl"]["loss"] is None else e["cl"]["loss"] for e in lg[-250:]])
            ent = np.mean([e["cl"]["entropy"] for e in lg[-250:]])
            dr = lg[-1]["cl"].get("trunk_drift"), lg[-1]["cl"].get("actor_drift")
            pbv[k] = {"best_val": v[bi]["hv"], "best_update": v[bi]["update"], "final_val": v[-1]["hv"],
                      "final_greedy": fe["hv"] if fe else None, "final_s16": fe["s16"] if fe else None,
                      "best_greedy": be["hv"] if be else None, "best_s16": be["s16"] if be else None,
                      "loss_late": float(loss), "entropy_late": float(ent), "train_h": st["train_seconds"] / 3600,
                      "process_h": proc / 3600 if proc else None, "drift_end": dr}
            x = pbv[k]
            L.append(f"| {pb['arms'][k]['label']} | {x['best_val']:.3f} ({x['best_update']}) | {x['final_val']:.3f} | "
                     f"{f3(x['final_greedy'])} / {f3(x['final_s16'])} | {f3(x['best_greedy'])} / {f3(x['best_s16'])} | "
                     f"{x['loss_late']:.3f} | {x['entropy_late']:.3f} | trunk {f3(dr[0])}, actor {f3(dr[1])} | "
                     f"{x['train_h']:.2f} (process {f3(x['process_h'], 2)}) |")
        # control vs the M3 run's first updates
        if "ctrl" in runs:
            m3v = {e["update"]: e["hv"] for e in vq["wm"]}
            cv = json.loads((runs["ctrl"] / "valiquality.json").read_text())
            same = [e["update"] for e in cv if e["update"] in m3v]
            exact = all(m3v[e["update"]] == e["hv"] for e in cv if e["update"] in m3v)
            pbv["control_reproduces_m3"] = {"compared_updates": len(same), "exact": exact,
                                            "max_abs_diff": max((abs(m3v[e["update"]] - e["hv"]) for e in cv
                                                                 if e["update"] in m3v), default=None)}
            L += ["", f"Control vs the M3 World-model base mix run, validation HV at the {len(same)} shared validation "
                  f"points: {'identical' if exact else 'not identical'} (largest difference "
                  f"{f3(pbv['control_reproduces_m3']['max_abs_diff'], 6)})."]
        f, axs = plt.subplots(1, 3, figsize=(18, 4.8), facecolor="white")
        for k, d in runs.items():
            v = json.loads((d / "valiquality.json").read_text()); lg = json.loads((d / "reward.json").read_text())
            c = COL.get(k, "#555") if k != "ctrl" else "#2a2a2a"
            axs[0].plot([e["update"] for e in v], rolling([e["hv"] for e in v], 5), color=c, lw=1.6,
                        label=pb["arms"][k]["label"])
            axs[0].plot([e["update"] for e in v], [e["hv"] for e in v], color=c, lw=.6, alpha=.3)
            axs[1].plot(np.arange(1, len(lg) + 1), rolling([np.nan if e["cl"]["loss"] is None else e["cl"]["loss"]
                                                            for e in lg], W), color=c, lw=1.4)
            axs[2].plot(np.arange(1, len(lg) + 1), rolling([e["cl"]["entropy"] for e in lg], W), color=c, lw=1.4)
        axs[1].axhline(cfg["part_a"]["log2"], color="#555", ls="--", lw=.8)
        for ax, t in zip(axs, ("validation HV (smoothed over 5 points; faint: raw)", f"contrastive loss (smoothed over {W})",
                               f"policy entropy (smoothed over {W})")):
            ax.set_title(t, fontsize=11); ax.set_xlabel("updates"); ax.grid(alpha=.3)
        axs[0].legend(fontsize=9)
        f.tight_layout(); f.savefig(fig / "part_b.png", dpi=110, facecolor="white"); plt.close(f)
        L += ["", "![Part B](plateau_diagnosis_v3/figures/part_b.png)", ""]
    verdict["part_b"] = pbv

    # ------------------------------------------------------------------ verdict
    arms = [k for k in ("beta05", "lr1e3", "k16") if pbv.get(k, {}).get("final_greedy") is not None]
    ctrl_g = pbv.get("ctrl", {}).get("final_greedy")
    if ctrl_g is not None and len(arms) == 3:
        beats = [k for k in arms if pbv[k]["final_greedy"] >= ctrl_g + 0.02]
        weak = "yes" if beats else "no"
        weak_txt = (f"{'yes' if beats else 'no'}: " + (", ".join(pb['arms'][k]['label'] for k in beats) +
                    " beat the Control by ≥ 0.02 greedy evaluation HV" if beats else
                    "no stronger push, faster learning or more samples beats the Control by ≥ 0.02 greedy evaluation HV")
                    + f" (Control {ctrl_g:.3f}; " + ", ".join(f"{pb['arms'][k]['label']} {pbv[k]['final_greedy']:.3f}"
                                                            for k in arms) + ").")
    else:
        weak = "pending"
        shr = [lab[r] for r in ORDER if rules[r]["shrinks"]]
        weak_txt = ("pending Part B. From the logs: " + (f"the signal shrinks (S2/S3) for {', '.join(shr)}; "
                                                          if shr else "no run's signal shrinks by the rules (S2/S3); ")
                    + "late loss " + ", ".join(f"{lab[r]} {rules[r]['loss_late']:.3f}" for r in ORDER) + ".")
    kr = [lab[r] for r in ("wm", "fs") if rise.get(r, {}).get("keeps_rising")]
    gaps = [rise[r]["gap_final"] for r in rise]
    if kr:
        greedy = "yes"; gtxt = f"yes: best-of-16 keeps rising while greedy stays flat for {', '.join(kr)}."
    elif gaps and min(gaps) >= 0.05:
        greedy = "partly"
        gtxt = (f"partly: best-of-16 is {min(gaps):.3f}–{max(gaps):.3f} HV above greedy at every final checkpoint, so "
                "greedy decoding leaves HV unused, but best-of-16 also stops rising, so it does not explain the flat curve.")
    else:
        greedy = "no"; gtxt = "no: best-of-16 does not keep rising and its gap to greedy is small."
    fin = [(E[(r, final_u[r])]["hv"], E[(r, final_u[r])]["s16"]) for r in ORDER
           if final_u.get(r) is not None and E.get((r, final_u[r]))]
    if len(fin) == len(ORDER) and all(s is not None for _, s in fin):
        rg = max(g for g, _ in fin) - min(g for g, _ in fin); rs = max(s for _, s in fin) - min(s for _, s in fin)
        shared = "likely" if rg <= 0.05 and rs <= 0.05 else "no"
        stxt = (f"{shared}: the five final checkpoints span {rg:.3f} greedy and {rs:.3f} best-of-16 evaluation HV "
                f"({'both within' if shared == 'likely' else 'not both within'} 0.05).")
        if ctx:
            g_best = max(ctx[r]["greedy"] for r in ORDER if r in ctx)
            stxt += (f" On the {ctx['n']} instances with a CP-SAT front: CP-SAT B2 (waiting allowed) {ctx['b2']:.3f}; "
                     f"all non-delay schedules found here, pooled, {ctx['union']:.3f}; the best single method's final "
                     f"greedy front {g_best:.3f}. The pooled non-delay front is only a lower bound on what the no-waiting "
                     "rule allows.")
        stxt += " Model size was not varied, so it is not tested."
    else:
        shared, stxt = "pending", "pending (evaluations missing)."
    verdict["verdict"] = {"weak_signal": weak, "greedy_decoding": greedy, "shared": shared}
    L[4:4] = ["## Verdict", "", f"- **Weak training signal:** {weak_txt}", f"- **Greedy decoding:** {gtxt}",
              f"- **Something shared by all methods:** {stxt}", "",
              "Rules fixed in the protocol before the run; single seed, numbers only. Details below.", ""]
    md.write_text("\n".join(L) + "\n")
    (res / "verdict.json").write_text(json.dumps(verdict, indent=1, default=float) + "\n")
    print(f"wrote {md.relative_to(REPO)}")


if __name__ == "__main__":
    main()
