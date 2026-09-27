"""Analysis of the blend fine-tuning study (pre-registered in results/blend_finetune.md, commit d05b2b6).

Reads results/blend_finetune/raw/<arm>_s<seed>.{npz,json} and the Step 4 evaluation files (C1 = Step 4 makespan
arm; Step 4 merge and scratch for the figure and the merge-stall test). Writes analysis_tables.md, summary.json and
score_vs_updates.png under results/blend_finetune/. No GPU.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))
from preference3 import Preference                   # noqa: E402

OUT = REPO / "results/blend_finetune"
S4 = REPO / "daniel/test_results/SD2/trainvali-k125s060_10x5+carbon+priority"
SEEDS, TAG = [111, 222, 333, 444], "k125s060"
BUDGETS = [0, 50, 100, 250, 500, 1000]
PREF = Preference(1 / 3, 1 / 3, 1 / 3, third="t")
NBOOT, BSEED = 10_000, 20260927
HELD = np.array(json.loads((REPO / "results/blend_zero_training/split.json").read_text())["held_idx"])


def s4(name):
    return np.load(S4 / f"Result_DANIELG+10x5+carbon+priority+{name}_trainvali-{TAG}_10x5+carbon+priority.npy")[:, [0, 1, 3]]


REF = {s: s4(f"simplex_m333c333t334_{TAG}_s{s}") for s in SEEDS}          # Step 4 reference, all 100 instances


def score_held(s, out3):
    """Step 4 score of HELD-OUT results out3 [3, 50]: per-instance makespan/carbon ratios, tardiness over the
    centroid's pool mean across ALL 100 instances (Step 4's definition), weights as preference3.scalarise."""
    r = REF[s]; tbar = float(np.mean(r[:, 2]))
    return PREF.w_m * (out3[0] / r[HELD, 0]) + PREF.w_c * (out3[1] / r[HELD, 1]) + PREF.w_p * (out3[2] / tbar)


# ------------------------------------------------------------------------------------------ load
RAW, SC, META = {}, {}, {}
for s in SEEDS:
    for arm, stem in (("C1 makespan spec (Step 4)", "spec_m"), ("Step 4 merge", "merge"), ("Step 4 scratch", "scratch")):
        for b in BUDGETS:
            RAW.setdefault(arm, {}).setdefault(s, {})[b] = s4(f"s4t_{stem}_{TAG}_s{s}@u{b}")[HELD].T
    for arm, lab in (("F1", "F1 blend, joint fine-tune"), ("F2", "F2 blend + residual"),
                     ("C2", "C2 makespan spec + fresh critic"), ("C3", "C3 merge + fresh critic")):
        f = OUT / "raw" / f"{arm}_s{s}.npz"
        if f.exists():
            d = np.load(f)
            RAW.setdefault(lab, {})[s] = {b: d[f"u{b}"] for b in BUDGETS}
            META.setdefault(lab, {})[s] = json.loads((OUT / "raw" / f"{arm}_s{s}.json").read_text())
for arm, d in RAW.items():
    SC[arm] = {s: {b: score_held(s, v) for b, v in dd.items()} for s, dd in d.items()}
ARMS = [a for a in ("C1 makespan spec (Step 4)", "C2 makespan spec + fresh critic", "C3 merge + fresh critic",
                    "F1 blend, joint fine-tune", "F2 blend + residual", "Step 4 merge", "Step 4 scratch") if a in SC]
complete = {a: sorted(SC[a]) for a in ARMS}

L, summary = [], {"seeds_available": complete}
P_ = L.append


def pooled(arm, b):
    return float(np.mean([SC[arm][s][b].mean() for s in SC[arm]]))


P_("### Step 4 score on HELD-OUT at every snapshot (lower is better): pooled mean over seeds (sd across seeds)\n")
P_("| arm | " + " | ".join(f"u{b}" for b in BUDGETS) + " |")
P_("|---|" + "---|" * len(BUDGETS))
for a in ARMS:
    P_(f"| {a} | " + " | ".join(f"{pooled(a, b):.4f} ({np.std([SC[a][s][b].mean() for s in SC[a]], ddof=1):.3f})"
                              for b in BUDGETS) + " |")
P_("\nPer seed:\n")
P_("| arm | seed | " + " | ".join(f"u{b}" for b in BUDGETS) + " |")
P_("|---|---|" + "---|" * len(BUDGETS))
for a in ARMS:
    for s in SC[a]:
        P_(f"| {a} | {s} | " + " | ".join(f"{SC[a][s][b].mean():.4f}" for b in BUDGETS) + " |")

P_("\n### Raw objectives at update 1000 on HELD-OUT (mean over instances, then seeds)\n")
P_("| arm | makespan | carbon | tardiness | score |")
P_("|---|---|---|---|---|")
for a in ARMS:
    M = np.mean([RAW[a][s][1000].mean(1) for s in RAW[a]], 0)
    P_(f"| {a} | {M[0]:.1f} | {M[1]:.1f} | {M[2]:.1f} | {pooled(a, 1000):.4f} |")

# ------------------------------------------------------------------------------------------ comparisons
rng = np.random.default_rng(BSEED)
BOOT = rng.integers(0, len(HELD), size=(NBOOT, len(HELD)))


def gain(x, bx, y, by, seeds):
    X = np.stack([SC[x][s][bx] for s in seeds]); Y = np.stack([SC[y][s][by] for s in seeds])
    g = (Y.mean() - X.mean()) / Y.mean()
    xb, yb = X[:, BOOT].mean(axis=(0, 2)), Y[:, BOOT].mean(axis=(0, 2))
    gb = (yb - xb) / yb
    return float(g), float(np.percentile(gb, 2.5)), float(np.percentile(gb, 97.5))


cands = [a for a in ("C1 makespan spec (Step 4)", "C2 makespan spec + fresh critic") if a in SC and len(SC[a]) == 4]
CTRL = min(cands, key=lambda a: pooled(a, 1000))
summary["best_single_control"] = CTRL
P_(f"\n### Paired comparisons at update 1000 on HELD-OUT\n\nBest single-specialist control (lower pooled score at "
   f"update 1000 among C1 and C2): **{CTRL}** ({pooled(CTRL, 1000):.4f}; "
   + ", ".join(f"{a}: {pooled(a, 1000):.4f}" for a in cands) + ").\n")
P_("Relative gain of X over Y = (mean Y - mean X)/mean Y, positive = X better; 95% CI from 10,000 held-out instance "
   "resamples. * = gain >= 1% with CI above 0; ! = gain <= -1% with CI below 0 (Y beats X).\n")
P_("| X vs Y | " + " | ".join(f"seed {s}" for s in SEEDS) + " | pooled | X beats Y | Y beats X |")
P_("|---|" + "---|" * (len(SEEDS) + 3))
comp = {}
pairs = [(f, CTRL) for f in ("F1 blend, joint fine-tune", "F2 blend + residual") if f in SC]
if "C3 merge + fresh critic" in SC:
    pairs.append(("C3 merge + fresh critic", "Step 4 merge"))
for x, y in pairs:
    seeds = [s for s in SEEDS if s in SC[x] and s in SC[y]]
    cells, nx, ny = [], 0, 0
    for s in SEEDS:
        if s not in seeds:
            cells.append("n/a"); continue
        g, lo, hi = gain(x, 1000, y, 1000, [s])
        bx, byy = g >= 0.01 and lo > 0, g <= -0.01 and hi < 0
        nx += bx; ny += byy; comp[(x, y, s)] = (g, lo, hi, bx, byy)
        cells.append(f"{100 * g:+.2f}% [{100 * lo:+.2f}, {100 * hi:+.2f}]{' *' if bx else ''}{' !' if byy else ''}")
    g, lo, hi = gain(x, 1000, y, 1000, seeds)
    comp[(x, y, "pooled")] = (g, lo, hi, None, None)
    P_(f"| {x} vs {y} | " + " | ".join(cells) + f" | {100 * g:+.2f}% [{100 * lo:+.2f}, {100 * hi:+.2f}] | {nx}/{len(seeds)} | {ny}/{len(seeds)} |")
summary["comparisons"] = {f"{x} vs {y} | {s}": v for (x, y, s), v in comp.items()}

# ------------------------------------------------------------------------------------------ updates-to-threshold
P_(f"\n### Updates-to-threshold: first snapshot at which the arm's score is <= {CTRL} at update 1000\n")
P_("| arm | pooled | " + " | ".join(f"seed {s}" for s in SEEDS) + " | seeds reaching it by update 500 |")
P_("|---|---|" + "---|" * len(SEEDS) + "---|")
thr = {}
for a in ARMS:
    pt = next((b for b in BUDGETS if pooled(a, b) <= pooled(CTRL, 1000)), None)
    per = {s: next((b for b in BUDGETS if SC[a][s][b].mean() <= SC[CTRL][s][1000].mean()), None) for s in SC[a]}
    by500 = sum(1 for v in per.values() if v is not None and v <= 500)
    thr[a] = dict(pooled=pt, per_seed=per, by500=by500)
    P_(f"| {a} | {pt if pt is not None else 'never'} | " + " | ".join(
        f"{per[s] if s in per and per[s] is not None else ('never' if s in per else 'n/a')}" for s in SEEDS) + f" | {by500} |")
summary["threshold"] = thr

# ------------------------------------------------------------------------------------------ cost
P_("\n### Cost (training wall-clock on one RTX 5060 Ti; GPU-bound, so wall-clock ~ GPU time)\n")
P_("| arm | trainable parameters | training min per seed (mean, range) | GPU-hours, all seeds | x single-network cost |")
P_("|---|---|---|---|---|")
base = None
if "C2 makespan spec + fresh critic" in META:
    base = np.mean([m["train_seconds"] for m in META["C2 makespan spec + fresh critic"].values()])
for a, M in META.items():
    t = [m["train_seconds"] / 60 for m in M.values()]
    ratio = f"{np.mean(t) * 60 / base:.1f}x" if base else "n/a"
    P_(f"| {a} | {list(M.values())[0]['trainable_parameters']:,} | {np.mean(t):.1f} ({min(t):.1f}-{max(t):.1f}) | "
       f"{sum(t) / 60:.2f} | {ratio} |")
P_("| C1 makespan spec (Step 4) | 29,154 | reused (Step 4 took 11.8-15 min per run) | 0 (reused) | 1.0x |")

# ------------------------------------------------------------------------------------------ verdicts
win_parts, lose_seeds = {}, 0
for f in ("F1 blend, joint fine-tune", "F2 blend + residual"):
    if f in SC:
        beats = sum(1 for s in SEEDS if (f, CTRL, s) in comp and comp[(f, CTRL, s)][3])
        win_parts[f] = dict(beats_at_1000=beats, reaches_by_500=thr[f]["by500"])
for s in SEEDS:
    if all((f, CTRL, s) in comp and comp[(f, CTRL, s)][4] for f in ("F1 blend, joint fine-tune", "F2 blend + residual")):
        lose_seeds += 1
if any(v["beats_at_1000"] >= 3 or v["reaches_by_500"] >= 3 for v in win_parts.values()):
    verdict = "WIN"
elif lose_seeds >= 3:
    verdict = "LOSE"
else:
    verdict = "TIE"
c3 = sum(1 for s in SEEDS if ("C3 merge + fresh critic", "Step 4 merge", s) in comp
         and comp[("C3 merge + fresh critic", "Step 4 merge", s)][3])
secondary = "critic caused the stall" if c3 >= 3 else "not supported"
summary.update(verdict=verdict, win_parts=win_parts, lose_seeds=lose_seeds, c3_seeds=c3, secondary=secondary)
P_(f"\n### Verdict inputs\n")
for f, v in win_parts.items():
    P_(f"* {f}: beats {CTRL} at update 1000 (gain >= 1%, CI > 0) at {v['beats_at_1000']}/4 seeds; reaches its "
       f"update-1000 score by update 500 at {v['reaches_by_500']}/4 seeds (WIN needs 3 on either)")
P_(f"* {CTRL} beats BOTH F1 and F2 at the same seed at {lose_seeds}/4 seeds (LOSE needs 3)")
P_(f"* **Primary outcome: {verdict}**")
P_(f"* C3 beats Step 4 merge at update 1000 at {c3}/4 seeds (needs 3) -> **merge stall: {secondary}**")

# ------------------------------------------------------------------------------------------ figure
style = {"C1 makespan spec (Step 4)": ("tab:orange", "-o"), "C2 makespan spec + fresh critic": ("tab:orange", "--s"),
         "C3 merge + fresh critic": ("tab:blue", "--s"), "F1 blend, joint fine-tune": ("purple", "-D"),
         "F2 blend + residual": ("tab:pink", "-D"), "Step 4 merge": ("tab:blue", "-o"), "Step 4 scratch": ("tab:gray", "-o")}
fig, axs = plt.subplots(1, 2, figsize=(13, 4.8))
for k, ax in enumerate(axs):
    for a in ARMS:
        if k == 1 and a == "Step 4 scratch":
            continue
        Y = np.array([[SC[a][s][b].mean() for b in BUDGETS] for s in SC[a]])
        col, mk = style[a]
        ax.plot(BUDGETS, Y.mean(0), mk, ms=3, color=col, lw=1.4, label=a)
        ax.fill_between(BUDGETS, Y.min(0), Y.max(0), color=col, alpha=0.10 if k == 0 else 0.07)
    if k == 1:
        ax.set_ylim(0.83, 0.98)
    ax.axvspan(0, 50, color="k", alpha=0.04, lw=0)
    ax.text(25, ax.get_ylim()[1], "critic\nwarm-up", ha="center", va="top", fontsize=7, color="0.4")
    ax.set_xscale("symlog", linthresh=50); ax.set_xticks(BUDGETS); ax.set_xticklabels([str(b) for b in BUDGETS])
    ax.set_xlabel("PPO updates (symlog, linthresh 50)"); ax.grid(alpha=0.3)
    ax.set_title(["all arms", "zoom: without scratch"][k], fontsize=9)
axs[0].set_ylabel("Step 4 score on HELD-OUT (lower is better)")
axs[0].legend(fontsize=7, loc="upper right")
fig.suptitle("10x5, HELD-OUT: fine-tuning the blend vs one specialist (line = mean over seeds, band = seed range)", fontsize=10)
fig.tight_layout(); fig.savefig(OUT / "score_vs_updates.png", dpi=130); plt.close(fig)

(OUT / "analysis_tables.md").write_text("\n".join(L) + "\n")
json.dump(summary, open(OUT / "summary.json", "w"), indent=1, default=str)
print("\n".join(L))
