"""Analysis of the zero-training three-specialist blend (pre-registered in results/blend_zero_training.md).

Reads results/blend_zero_training/raw/seed<s>.npz and the Step 4 evaluation files (for the budget-curve figure);
writes results/blend_zero_training/analysis_tables.md, summary.json and step4_with_blend.png. No GPU.
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
from preference3 import Preference, scalarise          # noqa: E402

OUT = REPO / "results/blend_zero_training"
S4 = REPO / "daniel/test_results/SD2/trainvali-k125s060_10x5+carbon+priority"
SEEDS, TAG = [111, 222, 333, 444], "k125s060"
PREF = Preference(1 / 3, 1 / 3, 1 / 3, third="t")
NBOOT, BSEED = 10_000, 20260927
BUDGETS = [0, 50, 100, 250, 500, 1000]
ARMS_R = ["merge", "scratch", "spec_m", "spec_c", "spec_t"]
LBL = {"merge": "merge", "scratch": "scratch", "spec_m": "makespan spec", "spec_c": "carbon spec", "spec_t": "tardiness spec"}

split = json.loads((OUT / "split.json").read_text())
FIT, HELD = np.array(split["fit_idx"]), np.array(split["held_idx"])


def s4(name):
    return np.load(S4 / f"Result_DANIELG+10x5+carbon+priority+{name}_trainvali-{TAG}_10x5+carbon+priority.npy")[:, [0, 1, 3]]


REF = {s: s4(f"simplex_m333c333t334_{TAG}_s{s}") for s in SEEDS}


def sc(s, out3):                    # out3: [3, n] (makespan, carbon, tardiness) -> per-instance Step 4 score
    return scalarise(out3.T, REF[s], PREF, True)


D = {s: dict(np.load(OUT / "raw" / f"seed{s}.npz")) for s in SEEDS}
ARM = {}                            # ARM[name][s] -> [3, n] raw objectives
chosen3 = {}
for s in SEEDS:
    d = D[s]
    fit_score = np.array([sc(s, d["G"][:, g, :])[FIT].mean() for g in range(len(d["grid"]))])
    g3 = int(fit_score.argmin()); chosen3[s] = d["grid"][g3].tolist()
    put = lambda k, v: ARM.setdefault(k, {}).__setitem__(s, v)
    put("B1 logit, alpha = w", d["b1"]); put("B2 prob, alpha = w", d["b2"]); put("B3 logit, tuned alpha", d["G"][:, g3, :])
    for lab, v in (("B4 makespan spec alone", (1000, 0, 0)), ("B4 carbon spec alone", (0, 1000, 0)),
                   ("B4 tardiness spec alone", (0, 0, 1000))):
        put(lab, d["G"][:, int(np.flatnonzero((d["grid"] == v).all(1))[0]), :])
    for a in ARMS_R:
        for b in (0, 1000):
            key = f"R_{a}@{b}"
            put(f"R {LBL[a]} @{b}", d[key] if key in d else None)
SCORE = {k: {s: sc(s, v) for s, v in d_.items() if v is not None} for k, d_ in ARM.items()}

L, summary = [], {"b3_alpha_permille_mct": {str(s): v for s, v in chosen3.items()}}
P_ = L.append
P_("### Step 4 score on HELD-OUT (lower is better), per seed and pooled\n")
P_("| arm | " + " | ".join(f"seed {s}" for s in SEEDS) + " | pooled |")
P_("|---|" + "---|" * (len(SEEDS) + 1))
for k, v in SCORE.items():
    P_(f"| {k} | " + " | ".join(f"{v[s][HELD].mean():.4f}" for s in SEEDS) + f" | **{np.mean([v[s][HELD].mean() for s in SEEDS]):.4f}** |")
P_("\nB3's alpha chosen on FIT (permille m/c/t): " + ", ".join(f"seed {s}: {'/'.join(str(x) for x in chosen3[s])}" for s in SEEDS))

P_("\n### Raw objectives on HELD-OUT (mean over held-out instances, then over seeds)\n")
P_("| arm | makespan | carbon | tardiness |")
P_("|---|---|---|---|")
for k, v in ARM.items():
    M = np.mean([v[s][:, HELD].mean(1) for s in SEEDS], 0)
    P_(f"| {k} | {M[0]:.1f} | {M[1]:.1f} | {M[2]:.1f} |")

rng = np.random.default_rng(BSEED)
BOOT = rng.integers(0, len(HELD), size=(NBOOT, len(HELD)))


def gain(x_key, y_key, seeds):
    X = np.stack([SCORE[x_key][s][HELD] for s in seeds]); Y = np.stack([SCORE[y_key][s][HELD] for s in seeds])
    g = (Y.mean() - X.mean()) / Y.mean()
    xb = X[:, BOOT].mean(axis=(0, 2)); yb = Y[:, BOOT].mean(axis=(0, 2))
    gb = (yb - xb) / yb
    return float(g), float(np.percentile(gb, 2.5)), float(np.percentile(gb, 97.5)), float((Y - X).mean())


COMP = [("B1 logit, alpha = w", "R makespan spec @1000"), ("B1 logit, alpha = w", "R merge @0"),
        ("B1 logit, alpha = w", "R scratch @1000"), ("B3 logit, tuned alpha", "R makespan spec @1000"),
        ("B3 logit, tuned alpha", "R merge @0"), ("B3 logit, tuned alpha", "R scratch @1000"),
        ("B1 logit, alpha = w", "B2 prob, alpha = w")]
P_("\n### Paired comparisons on HELD-OUT: relative gain of X over Y = (mean Y - mean X)/mean Y (positive = X better), "
   "95% CI from 10,000 instance resamples\n")
P_("| X vs Y | " + " | ".join(f"seed {s}" for s in SEEDS) + " | pooled | seeds beating Y (gain >= 1%, CI > 0) |")
P_("|---|" + "---|" * (len(SEEDS) + 2))
res = {}
for x, y in COMP:
    cells, n_beat = [], 0
    for s in SEEDS:
        g, lo, hi, dm = gain(x, y, [s]); b = g >= 0.01 and lo > 0; n_beat += b
        res[(x, y, s)] = (g, lo, hi, dm, b)
        cells.append(f"{100 * g:+.2f}% [{100 * lo:+.2f}, {100 * hi:+.2f}]{' *' if b else ''}")
    g, lo, hi, dm = gain(x, y, SEEDS)
    res[(x, y, "pooled")] = (g, lo, hi, dm, None)
    P_(f"| {x.split(',')[0]} vs {y} | " + " | ".join(cells) + f" | {100 * g:+.2f}% [{100 * lo:+.2f}, {100 * hi:+.2f}] | {n_beat}/4 |")
P_("\n(* = gain >= 1% with the CI above 0 at that seed. Mean per-instance score differences Y - X are in summary.json.)")
summary["comparisons"] = {f"{x} vs {y} | {s}": dict(gain=v[0], lo=v[1], hi=v[2], mean_diff=v[3], beats=v[4])
                          for (x, y, s), v in res.items()}

# ---------------------------------------------------------------- compromise analysis (B1, all 100 instances)
top = np.concatenate([D[s]["diag_top"] for s in SEEDS]); ent = np.concatenate([D[s]["diag_ent"] for s in SEEDS])
cho = np.concatenate([D[s]["diag_chosen"] for s in SEEDS])                      # [400, 50(, 3)]
match = top == cho[..., None]; nm = match.sum(-1)
unan = (top[..., 0] == top[..., 1]) & (top[..., 1] == top[..., 2])
cat = {"unanimous (all three top-1s equal, blend takes it)": unan & (nm == 3),
       "majority (blend takes a top-1 shared by exactly two)": nm == 2,
       "follows M only": (nm == 1) & match[..., 0], "follows C only": (nm == 1) & match[..., 1],
       "follows T only": (nm == 1) & match[..., 2], "compromise (none of the top-1s)": nm == 0}
P_("\n### Confidence / compromise analysis (B1 rollouts, all 100 instances x 4 seeds = 20,000 decisions)\n")
P_("| category | overall | fifth 1 | 2 | 3 | 4 | 5 |")
P_("|---|---|---|---|---|---|---|")
for k, m in cat.items():
    P_(f"| {k} | {100 * m.mean():.1f}% | " + " | ".join(f"{100 * m[:, i * 10:(i + 1) * 10].mean():.1f}%" for i in range(5)) + " |")
one = nm == 1
followed = match.argmax(-1)
lowest = ent.min(-1)
obs = float((np.take_along_axis(ent, followed[..., None], -1)[..., 0] <= lowest + 1e-12)[one].mean())
r2 = np.random.default_rng(BSEED); null = []
E1, F1 = ent[one], followed[one]
for _ in range(200):
    perm = np.argsort(r2.random(E1.shape), axis=1)
    Ep = np.take_along_axis(E1, perm, 1)
    null.append(float((Ep[np.arange(len(F1)), F1] <= Ep.min(1) + 1e-12).mean()))
nonun = ~unan
comp_share = float((nm == 0)[nonun].mean())
P_(f"\n* Decisions where the blend follows exactly one expert: {100 * one.mean():.1f}%. Among them, the followed expert is "
   f"the MOST CONFIDENT (lowest entropy) in **{100 * obs:.1f}%**; permutation null (entropies shuffled across experts within "
   f"each decision, 200x): mean {100 * np.mean(null):.1f}%, 95th percentile {100 * np.percentile(null, 95):.1f}% -> "
   f"{'the blend follows the most confident expert MORE than chance' if obs > np.percentile(null, 95) else 'no more than chance'}.")
P_(f"* Among decisions where the experts' top-1s are not unanimous ({100 * nonun.mean():.1f}% of all decisions), "
   f"**{100 * comp_share:.1f}%** are resolved by compromise.")
P_(f"* Mean entropy (bits): M {ent[..., 0].mean():.2f}, C {ent[..., 1].mean():.2f}, T {ent[..., 2].mean():.2f}.")
summary.update(follow_most_confident=obs, follow_null_mean=float(np.mean(null)), follow_null_95=float(np.percentile(null, 95)),
               compromise_share_nonunanimous=comp_share, category_shares={k: float(m.mean()) for k, m in cat.items()})

# ---------------------------------------------------------------- verdict
def meets(arm, y):
    return sum(res[(arm, y, s)][4] for s in SEEDS)


strong = {a: meets(a, "R makespan spec @1000") for a in ("B1 logit, alpha = w", "B3 logit, tuned alpha")}
meaningful = {a: sum(res[(a, "R merge @0", s)][4] and res[(a, "R scratch @1000", s)][4] for s in SEEDS)
              for a in ("B1 logit, alpha = w", "B3 logit, tuned alpha")}
if max(strong.values()) >= 3:
    verdict = "STRONG"
elif max(meaningful.values()) >= 3:
    verdict = "MEANINGFUL"
else:
    verdict = "NOT MEANINGFUL"
summary.update(verdict=verdict, strong_seed_counts=strong, meaningful_seed_counts=meaningful)
P_(f"\n### Verdict inputs\n\n* Seeds where the arm beats R[makespan spec @1000] (gain >= 1%, CI > 0): B1 {strong['B1 logit, alpha = w']}/4, "
   f"B3 {strong['B3 logit, tuned alpha']}/4 (STRONG needs 3)")
P_(f"* Seeds where the arm beats BOTH R[merge @0] and R[scratch @1000]: B1 {meaningful['B1 logit, alpha = w']}/4, "
   f"B3 {meaningful['B3 logit, tuned alpha']}/4 (MEANINGFUL needs 3)")
P_(f"* **Outcome: {verdict}**")

# ---------------------------------------------------------------- figure
fig, ax = plt.subplots(figsize=(7.2, 4.6))
cols = {"merge": "tab:blue", "scratch": "tab:gray", "spec_m": "tab:orange", "spec_c": "tab:green", "spec_t": "tab:red"}
for a in ARMS_R:
    Y = np.array([[scalarise(s4(f"s4t_{a}_{TAG}_s{s}@u{b}"), REF[s], PREF, True)[HELD].mean() for b in BUDGETS] for s in SEEDS])
    ax.plot(BUDGETS, Y.mean(0), "-o", ms=3, color=cols[a], label=f"Step 4: {LBL[a]}")
    ax.fill_between(BUDGETS, Y.min(0), Y.max(0), color=cols[a], alpha=0.12)
for k, ls, c in (("B1 logit, alpha = w", "--", "k"), ("B3 logit, tuned alpha", ":", "purple")):
    v = np.mean([SCORE[k][s][HELD].mean() for s in SEEDS])
    ax.axhline(v, ls=ls, color=c, lw=1.6, label=f"{k.split(',')[0]} blend, zero training ({v:.3f})")
ax.set_xscale("symlog", linthresh=50); ax.set_xticks(BUDGETS); ax.set_xticklabels([str(b) for b in BUDGETS])
ax.set_xlabel("PPO updates (symlog, linthresh 50)"); ax.set_ylabel("Step 4 score on HELD-OUT (lower is better)")
ax.set_title("10x5, HELD-OUT: Step 4 fine-tuning curves vs zero-training blends\n(line = mean over 4 seeds, band = seed range)", fontsize=10)
ax.grid(alpha=0.3); ax.legend(fontsize=7, loc="upper right")
fig.tight_layout(); fig.savefig(OUT / "step4_with_blend.png", dpi=130); plt.close(fig)

(OUT / "analysis_tables.md").write_text("\n".join(L) + "\n")
json.dump(summary, open(OUT / "summary.json", "w"), indent=1, default=float)
print("\n".join(L))
