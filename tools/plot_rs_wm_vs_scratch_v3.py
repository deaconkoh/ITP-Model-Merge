"""Validation-HV convergence: world-model shared-base Rewarded Soups vs random shared-base Rewarded Soups vs From-scratch
DANIEL (10x5, seed 111, waiting allowed). Plot only, from the saved valiquality.json and state.json files; nothing is
re-run. Left: validation HV at every validation point. Right: best validation HV so far (what checkpoint selection
keeps). Ringed markers: the selected checkpoints; vertical ticks: the stopping updates.

    python tools/plot_rs_wm_vs_scratch_v3.py   # -> results/convergence_rs_wm_vs_scratch_v3/
"""
from __future__ import annotations
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                     # noqa: E402

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "results" / "convergence_rs_wm_vs_scratch_v3"
RUNS = [  # label, valiquality, state file, unit key, colour (categorical slots 1-3), marker
    ("Rewarded Soups, random shared base", "results/specialist_reuse_v3/runs/10x5+mix+bv3_sr_rs_s111/valiquality.json",
     "results/specialist_reuse_v3/state.json", "final:rs", "#2a78d6", "o"),
    ("Rewarded Soups, world-model shared base", "results/specialist_reuse_wm_v3/runs/10x5+mix+bv3_srwm_rs_s111/valiquality.json",
     "results/specialist_reuse_wm_v3/state.json", "final:rs", "#eb6834", "s"),
    ("From-scratch DANIEL", "results/specialist_reuse_v3/runs/10x5+mix+bv3_sr_scratch_s111/valiquality.json",
     "results/specialist_reuse_v3/state.json", "final:scratch", "#1baf7a", "^"),
]
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3de"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 9.5, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                         "ytick.color": MUTED, "text.color": INK})
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13, 5.2), sharey=True, facecolor="#fcfcfb")
    rows = []
    for lab, f, sf, key, col, mk in RUNS:
        v = json.loads((REPO / f).read_text())
        u = json.loads((REPO / sf).read_text())["units"][key]
        x = [e["update"] for e in v]; y = [e["hv"] for e in v]
        best = [max(y[:i + 1]) for i in range(len(y))]
        b = max(range(len(y)), key=lambda i: y[i])
        rows.append({"run": lab, "update0_hv": y[0], "best_hv": y[b], "best_update": x[b], "stop_update": u["update"]})
        ax.plot(x, y, color=col, lw=1.6, marker=mk, ms=3.2, markevery=4, label=lab, zorder=3)
        ax.plot(x[b], y[b], marker=mk, ms=9, mfc=col, mec="#fcfcfb", mew=2, zorder=4)
        ax2.step(x, best, where="post", color=col, lw=2, zorder=3,
                 label=f"{lab}\nbest {y[b]:.4f} @ {x[b]:,}; stopped at {u['update']:,}")
        ax2.plot(x[b], y[b], marker=mk, ms=9, mfc=col, mec="#fcfcfb", mew=2, zorder=4)
        for a in (ax, ax2):
            a.axvline(u["update"], color=col, lw=.9, ls=":", alpha=.9, zorder=1)
    ax.set_title("Validation HV at each validation point", fontsize=10.5, loc="left")
    ax2.set_title("Best validation HV so far (checkpoint selection)", fontsize=10.5, loc="left")
    for a in (ax, ax2):
        a.set_xlabel("training updates")
        a.grid(color=GRID, lw=.8); a.set_axisbelow(True)
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
        a.set_facecolor("#fcfcfb")
    ax.set_ylabel("validation HV (18 instances x 15 preferences)")
    ax.set_ylim(0.45, 0.99)
    ax.legend(frameon=False, loc="lower right", fontsize=8.5)
    ax2.legend(frameon=False, loc="lower right", fontsize=8.5, labelspacing=0.9)
    fig.text(0.01, 0.01, "Rewarded Soups x-axis = merge fine-tuning updates, counted after specialist training.\nDotted verticals: stopping update. Large markers: validation-selected checkpoint. "
                         "From scratch starts at HV 0.000 (below the axis). Single seed (111), 10x5, waiting allowed.",
             fontsize=8, color=MUTED)
    fig.suptitle("Convergence: Rewarded Soups (world-model vs random shared base) and From-scratch DANIEL",
                 fontsize=12, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.05, 1, 0.97))
    fig.savefig(OUT / "convergence.png", dpi=150, facecolor="#fcfcfb"); plt.close(fig)
    (OUT / "convergence.json").write_text(json.dumps(rows, indent=1) + "\n")
    print(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
