"""Figure of the specialist-reuse results table (results/specialist_reuse_v3.md): greedy HV, best-of-128 HV, best-of-128
IGD+ and multi-objective policy training time for the four methods, as four small-multiple dot panels with a shared
method axis (one panel per measure, each its own scale; every value labelled). Colours as the convergence figure.
Reads only the committed results/specialist_reuse_v3/summary.json.

    python tools/plot_specialist_reuse_v3_table.py   # -> results/specialist_reuse_v3/figures/results_table.png
"""
from __future__ import annotations
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                    # noqa: E402

REPO = Path(__file__).resolve().parents[1]
RES = REPO / "results" / "specialist_reuse_v3"
METHODS = [("scratch", "From-scratch DANIEL", "#ff7f0e"), ("distill", "Distillation", "#1f77b4"),
           ("rs", "Rewarded Soups", "#2ca02c"), ("ties", "Preference-weighted\nTIES Merge", "#d62728")]
PANELS = [  # key in summary rows, title, better, number format
    ("greedy_hv", "Greedy HV  ↑", "higher", "{:.3f}"),
    ("sample_hv", "Best-of-128 HV  ↑", "higher", "{:.3f}"),
    ("sample_igd", "Best-of-128 IGD+  ↓", "lower", "{:.3f}"),
    ("add_h", "Multi-objective policy\ntraining time (h)  ↓", "lower", "{:.2f} h"),
]
INK, MUTED, GRID = "#1f1f1f", "#6b6b6b", "#e6e6e6"


def main():
    rows = json.loads((RES / "summary.json").read_text())["rows"]
    plt.rcParams.update({"font.size": 11, "axes.edgecolor": GRID})
    fig, axes = plt.subplots(1, 4, figsize=(13.5, 3.7), facecolor="white", sharey=True)
    ys = list(range(len(METHODS)))[::-1]
    for ax, (key, title, better, fmt) in zip(axes, PANELS):
        vals = [rows[m][key] for m, _, _ in METHODS]
        lo, hi = min(vals), max(vals)
        span = hi - lo
        if key == "add_h":
            x0, x1 = 0.0, hi * 1.35
        else:
            x0, x1 = lo - span * 0.35, hi + span * 0.6
        best = (max if better == "higher" else min)(vals)
        ref = rows["scratch"][key]
        ax.axvline(ref, color=MUTED, lw=.8, ls=(0, (3, 3)), zorder=1)
        for y, (m, _, col), v in zip(ys, METHODS, vals):
            ax.hlines(y, x0, v, color=GRID, lw=1.2, zorder=1)
            ax.plot(v, y, "o", ms=10, color=col, mec="white", mew=2, zorder=3)
            ax.annotate(fmt.format(v), (v, y), xytext=(9, 0), textcoords="offset points", va="center",
                        color=INK, fontsize=10.5, fontweight="bold" if v == best else "normal", zorder=4,
                        bbox=dict(boxstyle="square,pad=0.15", fc="white", ec="none"))
        ax.set_xlim(x0, x1)
        ax.set_title(title, fontsize=11.5, color=INK, loc="left")
        ax.tick_params(axis="x", colors=MUTED, labelsize=9); ax.tick_params(axis="y", length=0)
        ax.grid(axis="x", color=GRID, lw=.6); ax.set_axisbelow(True)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
    axes[0].set_yticks(ys); axes[0].set_yticklabels([lab for _, lab, _ in METHODS], color=INK, fontsize=11)
    axes[0].set_ylim(-0.6, len(METHODS) - 0.4)
    fig.suptitle("Specialist reuse vs from scratch, waiting allowed (10x5, seed 111): evaluation results",
                 x=0.01, ha="left", fontsize=12.5, color=INK)
    fig.text(0.01, 0.012, "100 evaluation instances x 66 preferences (IGD+: the 20 instances with a CP-SAT reference "
             "front); best-of-128 sampling seed 51. Bold: best per measure. Dashed: From-scratch DANIEL.\n"
             "Training time excludes the specialists (8.0 h, needed by the three reuse methods). HV and IGD+ axes are "
             "zoomed; time starts at 0.",
             ha="left", fontsize=8.8, color=MUTED)
    fig.tight_layout(rect=(0, 0.08, 1, 0.97))
    out = RES / "figures" / "results_table.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, facecolor="white"); plt.close(fig)
    print(f"wrote {out.relative_to(REPO)}")


if __name__ == "__main__":
    main()
