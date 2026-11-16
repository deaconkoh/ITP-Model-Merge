"""Simple comparison chart of the best-of-128 evaluation HV of the specialist-reuse experiment (results table column).
Saved results only (results/specialist_reuse_v3/summary.json); no training or evaluation.

    python tools/plot_specialist_reuse_v3_best128.py   # -> results/specialist_reuse_v3/figures/best_of_128_hv.png
"""
from __future__ import annotations
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                    # noqa: E402

REPO = Path(__file__).resolve().parents[1]
RES = REPO / "results" / "specialist_reuse_v3"
METHODS = [("scratch", "From scratch", "#ff7f0e"), ("distill", "Distillation", "#1f77b4"),
           ("rs", "Rewarded Soups", "#2ca02c"), ("ties", "Preference-weighted TIES", "#d62728")]
INK, MUTED, GRID = "#1f1f1f", "#6b6b6b", "#e6e6e6"


def main():
    rows = json.loads((RES / "summary.json").read_text())["rows"]
    vals = [rows[m]["sample_hv"] for m, _, _ in METHODS]
    base = vals[0]
    plt.rcParams.update({"font.size": 11})
    fig, ax = plt.subplots(figsize=(9.5, 3.8), facecolor="white")
    ys = list(range(len(METHODS)))[::-1]
    for y, (m, lab, col), v in zip(ys, METHODS, vals):
        ax.barh(y, v, height=0.6, color=col)
        txt = f"{v:.3f}" + ("" if m == "scratch" else f"   ({v - base:+.3f} vs from scratch)")
        ax.text(v + 0.012, y, txt, va="center", color=INK, fontsize=10.5,
                fontweight="bold" if v == max(vals) else "normal")
    ax.set_yticks(ys); ax.set_yticklabels([lab for _, lab, _ in METHODS], color=INK)
    ax.set_xlim(0, 1.75)
    ax.set_xlabel("best-of-128 evaluation hypervolume (higher is better)", color=MUTED)
    ax.tick_params(axis="x", colors=MUTED); ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color=GRID, lw=.6); ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    fig.suptitle("Best-of-128 HV: specialist reuse vs from scratch (10x5, waiting allowed)",
                 x=0.01, ha="left", fontsize=11.5, color=INK)
    fig.text(0.01, 0.02, "100 evaluation instances x 66 preferences; sampling seed 51; frozen normalisation, "
             "reference point 1.1. Seed 111 only.", fontsize=8.8, color=MUTED)
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    out = RES / "figures" / "best_of_128_hv.png"
    fig.savefig(out, dpi=150, facecolor="white"); plt.close(fig)
    print(f"wrote {out.relative_to(REPO)}")


if __name__ == "__main__":
    main()
