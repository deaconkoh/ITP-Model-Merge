"""Convergence figure of world-model M3 Part B, in the style of results/merge_categories_v3/figures/curves_10x5.png:
validation HV (15 preferences, 18 instances) per update for the three merge fine-tuning runs and the converged From
scratch and Distillation references, smoothed over 5 points, each labelled at its right end; dashed line at the
Faster target (0.542). Reads only committed run logs.

    python tools/plot_world_model_m3_v3_curves.py   # -> results/world_model_m3_v3/figures/curves_10x5.png
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                    # noqa: E402
import numpy as np                                                 # noqa: E402

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))
from analyze_merge_categories_v3 import rolling                   # noqa: E402
import freeze_world_model_m3_v3 as F                               # noqa: E402

RES = REPO / "results" / "world_model_m3_v3"
RUNS = [  # label, valiquality.json, colour, raw values drawn faint
    ("World-model base mix", RES / "runs/10x5+mix+bv3_wm3_wm_s111/valiquality.json", "#d62728", True),
    ("Random shared base mix", RES / "runs/10x5+mix+bv3_wm3_ctrl_s111/valiquality.json", "#2ca02c", True),
    ("Original specialists mix", RES / "runs/10x5+mix+bv3_wm3_orig_s111/valiquality.json", "#9467bd", False),
    ("Distillation (converged)", None, "#1f77b4", False),
    ("From scratch (converged)", None, "#ff7f0e", False),
]


def draw(cfg, out, ylim, gap):
    """One figure; ylim (low, high) or None for a full axis from 0."""
    refs = {"Distillation (converged)": REPO / cfg["references"]["Distillation"]["valiquality"],
            "From scratch (converged)": REPO / cfg["references"]["From scratch"]["valiquality"]}
    target = cfg["criteria"]["faster"]["target_hv"]
    plt.rcParams.update({"font.size": 11})
    fig, ax = plt.subplots(figsize=(10.5, 5.8), facecolor="white")
    ax.set_facecolor("white"); ax.grid(alpha=.3, lw=.6)
    ends = []
    for name, path, col, raw in RUNS:
        v = json.loads((path or refs[name]).read_text())
        u, hv = [e["update"] for e in v], [e["hv"] for e in v]
        if raw:
            ax.plot(u, hv, color=col, lw=.7, alpha=.3)
        sm = rolling(hv); ax.plot(u, sm, color=col, lw=2)
        ends.append([u[-1], sm[-1], sm[-1], name, col])
    ax.axhline(target, color="#555555", lw=.9, ls="--")
    xmax = max(e[0] for e in ends)
    ends.append([xmax, target, target, f"target {target} (From scratch's best)", "#555555"])
    ends.sort(key=lambda e: e[1])
    for lo, hi in zip(ends, ends[1:]):
        hi[2] = max(hi[2], lo[2] + gap)
    shift = sum(e[1] - e[2] for e in ends) / len(ends)            # centre the label stack on the line ends
    for e in ends:
        e[2] += shift
    for xe, ye, yl, name, col in ends:                             # labels in the right margin, spread apart
        ax.annotate(name, (xe, ye), xytext=(xmax * 1.04, yl), textcoords="data", va="center", color=col,
                    fontsize=9.5 if name.startswith("target") else 11,
                    arrowprops=dict(arrowstyle="-", color=col, lw=.6, shrinkA=0, shrinkB=2, alpha=.6))
    ax.set_xlim(0, xmax * 1.45)
    ax.set_xticks(np.arange(0, xmax + 1, 500))
    if ylim:
        ax.set_ylim(*ylim)
    else:
        ax.set_ylim(bottom=0)
    ax.set_xlabel("updates"); ax.set_ylabel("validation hypervolume")
    ax.set_title("Validation hypervolume during merge fine-tuning (10x5, seed 111)" + (" - zoomed" if ylim else ""))
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.text(0.5, 0.01, "Smoothed over 5 points; faint lines: raw values. Dashed: target for the Faster criterion.",
             ha="center", fontsize=11)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130, facecolor="white"); plt.close(fig)
    print(f"wrote {out.relative_to(REPO)}")


def main():
    cfg = F.load()
    draw(cfg, RES / "figures" / "curves_10x5.png", None, 0.026)
    draw(cfg, RES / "figures" / "curves_10x5_zoom.png", (0.46, 0.61), 0.009)


if __name__ == "__main__":
    main()
