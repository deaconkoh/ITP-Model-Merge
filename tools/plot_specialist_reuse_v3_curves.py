"""Convergence figure of the specialist-reuse experiment (docs/specialist_reuse_protocol_v3.md), in the style of
results/merge_categories_v3/figures/curves_10x5.png: validation HV (greedy, 15 preferences, 18 instances, every 50
updates) of the four full runs (From scratch, Distillation, Rewarded Soups, preference-weighted TIES), smoothed over 5
points with the raw values faint, each labelled at its right end; dashed line at From scratch's best validation HV
(the time-to-target target). Reads only committed run logs.

    python tools/plot_specialist_reuse_v3_curves.py   # -> results/specialist_reuse_v3/figures/curves_10x5{,_zoom}.png
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

RES = REPO / "results" / "specialist_reuse_v3"
RUNS = [  # label, run, colour
    ("From scratch", "scratch", "#ff7f0e"),
    ("Distillation", "distill", "#1f77b4"),
    ("Rewarded Soups", "rs", "#2ca02c"),
    ("Preference-weighted TIES", "ties", "#d62728"),
]


def draw(out, ylim, gap):
    st = json.loads((RES / "state.json").read_text())["units"]
    target = -st["final:scratch"]["best"]
    plt.rcParams.update({"font.size": 11})
    fig, ax = plt.subplots(figsize=(10.5, 5.8), facecolor="white")
    ax.set_facecolor("white"); ax.grid(alpha=.3, lw=.6)
    ends = []
    for name, key, col in RUNS:
        v = json.loads((RES / "runs" / f"10x5+mix+bv3_sr_{key}_s111" / "valiquality.json").read_text())
        u, hv = [e["update"] for e in v], [e["hv"] for e in v]
        ax.plot(u, hv, color=col, lw=.7, alpha=.3)
        sm = rolling(hv); ax.plot(u, sm, color=col, lw=2)
        stop = st[f"final:{key}"]
        ends.append([u[-1], sm[-1], sm[-1], f"{name} ({stop['status'].lower()}, {stop['update']})", col])
    xmax = max(e[0] for e in ends)
    ax.hlines(target, 0, xmax, color="#555555", lw=.9, ls="--")
    ends.append([xmax, target, target, f"target {target:.3f} (From scratch's best)", "#555555"])
    ends.sort(key=lambda e: e[1])
    for lo, hi in zip(ends, ends[1:]):
        hi[2] = max(hi[2], lo[2] + gap)
    shift = sum(e[1] - e[2] for e in ends) / len(ends)            # centre the label stack on the line ends
    for e in ends:
        e[2] += shift
    top = (ylim[1] if ylim else 1.0) - gap / 2                     # keep the stack inside the axes
    excess = max(e[2] for e in ends) - top
    for e in ends:
        e[2] -= max(0.0, excess)
    for xe, ye, yl, name, col in ends:                             # labels in the right margin, spread apart
        ax.annotate(name, (xe, ye), xytext=(xmax * 1.04, yl), textcoords="data", va="center", color=col,
                    fontsize=9.5 if name.startswith("target") else 11,
                    arrowprops=dict(arrowstyle="-", color=col, lw=.6, shrinkA=0, shrinkB=2, alpha=.6))
    ax.set_xlim(0, xmax * 1.5)
    ax.set_xticks(np.arange(0, xmax + 1, 500))
    if ylim:
        ax.set_ylim(*ylim)
    else:
        ax.set_ylim(0, 1.0)
    ax.set_xlabel("updates"); ax.set_ylabel("validation hypervolume")
    ax.set_title("Validation hypervolume, specialist reuse vs from scratch, waiting allowed (10x5, seed 111)"
                 + (" - zoomed" if ylim else ""), fontsize=12)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.text(0.5, 0.01, "Smoothed over 5 points; faint lines: raw values. Dashed: From scratch's best validation HV "
             "(time-to-target). Specialist training (8.0 h) not shown.", ha="center", fontsize=10)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130, facecolor="white"); plt.close(fig)
    print(f"wrote {out.relative_to(REPO)}")


def main():
    draw(RES / "figures" / "curves_10x5.png", None, 0.045)
    draw(RES / "figures" / "curves_10x5_zoom.png", (0.86, 0.97), 0.008)


if __name__ == "__main__":
    main()
