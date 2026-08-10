"""Compare the fine-tuned merged model against normal DANIEL.

Normal DANIEL here means the merge-compatible speed/normal-objective models:

* trained with the normal makespan objective
* compatible with the 11-operation-feature merged-model shape
* evaluated on the same carbon+priority test sets

For each test size, this script chooses the normal-DANIEL row with the lowest
makespan, then compares the fine-tuned merged model against that same baseline
for makespan, carbon, and priority-weighted completion.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


TEST_ORDER = [
    "20x5+carbon+priority",
    "15x10+carbon+priority",
    "20x10+carbon+priority",
    "30x10+carbon+priority",
    "40x10+carbon+priority",
]

DISPLAY_SIZE = {
    "20x5+carbon+priority": "20x5",
    "15x10+carbon+priority": "15x10",
    "20x10+carbon+priority": "20x10",
    "30x10+carbon+priority": "30x10",
    "40x10+carbon+priority": "40x10",
}


def pct_change(candidate: float, baseline: float) -> float:
    return (candidate - baseline) / baseline * 100.0


def build_summary(args: argparse.Namespace) -> pd.DataFrame:
    merged = pd.read_csv(args.merged_eval)
    merged = merged[merged["model"].eq(args.merged_model)].copy()
    merged = merged[merged["test_data"].isin(TEST_ORDER)].copy()

    normal = pd.read_csv(args.normal_eval)
    normal = normal[normal["model"].str.contains("speed_compatible", na=False)].copy()
    normal = normal[normal["test_data"].isin(TEST_ORDER)].copy()

    best_idx = normal.groupby("test_data")["mean_makespan"].idxmin()
    best_normal = normal.loc[
        best_idx,
        [
            "test_data",
            "model",
            "mean_makespan",
            "mean_total_carbon",
            "mean_priority_weighted_completion",
        ],
    ].rename(
        columns={
            "model": "normal_daniel_model",
            "mean_makespan": "normal_makespan",
            "mean_total_carbon": "normal_carbon",
            "mean_priority_weighted_completion": "normal_priority",
        }
    )

    summary = merged[
        [
            "test_data",
            "mean_makespan",
            "mean_total_carbon",
            "mean_priority_weighted_completion",
        ]
    ].rename(
        columns={
            "mean_makespan": "merged_makespan",
            "mean_total_carbon": "merged_carbon",
            "mean_priority_weighted_completion": "merged_priority",
        }
    )
    summary = summary.merge(best_normal, on="test_data", how="left")

    summary["makespan_pct_vs_normal"] = summary.apply(
        lambda r: pct_change(r["merged_makespan"], r["normal_makespan"]), axis=1
    )
    summary["carbon_pct_vs_normal"] = summary.apply(
        lambda r: pct_change(r["merged_carbon"], r["normal_carbon"]), axis=1
    )
    summary["priority_pct_vs_normal"] = summary.apply(
        lambda r: pct_change(r["merged_priority"], r["normal_priority"]), axis=1
    )

    summary["test_size"] = summary["test_data"].map(DISPLAY_SIZE)
    summary["lower_is_better_note"] = "negative % means merged fine-tuned is better"

    order = {name: i for i, name in enumerate(TEST_ORDER)}
    return summary.sort_values("test_data", key=lambda s: s.map(order)).reset_index(drop=True)


def draw_chart(summary: pd.DataFrame, out_png: Path) -> None:
    labels = summary["test_size"].tolist()
    metrics = [
        ("Makespan", "makespan_pct_vs_normal"),
        ("Carbon", "carbon_pct_vs_normal"),
        ("Priority", "priority_pct_vs_normal"),
    ]
    colors = ["#2563eb", "#16a34a", "#f59e0b"]

    x = np.arange(len(labels))
    width = 0.24

    fig, ax = plt.subplots(figsize=(12, 6.6))
    fig.patch.set_facecolor("#f8fafc")
    ax.set_facecolor("#ffffff")

    for i, ((name, col), color) in enumerate(zip(metrics, colors)):
        vals = summary[col].to_numpy(dtype=float)
        offset = (i - 1) * width
        bars = ax.bar(x + offset, vals, width, label=name, color=color, alpha=0.9)
        for bar, val in zip(bars, vals):
            y = bar.get_height()
            va = "bottom" if y >= 0 else "top"
            pad = 0.5 if y >= 0 else -0.5
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                y + pad,
                f"{val:+.1f}%",
                ha="center",
                va=va,
                fontsize=9,
                fontweight="bold",
                color="#0f172a",
            )

    finite_vals = summary[
        ["makespan_pct_vs_normal", "carbon_pct_vs_normal", "priority_pct_vs_normal"]
    ].to_numpy(dtype=float)
    min_val = np.nanmin(finite_vals)
    max_val = np.nanmax(finite_vals)
    ax.set_ylim(min(min_val - 2.0, -5.0), max(max_val + 2.0, 5.0))
    ax.axhline(0, color="#0f172a", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11, fontweight="bold")
    ax.set_ylabel("Fine-tuned merged model difference (%)", fontsize=11)
    fig.suptitle(
        "Fine-tuned merged model vs normal DANIEL",
        fontsize=16,
        fontweight="bold",
        y=0.97,
    )
    ax.text(
        0,
        1.03,
        "Lower is better for all metrics. Negative bars mean the merged model is better than normal DANIEL.",
        transform=ax.transAxes,
        fontsize=10,
        color="#475569",
    )
    ax.legend(loc="upper left", bbox_to_anchor=(0, -0.13), ncol=3, frameon=False, fontsize=10)
    ax.grid(axis="y", color="#e2e8f0", linewidth=0.8)
    for spine in ax.spines.values():
        spine.set_visible(False)

    fig.tight_layout(rect=[0, 0.04, 1, 0.93])
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[1]
    flex = root / "experiments" / "flexibility"
    parser.add_argument("--merged-model", default="20x5+carbon+priority+merged_mcp_finetune")
    parser.add_argument("--merged-eval", type=Path, default=flex / "merged_mcp_finetune_eval.csv")
    parser.add_argument(
        "--normal-eval",
        type=Path,
        default=flex / "carbon_normal_objective_baseline_eval.csv",
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=flex / "merged_vs_normal_daniel_summary.csv",
    )
    parser.add_argument(
        "--out-png",
        type=Path,
        default=flex / "merged_vs_normal_daniel_chart.png",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = build_summary(args)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.out_csv, index=False)
    draw_chart(summary, args.out_png)
    print(f"Wrote {args.out_csv}")
    print(f"Wrote {args.out_png}")


if __name__ == "__main__":
    main()
