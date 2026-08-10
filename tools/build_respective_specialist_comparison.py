"""Build a corrected merged-vs-specialists comparison.

The earlier presentation table compared the fine-tuned merged model against the
best carbon sweep row for every metric. This script compares each metric against
the matching standalone specialist instead:

* makespan  -> best speed-compatible standalone model
* carbon    -> best carbon-specialist standalone model
* priority  -> best priority-specialist standalone model

Lower is better for all three metrics, so negative percentage means the merged
fine-tuned model is better than that specialist baseline.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_TEST_ORDER = [
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


def best_by_metric(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    idx = df.groupby("test_data")[metric].idxmin()
    return df.loc[idx, ["test_data", "model", metric]].copy()


def read_optional_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if df.empty:
        return None
    return df


def build_summary(args: argparse.Namespace) -> pd.DataFrame:
    merged = pd.read_csv(args.merged_eval)
    merged = merged[merged["model"] == args.merged_model].copy()
    merged = merged[merged["test_data"].isin(DEFAULT_TEST_ORDER)].copy()

    speed = pd.read_csv(args.speed_eval)
    speed = speed[speed["model"].str.contains("speed_compatible", na=False)].copy()
    best_speed = best_by_metric(speed, "mean_makespan").rename(
        columns={
            "model": "speed_specialist_model",
            "mean_makespan": "speed_specialist_makespan",
        }
    )

    carbon = pd.read_csv(args.carbon_eval)
    carbon = carbon[carbon["model"].str.contains("carbon_reward", na=False)].copy()
    best_carbon = best_by_metric(carbon, "mean_total_carbon").rename(
        columns={
            "model": "carbon_specialist_model",
            "mean_total_carbon": "carbon_specialist_carbon",
        }
    )

    priority = read_optional_csv(args.priority_eval)
    if priority is not None:
        priority = priority[priority["model"].str.contains("job_priority", na=False)].copy()
        best_priority = best_by_metric(priority, "mean_priority_weighted_completion").rename(
            columns={
                "model": "priority_specialist_model",
                "mean_priority_weighted_completion": "priority_specialist_priority",
            }
        )
    else:
        best_priority = pd.DataFrame(
            columns=["test_data", "priority_specialist_model", "priority_specialist_priority"]
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
    summary = summary.merge(best_speed, on="test_data", how="left")
    summary = summary.merge(best_carbon, on="test_data", how="left")
    summary = summary.merge(best_priority, on="test_data", how="left")

    summary["makespan_pct_vs_speed_specialist"] = summary.apply(
        lambda r: pct_change(r["merged_makespan"], r["speed_specialist_makespan"]), axis=1
    )
    summary["carbon_pct_vs_carbon_specialist"] = summary.apply(
        lambda r: pct_change(r["merged_carbon"], r["carbon_specialist_carbon"]), axis=1
    )
    summary["priority_pct_vs_priority_specialist"] = summary.apply(
        lambda r: (
            pct_change(r["merged_priority"], r["priority_specialist_priority"])
            if pd.notna(r["priority_specialist_priority"])
            else np.nan
        ),
        axis=1,
    )

    summary["test_size"] = summary["test_data"].map(DISPLAY_SIZE)
    summary["lower_is_better_note"] = "negative % means merged fine-tuned is better"
    summary["priority_status"] = np.where(
        summary["priority_specialist_priority"].notna(),
        "available",
        f"pending: create {args.priority_eval.name}",
    )

    order = {name: i for i, name in enumerate(DEFAULT_TEST_ORDER)}
    summary = summary.sort_values("test_data", key=lambda s: s.map(order)).reset_index(drop=True)
    return summary


def draw_chart(summary: pd.DataFrame, out_png: Path) -> None:
    labels = summary["test_size"].tolist()
    metrics = [
        ("Makespan vs speed specialist", "makespan_pct_vs_speed_specialist"),
        ("Carbon vs carbon specialist", "carbon_pct_vs_carbon_specialist"),
        ("Priority vs priority specialist", "priority_pct_vs_priority_specialist"),
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
            if np.isnan(val):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    0.8,
                    "pending",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    color="#64748b",
                    rotation=90,
                )
                continue
            y = bar.get_height()
            va = "bottom" if y >= 0 else "top"
            pad = 0.6 if y >= 0 else -0.6
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

    ax.axhline(0, color="#0f172a", linewidth=1)
    finite_vals = summary[
        [
            "makespan_pct_vs_speed_specialist",
            "carbon_pct_vs_carbon_specialist",
            "priority_pct_vs_priority_specialist",
        ]
    ].to_numpy(dtype=float)
    max_val = np.nanmax(finite_vals)
    min_val = np.nanmin(finite_vals)
    ax.set_ylim(min(min_val - 1.5, -1.0), max(max_val + 2.0, 3.0))
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11, fontweight="bold")
    ax.set_ylabel("Fine-tuned merged model difference (%)", fontsize=11)
    fig.suptitle(
        "Fine-tuned merged model vs matching standalone specialists",
        fontsize=16,
        fontweight="bold",
        y=0.97,
    )
    ax.text(
        0,
        1.03,
        "Lower is better for all metrics. Negative bars mean the merged model is better than that specialist.",
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


def write_handoff(args: argparse.Namespace) -> None:
    text = f"""# Corrected Specialist Comparison Handoff

This comparison fixes the earlier chart issue by comparing each metric against
the correct standalone specialist:

- makespan is compared against the best speed-compatible standalone model
- carbon is compared against the best carbon-specialist model
- priority is compared against the best priority-specialist model

The priority-specialist CSV was not available on this laptop because the CPU
evaluation was too slow. Generate it on the GPU PC with:

```powershell
python tools/eval_carbon_models.py --device cuda --data-source SD2 --models 10x5+mix+job_priority_w050 20x5+mix+job_priority_w050 15x10+mix+job_priority_w050 20x10+mix+job_priority_w050 --test-data 20x5+carbon+priority 15x10+carbon+priority 20x10+carbon+priority 30x10+carbon+priority 40x10+carbon+priority --limit 0 --enable-carbon --out experiments/flexibility/priority_specialist_carbon_priority_eval.csv
```

Then regenerate the corrected summary and chart:

```powershell
python tools/build_respective_specialist_comparison.py
```

Outputs:

- `{args.out_csv}`
- `{args.out_png}`

Interpretation:

- Negative percentage = fine-tuned merged model is better.
- Positive percentage = fine-tuned merged model is worse.
- A good merged model does not need to beat every specialist in every metric;
  it should be more balanced across makespan, carbon, and priority.
"""
    args.handoff_md.parent.mkdir(parents=True, exist_ok=True)
    args.handoff_md.write_text(text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[1]
    flex = root / "experiments" / "flexibility"
    parser.add_argument("--merged-model", default="20x5+carbon+priority+merged_mcp_finetune")
    parser.add_argument("--merged-eval", type=Path, default=flex / "merged_mcp_finetune_eval.csv")
    parser.add_argument("--speed-eval", type=Path, default=flex / "carbon_normal_objective_baseline_eval.csv")
    parser.add_argument("--carbon-eval", type=Path, default=flex / "carbon_model_sweep_eval.csv")
    parser.add_argument(
        "--priority-eval",
        type=Path,
        default=flex / "priority_specialist_carbon_priority_eval.csv",
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=flex / "merged_vs_respective_specialists_summary.csv",
    )
    parser.add_argument(
        "--out-png",
        type=Path,
        default=flex / "merged_vs_respective_specialists_chart.png",
    )
    parser.add_argument(
        "--handoff-md",
        type=Path,
        default=root / "docs" / "RESPECTIVE_SPECIALIST_COMPARISON_HANDOFF.md",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = build_summary(args)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.out_csv, index=False)
    draw_chart(summary, args.out_png)
    write_handoff(args)
    print(f"Wrote {args.out_csv}")
    print(f"Wrote {args.out_png}")
    print(f"Wrote {args.handoff_md}")


if __name__ == "__main__":
    main()
