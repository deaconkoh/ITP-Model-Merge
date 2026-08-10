"""Summarize job-priority multi-seed evaluation results.

Compares each priority-aware checkpoint against the normal DANIEL checkpoint
trained on the same problem size, then reports mean/std across training seeds.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

WEIGHT_FROM_CODE = {"w010": 0.10, "w025": 0.25, "w050": 0.50, "w075": 0.75, "w100": 1.00}
MODEL_RE = re.compile(r"(?P<size>\d+x\d+)\+mix\+job_priority_s(?P<seed>\d+)_(?P<weight>w\d{3})")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize job-priority multi-seed results")
    parser.add_argument("--priority-eval", default="experiments/flexibility/job_priority_multiseed_eval.csv")
    parser.add_argument("--baseline-eval", default="experiments/flexibility/job_priority_multiseed_baseline_eval.csv")
    parser.add_argument("--detail-out", default="experiments/flexibility/job_priority_multiseed_summary.csv")
    parser.add_argument("--weight-out", default="experiments/flexibility/job_priority_multiseed_by_weight.csv")
    return parser.parse_args()


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def add_model_parts(df: pd.DataFrame) -> pd.DataFrame:
    parts = df["model"].str.extract(MODEL_RE)
    if parts.isna().any().any():
        bad = df.loc[parts.isna().any(axis=1), "model"].drop_duplicates().tolist()
        raise ValueError(f"Could not parse priority model names: {bad[:5]}")
    out = df.copy()
    out["train_size"] = parts["size"]
    out["seed_train"] = parts["seed"].astype(int)
    out["priority_reward_weight"] = parts["weight"].map(WEIGHT_FROM_CODE)
    out["same_size_test"] = out.apply(
        lambda row: str(row["test_data"]).startswith(f"{row['train_size']}+"),
        axis=1,
    )
    return out


def main() -> None:
    args = parse_args()
    repo = repo_root()
    priority = pd.read_csv(repo / args.priority_eval)
    baseline = pd.read_csv(repo / args.baseline_eval)
    priority = add_model_parts(priority)
    baseline = baseline[baseline["action_mode"].eq("normal")].copy()
    baseline["train_size"] = baseline["model"].str.replace("+mix", "", regex=False)

    merged = priority.merge(
        baseline[[
            "train_size",
            "test_data",
            "mean_makespan",
            "mean_weighted_completion",
            "mean_urgent_completion",
            "mean_normal_job_completion",
        ]],
        on=["train_size", "test_data"],
        how="left",
        suffixes=("", "_baseline"),
    )
    metrics = ["makespan", "weighted_completion", "urgent_completion", "normal_job_completion"]
    for metric in metrics:
        src = f"mean_{metric}"
        base = f"mean_{metric}_baseline"
        merged[f"{metric}_delta"] = merged[src] - merged[base]

    merged["balanced_score"] = merged["weighted_completion_delta"] + 0.25 * merged["makespan_delta"]
    detail_path = repo / args.detail_out
    detail_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(detail_path, index=False)

    grouped = merged.groupby(["same_size_test", "priority_reward_weight"]).agg(
        runs=("model", "count"),
        makespan_delta_mean=("makespan_delta", "mean"),
        makespan_delta_std=("makespan_delta", "std"),
        weighted_delta_mean=("weighted_completion_delta", "mean"),
        weighted_delta_std=("weighted_completion_delta", "std"),
        urgent_delta_mean=("urgent_completion_delta", "mean"),
        urgent_delta_std=("urgent_completion_delta", "std"),
        balanced_score_mean=("balanced_score", "mean"),
        balanced_score_std=("balanced_score", "std"),
    ).reset_index()
    weight_path = repo / args.weight_out
    grouped.to_csv(weight_path, index=False)

    same = grouped[grouped["same_size_test"].eq(True)].copy()
    print("Same-size average by reward weight. Negative deltas are better.")
    print(same.to_string(index=False))
    print(f"Wrote {detail_path}")
    print(f"Wrote {weight_path}")


if __name__ == "__main__":
    main()
