"""Summarize carbon/speed model results against normal DANIEL baselines.

This script is intentionally dependency-free so it can run even on machines
without pandas installed.

Speed comparison:
  Uses experiments/flexibility/speed_compatible_eval.csv and compares each
  merge-compatible speed checkpoint against the matching original normal
  DANIEL checkpoint from job_priority_multisize_summary.csv.

Carbon comparison:
  If experiments/flexibility/carbon_normal_objective_baseline_eval.csv exists,
  compares carbon reward checkpoints against normal-objective, merge-compatible
  checkpoints evaluated on the same carbon+priority datasets.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def speed_model_to_normal(model: str) -> str:
    return model.replace("+speed_compatible", "")


def carbon_model_to_normal_objective(model: str) -> str:
    return model.replace("+carbon+priority+carbon_reward", "+mix+speed_compatible")


def f(value: str) -> float:
    return float(value)


def summarize_speed(repo: Path, out: Path) -> None:
    speed_eval = read_csv(repo / "experiments/flexibility/speed_compatible_eval.csv")
    multisize = read_csv(repo / "experiments/flexibility/job_priority_multisize_summary.csv")

    baseline: dict[tuple[str, str], float] = {}
    for row in multisize:
        key = (row["baseline_model"], row["test_data"])
        baseline.setdefault(key, f(row["baseline_makespan"]))

    rows: list[dict[str, object]] = []
    for row in speed_eval:
        normal_model = speed_model_to_normal(row["model"])
        key = (normal_model, row["test_data"])
        if key not in baseline:
            continue
        model_makespan = f(row["mean_makespan"])
        baseline_makespan = baseline[key]
        rows.append(
            {
                "model": row["model"],
                "normal_daniel_model": normal_model,
                "test_data": row["test_data"],
                "instances": row["instances"],
                "mean_makespan": round(model_makespan, 4),
                "normal_daniel_makespan": round(baseline_makespan, 4),
                "makespan_delta_vs_normal": round(model_makespan - baseline_makespan, 4),
                "makespan_pct_vs_normal": round((model_makespan - baseline_makespan) / baseline_makespan * 100, 4),
                "mean_runtime_seconds": row["mean_runtime_seconds"],
            }
        )

    write_csv(
        out,
        rows,
        [
            "model",
            "normal_daniel_model",
            "test_data",
            "instances",
            "mean_makespan",
            "normal_daniel_makespan",
            "makespan_delta_vs_normal",
            "makespan_pct_vs_normal",
            "mean_runtime_seconds",
        ],
    )
    print(f"Wrote {out}")


def summarize_carbon(repo: Path, baseline_path: Path, out: Path) -> bool:
    carbon_eval_path = repo / "experiments/flexibility/carbon_model_sweep_eval.csv"
    if not baseline_path.exists():
        print(f"SKIP carbon comparison: missing {baseline_path}")
        return False

    carbon_eval = read_csv(carbon_eval_path)
    baseline_eval = read_csv(baseline_path)

    baseline: dict[tuple[str, str], dict[str, str]] = {}
    for row in baseline_eval:
        baseline[(row["model"], row["test_data"])] = row

    rows: list[dict[str, object]] = []
    for row in carbon_eval:
        normal_model = carbon_model_to_normal_objective(row["model"])
        key = (normal_model, row["test_data"])
        if key not in baseline:
            continue
        base = baseline[key]
        carbon = f(row["mean_total_carbon"])
        base_carbon = f(base["mean_total_carbon"])
        makespan = f(row["mean_makespan"])
        base_makespan = f(base["mean_makespan"])
        priority = f(row["mean_priority_weighted_completion"])
        base_priority = f(base["mean_priority_weighted_completion"])
        rows.append(
            {
                "model": row["model"],
                "normal_objective_model": normal_model,
                "test_data": row["test_data"],
                "instances": row["instances"],
                "mean_total_carbon": round(carbon, 4),
                "normal_total_carbon": round(base_carbon, 4),
                "carbon_delta_vs_normal_objective": round(carbon - base_carbon, 4),
                "carbon_pct_vs_normal_objective": round((carbon - base_carbon) / base_carbon * 100, 4),
                "mean_makespan": round(makespan, 4),
                "normal_makespan": round(base_makespan, 4),
                "makespan_delta_vs_normal_objective": round(makespan - base_makespan, 4),
                "mean_priority_weighted_completion": round(priority, 4),
                "normal_priority_weighted_completion": round(base_priority, 4),
                "priority_weighted_delta_vs_normal_objective": round(priority - base_priority, 4),
            }
        )

    write_csv(
        out,
        rows,
        [
            "model",
            "normal_objective_model",
            "test_data",
            "instances",
            "mean_total_carbon",
            "normal_total_carbon",
            "carbon_delta_vs_normal_objective",
            "carbon_pct_vs_normal_objective",
            "mean_makespan",
            "normal_makespan",
            "makespan_delta_vs_normal_objective",
            "mean_priority_weighted_completion",
            "normal_priority_weighted_completion",
            "priority_weighted_delta_vs_normal_objective",
        ],
    )
    print(f"Wrote {out}")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize carbon/speed results against baselines")
    parser.add_argument("--repo", default=Path(__file__).resolve().parents[1])
    parser.add_argument(
        "--carbon-baseline",
        default="experiments/flexibility/carbon_normal_objective_baseline_eval.csv",
    )
    parser.add_argument(
        "--speed-out",
        default="experiments/flexibility/speed_vs_normal_daniel_summary.csv",
    )
    parser.add_argument(
        "--carbon-out",
        default="experiments/flexibility/carbon_vs_normal_objective_summary.csv",
    )
    args = parser.parse_args()

    repo = Path(args.repo).resolve()
    summarize_speed(repo, repo / args.speed_out)
    summarize_carbon(repo, repo / args.carbon_baseline, repo / args.carbon_out)


if __name__ == "__main__":
    main()
