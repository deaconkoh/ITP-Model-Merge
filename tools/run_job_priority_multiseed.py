"""Run or print the job-priority multi-seed experiment.

Purpose:
    Check whether low priority reward weights (0.10 and 0.25) are genuinely
    weak, or whether the previous single-run result was caused by RL training
    randomness.

Default behavior is dry-run. Add --execute on the GPU PC to actually train and
run evaluation.
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import time
from pathlib import Path

SIZES = {
    "10x5": (10, 5),
    "20x5": (20, 5),
    "15x10": (15, 10),
    "20x10": (20, 10),
}
WEIGHT_CODES = {
    0.10: "w010",
    0.25: "w025",
    0.50: "w050",
    0.75: "w075",
    1.00: "w100",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run job-priority multi-seed training/evaluation")
    parser.add_argument("--execute", action="store_true", help="Actually run commands. Omit for dry-run.")
    parser.add_argument("--device", default="cuda", help="Torch device for training/evaluation")
    parser.add_argument("--python", default=None, help="Python executable. Default: .venv/Scripts/python.exe if present, else current Python.")
    parser.add_argument("--sizes", nargs="+", default=list(SIZES), choices=list(SIZES), help="Problem sizes to train")
    parser.add_argument("--weights", nargs="+", type=float, default=list(WEIGHT_CODES), help="Priority reward weights")
    parser.add_argument("--seeds", nargs="+", type=int, default=[300, 301, 302], help="Training seeds")
    parser.add_argument("--max-updates", type=int, default=1000)
    parser.add_argument("--num-envs", type=int, default=20)
    parser.add_argument("--validate-timestep", type=int, default=10)
    parser.add_argument("--reset-env-timestep", type=int, default=20)
    parser.add_argument("--urgent-jobs", type=int, default=2)
    parser.add_argument("--priority-weight", type=float, default=3.0, help="Urgent job multiplier inside each instance")
    parser.add_argument("--eval-seed", type=int, default=50, help="Fixed evaluation seed for fair comparison")
    parser.add_argument("--limit", type=int, default=0, help="Evaluation instance limit; 0 means all")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--training-log", default="experiments/flexibility/job_priority_multiseed_training_runs.csv")
    parser.add_argument("--priority-eval-out", default="experiments/flexibility/job_priority_multiseed_eval.csv")
    parser.add_argument("--baseline-eval-out", default="experiments/flexibility/job_priority_multiseed_baseline_eval.csv")
    return parser.parse_args()


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def python_exe(repo: Path, requested: str | None) -> str:
    if requested:
        return requested
    venv_python = repo / ".venv" / "Scripts" / "python.exe"
    if venv_python.exists():
        return str(venv_python)
    return sys.executable


def weight_code(weight: float) -> str:
    rounded = round(float(weight), 2)
    if rounded not in WEIGHT_CODES:
        raise ValueError(f"Unsupported weight {weight}. Use one of: {sorted(WEIGHT_CODES)}")
    return WEIGHT_CODES[rounded]


def model_suffix(seed: int, weight: float) -> str:
    return f"job_priority_s{seed}_{weight_code(weight)}"


def model_name(size: str, seed: int, weight: float) -> str:
    return f"{size}+mix+{model_suffix(seed, weight)}"


def train_command(py: str, args: argparse.Namespace, size: str, seed: int, weight: float) -> list[str]:
    jobs, machines = SIZES[size]
    suffix = model_suffix(seed, weight)
    return [
        py,
        "train.py",
        "--device", args.device,
        "--data_source", "SD2",
        "--data_suffix", "mix",
        "--max_updates", str(args.max_updates),
        "--num_envs", str(args.num_envs),
        "--validate_timestep", str(args.validate_timestep),
        "--reset_env_timestep", str(args.reset_env_timestep),
        "--enable_priority", "True",
        "--priority_scope", "legacy_job",
        "--feature_schema", "legacy_f11_p8_v1",
        "--fea_pair_input_dim", "8",
        "--carbon_feature", "False",
        "--urgent_jobs", str(args.urgent_jobs),
        "--priority_weight", str(args.priority_weight),
        "--priority_reward_weight", f"{weight:.2f}",
        "--priority_seed", str(seed),
        "--seed_train", str(seed),
        "--n_j", str(jobs),
        "--n_m", str(machines),
        "--model_suffix", suffix,
    ]


def eval_priority_command(py: str, args: argparse.Namespace, models: list[str]) -> list[str]:
    test_data = [f"{size}+mix" for size in args.sizes]
    return [
        py,
        "tools/eval_priority_flexibility.py",
        "--device", args.device,
        "--data-source", "SD2",
        "--models", *models,
        "--test-data", *test_data,
        "--limit", str(args.limit),
        "--urgent-jobs", str(args.urgent_jobs),
        "--priority-weight", str(args.priority_weight),
        "--seed", str(args.eval_seed),
        "--priority-model",
        "--out", args.priority_eval_out,
    ]


def eval_baseline_command(py: str, args: argparse.Namespace) -> list[str]:
    normal_models = [f"{size}+mix" for size in args.sizes]
    test_data = [f"{size}+mix" for size in args.sizes]
    return [
        py,
        "tools/eval_priority_flexibility.py",
        "--device", args.device,
        "--data-source", "SD2",
        "--models", *normal_models,
        "--test-data", *test_data,
        "--limit", str(args.limit),
        "--urgent-jobs", str(args.urgent_jobs),
        "--priority-weight", str(args.priority_weight),
        "--seed", str(args.eval_seed),
        "--out", args.baseline_eval_out,
    ]


def show_command(cmd: list[str], cwd: Path) -> None:
    print(f"cwd: {cwd}")
    print(" ".join(f'\"{part}\"' if " " in part else part for part in cmd))


def run_command(cmd: list[str], cwd: Path) -> int:
    show_command(cmd, cwd)
    completed = subprocess.run(cmd, cwd=cwd)
    return completed.returncode


def main() -> None:
    args = parse_args()
    repo = repo_root()
    py = python_exe(repo, args.python)
    daniel_dir = repo / "daniel"
    training_log = repo / args.training_log
    training_log.parent.mkdir(parents=True, exist_ok=True)

    planned_models = [model_name(size, seed, weight) for size in args.sizes for seed in args.seeds for weight in args.weights]
    print(f"Planned priority checkpoints: {len(planned_models)}")
    print(f"Sizes={args.sizes} weights={args.weights} seeds={args.seeds}")
    print("Dry run only. Add --execute on the GPU PC to run." if not args.execute else "EXECUTE mode enabled.")

    rows = []
    if not args.skip_train:
        for size in args.sizes:
            for seed in args.seeds:
                for weight in args.weights:
                    cmd = train_command(py, args, size, seed, weight)
                    start = time.time()
                    rc = 0
                    if args.execute:
                        rc = run_command(cmd, daniel_dir)
                        if rc != 0:
                            raise SystemExit(rc)
                    else:
                        show_command(cmd, daniel_dir)
                    rows.append({
                        "size": size,
                        "seed_train": seed,
                        "priority_seed": seed,
                        "priority_reward_weight": f"{weight:.2f}",
                        "model": model_name(size, seed, weight),
                        "returncode": rc,
                        "elapsed_seconds": round(time.time() - start, 2),
                    })

        if args.execute:
            with training_log.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            print(f"Wrote {training_log}")

    if not args.skip_eval:
        priority_cmd = eval_priority_command(py, args, planned_models)
        baseline_cmd = eval_baseline_command(py, args)
        if args.execute:
            rc = run_command(priority_cmd, repo)
            if rc != 0:
                raise SystemExit(rc)
            rc = run_command(baseline_cmd, repo)
            if rc != 0:
                raise SystemExit(rc)
        else:
            show_command(priority_cmd, repo)
            show_command(baseline_cmd, repo)

    print("Next: run tools/summarize_job_priority_multiseed.py after the two eval CSVs exist.")


if __name__ == "__main__":
    main()
