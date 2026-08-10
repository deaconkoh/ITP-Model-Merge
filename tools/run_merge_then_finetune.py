"""Plan or run a merge -> fine-tune -> evaluate experiment.

The default is a dry run. Add --execute on the GPU PC after checking the
printed commands.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one DANIEL merge-then-fine-tune experiment")
    parser.add_argument("--execute", action="store_true", help="Actually run the commands")
    parser.add_argument("--method", choices=["soup", "task_arithmetic"], default="soup")
    parser.add_argument("--data-source", default="SD2")
    parser.add_argument("--checkpoints", nargs="+", required=True, help="Checkpoint names or .pth paths")
    parser.add_argument("--weights", nargs="+", type=float, help="Merge weights")
    parser.add_argument("--base", help="Base checkpoint for task_arithmetic")
    parser.add_argument("--merged-name", required=True, help="Output merged checkpoint name without .pth")
    parser.add_argument("--fine-tune-suffix", required=True, help="Fine-tuned model suffix for daniel/train.py")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--device-id", default="0")
    parser.add_argument("--n-j", type=int, required=True, help="Training jobs, for example 20")
    parser.add_argument("--n-m", type=int, required=True, help="Training machines, for example 5")
    parser.add_argument("--data-suffix", default="mix")
    parser.add_argument("--max-updates", type=int, default=200)
    parser.add_argument("--num-envs", type=int, default=20)
    parser.add_argument("--validate-timestep", type=int, default=10)
    parser.add_argument("--reset-env-timestep", type=int, default=20)
    parser.add_argument("--urgent-jobs", type=int, default=2)
    parser.add_argument("--priority-weight", type=float, default=3.0)
    parser.add_argument("--priority-reward-weight", type=float, default=0.5)
    parser.add_argument("--priority-seed", type=int, default=50)
    parser.add_argument(
        "--test-data",
        nargs="+",
        default=["10x5+mix", "20x5+mix", "15x10+mix", "20x10+mix"],
        help="Datasets for post-fine-tune priority evaluation",
    )
    parser.add_argument("--eval-limit", type=int, default=0, help="0 means full evaluation")
    parser.add_argument("--out-csv", default="", help="Optional evaluation CSV path")
    return parser.parse_args()


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def show_command(cmd: list[str], cwd: Path) -> None:
    print(f"\n# cwd: {cwd}")
    print(" ".join(f'"{part}"' if " " in part else part for part in cmd))


def run_or_print(cmd: list[str], cwd: Path, execute: bool) -> None:
    show_command(cmd, cwd)
    if execute:
        subprocess.run(cmd, cwd=str(cwd), check=True)


def str_to_suffix(value: str) -> str:
    return "" if value == "" else f"+{value}"


def main() -> None:
    args = parse_args()
    root = repo_root()
    daniel_dir = root / "daniel"
    py = sys.executable

    merge_cmd = [
        py,
        str(root / "tools" / "merge_daniel_checkpoints.py"),
        "--method",
        args.method,
        "--data-source",
        args.data_source,
        "--checkpoints",
        *args.checkpoints,
        "--out-name",
        args.merged_name,
    ]
    if args.weights:
        merge_cmd.extend(["--weights", *[str(weight) for weight in args.weights]])
    if args.base:
        merge_cmd.extend(["--base", args.base])

    train_cmd = [
        py,
        "train.py",
        "--device",
        args.device,
        "--device_id",
        args.device_id,
        "--data_source",
        args.data_source,
        "--data_suffix",
        args.data_suffix,
        "--n_j",
        str(args.n_j),
        "--n_m",
        str(args.n_m),
        "--max_updates",
        str(args.max_updates),
        "--num_envs",
        str(args.num_envs),
        "--validate_timestep",
        str(args.validate_timestep),
        "--reset_env_timestep",
        str(args.reset_env_timestep),
        "--model_suffix",
        args.fine_tune_suffix,
        "--init_from",
        args.merged_name,
        "--enable_priority",
        "True",
        "--urgent_jobs",
        str(args.urgent_jobs),
        "--priority_weight",
        str(args.priority_weight),
        "--priority_reward_weight",
        str(args.priority_reward_weight),
        "--priority_seed",
        str(args.priority_seed),
    ]

    fine_tuned_model = f"{args.n_j}x{args.n_m}{str_to_suffix(args.data_suffix)}{str_to_suffix(args.fine_tune_suffix)}"
    eval_cmd = [
        py,
        str(root / "tools" / "eval_priority_flexibility.py"),
        "--device",
        args.device,
        "--data-source",
        args.data_source,
        "--models",
        fine_tuned_model,
        "--test-data",
        *args.test_data,
        "--limit",
        str(args.eval_limit),
        "--urgent-jobs",
        str(args.urgent_jobs),
        "--priority-weight",
        str(args.priority_weight),
        "--priority-model",
    ]
    if args.out_csv:
        eval_cmd.extend(["--out", args.out_csv])

    print("Merge-then-fine-tune experiment")
    print("Dry run only. Add --execute to run these commands." if not args.execute else "Executing commands.")
    run_or_print(merge_cmd, root, args.execute)
    run_or_print(train_cmd, daniel_dir, args.execute)
    run_or_print(eval_cmd, root, args.execute)


if __name__ == "__main__":
    main()
