"""Plan or run carbon-aware DANIEL training across SD2 sizes.

Default behavior is a dry run. Add --execute on the GPU PC.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


DEFAULT_TRAIN_SIZES = ["20x5", "15x10", "20x10"]
DEFAULT_TEST_DATA = [
    "20x5+carbon+priority",
    "15x10+carbon+priority",
    "20x10+carbon+priority",
    "30x10+carbon+priority",
    "40x10+carbon+priority",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run carbon model training/evaluation sweep")
    parser.add_argument("--execute", action="store_true", help="Actually run commands")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--device-id", default="0")
    parser.add_argument("--sizes", nargs="+", default=DEFAULT_TRAIN_SIZES)
    parser.add_argument("--test-data", nargs="+", default=DEFAULT_TEST_DATA)
    parser.add_argument("--max-updates", type=int, default=1000)
    parser.add_argument("--num-envs", type=int, default=20)
    parser.add_argument("--validate-timestep", type=int, default=10)
    parser.add_argument("--reset-env-timestep", type=int, default=20)
    parser.add_argument("--carbon-reward-weight", type=float, default=0.01)
    parser.add_argument(
        "--carbon-feature",
        action="store_true",
        help="A-style carbon model: carbon is included as an input feature. Not merge-compatible with current job-priority checkpoints.",
    )
    parser.add_argument(
        "--model-tag",
        default="carbon_reward",
        help="Suffix tag. The size is added by train.py, so this becomes <size>+carbon+priority+<tag>.",
    )
    parser.add_argument("--eval-limit", type=int, default=0)
    parser.add_argument("--out", default="experiments/flexibility/carbon_model_sweep_eval.csv")
    return parser.parse_args()


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def parse_size(size: str) -> tuple[int, int]:
    jobs, machines = size.lower().split("x")
    return int(jobs), int(machines)


def show_command(cmd: list[str], cwd: Path) -> None:
    print(f"\n# cwd: {cwd}")
    print(" ".join(f'"{part}"' if " " in part else part for part in cmd))


def run_or_print(cmd: list[str], cwd: Path, execute: bool) -> None:
    show_command(cmd, cwd)
    if execute:
        subprocess.run(cmd, cwd=str(cwd), check=True)


def model_name(size: str, tag: str) -> str:
    return f"{size}+carbon+priority+{tag}"


def main() -> None:
    args = parse_args()
    root = repo_root()
    daniel_dir = root / "daniel"
    py = sys.executable

    print("Carbon model training sweep")
    print("Dry run only. Add --execute to run these commands." if not args.execute else "Executing commands.")
    if args.carbon_feature:
        print("Mode: A-style carbon input model. This changes checkpoint shape.")
    else:
        print("Mode: merge-compatible carbon reward model. Carbon is used in reward, not as an extra model input.")

    trained_models = []
    for size in args.sizes:
        n_j, n_m = parse_size(size)
        data_name = f"{size}+carbon+priority"
        vali_dir = daniel_dir / "data" / "data_train_vali" / "SD2" / data_name
        if not vali_dir.exists():
            print(f"\nSKIP training {size}: missing validation folder {vali_dir}")
            continue

        train_cmd = [
            py,
            "train.py",
            "--device",
            args.device,
            "--device_id",
            args.device_id,
            "--data_source",
            "SD2",
            "--data_suffix",
            "carbon+priority",
            "--n_j",
            str(n_j),
            "--n_m",
            str(n_m),
            "--train_from_files",
            "True",
            "--enable_priority",
            "True",
            "--enable_carbon",
            "True",
            "--carbon_feature",
            "True" if args.carbon_feature else "False",
            "--goal",
            "c",
            "--carbon_reward_weight",
            str(args.carbon_reward_weight),
            "--max_updates",
            str(args.max_updates),
            "--num_envs",
            str(args.num_envs),
            "--validate_timestep",
            str(args.validate_timestep),
            "--reset_env_timestep",
            str(args.reset_env_timestep),
            "--model_suffix",
            args.model_tag,
        ]
        run_or_print(train_cmd, daniel_dir, args.execute)
        trained_models.append(model_name(size, args.model_tag))

    if trained_models:
        eval_cmd = [
            py,
            str(root / "tools" / "eval_carbon_models.py"),
            "--device",
            args.device,
            "--data-source",
            "SD2",
            "--models",
            *trained_models,
            "--test-data",
            *args.test_data,
            "--limit",
            str(args.eval_limit),
            "--enable-carbon",
            "--out",
            args.out,
        ]
        if args.carbon_feature:
            eval_cmd.append("--carbon-feature")
        run_or_print(eval_cmd, root, args.execute)
    else:
        print("No trainable sizes were found.")


if __name__ == "__main__":
    main()
