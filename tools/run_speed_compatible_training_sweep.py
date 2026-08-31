"""Plan or run merge-compatible speed/makespan DANIEL training.

These models are trained for inherited makespan minimization with the canonical
F11/P9 observation schema shared by all new specialists.

Default behavior is a dry run. Add --execute on the GPU PC.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


DEFAULT_TRAIN_SIZES = ["10x5", "20x5", "15x10", "20x10"]
DEFAULT_TEST_DATA = ["10x5+carbon+priority", "20x5+carbon+priority",
                     "15x10+carbon+priority", "20x10+carbon+priority"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run merge-compatible speed model training/evaluation sweep")
    parser.add_argument("--execute", action="store_true", help="Actually run commands")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--device-id", default="0")
    parser.add_argument("--sizes", nargs="+", default=DEFAULT_TRAIN_SIZES)
    parser.add_argument("--seeds", nargs="+", type=int, default=[300, 301, 302, 303, 304])
    parser.add_argument("--validation-data", nargs="+", default=DEFAULT_TEST_DATA)
    parser.add_argument("--data-root", default="daniel/data")
    parser.add_argument("--max-updates", type=int, default=1000)
    parser.add_argument("--num-envs", type=int, default=20)
    parser.add_argument("--validate-timestep", type=int, default=10)
    parser.add_argument("--reset-env-timestep", type=int, default=20)
    parser.add_argument(
        "--model-tag",
        default="speed_canonical",
        help="Suffix tag for the canonical pure-makespan specialist.",
    )
    parser.add_argument("--eval-limit", type=int, default=0)
    parser.add_argument("--out", default="experiments/flexibility/speed_compatible_eval.csv")
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
    data_root = (root / args.data_root).resolve()
    py = sys.executable

    print("Merge-compatible speed/makespan training sweep")
    print("Dry run only. Add --execute to run these commands." if not args.execute else "Executing commands.")
    print("Mode: canonical F11/P9 observations, pure makespan reward.")

    trained_models = []
    for size in args.sizes:
        n_j, n_m = parse_size(size)
        data_name = f"{size}+carbon+priority"
        train_dir = data_root / "data_train" / "SD2" / data_name
        vali_dir = data_root / "data_validation" / "SD2" / data_name
        if not train_dir.exists() or not vali_dir.exists():
            print(f"\nSKIP training {size}: missing separate train/validation folders")
            continue

        for seed in args.seeds:
            model_tag = f"{args.model_tag}_s{seed}"
            train_cmd = [
            py, "train.py",
            "--device",
            args.device,
            "--device_id",
            args.device_id,
            "--data_source",
            "SD2",
            "--data_suffix",
            "carbon+priority",
            "--train_data_path",
            str(train_dir),
            "--validation_data_path",
            str(vali_dir),
            "--n_j",
            str(n_j),
            "--n_m",
            str(n_m),
            "--enable_priority",
            "True",
            "--enable_carbon",
            "True",
            "--carbon_feature",
            "True",
            "--fea_pair_input_dim",
            "9",
            "--feature_schema",
            "canonical_f11_p9_v2",
            "--goal",
            "m",
            "--max_updates",
            str(args.max_updates),
            "--num_envs",
            str(args.num_envs),
            "--validate_timestep",
            str(args.validate_timestep),
            "--reset_env_timestep",
            str(args.reset_env_timestep),
            "--seed_train",
            str(seed),
            "--model_suffix",
            model_tag,
            ]
            run_or_print(train_cmd, daniel_dir, args.execute)
            trained_models.append(model_name(size, model_tag))

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
            *args.validation_data,
            "--data-root",
            str(data_root / "data_validation"),
            "--enable-carbon",
            "--carbon-feature",
            "--limit",
            str(args.eval_limit),
            "--out",
            args.out,
        ]
        run_or_print(eval_cmd, root, args.execute)
    else:
        print("No trainable sizes were found.")


if __name__ == "__main__":
    main()
