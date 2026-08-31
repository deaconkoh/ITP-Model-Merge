"""Plan or run canonical F11/P9 operation-priority specialist training."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Canonical operation-priority multi-seed sweep")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--device-id", default="0")
    parser.add_argument("--data-root", default="daniel/data")
    parser.add_argument("--sizes", nargs="+", default=["20x5", "15x10", "20x10"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[300, 301, 302, 303, 304])
    parser.add_argument("--priority-reward-weight", type=float, default=1.0)
    parser.add_argument("--max-updates", type=int, default=1000)
    parser.add_argument("--num-envs", type=int, default=20)
    parser.add_argument("--validate-timestep", type=int, default=10)
    parser.add_argument("--reset-env-timestep", type=int, default=20)
    parser.add_argument("--eval-limit", type=int, default=0)
    parser.add_argument("--out", default="experiments/canonical/operation_priority_validation.csv")
    return parser.parse_args()


def run_or_print(command: list[str], cwd: Path, execute: bool) -> None:
    print(f"\n# cwd: {cwd}")
    print(" ".join(f'"{part}"' if " " in part else part for part in command))
    if execute:
        subprocess.run(command, cwd=cwd, check=True)


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    daniel_dir = root / "daniel"
    data_root = (root / args.data_root).resolve()
    python = sys.executable
    models = []
    validation_data = []

    for size in args.sizes:
        jobs, machines = (int(value) for value in size.split("x"))
        data_name = f"{size}+carbon+priority"
        train_path = data_root / "data_train" / "SD2" / data_name
        validation_path = data_root / "data_validation" / "SD2" / data_name
        if not train_path.is_dir() or not validation_path.is_dir():
            print(f"SKIP {size}: missing separate train/validation directories")
            continue
        validation_data.append(data_name)
        for seed in args.seeds:
            suffix = f"operation_priority_s{seed}"
            model = f"{data_name}+{suffix}"
            command = [
                python, "train.py",
                "--config", str(root / "configs" / "canonical" / "p.json"),
                "--device", args.device,
                "--device_id", args.device_id,
                "--data_source", "SD2",
                "--data_suffix", "carbon+priority",
                "--train_data_path", str(train_path),
                "--validation_data_path", str(validation_path),
                "--n_j", str(jobs), "--n_m", str(machines),
                "--seed_train", str(seed),
                "--priority_reward_weight", str(args.priority_reward_weight),
                "--max_updates", str(args.max_updates),
                "--num_envs", str(args.num_envs),
                "--validate_timestep", str(args.validate_timestep),
                "--reset_env_timestep", str(args.reset_env_timestep),
                "--model_suffix", suffix,
            ]
            run_or_print(command, daniel_dir, args.execute)
            models.append(model)

    if models:
        command = [
            python, str(root / "tools" / "eval_operation_priority.py"),
            "--device", args.device,
            "--data-source", "SD2",
            "--data-root", str(data_root / "data_validation"),
            "--models", *models,
            "--test-data", *sorted(set(validation_data)),
            "--limit", str(args.eval_limit),
            "--priority-model",
            "--feature-schema", "canonical_f11_p9_v2",
            "--out", args.out,
        ]
        run_or_print(command, root, args.execute)


if __name__ == "__main__":
    main()
