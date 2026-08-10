"""Quick cross-size DANIEL evaluation for the flexibility-model baseline.

This intentionally runs greedy evaluation once and can limit the number of
instances per (checkpoint, dataset) pair so CPU laptops can produce a first
result table quickly.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run quick DANIEL cross-size flexibility evaluation")
    parser.add_argument("--device", default="cpu", help="Torch device, e.g. cpu or cuda")
    parser.add_argument("--data-source", default="SD2", help="DANIEL data source folder")
    parser.add_argument("--enable-priority", default="False", help="Pass-through params.py priority flag")
    parser.add_argument("--enable-carbon", default="False", help="Pass-through params.py carbon flag")
    parser.add_argument("--carbon-feature", default="False", help="Pass-through params.py carbon feature flag")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["10x5+mix", "20x5+mix", "15x10+mix", "20x10+mix"],
        help="Checkpoint names under daniel/trained_network/<data-source>",
    )
    parser.add_argument(
        "--test-data",
        nargs="+",
        default=["10x5+mix", "20x5+mix", "15x10+mix", "20x10+mix", "30x10+mix", "40x10+mix"],
        help="Dataset names under daniel/data/<data-source>",
    )
    parser.add_argument("--limit", type=int, default=10, help="Instances per pair; use 0 for all instances")
    parser.add_argument(
        "--out",
        default="experiments/flexibility/quick_flex_eval.csv",
        help="Output CSV path relative to repo root",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    daniel_dir = repo_root / "daniel"
    out_path = repo_root / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # DANIEL modules use params.py at import time, so set its argv before import.
    os.chdir(daniel_dir)
    sys.path.insert(0, str(daniel_dir))
    sys.argv = [
        "quick_flex_eval",
        "--device",
        args.device,
        "--data_source",
        args.data_source,
        "--enable_priority",
        args.enable_priority,
        "--enable_carbon",
        args.enable_carbon,
        "--carbon_feature",
        args.carbon_feature,
        "--fea_j_input_dim",
        "10" if args.enable_priority.lower() == "false" and args.enable_carbon.lower() == "false" else "11",
        "--fea_pair_input_dim",
        "9" if args.enable_carbon.lower() == "true" and args.carbon_feature.lower() == "true" else "8",
    ]

    import numpy as np
    import torch

    from common_utils import greedy_select_action, setup_seed
    from data_utils import pack_data_from_config
    from fjsp_env_same_op_nums import FJSPEnvForSameOpNums
    from model.PPO import PPO_initialize

    setup_seed(50)
    ppo = PPO_initialize()
    rows: list[dict[str, object]] = []

    for model_name in args.models:
        model_path = daniel_dir / "trained_network" / args.data_source / f"{model_name}.pth"
        if not model_path.exists():
            print(f"SKIP missing model: {model_path}")
            continue

        ppo.policy.load_state_dict(torch.load(model_path, map_location=args.device))
        ppo.policy.eval()

        for data_name in args.test_data:
            data_items = pack_data_from_config(args.data_source, [data_name])
            (job_lengths, op_pts), _ = data_items[0]
            total = len(job_lengths) if args.limit == 0 else min(args.limit, len(job_lengths))
            if total == 0:
                continue

            n_j = job_lengths[0].shape[0]
            _, n_m = op_pts[0].shape
            env = FJSPEnvForSameOpNums(n_j=n_j, n_m=n_m)
            makespans = []
            times = []
            print(f"Evaluating model={model_name} data={data_name} instances={total}")

            for idx in range(total):
                state = env.set_initial_data([job_lengths[idx]], [op_pts[idx]])
                start = time.time()
                while True:
                    with torch.no_grad():
                        pi, _ = ppo.policy(
                            fea_j=state.fea_j_tensor,
                            op_mask=state.op_mask_tensor,
                            candidate=state.candidate_tensor,
                            fea_m=state.fea_m_tensor,
                            mch_mask=state.mch_mask_tensor,
                            comp_idx=state.comp_idx_tensor,
                            dynamic_pair_mask=state.dynamic_pair_mask_tensor,
                            fea_pairs=state.fea_pairs_tensor,
                        )
                    action = greedy_select_action(pi)
                    state, _, done = env.step(actions=action.cpu().numpy())
                    if done:
                        break
                times.append(time.time() - start)
                makespans.append(float(env.current_makespan[0]))

            rows.append(
                {
                    "model": model_name,
                    "test_data": data_name,
                    "instances": total,
                    "mean_makespan": round(float(np.mean(makespans)), 4),
                    "mean_seconds": round(float(np.mean(times)), 4),
                }
            )

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "test_data", "instances", "mean_makespan", "mean_seconds"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
