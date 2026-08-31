"""Evaluate merge-compatible speed/makespan DANIEL checkpoints.

The original shipped DANIEL checkpoints use 10 operation features. The
priority/carbon-compatible models use 11 operation features. This evaluator
loads speed models trained with the 11-feature setup and reports makespan on
normal SD2 mix datasets.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate 11-feature speed-compatible DANIEL models")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--data-source", default="SD2")
    parser.add_argument("--data-root", default="./data")
    parser.add_argument(
        "--models",
        nargs="+",
        required=True,
        help="Checkpoint names under daniel/trained_network/<data-source>",
    )
    parser.add_argument(
        "--test-data",
        nargs="+",
        required=True,
        help="Dataset names under daniel/data/<data-source>",
    )
    parser.add_argument("--limit", type=int, default=0, help="Instances per dataset; 0 means all")
    parser.add_argument("--out", default="experiments/flexibility/speed_compatible_eval.csv")
    return parser.parse_args()


def normal_greedy_action(policy, state, torch):
    from common_utils import greedy_select_action

    with torch.no_grad():
        pi, _ = policy(
            fea_j=state.fea_j_tensor,
            op_mask=state.op_mask_tensor,
            candidate=state.candidate_tensor,
            fea_m=state.fea_m_tensor,
            mch_mask=state.mch_mask_tensor,
            comp_idx=state.comp_idx_tensor,
            dynamic_pair_mask=state.dynamic_pair_mask_tensor,
            fea_pairs=state.fea_pairs_tensor,
        )
    return greedy_select_action(pi)


def run_schedule(policy, env, job_length, op_pt, torch) -> dict[str, float]:
    state = env.set_initial_data([job_length], [op_pt])
    start = time.time()

    while True:
        action = normal_greedy_action(policy, state, torch)
        state, _, done = env.step(actions=action.cpu().numpy())
        if done:
            break

    return {
        "makespan": float(env.current_makespan[0]),
        "runtime_seconds": time.time() - start,
    }


def mean_metric(rows: list[dict[str, float]], metric: str) -> float:
    import numpy as np

    return float(np.mean([row[metric] for row in rows])) if rows else float("nan")


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    daniel_dir = repo_root / "daniel"
    out_path = repo_root / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    os.chdir(daniel_dir)
    sys.path.insert(0, str(daniel_dir))
    sys.argv = [
        "eval_speed_compatible_models",
        "--device",
        args.device,
        "--data_source",
        args.data_source,
        "--data_root",
        args.data_root,
        "--enable_carbon",
        "True",
        "--carbon_feature",
        "False",
        "--fea_pair_input_dim",
        "8",
        "--feature_schema",
        "legacy_f11_p8_v1",
        "--goal",
        "m",
    ]

    import torch

    from data_utils import pack_data_from_config
    from fjsp_env_same_op_nums import FJSPEnvForSameOpNums
    from model.main_model import DANIEL
    from params import configs
    from feature_schemas import validate_config_against_schema
    validate_config_against_schema(configs)

    rows: list[dict[str, object]] = []
    for model_name in args.models:
        model_path = daniel_dir / "trained_network" / args.data_source / f"{model_name}.pth"
        if not model_path.exists():
            print(f"SKIP missing model: {model_path}")
            continue

        policy = DANIEL(configs)
        from common_utils import load_checkpoint_state_dict
        policy.load_state_dict(load_checkpoint_state_dict(
            model_path, map_location=args.device, expected_schema=configs.feature_schema))
        policy.to(torch.device(args.device))
        policy.eval()

        for data_name in args.test_data:
            data_items = pack_data_from_config(args.data_source, [data_name])
            (job_lengths, op_pts), _ = data_items[0]
            total = len(job_lengths) if args.limit == 0 else min(args.limit, len(job_lengths))
            if total == 0:
                continue

            n_j = job_lengths[0].shape[0]
            _, n_m = op_pts[0].shape
            env = FJSPEnvForSameOpNums(n_j=n_j, n_m=n_m)
            results = []
            print(f"Evaluating speed-compatible model={model_name} data={data_name} instances={total}")

            for idx in range(total):
                results.append(run_schedule(policy, env, job_lengths[idx], op_pts[idx], torch))

            rows.append(
                {
                    "model": model_name,
                    "test_data": data_name,
                    "instances": total,
                    "mean_makespan": round(mean_metric(results, "makespan"), 4),
                    "mean_runtime_seconds": round(mean_metric(results, "runtime_seconds"), 4),
                }
            )

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "model",
                "test_data",
                "instances",
                "mean_makespan",
                "mean_runtime_seconds",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
