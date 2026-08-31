"""Evaluate DANIEL checkpoints on carbon+priority datasets.

This reports three simple outcomes:

* makespan: how late the whole schedule finishes
* total_carbon: total carbon value accumulated by chosen operation-machine pairs
* priority_weighted_completion: whether high-priority operations finish earlier
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate carbon-aware DANIEL checkpoints")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--data-source", default="SD2")
    parser.add_argument("--data-root", default="./data", help="Data root as seen from the daniel/ directory")
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--test-data", nargs="+", required=True)
    parser.add_argument("--limit", type=int, default=0, help="Instances per dataset; 0 means all")
    parser.add_argument("--enable-carbon", action="store_true", help="Use carbon-capable environment/model config")
    parser.add_argument("--carbon-feature", action="store_true", help="Use 9 pair features, matching A-style carbon models")
    parser.add_argument("--feature-schema", default="canonical_f11_p9_v2")
    parser.add_argument("--final-evaluation-manifest",
                        help="Frozen manifest required when these results are used as final-test evidence")
    parser.add_argument("--out", default="experiments/flexibility/carbon_model_eval.csv")
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


def run_schedule(policy, env, job_length, op_pt, op_priority, op_carbon, torch):
    state = env.set_initial_data([job_length], [op_pt], [op_priority], [op_carbon])
    start = time.time()
    while True:
        action = normal_greedy_action(policy, state, torch)
        state, _, done = env.step(actions=action.cpu().numpy())
        if done:
            break

    import numpy as np

    op_completion = env.true_op_ct[0].astype(float)
    priorities = op_priority.astype(float)
    from objectives import operation_priority_weighted_completion
    weighted_completion = float(operation_priority_weighted_completion(op_completion, priorities))
    return {
        "makespan": float(env.current_makespan[0]),
        "total_carbon": float(env.total_carbon[0]),
        "priority_weighted_completion": weighted_completion,
        "runtime_seconds": time.time() - start,
    }


def run_schedule_batch(policy, env, job_lengths, op_pts, op_priorities, op_carbons, torch):
    """Run one same-size dataset batch through the greedy policy."""
    import numpy as np

    state = env.set_initial_data(job_lengths, op_pts, op_priorities, op_carbons)
    start = time.time()
    while True:
        action = normal_greedy_action(policy, state, torch)
        state, _, done = env.step(actions=action.cpu().numpy())
        if np.all(done):
            break

    elapsed_per_instance = (time.time() - start) / len(job_lengths)
    priorities = np.asarray(op_priorities, dtype=float)
    from objectives import operation_priority_weighted_completion
    weighted_completion = operation_priority_weighted_completion(env.true_op_ct, priorities)

    return [
        {
            "makespan": float(env.current_makespan[idx]),
            "total_carbon": float(env.total_carbon[idx]),
            "priority_weighted_completion": float(weighted_completion[idx]),
            "runtime_seconds": elapsed_per_instance,
        }
        for idx in range(len(job_lengths))
    ]


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
        "eval_carbon_models",
        "--device",
        args.device,
        "--data_source",
        args.data_source,
        "--data_root",
        args.data_root,
        "--feature_schema",
        args.feature_schema,
        "--enable_priority",
        "True",
        "--enable_carbon",
        "True" if args.enable_carbon else "False",
        "--carbon_feature",
        "True" if args.carbon_feature else "False",
        "--fea_pair_input_dim",
        "9" if args.carbon_feature else "8",
    ]

    import torch

    from data_utils import load_priority_carbon_data_from_files
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
            data_path = Path(args.data_root) / args.data_source / data_name
            if args.final_evaluation_manifest:
                from evaluation_protocol import verify_frozen_evaluation
                verify_frozen_evaluation(args.final_evaluation_manifest, model_path, data_path)
            job_lengths, op_pts, op_priorities, op_carbons = load_priority_carbon_data_from_files(str(data_path))
            total = len(job_lengths) if args.limit == 0 else min(args.limit, len(job_lengths))
            if total == 0:
                print(f"SKIP no data: {data_path}")
                continue

            n_j = job_lengths[0].shape[0]
            _, n_m = op_pts[0].shape
            env = FJSPEnvForSameOpNums(n_j=n_j, n_m=n_m)
            print(f"Evaluating carbon model={model_name} data={data_name} instances={total}", flush=True)
            results = run_schedule_batch(
                policy=policy,
                env=env,
                job_lengths=job_lengths[:total],
                op_pts=op_pts[:total],
                op_priorities=op_priorities[:total],
                op_carbons=op_carbons[:total],
                torch=torch,
            )

            rows.append(
                {
                    "model": model_name,
                    "test_data": data_name,
                    "instances": total,
                    "carbon_feature": args.carbon_feature,
                    "mean_makespan": round(mean_metric(results, "makespan"), 4),
                    "mean_total_carbon": round(mean_metric(results, "total_carbon"), 4),
                    "mean_priority_weighted_completion": round(
                        mean_metric(results, "priority_weighted_completion"), 4
                    ),
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
                "carbon_feature",
                "mean_makespan",
                "mean_total_carbon",
                "mean_priority_weighted_completion",
                "mean_runtime_seconds",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
