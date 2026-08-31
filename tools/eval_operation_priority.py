"""Evaluate DANIEL on operation-level priority datasets.

The carbon+priority .fjs files add one priority score to every operation.
This script compares how well a checkpoint handles those operation priorities.
It intentionally evaluates carbon values only as loaded data for now; the
current metric focus is priority-weighted operation completion.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate operation-level priority DANIEL")
    parser.add_argument("--device", default="cpu", help="Torch device, e.g. cpu or cuda")
    parser.add_argument("--data-source", default="SD2", help="DANIEL data/checkpoint source")
    parser.add_argument("--data-root", default="../data", help="Data root as seen from the daniel/ directory")
    parser.add_argument("--models", nargs="+", default=["10x5+mix"], help="Checkpoint names")
    parser.add_argument("--test-data", nargs="+", default=["10x5+carbon+priority"], help="Extended dataset names")
    parser.add_argument("--limit", type=int, default=0, help="Instances per dataset; 0 means all")
    parser.add_argument("--priority-model", action="store_true", help="Use 11-feature priority-aware checkpoints")
    parser.add_argument("--feature-schema", default="canonical_f11_p9_v2")
    parser.add_argument("--final-evaluation-manifest",
                        help="Frozen manifest required when these results are used as final-test evidence")
    parser.add_argument("--high-priority-threshold", type=float, default=80.0)
    parser.add_argument("--out", default="experiments/flexibility/operation_priority_eval.csv")
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


def run_schedule(policy, env, job_length, op_pt, op_priority, op_carbon, args, torch):
    if args.priority_model:
        state = env.set_initial_data([job_length], [op_pt], [op_priority], [op_carbon])
    else:
        state = env.set_initial_data([job_length], [op_pt])

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
    high_mask = priorities >= args.high_priority_threshold
    low_mask = ~high_mask
    high_completion = float(np.mean(op_completion[high_mask])) if np.any(high_mask) else float("nan")
    low_completion = float(np.mean(op_completion[low_mask])) if np.any(low_mask) else float("nan")

    return {
        "makespan": float(env.current_makespan[0]),
        "priority_weighted_completion": weighted_completion,
        "high_priority_mean_completion": high_completion,
        "low_priority_mean_completion": low_completion,
        "high_priority_operation_count": int(np.sum(high_mask)),
        "runtime_seconds": time.time() - start,
    }


def mean_metric(rows: list[dict[str, float]], metric: str):
    import numpy as np

    values = [row[metric] for row in rows]
    return float(np.nanmean(values)) if values else float("nan")


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    daniel_dir = repo_root / "daniel"
    out_path = repo_root / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    os.chdir(daniel_dir)
    sys.path.insert(0, str(daniel_dir))
    sys.argv = [
        "eval_operation_priority",
        "--device",
        args.device,
        "--data_source",
        args.data_source,
        "--data_root",
        args.data_root,
        "--feature_schema",
        args.feature_schema,
        "--fea_pair_input_dim",
        "8" if args.feature_schema == "legacy_f11_p8_v1" else "9",
    ]
    if args.priority_model:
        sys.argv.extend(["--enable_priority", "True"])

    import torch

    from data_utils import load_priority_carbon_data_from_files
    from fjsp_env_same_op_nums import FJSPEnvForSameOpNums
    from model.main_model import DANIEL
    from params import configs
    from feature_schemas import validate_config_against_schema
    validate_config_against_schema(configs)

    action_mode = "operation_priority_aware" if args.priority_model else "normal"
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
            results = []
            print(f"Evaluating operation priorities model={model_name} data={data_name} instances={total}")

            for idx in range(total):
                results.append(
                    run_schedule(
                        policy=policy,
                        env=env,
                        job_length=job_lengths[idx],
                        op_pt=op_pts[idx],
                        op_priority=op_priorities[idx],
                        op_carbon=op_carbons[idx],
                        args=args,
                        torch=torch,
                    )
                )

            rows.append(
                {
                    "model": model_name,
                    "test_data": data_name,
                    "action_mode": action_mode,
                    "instances": total,
                    "high_priority_threshold": args.high_priority_threshold,
                    "mean_makespan": round(mean_metric(results, "makespan"), 4),
                    "mean_priority_weighted_completion": round(
                        mean_metric(results, "priority_weighted_completion"), 4
                    ),
                    "mean_high_priority_completion": round(
                        mean_metric(results, "high_priority_mean_completion"), 4
                    ),
                    "mean_low_priority_completion": round(
                        mean_metric(results, "low_priority_mean_completion"), 4
                    ),
                    "mean_high_priority_operation_count": round(
                        mean_metric(results, "high_priority_operation_count"), 4
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
                "action_mode",
                "instances",
                "high_priority_threshold",
                "mean_makespan",
                "mean_priority_weighted_completion",
                "mean_high_priority_completion",
                "mean_low_priority_completion",
                "mean_high_priority_operation_count",
                "mean_runtime_seconds",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
