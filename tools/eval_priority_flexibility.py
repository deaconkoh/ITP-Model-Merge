"""Evaluate DANIEL under fixed job-priority scenarios.

This is the first baseline for priority-flexibility:

1. Mark some jobs as urgent from the beginning of scheduling.
2. Run normal greedy DANIEL.
3. Run a priority-biased DANIEL decision rule.
4. Compare makespan, urgent-job completion time, and weighted completion.

The priority-biased rule does not train a new model. It keeps DANIEL unchanged
and adds a small score boost to urgent jobs before selecting the greedy action.
This gives us a simple baseline before changing neural network inputs/rewards.
With --priority-model, the same script also evaluates an 11-feature checkpoint
that was trained with job-priority information in the environment state.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate DANIEL under fixed job priorities")
    parser.add_argument("--device", default="cpu", help="Torch device, e.g. cpu or cuda")
    parser.add_argument("--data-source", default="SD2", help="DANIEL data/checkpoint source, e.g. SD1 or SD2")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["10x5+mix", "20x10+mix"],
        help="Checkpoint names under daniel/trained_network/<data-source>",
    )
    parser.add_argument(
        "--test-data",
        nargs="+",
        default=["10x5+mix", "20x10+mix"],
        help="Dataset names under daniel/data/<data-source>",
    )
    parser.add_argument("--limit", type=int, default=10, help="Instances per dataset; use 0 for all instances")
    parser.add_argument("--urgent-jobs", type=int, default=1, help="Number of urgent jobs per instance")
    parser.add_argument("--priority-weight", type=float, default=3.0, help="Weight assigned to urgent jobs")
    parser.add_argument(
        "--priority-model",
        action="store_true",
        help="Evaluate checkpoints trained with --enable_priority True and 11 operation features",
    )
    parser.add_argument(
        "--priority-boost",
        type=float,
        default=1.0,
        help="Multiplier for the log-priority score boost used by the biased baseline",
    )
    parser.add_argument("--seed", type=int, default=50, help="Random seed for selecting urgent jobs")
    parser.add_argument(
        "--out",
        default="experiments/flexibility/priority_flex_eval.csv",
        help="Output CSV path relative to repo root",
    )
    return parser.parse_args()


def choose_urgent_jobs(num_jobs: int, urgent_jobs: int, rng):
    urgent_count = min(max(urgent_jobs, 1), num_jobs)
    return sorted(rng.choice(num_jobs, size=urgent_count, replace=False).tolist())


def priority_vector(num_jobs: int, urgent_job_ids: list[int], priority_weight: float):
    import numpy as np

    priorities = np.ones(num_jobs, dtype=float)
    priorities[urgent_job_ids] = priority_weight
    return priorities


def completion_metrics(env, priorities, urgent_job_ids: list[int]) -> dict[str, float]:
    import numpy as np

    job_completion_times = env.true_candidate_free_time[0].astype(float)
    weighted_completion = float(np.sum(job_completion_times * priorities) / np.sum(priorities))
    urgent_completion = float(np.mean(job_completion_times[urgent_job_ids]))
    normal_job_ids = [idx for idx in range(len(job_completion_times)) if idx not in urgent_job_ids]
    normal_completion = float(np.mean(job_completion_times[normal_job_ids])) if normal_job_ids else 0.0
    return {
        "makespan": float(env.current_makespan[0]),
        "weighted_completion": weighted_completion,
        "urgent_mean_completion": urgent_completion,
        "normal_mean_completion": normal_completion,
    }


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


def priority_biased_action(policy, state, priorities, priority_boost: float, torch):
    with torch.no_grad():
        logits, _, _ = policy.forward_logits(
            fea_j=state.fea_j_tensor,
            op_mask=state.op_mask_tensor,
            candidate=state.candidate_tensor,
            fea_m=state.fea_m_tensor,
            mch_mask=state.mch_mask_tensor,
            comp_idx=state.comp_idx_tensor,
            dynamic_pair_mask=state.dynamic_pair_mask_tensor,
            fea_pairs=state.fea_pairs_tensor,
        )

    _, num_actions = logits.shape
    num_jobs = len(priorities)
    num_machines = num_actions // num_jobs
    priority_bonus = torch.log(torch.tensor(priorities, dtype=logits.dtype, device=logits.device))
    priority_bonus = priority_bonus.repeat_interleave(num_machines).unsqueeze(0)
    biased_logits = logits + priority_boost * priority_bonus
    return torch.argmax(biased_logits, dim=1)


def run_schedule(policy, env, job_length, op_pt, priorities, urgent_job_ids, action_mode, args, torch):
    if args.priority_model:
        state = env.set_initial_data([job_length], [op_pt], [priorities])
    else:
        state = env.set_initial_data([job_length], [op_pt])
    start = time.time()

    while True:
        if action_mode in ("normal", "priority_aware"):
            action = normal_greedy_action(policy, state, torch)
        elif action_mode == "priority_biased":
            action = priority_biased_action(policy, state, priorities, args.priority_boost, torch)
        else:
            raise ValueError(f"Unknown action mode: {action_mode}")

        state, _, done = env.step(actions=action.cpu().numpy())
        if done:
            break

    metrics = completion_metrics(env, priorities, urgent_job_ids)
    metrics["runtime_seconds"] = time.time() - start
    return metrics


def mean_metric(rows: list[dict[str, float]], metric: str):
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
    sys.argv = ["eval_priority_flexibility", "--device", args.device, "--data_source", args.data_source]
    if args.priority_model:
        sys.argv.extend([
            "--enable_priority", "True",
            "--enable_carbon", "True",
            "--carbon_feature", "False",
            "--fea_pair_input_dim", "8",
            "--feature_schema", "legacy_f11_p8_v1",
            "--priority_scope", "legacy_job",
            "--urgent_jobs", str(args.urgent_jobs),
            "--priority_weight", str(args.priority_weight),
        ])
    else:
        sys.argv.extend([
            "--enable_priority", "False",
            "--enable_carbon", "False",
            "--carbon_feature", "False",
            "--fea_j_input_dim", "10",
            "--fea_pair_input_dim", "8",
            "--feature_schema", "legacy_f10_p8_v1",
        ])

    import numpy as np
    import torch

    from common_utils import setup_seed
    from data_utils import pack_data_from_config
    from fjsp_env_same_op_nums import FJSPEnvForSameOpNums
    from model.main_model import DANIEL
    from params import configs
    from feature_schemas import validate_config_against_schema
    validate_config_against_schema(configs)

    setup_seed(args.seed)
    rng = np.random.default_rng(args.seed)
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

            if args.priority_model:
                mode_results = {"priority_aware": []}
            else:
                mode_results = {"normal": [], "priority_biased": []}
            print(f"Evaluating priorities model={model_name} data={data_name} instances={total}")

            for idx in range(total):
                job_length = job_lengths[idx]
                op_pt = op_pts[idx]
                urgent_job_ids = choose_urgent_jobs(n_j, args.urgent_jobs, rng)
                priorities = priority_vector(n_j, urgent_job_ids, args.priority_weight)

                for action_mode in mode_results:
                    result = run_schedule(
                        policy=policy,
                        env=env,
                        job_length=job_length,
                        op_pt=op_pt,
                        priorities=priorities,
                        urgent_job_ids=urgent_job_ids,
                        action_mode=action_mode,
                        args=args,
                        torch=torch,
                    )
                    mode_results[action_mode].append(result)

            for action_mode, results in mode_results.items():
                rows.append(
                    {
                        "model": model_name,
                        "test_data": data_name,
                        "action_mode": action_mode,
                        "instances": total,
                        "urgent_jobs_per_instance": min(max(args.urgent_jobs, 1), n_j),
                        "priority_weight": args.priority_weight,
                        "priority_boost": args.priority_boost if action_mode == "priority_biased" else 0.0,
                        "mean_makespan": round(mean_metric(results, "makespan"), 4),
                        "mean_weighted_completion": round(mean_metric(results, "weighted_completion"), 4),
                        "mean_urgent_completion": round(mean_metric(results, "urgent_mean_completion"), 4),
                        "mean_normal_job_completion": round(mean_metric(results, "normal_mean_completion"), 4),
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
                "urgent_jobs_per_instance",
                "priority_weight",
                "priority_boost",
                "mean_makespan",
                "mean_weighted_completion",
                "mean_urgent_completion",
                "mean_normal_job_completion",
                "mean_runtime_seconds",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
