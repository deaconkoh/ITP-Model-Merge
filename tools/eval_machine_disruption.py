"""Evaluate DANIEL under simple machine-unavailability disruptions.

This is the first baseline for the disruption-flexibility task:

1. Run a normal greedy schedule.
2. Remove one machine from the instance by marking all operations on that
   machine as incompatible.
3. Run greedy scheduling again.
4. Measure the robustness gap: disrupted makespan - normal makespan.

This models "machine unavailable from the start". It does not yet model a
machine breaking halfway through an already-started schedule.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate DANIEL under machine-unavailability disruption")
    parser.add_argument("--device", default="cpu", help="Torch device, e.g. cpu or cuda")
    parser.add_argument("--data-source", default="SD2", help="DANIEL data/checkpoint source, e.g. SD1 or SD2")
    parser.add_argument("--enable-priority", default="False", help="Pass-through params.py priority flag")
    parser.add_argument("--enable-carbon", default="False", help="Pass-through params.py carbon flag")
    parser.add_argument("--carbon-feature", default="False", help="Pass-through params.py carbon feature flag")
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
    parser.add_argument(
        "--max-machines",
        type=int,
        default=0,
        help="Max machine removals to test per instance; 0 means test every single-machine removal",
    )
    parser.add_argument(
        "--out",
        default="experiments/flexibility/machine_disruption_eval.csv",
        help="Output CSV path relative to repo root",
    )
    return parser.parse_args()


def is_feasible_after_removal(op_pt, removed_machine: int) -> bool:
    disrupted = op_pt.copy()
    disrupted[:, removed_machine] = 0
    return bool((disrupted > 0).any(axis=1).all())


def remove_machine(op_pt, removed_machine: int):
    disrupted = op_pt.copy()
    disrupted[:, removed_machine] = 0
    return disrupted


def run_greedy(policy, env, job_length, op_pt, torch, greedy_select_action) -> tuple[float, float]:
    state = env.set_initial_data([job_length], [op_pt])
    start = time.time()
    while True:
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
        action = greedy_select_action(pi)
        state, _, done = env.step(actions=action.cpu().numpy())
        if done:
            break
    return float(env.current_makespan[0]), time.time() - start


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    daniel_dir = repo_root / "daniel"
    out_path = repo_root / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    os.chdir(daniel_dir)
    sys.path.insert(0, str(daniel_dir))
    sys.argv = [
        "eval_machine_disruption",
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
        "--feature_schema",
        ("canonical_f11_p9_v2" if args.enable_carbon.lower() == "true" and args.carbon_feature.lower() == "true"
         else "legacy_f11_p8_v1" if args.enable_priority.lower() == "true" or args.enable_carbon.lower() == "true"
         else "legacy_f10_p8_v1"),
    ]

    import numpy as np
    import torch

    from common_utils import greedy_select_action, setup_seed
    from data_utils import pack_data_from_config
    from fjsp_env_same_op_nums import FJSPEnvForSameOpNums
    from model.main_model import DANIEL
    from params import configs
    from feature_schemas import validate_config_against_schema
    validate_config_against_schema(configs)

    setup_seed(50)
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

            normal_makespans = []
            disrupted_makespans = []
            skipped_removals = 0
            tested_removals = 0
            print(f"Evaluating disruptions model={model_name} data={data_name} instances={total}")

            for idx in range(total):
                job_length = job_lengths[idx]
                op_pt = op_pts[idx]
                normal_makespan, _ = run_greedy(policy, env, job_length, op_pt, torch, greedy_select_action)
                normal_makespans.append(normal_makespan)

                machine_ids = list(range(n_m))
                if args.max_machines > 0:
                    machine_ids = machine_ids[: args.max_machines]

                for removed_machine in machine_ids:
                    if not is_feasible_after_removal(op_pt, removed_machine):
                        skipped_removals += 1
                        continue
                    disrupted_pt = remove_machine(op_pt, removed_machine)
                    disrupted_makespan, _ = run_greedy(
                        policy, env, job_length, disrupted_pt, torch, greedy_select_action
                    )
                    disrupted_makespans.append(disrupted_makespan)
                    tested_removals += 1

            normal_mean = float(np.mean(normal_makespans)) if normal_makespans else float("nan")
            disrupted_mean = float(np.mean(disrupted_makespans)) if disrupted_makespans else float("nan")
            rows.append(
                {
                    "model": model_name,
                    "test_data": data_name,
                    "instances": total,
                    "tested_single_machine_removals": tested_removals,
                    "skipped_infeasible_removals": skipped_removals,
                    "normal_mean_makespan": round(normal_mean, 4),
                    "disrupted_mean_makespan": round(disrupted_mean, 4),
                    "robustness_gap": round(disrupted_mean - normal_mean, 4),
                }
            )

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "model",
                "test_data",
                "instances",
                "tested_single_machine_removals",
                "skipped_infeasible_removals",
                "normal_mean_makespan",
                "disrupted_mean_makespan",
                "robustness_gap",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
