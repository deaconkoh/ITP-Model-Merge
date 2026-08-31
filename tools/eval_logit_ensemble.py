"""Evaluate single DANIEL checkpoints and logit ensembles across datasets.

This supports the "merge models trained on different data sizes" experiment:

1. Evaluate submodel A.
2. Evaluate submodel B.
3. Evaluate an integrated model that averages A/B raw action scores before
   softmax, then chooses the greedy action.

The integrated model is an inference-time ensemble, not a new weight checkpoint.
It is the safest first merging baseline because it does not alter model weights.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate DANIEL single models and logit ensembles")
    parser.add_argument("--device", default="cpu", help="Torch device, e.g. cpu or cuda")
    parser.add_argument("--data-source", default="SD1", help="DANIEL data/checkpoint source, e.g. SD1 or SD2")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["10x5", "20x10"],
        help="Single checkpoint names to evaluate",
    )
    parser.add_argument(
        "--ensembles",
        nargs="+",
        default=["10x5+20x10"],
        help="Ensembles using + between checkpoint names, e.g. 10x5+20x10",
    )
    parser.add_argument(
        "--test-data",
        nargs="+",
        default=["10x5", "20x5", "15x10", "20x10", "30x10", "40x10"],
        help="Dataset names under daniel/data/<data-source>",
    )
    parser.add_argument("--limit", type=int, default=10, help="Instances per pair; use 0 for all instances")
    parser.add_argument(
        "--out",
        default="experiments/flexibility/logit_ensemble_eval.csv",
        help="Output CSV path relative to repo root",
    )
    return parser.parse_args()


def run_policy(policy, state, torch):
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
    return pi


def run_ensemble(policies, state, torch, F):
    logits = []
    with torch.no_grad():
        for policy in policies:
            model_logits, _, _ = policy.forward_logits(
                fea_j=state.fea_j_tensor,
                op_mask=state.op_mask_tensor,
                candidate=state.candidate_tensor,
                fea_m=state.fea_m_tensor,
                mch_mask=state.mch_mask_tensor,
                comp_idx=state.comp_idx_tensor,
                dynamic_pair_mask=state.dynamic_pair_mask_tensor,
                fea_pairs=state.fea_pairs_tensor,
            )
            logits.append(model_logits)
        mean_logits = torch.stack(logits, dim=0).mean(dim=0)
        pi = F.softmax(mean_logits, dim=1)
    return pi


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    daniel_dir = repo_root / "daniel"
    out_path = repo_root / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    os.chdir(daniel_dir)
    sys.path.insert(0, str(daniel_dir))
    sys.argv = ["eval_logit_ensemble", "--device", args.device, "--data_source", args.data_source]

    import numpy as np
    import torch
    import torch.nn.functional as F

    from common_utils import greedy_select_action, setup_seed
    from data_utils import pack_data_from_config
    from fjsp_env_same_op_nums import FJSPEnvForSameOpNums
    from model.main_model import DANIEL
    from params import configs

    setup_seed(50)
    rows: list[dict[str, object]] = []
    policy_cache = {}

    def load_policy(model_name: str):
        if model_name in policy_cache:
            return policy_cache[model_name]
        model_path = daniel_dir / "trained_network" / args.data_source / f"{model_name}.pth"
        if not model_path.exists():
            raise FileNotFoundError(model_path)
        policy = DANIEL(configs)
        from common_utils import load_checkpoint_state_dict
        policy.load_state_dict(load_checkpoint_state_dict(
            model_path, map_location=args.device, expected_schema=configs.feature_schema))
        policy.to(torch.device(args.device))
        policy.eval()
        policy_cache[model_name] = policy
        return policy

    eval_specs: list[tuple[str, list[str]]] = []
    eval_specs.extend((model_name, [model_name]) for model_name in args.models)
    for ensemble_name in args.ensembles:
        eval_specs.append((f"ensemble:{ensemble_name}", ensemble_name.split("+")))

    for label, model_names in eval_specs:
        try:
            policies = [load_policy(model_name) for model_name in model_names]
        except FileNotFoundError as exc:
            print(f"SKIP {label}: missing checkpoint {exc}")
            continue

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
            print(f"Evaluating {label} on {data_name} instances={total}")

            for idx in range(total):
                state = env.set_initial_data([job_lengths[idx]], [op_pts[idx]])
                start = time.time()
                while True:
                    if len(policies) == 1:
                        pi = run_policy(policies[0], state, torch)
                    else:
                        pi = run_ensemble(policies, state, torch, F)
                    action = greedy_select_action(pi)
                    state, _, done = env.step(actions=action.cpu().numpy())
                    if done:
                        break
                times.append(time.time() - start)
                makespans.append(float(env.current_makespan[0]))

            rows.append(
                {
                    "policy": label,
                    "source_models": "+".join(model_names),
                    "test_data": data_name,
                    "instances": total,
                    "mean_makespan": round(float(np.mean(makespans)), 4),
                    "mean_seconds": round(float(np.mean(times)), 4),
                }
            )

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["policy", "source_models", "test_data", "instances", "mean_makespan", "mean_seconds"],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
