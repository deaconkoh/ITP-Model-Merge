r"""Validate merge-compatible carbon model schedules on 20x5+carbon+priority data.

Three checks are run in sequence:

  1. Feasibility    -- verify no machine is double-booked and all ops appear
  2. Carbon choices -- how often does the model pick the min-carbon machine?
  3. Distribution   -- per-instance stats across all instances (not just the mean)

Usage (from repo root):
  .\.venv\Scripts\python.exe tools\validate_carbon_models.py
  .\.venv\Scripts\python.exe tools\validate_carbon_models.py --device cuda
  .\.venv\Scripts\python.exe tools\validate_carbon_models.py --feasibility-limit 5
  .\.venv\Scripts\python.exe tools\validate_carbon_models.py --feasibility-limit 0  # skip checks 1+2

Outputs:
  experiments/flexibility/carbon_validation_feasibility.csv
  experiments/flexibility/carbon_validation_choices.csv
  experiments/flexibility/carbon_validation_distribution.csv
  experiments/flexibility/carbon_validation_distribution_summary.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Validate merge-compatible carbon models")
    p.add_argument("--device", default="cpu")
    p.add_argument("--data-source", default="SD2")
    p.add_argument(
        "--models",
        nargs="+",
        default=[
            "20x10+carbon+priority+carbon_reward",
            "20x5+carbon+priority+carbon_reward",
            "15x10+carbon+priority+carbon_reward",
        ],
    )
    p.add_argument("--test-data", default="20x5+carbon+priority")
    p.add_argument(
        "--feasibility-limit",
        type=int,
        default=10,
        help="Instances for feasibility + carbon-choice checks (0 = skip both).",
    )
    p.add_argument(
        "--distribution-limit",
        type=int,
        default=0,
        help="Instances for distribution check (0 = all available).",
    )
    p.add_argument("--out-dir", default="experiments/flexibility")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Feasibility helpers
# ---------------------------------------------------------------------------

def _compatible_mask(true_op_pt_row: np.ndarray) -> np.ndarray:
    """Boolean mask [M]: True where the machine can process this operation."""
    return true_op_pt_row > 0


def _check_feasibility(
    schedule: list[tuple[int, int, float, float]],
    n_ops: int,
    true_op_ct: np.ndarray,
) -> list[str]:
    """
    Check one instance's schedule for validity.

    schedule : list of (op_idx, mch_idx, start_time, end_time)
    n_ops    : expected total number of operations
    true_op_ct: env.true_op_ct[0] after episode ends — used to cross-check end times

    Returns a list of error strings (empty list = PASS).
    """
    errors: list[str] = []

    # 1. All operations should appear exactly once
    op_ids = [s[0] for s in schedule]
    if len(op_ids) != n_ops:
        errors.append(f"Step count {len(op_ids)} != n_ops {n_ops}")
    unique_ops = set(op_ids)
    if len(unique_ops) != n_ops:
        missing = set(range(n_ops)) - unique_ops
        errors.append(f"{len(missing)} ops never scheduled: {sorted(missing)[:10]}")

    # 2. No machine double-booking
    mch_intervals: dict[int, list[tuple[float, float, int]]] = defaultdict(list)
    for op, mch, start, end in schedule:
        mch_intervals[mch].append((start, end, op))

    for mch, intervals in sorted(mch_intervals.items()):
        sorted_intervals = sorted(intervals, key=lambda x: x[0])
        for i in range(len(sorted_intervals) - 1):
            s0, e0, op0 = sorted_intervals[i]
            s1, e1, op1 = sorted_intervals[i + 1]
            if s1 < e0 - 1e-4:
                errors.append(
                    f"Machine {mch}: Op{op0} [{s0:.2f},{e0:.2f}) overlaps "
                    f"Op{op1} [{s1:.2f},{e1:.2f})"
                )

    # 3. End times should match env.true_op_ct (cross-check internal consistency)
    for op, mch, start, end in schedule:
        expected = float(true_op_ct[op])
        if abs(end - expected) > 1e-4:
            errors.append(
                f"Op{op}: tracked end {end:.4f} != env true_op_ct {expected:.4f}"
            )

    return errors


# ---------------------------------------------------------------------------
# Core episode runner
# ---------------------------------------------------------------------------

def _run_episode(
    policy,
    env,
    job_length,
    op_pt,
    op_priority,
    op_carbon,
    torch,
    greedy_select_action,
    track: bool = False,
):
    """
    Run one scheduling episode.

    track=False : fast path (no per-step bookkeeping)
    track=True  : also return schedule + carbon choices

    Returns
    -------
    metrics : dict {makespan, total_carbon, priority_weighted_completion}
    schedule: list of (op, mch, start, end)           -- only if track=True else []
    choices : list of (op, mch, chosen_c, min_c, chosen_pt, min_pt) -- only if track=True else []
    """
    state = env.set_initial_data([job_length], [op_pt], [op_priority], [op_carbon])
    n_m = env.number_of_machines
    schedule: list[tuple[int, int, float, float]] = []
    choices: list[tuple[int, int, float, float, float, float]] = []

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
        action_val = int(action.cpu().numpy()[0])

        if track:
            j = action_val // n_m
            m = action_val % n_m
            op = int(env.candidate[0, j])

            # Capture BEFORE step so values are not yet updated
            mch_free = float(env.true_mch_free_time[0, m])
            job_free = float(env.true_candidate_free_time[0, j])
            start = max(mch_free, job_free)
            pt_row = env.true_op_pt[0, op, :]
            pt = float(pt_row[m])
            end = start + pt
            schedule.append((op, m, start, end))

            compat = _compatible_mask(pt_row)
            c_row = env.op_carbon[0, op, :]
            chosen_c = float(c_row[m])
            chosen_pt = float(pt_row[m])
            min_c = float(c_row[compat].min()) if compat.any() else chosen_c
            min_pt = float(pt_row[compat].min()) if compat.any() else chosen_pt
            choices.append((op, m, chosen_c, min_c, chosen_pt, min_pt))

        state, _, done = env.step(actions=action.cpu().numpy())
        if done:
            break

    op_ct = env.true_op_ct[0].astype(float)
    prio = op_priority.astype(float)
    from objectives import operation_priority_weighted_completion
    pwc = float(operation_priority_weighted_completion(op_ct, prio))
    metrics = {
        "makespan": float(env.current_makespan[0]),
        "total_carbon": float(env.total_carbon[0]),
        "priority_weighted_completion": pwc,
    }
    return metrics, schedule, choices


# ---------------------------------------------------------------------------
# Section 1: Feasibility
# ---------------------------------------------------------------------------

def run_feasibility(
    model_names, policy_map, env, data, limit, out_path, torch, greedy_select_action, test_data_name
):
    job_lengths, op_pts, op_priorities, op_carbons = data
    n_total = min(limit, len(job_lengths))

    print(f"\n{'='*70}")
    print("SECTION 1: FEASIBILITY CHECK")
    print(f"{'='*70}")
    print(f"Models     : {', '.join(model_names)}")
    print(f"Test data  : {test_data_name}")
    print(f"Instances  : {n_total} (of {len(job_lengths)} available)")

    rows: list[dict] = []
    for model_name in model_names:
        if model_name not in policy_map:
            continue
        policy = policy_map[model_name]
        pass_count = 0

        for i in range(n_total):
            n_ops = int(np.sum(job_lengths[i]))
            metrics, schedule, _ = _run_episode(
                policy, env, job_lengths[i], op_pts[i], op_priorities[i], op_carbons[i],
                torch, greedy_select_action, track=True,
            )
            errors = _check_feasibility(schedule, n_ops, env.true_op_ct[0])
            passed = len(errors) == 0
            if passed:
                pass_count += 1

            tag = "PASS" if passed else "FAIL"
            print(
                f"  [{tag}] {model_name:42s} inst {i+1:3d}: "
                f"makespan={metrics['makespan']:8.1f}  carbon={metrics['total_carbon']:9.1f}"
            )
            for e in errors:
                print(f"         ERROR: {e}")

            rows.append({
                "model": model_name,
                "instance": i + 1,
                "n_ops": n_ops,
                "n_steps_recorded": len(schedule),
                "feasible": passed,
                "n_errors": len(errors),
                "makespan": round(metrics["makespan"], 4),
                "total_carbon": round(metrics["total_carbon"], 4),
                "priority_weighted_completion": round(metrics["priority_weighted_completion"], 4),
                "first_error": errors[0] if errors else "",
            })

        pct = 100.0 * pass_count / n_total if n_total else 0.0
        print(f"  --> {model_name}: {pass_count}/{n_total} PASS ({pct:.0f}%)\n")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "model", "instance", "n_ops", "n_steps_recorded", "feasible", "n_errors",
        "makespan", "total_carbon", "priority_weighted_completion", "first_error",
    ]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {out_path}")


# ---------------------------------------------------------------------------
# Section 2: Carbon choice analysis
# ---------------------------------------------------------------------------

def run_carbon_choices(
    model_names, policy_map, env, data, limit, out_path, torch, greedy_select_action, test_data_name
):
    job_lengths, op_pts, op_priorities, op_carbons = data
    n_total = min(limit, len(job_lengths))

    print(f"\n{'='*70}")
    print("SECTION 2: CARBON CHOICE ANALYSIS")
    print(f"{'='*70}")
    print(f"Models     : {', '.join(model_names)}")
    print(f"Test data  : {test_data_name}")
    print(f"Instances  : {n_total}")
    print(
        "At every scheduling step, did the model pick the lowest-carbon "
        "machine? Or the fastest machine?"
    )

    rows: list[dict] = []
    for model_name in model_names:
        if model_name not in policy_map:
            continue
        policy = policy_map[model_name]

        total_steps = 0
        min_c_count = 0       # steps where chosen == min carbon
        min_pt_count = 0      # steps where chosen == min processing time
        both_min_count = 0    # steps where chosen == both
        only_one_choice = 0   # steps where only one machine is compatible (no real choice)
        c_excess_sum = 0.0
        pt_excess_sum = 0.0

        for i in range(n_total):
            _, _, choices = _run_episode(
                policy, env, job_lengths[i], op_pts[i], op_priorities[i], op_carbons[i],
                torch, greedy_select_action, track=True,
            )
            for op, mch, c_chosen, c_min, pt_chosen, pt_min in choices:
                total_steps += 1
                is_min_c = abs(c_chosen - c_min) < 1e-6
                is_min_pt = abs(pt_chosen - pt_min) < 1e-6
                # Check if this was a forced choice (only one compatible machine)
                compat_count = int(np.sum(_compatible_mask(env.true_op_pt[0, op, :])))
                if compat_count <= 1:
                    only_one_choice += 1
                if is_min_c:
                    min_c_count += 1
                if is_min_pt:
                    min_pt_count += 1
                if is_min_c and is_min_pt:
                    both_min_count += 1
                c_excess_sum += c_chosen - c_min
                pt_excess_sum += pt_chosen - pt_min

        free_steps = total_steps - only_one_choice  # steps with a real choice
        pct_min_c = 100.0 * min_c_count / total_steps if total_steps else 0.0
        pct_min_pt = 100.0 * min_pt_count / total_steps if total_steps else 0.0
        pct_both = 100.0 * both_min_count / total_steps if total_steps else 0.0
        avg_c_excess = c_excess_sum / total_steps if total_steps else 0.0
        avg_pt_excess = pt_excess_sum / total_steps if total_steps else 0.0

        rows.append({
            "model": model_name,
            "instances": n_total,
            "total_steps": total_steps,
            "forced_steps_only_one_machine": only_one_choice,
            "free_choice_steps": free_steps,
            "pct_chose_min_carbon": round(pct_min_c, 2),
            "pct_chose_min_pt": round(pct_min_pt, 2),
            "pct_chose_both_min": round(pct_both, 2),
            "avg_carbon_excess_above_min": round(avg_c_excess, 6),
            "avg_pt_excess_above_min": round(avg_pt_excess, 6),
        })

        print(f"\n  Model: {model_name}")
        print(f"    Total steps          : {total_steps}")
        print(f"    Forced choices       : {only_one_choice}  (only 1 machine compatible)")
        print(f"    Free-choice steps    : {free_steps}")
        print(f"    Chose min carbon     : {min_c_count}/{total_steps} ({pct_min_c:.1f}%)")
        print(f"    Chose min time       : {min_pt_count}/{total_steps} ({pct_min_pt:.1f}%)")
        print(f"    Chose both min       : {both_min_count}/{total_steps} ({pct_both:.1f}%)")
        print(f"    Avg carbon excess    : {avg_c_excess:.4f} above available minimum")
        print(f"    Avg time excess      : {avg_pt_excess:.4f} above available minimum")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "model", "instances", "total_steps", "forced_steps_only_one_machine",
        "free_choice_steps", "pct_chose_min_carbon", "pct_chose_min_pt",
        "pct_chose_both_min", "avg_carbon_excess_above_min", "avg_pt_excess_above_min",
    ]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {out_path}")


# ---------------------------------------------------------------------------
# Section 3: Distribution check
# ---------------------------------------------------------------------------

def run_distribution(
    model_names, policy_map, env, data, limit, out_per_instance, out_summary, torch, greedy_select_action, test_data_name
):
    job_lengths, op_pts, op_priorities, op_carbons = data
    n_total = len(job_lengths) if limit == 0 else min(limit, len(job_lengths))

    print(f"\n{'='*70}")
    print("SECTION 3: DISTRIBUTION CHECK")
    print(f"{'='*70}")
    print(f"Models     : {', '.join(model_names)}")
    print(f"Test data  : {test_data_name}")
    print(f"Instances  : {n_total}")
    print("Reporting mean / std / min / median / max — not just the mean.\n")

    all_results: dict[str, list[dict]] = {}

    for model_name in model_names:
        if model_name not in policy_map:
            continue
        policy = policy_map[model_name]
        results: list[dict] = []

        t0 = time.time()
        for i in range(n_total):
            metrics, _, _ = _run_episode(
                policy, env, job_lengths[i], op_pts[i], op_priorities[i], op_carbons[i],
                torch, greedy_select_action, track=False,
            )
            metrics["instance"] = i + 1
            results.append(metrics)

        elapsed = time.time() - t0
        all_results[model_name] = results

        mk = np.array([r["makespan"] for r in results])
        ca = np.array([r["total_carbon"] for r in results])
        print(f"  {model_name}  ({elapsed:.1f}s)")
        print(
            f"    Makespan : mean={np.mean(mk):.2f}  std={np.std(mk):.2f}  "
            f"min={np.min(mk):.2f}  median={np.median(mk):.2f}  max={np.max(mk):.2f}"
        )
        print(
            f"    Carbon   : mean={np.mean(ca):.2f}  std={np.std(ca):.2f}  "
            f"min={np.min(ca):.2f}  median={np.median(ca):.2f}  max={np.max(ca):.2f}"
        )

    # Pairwise: 20x10 vs 20x5 model per instance
    m10 = "20x10+carbon+priority+carbon_reward"
    m5 = "20x5+carbon+priority+carbon_reward"
    if m10 in all_results and m5 in all_results:
        r10 = {r["instance"]: r for r in all_results[m10]}
        r5 = {r["instance"]: r for r in all_results[m5]}
        common = sorted(set(r10) & set(r5))
        mk_wins = sum(1 for i in common if r10[i]["makespan"] < r5[i]["makespan"])
        ca_wins = sum(1 for i in common if r10[i]["total_carbon"] < r5[i]["total_carbon"])
        mk_diffs = [r10[i]["makespan"] - r5[i]["makespan"] for i in common]
        print(f"\n  --- Pairwise: 20x10 vs 20x5 carbon model ({len(common)} instances) ---")
        print(f"  20x10 model has LOWER makespan on : {mk_wins}/{len(common)} instances")
        print(f"  20x10 model has LOWER carbon on   : {ca_wins}/{len(common)} instances")
        print(
            f"  Makespan delta (20x10 - 20x5)    : "
            f"mean={np.mean(mk_diffs):.2f}  std={np.std(mk_diffs):.2f}  "
            f"min={np.min(mk_diffs):.2f}  max={np.max(mk_diffs):.2f}"
        )

    # Write per-instance CSV
    per_inst_rows: list[dict] = []
    for model_name, results in all_results.items():
        for r in results:
            per_inst_rows.append({
                "model": model_name,
                "instance": r["instance"],
                "makespan": round(r["makespan"], 4),
                "total_carbon": round(r["total_carbon"], 4),
                "priority_weighted_completion": round(r["priority_weighted_completion"], 4),
            })

    out_per_instance.parent.mkdir(parents=True, exist_ok=True)
    with out_per_instance.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["model", "instance", "makespan", "total_carbon", "priority_weighted_completion"]
        )
        writer.writeheader()
        writer.writerows(per_inst_rows)
    print(f"\nWrote {out_per_instance}")

    # Write summary CSV
    summary_rows: list[dict] = []
    for model_name, results in all_results.items():
        mk = np.array([r["makespan"] for r in results])
        ca = np.array([r["total_carbon"] for r in results])
        pwc = np.array([r["priority_weighted_completion"] for r in results])
        summary_rows.append({
            "model": model_name,
            "n_instances": len(results),
            "makespan_mean": round(float(np.mean(mk)), 4),
            "makespan_std": round(float(np.std(mk)), 4),
            "makespan_min": round(float(np.min(mk)), 4),
            "makespan_median": round(float(np.median(mk)), 4),
            "makespan_max": round(float(np.max(mk)), 4),
            "carbon_mean": round(float(np.mean(ca)), 4),
            "carbon_std": round(float(np.std(ca)), 4),
            "carbon_min": round(float(np.min(ca)), 4),
            "carbon_median": round(float(np.median(ca)), 4),
            "carbon_max": round(float(np.max(ca)), 4),
            "pwc_mean": round(float(np.mean(pwc)), 4),
            "pwc_std": round(float(np.std(pwc)), 4),
        })

    with out_summary.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"Wrote {out_summary}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    repo = Path(__file__).resolve().parents[1]
    daniel_dir = repo / "daniel"
    os.chdir(daniel_dir)
    sys.path.insert(0, str(daniel_dir))

    # Must set sys.argv before importing daniel modules — params.py calls parse_args() globally
    sys.argv = [
        "validate_carbon_models",
        "--device", args.device,
        "--data_source", args.data_source,
        "--enable_priority", "True",
        "--enable_carbon", "True",
        "--carbon_feature", "False",
    ]

    import torch
    from common_utils import greedy_select_action
    from data_utils import load_priority_carbon_data_from_files
    from fjsp_env_same_op_nums import FJSPEnvForSameOpNums
    from model.main_model import DANIEL
    from params import configs

    out_dir = (repo / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load test data
    data_path = Path("./data") / args.data_source / args.test_data
    job_lengths, op_pts, op_priorities, op_carbons = load_priority_carbon_data_from_files(str(data_path))
    print(f"Loaded {len(job_lengths)} instances from {data_path}")

    n_j = job_lengths[0].shape[0]
    _, n_m = op_pts[0].shape
    env = FJSPEnvForSameOpNums(n_j=n_j, n_m=n_m)
    print(f"Environment: {n_j} jobs, {n_m} machines")

    # Load models
    policy_map: dict = {}
    for model_name in args.models:
        model_path = daniel_dir / "trained_network" / args.data_source / f"{model_name}.pth"
        if not model_path.exists():
            print(f"SKIP missing: {model_path}")
            continue
        policy = DANIEL(configs)
        from common_utils import load_checkpoint_state_dict
        policy.load_state_dict(load_checkpoint_state_dict(
            str(model_path), map_location=args.device, expected_schema=configs.feature_schema))
        policy.to(torch.device(args.device))
        policy.eval()
        policy_map[model_name] = policy
        print(f"Loaded model: {model_name}")

    if not policy_map:
        print("No models loaded — check trained_network/ directory.")
        return

    data = (job_lengths, op_pts, op_priorities, op_carbons)

    # Sections 1 and 2 share the same feasibility-limit instances
    if args.feasibility_limit > 0:
        run_feasibility(
            args.models, policy_map, env, data,
            limit=args.feasibility_limit,
            out_path=out_dir / "carbon_validation_feasibility.csv",
            torch=torch,
            greedy_select_action=greedy_select_action,
            test_data_name=args.test_data,
        )
        run_carbon_choices(
            args.models, policy_map, env, data,
            limit=args.feasibility_limit,
            out_path=out_dir / "carbon_validation_choices.csv",
            torch=torch,
            greedy_select_action=greedy_select_action,
            test_data_name=args.test_data,
        )
    else:
        print("Skipping feasibility + carbon-choice checks (--feasibility-limit 0).")

    # Section 3: distribution
    run_distribution(
        args.models, policy_map, env, data,
        limit=args.distribution_limit,
        out_per_instance=out_dir / "carbon_validation_distribution.csv",
        out_summary=out_dir / "carbon_validation_distribution_summary.csv",
        torch=torch,
        greedy_select_action=greedy_select_action,
        test_data_name=args.test_data,
    )

    print("\nValidation complete.")


if __name__ == "__main__":
    main()
