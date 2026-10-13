"""Benchmark v3 evaluation (docs/benchmark_v3.md): greedy rollouts of benchmark v3 checkpoints on a frozen v3 split.

Every schedule is re-scored by benchmark_v3.schedule_objectives (feasibility + objectives, independent of the
environment) and must agree with the environment's makespan, carbon and tardiness. Per instance it reports the raw
objectives, the ratios MK / MKref, C / Cref, T / Tref and the normalised score for the checkpoint's own goal and for the
balanced preference (1/3, 1/3, 1/3). The observations use fixed scales, so the whole split runs as one batch.

    python evaluate_v3.py --models 10x5+mix+bv3_m_s1 [...] --split validation --size 10x5 [--subset heldout_49]

Outputs test_results/benchmark_v3/<split>_<size>[_<subset>]/<model>.npz and summary.json. The reserved final-test
sets are not part of benchmark v3 and cannot be selected.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
ap.add_argument("--models", nargs="+", required=True, help="checkpoint names in trained_network/SD2 or .pth paths")
ap.add_argument("--split", default="validation", choices=["train", "validation", "development", "evaluation"])
ap.add_argument("--size", default="10x5", choices=["10x5", "20x10"])
ap.add_argument("--subset", default="", help="named subset of the split in the manifest, e.g. heldout_49")
ap.add_argument("--out", default="", help="output root (default: test_results/benchmark_v3)")
ap.add_argument("--device", default="cuda")
args, rest = ap.parse_known_args()                          # the rest (e.g. --benchmark_manifest) goes to params
sys.argv = [sys.argv[0], *rest, "--device", args.device, "--obs_version", "v3", "--reward_version", "v3",
            "--goal", "mct"]

import numpy as np                                          # noqa: E402
import torch                                                # noqa: E402
from params import configs                                  # noqa: E402
import benchmark_v3 as B                                    # noqa: E402
from common_utils import greedy_select_action, load_checkpoint_state_dict      # noqa: E402
from checkpointing import unwrap_checkpoint                 # noqa: E402
from feature_schemas import SCHEMA_V3                       # noqa: E402
from fjsp_env_same_op_nums import FJSPEnvForSameOpNums      # noqa: E402
from model.main_model import DANIEL                         # noqa: E402

HERE = Path(__file__).resolve().parent
BALANCED = (1 / 3, 1 / 3, 1 / 3)


def checkpoint_path(name):
    p = Path(name)
    return p if p.suffix == ".pth" else HERE / "trained_network" / "SD2" / f"{name}.pth"


@torch.no_grad()
def rollout(net, data, idx):
    """Greedy schedules for instances idx (one batch). Returns env objectives and (machine, start) per instance."""
    env = FJSPEnvForSameOpNums(n_j=B.SIZES[data.size][0], n_m=B.SIZES[data.size][1])
    state = env.set_initial_data(**data.env_kwargs(idx, "mct"))
    n_ops = env.number_of_ops
    machine = np.full((len(idx), n_ops), -1)
    while True:
        pi, _ = net(fea_j=state.fea_j_tensor, op_mask=state.op_mask_tensor, candidate=state.candidate_tensor,
                    fea_m=state.fea_m_tensor, mch_mask=state.mch_mask_tensor, comp_idx=state.comp_idx_tensor,
                    dynamic_pair_mask=state.dynamic_pair_mask_tensor, fea_pairs=state.fea_pairs_tensor)
        action = greedy_select_action(pi).cpu().numpy()
        op = env.candidate[env.env_idxs, action // env.number_of_machines]
        machine[env.env_idxs, op] = action % env.number_of_machines
        state, _, done = env.step(actions=action)
        if np.all(done):
            break
    pt = env.true_op_pt[env.env_idxs[:, None], np.arange(n_ops)[None, :], machine]
    start = env.true_op_ct - pt
    comp = env.true_op_ct[env.env_job_idx, env.job_last_op_id]
    tard = np.maximum(0.0, comp - env.true_due_dates).sum(1)
    return env.current_makespan.copy(), env.total_carbon.copy(), tard, machine, start


def main():
    manifest = B.load_manifest(configs.benchmark_manifest or None)
    data = B.load_split(args.split, args.size, manifest)
    idx = np.arange(len(data))
    tag = f"{args.split}_{args.size}"
    if args.subset:
        key = f"{args.split}/{args.size}/{args.subset}"
        if key not in manifest["subsets"]:
            raise KeyError(f"no subset {key} in the manifest (have {sorted(manifest['subsets'])})")
        idx = np.asarray(manifest["subsets"][key]["loader_idx"])
        assert [Path(data.keys[i]).name for i in idx] == manifest["subsets"][key]["files"]
        tag += f"_{args.subset}"
    out_dir = Path(args.out or HERE / "test_results" / "benchmark_v3") / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    mk_ref = np.asarray(data.mk_ref)[idx]; cb_ref = np.asarray(data.cb_ref)[idx]
    summary = {"split": args.split, "size": args.size, "subset": args.subset or None, "n": int(len(idx)),
               "manifest_sha256": B.file_sha256(configs.benchmark_manifest or B.MANIFEST), "t_ref": data.t_ref,
               "models": {}}
    for name in args.models:
        path = checkpoint_path(name)
        _, meta = unwrap_checkpoint(torch.load(path, map_location=configs.device))
        goal = meta.get("config", {}).get("goal", "mct")
        net = DANIEL(configs).to(configs.device)
        net.load_state_dict(load_checkpoint_state_dict(path, map_location=configs.device, expected_schema=SCHEMA_V3))
        net.eval()
        mk, cb, tard, machine, start = rollout(net, data, idx)
        for r, i in enumerate(idx):                         # independent re-scoring of every schedule
            ind = B.schedule_objectives(data.job_length[i], data.op_pt[i], data.carbon[i], data.due[i],
                                        machine[r], start[r])
            if (abs(ind["makespan"] - mk[r]) > 1e-9 or abs(ind["carbon"] - cb[r]) > 1e-9
                    or abs(ind["tardiness"] - tard[r]) > 1e-9):
                raise AssertionError(f"{name}: environment and independent scorer disagree on {data.keys[i]}")
        w_goal = B.goal_weights(goal)
        s_goal = B.score(w_goal, mk, cb, tard, mk_ref, cb_ref, data.t_ref)
        s_bal = B.score(BALANCED, mk, cb, tard, mk_ref, cb_ref, data.t_ref)
        np.savez_compressed(out_dir / f"{Path(name).stem}.npz", keys=np.asarray(data.keys)[idx], makespan=mk,
                            carbon=cb, tardiness=tard, score_goal=s_goal, score_balanced=s_bal, machine=machine,
                            start=start)
        summary["models"][name] = {
            "checkpoint_sha256": B.file_sha256(path), "goal": goal,
            "makespan": float(mk.mean()), "carbon": float(cb.mean()), "tardiness": float(tard.mean()),
            "mk_ratio": float((mk / mk_ref).mean()), "carbon_ratio": float((cb / cb_ref).mean()),
            "tardiness_ratio": float(tard.mean() / data.t_ref),
            "score_goal": float(s_goal.mean()), "score_balanced": float(s_bal.mean()),
            "independent_rescore": "all schedules feasible; objectives identical to the environment"}
        print(name, json.dumps(summary["models"][name]))
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
