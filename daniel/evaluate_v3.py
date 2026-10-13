"""Benchmark v3 evaluation (docs/benchmark_v3.md): greedy rollouts of benchmark v3 checkpoints on a frozen v3 split.

Every schedule is re-scored by benchmark_v3.schedule_objectives (feasibility + objectives, independent of the
environment) and must agree with the environment's makespan, carbon and tardiness. Per instance it reports the raw
objectives, the ratios MK / MKref, C / Cref, T / Tref and the normalised score for the checkpoint's own goal and for the
balanced preference (1/3, 1/3, 1/3), and the tardy-job share. The observations use fixed scales, so the whole split
runs as one batch. --sample N adds best-of-N sampling: N sampled rollouts per instance (torch generator seeded with
--sample_seed), the schedule with the lowest score for the checkpoint's own goal kept (ties: first sample).

    python evaluate_v3.py --models 10x5+mix+bv3_m_s1 [...] --split validation --size 10x5 [--subset heldout_49]

Outputs test_results/benchmark_v3/<split>_<size>[_<subset>]/<model>.npz and summary.json. The reserved final-test
sets are not part of benchmark v3 and cannot be selected.
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path

ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
ap.add_argument("--models", nargs="+", required=True, help="checkpoint names in trained_network/SD2 or .pth paths")
ap.add_argument("--split", default="validation", choices=["train", "validation", "development", "evaluation"])
ap.add_argument("--size", default="10x5", choices=["10x5", "20x10"])
ap.add_argument("--subset", default="", help="named subset of the split in the manifest, e.g. heldout_49")
ap.add_argument("--first", type=int, default=0, help="only the first N instances (smoke tests)")
ap.add_argument("--sample", type=int, default=0, help="also run best-of-N sampling (selected by the model's own goal)")
ap.add_argument("--sample_seed", type=int, default=50, help="torch generator seed of the sampling rollouts")
ap.add_argument("--sample_batch", type=int, default=1000, help="environments per sampling batch")
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
def rollout(net, data, idx, gen=None):
    """Greedy (gen None) or sampled schedules for instances idx (one batch, repeats allowed). Returns env objectives,
    (machine, start) and the tardy-job share per environment."""
    env = FJSPEnvForSameOpNums(n_j=B.SIZES[data.size][0], n_m=B.SIZES[data.size][1])
    state = env.set_initial_data(**data.env_kwargs(idx, "mct"))
    n_ops = env.number_of_ops
    machine = np.full((len(idx), n_ops), -1)
    while True:
        pi, _ = net(fea_j=state.fea_j_tensor, op_mask=state.op_mask_tensor, candidate=state.candidate_tensor,
                    fea_m=state.fea_m_tensor, mch_mask=state.mch_mask_tensor, comp_idx=state.comp_idx_tensor,
                    dynamic_pair_mask=state.dynamic_pair_mask_tensor, fea_pairs=state.fea_pairs_tensor)
        action = (greedy_select_action(pi) if gen is None else
                  torch.multinomial(pi, 1, generator=gen).squeeze(1)).cpu().numpy()
        op = env.candidate[env.env_idxs, action // env.number_of_machines]
        machine[env.env_idxs, op] = action % env.number_of_machines
        state, _, done = env.step(actions=action)
        if np.all(done):
            break
    pt = env.true_op_pt[env.env_idxs[:, None], np.arange(n_ops)[None, :], machine]
    start = env.true_op_ct - pt
    comp = env.true_op_ct[env.env_job_idx, env.job_last_op_id]
    tard = np.maximum(0.0, comp - env.true_due_dates).sum(1)
    tardy = (comp > env.true_due_dates).mean(1)
    return env.current_makespan.copy(), env.total_carbon.copy(), tard, machine, start, tardy


def best_of(net, data, idx, n, goal, seed, batch):
    """Best-of-n sampling per instance by the goal's score (ties: first sample)."""
    gen = torch.Generator(device=configs.device); gen.manual_seed(seed)
    per = max(1, batch // n)
    outs = []
    w = B.goal_weights(goal)
    for c0 in range(0, len(idx), per):
        chunk = np.asarray(idx[c0:c0 + per])
        mk, cb, td, mach, st, tardy = rollout(net, data, np.repeat(chunk, n), gen)
        sc = B.score(w, mk, cb, td, np.repeat(np.asarray(data.mk_ref)[chunk], n),
                     np.repeat(np.asarray(data.cb_ref)[chunk], n), data.t_ref).reshape(len(chunk), n)
        pick = np.arange(len(chunk)) * n + sc.argmin(1)
        outs.append((mk[pick], cb[pick], td[pick], mach[pick], st[pick], tardy[pick]))
    return tuple(np.concatenate([o[k] for o in outs]) for k in range(6))


def main():
    manifest = B.load_manifest(configs.benchmark_manifest or None)
    data = B.load_split(args.split, args.size, manifest)
    idx = np.arange(len(data) if not args.first else min(args.first, len(data)))
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
    summary_file = out_dir / "summary.json"
    summary = json.loads(summary_file.read_text()) if summary_file.exists() else {"models": {}}
    summary.update({"split": args.split, "size": args.size, "subset": args.subset or None, "n": int(len(idx)),
                    "manifest_sha256": B.file_sha256(configs.benchmark_manifest or B.MANIFEST), "t_ref": data.t_ref})
    for name in args.models:
        path = checkpoint_path(name)
        _, meta = unwrap_checkpoint(torch.load(path, map_location=configs.device))
        goal = meta.get("config", {}).get("goal", "mct")
        net = DANIEL(configs).to(configs.device)
        net.load_state_dict(load_checkpoint_state_dict(path, map_location=configs.device, expected_schema=SCHEMA_V3))
        net.eval()
        model_rec = {"checkpoint_sha256": B.file_sha256(path), "goal": goal, "update": meta.get("update")}
        modes = [("greedy", lambda: rollout(net, data, idx), "")]
        if args.sample > 0:
            modes.append((f"sample{args.sample}", lambda: best_of(net, data, idx, args.sample, goal, args.sample_seed,
                                                                  args.sample_batch), f"_sample{args.sample}"))
        for mode, run, suffix in modes:
            t0 = time.time()
            mk, cb, tard, machine, start, tardy = run()
            seconds = time.time() - t0
            for r, i in enumerate(idx):                     # independent re-scoring of every reported schedule
                ind = B.schedule_objectives(data.job_length[i], data.op_pt[i], data.carbon[i], data.due[i],
                                            machine[r], start[r])
                if (abs(ind["makespan"] - mk[r]) > 1e-9 or abs(ind["carbon"] - cb[r]) > 1e-9
                        or abs(ind["tardiness"] - tard[r]) > 1e-9):
                    raise AssertionError(f"{name}: environment and independent scorer disagree on {data.keys[i]}")
            w_goal = B.goal_weights(goal)
            s_goal = B.score(w_goal, mk, cb, tard, mk_ref, cb_ref, data.t_ref)
            s_bal = B.score(BALANCED, mk, cb, tard, mk_ref, cb_ref, data.t_ref)
            np.savez_compressed(out_dir / f"{Path(name).stem}{suffix}.npz", keys=np.asarray(data.keys)[idx],
                                makespan=mk, carbon=cb, tardiness=tard, tardy_share=tardy, score_goal=s_goal,
                                score_balanced=s_bal, machine=machine, start=start)
            model_rec[mode] = {
                "makespan": float(mk.mean()), "carbon": float(cb.mean()), "tardiness": float(tard.mean()),
                "tardy_share": float(tardy.mean()),
                "mk_ratio": float((mk / mk_ref).mean()), "carbon_ratio": float((cb / cb_ref).mean()),
                "tardiness_ratio": float(tard.mean() / data.t_ref),
                "score_goal": float(s_goal.mean()), "score_balanced": float(s_bal.mean()), "seconds": seconds,
                "independent_rescore": "all schedules feasible; objectives identical to the environment"}
            print(name, mode, json.dumps(model_rec[mode]), flush=True)
        summary["models"][name] = model_rec
        (out_dir / f"{Path(name).stem}.json").write_text(json.dumps(model_rec, indent=1))
    summary_file.write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
