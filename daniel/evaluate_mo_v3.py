"""MO-DANIEL final evaluation (docs/evaluation_protocol_v3.md, Addendum A, A5): greedy rollouts of a preference-
conditioned v3 checkpoint under every preference of the simplex grid (step 0.1, 66 vectors) plus the equal-weight
preference (1/3, 1/3, 1/3), on a frozen v3 split.

Per instance: the non-dominated set of the 66 grid schedules (the equal-weight schedule is reported separately and
is not part of the set), its normalised hypervolume (frozen ideal/nadir points, reference 1.1), and IGD+ against the
CP-SAT B2 reference front where one exists. Every schedule is re-scored by benchmark_v3.schedule_objectives and must
equal the environment's objectives.

    python evaluate_mo_v3.py --model 10x5+mix+bv3_mo_s111 --split evaluation --size 10x5 [--first N] [--grid_step 0.1]

Writes <out>/<split>_<size>/<model>.npz (objectives [instances, 67, 3]) and <model>.json.
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
ap.add_argument("--model", required=True)
ap.add_argument("--split", default="evaluation", choices=["validation", "development", "evaluation"])
ap.add_argument("--size", default="10x5", choices=["10x5", "20x10"])
ap.add_argument("--first", type=int, default=0)
ap.add_argument("--grid_step", type=float, default=0.1)
ap.add_argument("--norm", default=str(REPO / "results" / "baselines_v3" / "normalisation_points.json"))
ap.add_argument("--b2_root", default=str(REPO / "results" / "baselines_v3" / "cpsat" / "b2"))
ap.add_argument("--out", default=str(REPO / "results" / "mo_daniel_v3" / "eval"))
ap.add_argument("--batch", type=int, default=1400, help="environments per rollout batch")
ap.add_argument("--device", default="cuda")
args = ap.parse_args()
sys.argv = [sys.argv[0], "--device", args.device, "--obs_version", "v3", "--reward_version", "v3", "--goal", "mct",
            "--preference_dim", "3"]

import numpy as np                                          # noqa: E402
import torch                                                # noqa: E402
from params import configs                                  # noqa: E402
import benchmark_v3 as B                                    # noqa: E402
import mo_metrics as MM                                     # noqa: E402
from common_utils import greedy_select_action, load_checkpoint_state_dict      # noqa: E402
from checkpointing import unwrap_checkpoint                 # noqa: E402
from feature_schemas import SCHEMA_V3_MO                    # noqa: E402
from fjsp_env_same_op_nums import FJSPEnvForSameOpNums      # noqa: E402
from model.main_model import DANIEL                         # noqa: E402

EQUAL = np.array([[1 / 3, 1 / 3, 1 / 3]])
CORNERS = {"w=(1,0,0)": (1.0, 0.0, 0.0), "w=(0,1,0)": (0.0, 1.0, 0.0), "w=(0,0,1)": (0.0, 0.0, 1.0)}


@torch.no_grad()
def rollout(net, data, idx, prefs):
    """Greedy schedules of instances idx [n] under prefs [n, 3]; returns objectives [n, 3], machine, start."""
    env = FJSPEnvForSameOpNums(*B.SIZES[data.size])
    state = env.set_initial_data(**data.env_kwargs(idx, "mct"))
    state = env.set_preference(prefs)
    machine = np.full((len(idx), env.number_of_ops), -1)
    while True:
        pi, _ = net(fea_j=state.fea_j_tensor, op_mask=state.op_mask_tensor, candidate=state.candidate_tensor,
                    fea_m=state.fea_m_tensor, mch_mask=state.mch_mask_tensor, comp_idx=state.comp_idx_tensor,
                    dynamic_pair_mask=state.dynamic_pair_mask_tensor, fea_pairs=state.fea_pairs_tensor,
                    pref=state.pref_tensor)
        a = greedy_select_action(pi).cpu().numpy()
        op = env.candidate[env.env_idxs, a // env.number_of_machines]
        machine[env.env_idxs, op] = a % env.number_of_machines
        state, _, done = env.step(a)
        if np.all(done):
            break
    pt = env.true_op_pt[env.env_idxs[:, None], np.arange(env.number_of_ops)[None, :], machine]
    start = env.true_op_ct - pt
    comp = env.true_op_ct[env.env_job_idx, env.job_last_op_id]
    tard = np.maximum(0.0, comp - env.true_due_dates).sum(1)
    return np.stack([env.current_makespan, env.total_carbon, tard], 1), machine, start


def main():
    data = B.load_split(args.split, args.size)
    n = len(data) if args.first <= 0 else min(args.first, len(data))
    grid = MM.simplex_grid(args.grid_step)
    prefs = np.vstack([grid, EQUAL])                                      # [G+1, 3]; the last row is equal weights
    G = len(prefs)
    path = Path(args.model) if args.model.endswith(".pth") else HERE / "trained_network" / "SD2" / f"{args.model}.pth"
    name = path.stem
    _, meta = unwrap_checkpoint(torch.load(path, map_location=configs.device, weights_only=False))
    net = DANIEL(configs).to(configs.device)
    net.load_state_dict(load_checkpoint_state_dict(path, map_location=configs.device, expected_schema=SCHEMA_V3_MO))
    net.eval()
    norm = json.loads(Path(args.norm).read_text())["instances"]
    obj = np.zeros((n, G, 3))
    t0 = time.time()
    per = max(1, args.batch // G)
    for c0 in range(0, n, per):
        inst = list(range(c0, min(n, c0 + per)))
        idx = [i for i in inst for _ in range(G)]
        o, machine, start = rollout(net, data, idx, np.tile(prefs, (len(inst), 1)))
        for r, i in enumerate(idx):                                       # independent re-scoring
            ind = B.schedule_objectives(data.job_length[i], data.op_pt[i], data.carbon[i], data.due[i],
                                        machine[r], start[r])
            if max(abs(ind["makespan"] - o[r, 0]), abs(ind["carbon"] - o[r, 1]), abs(ind["tardiness"] - o[r, 2])) > 1e-9:
                raise AssertionError(f"environment and independent scorer disagree on {data.keys[i]}")
        obj[inst] = o.reshape(len(inst), G, 3)
    seconds = time.time() - t0
    hv, igd, igd_keys, nd_size = [], [], [], []
    b2_dir = Path(args.b2_root) / f"{args.split}_{args.size}"
    for i in range(n):
        key = data.keys[i]
        ideal, nadir = MM.ideal_nadir(norm[key])
        pts = obj[i, :len(grid)]
        hv.append(MM.instance_hv(pts, ideal, nadir))
        nd_size.append(int(MM.nondominated(pts).sum()))
        f = b2_dir / f"{Path(key).stem}.json"
        if f.exists():
            front = np.array([[p[k] for k in MM.OBJECTIVES] for p in json.loads(f.read_text())["front"]])
            igd.append(MM.instance_igd_plus(pts, front, ideal, nadir)); igd_keys.append(key)
    out_dir = Path(args.out) / f"{args.split}_{args.size}"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / f"{name}.npz", keys=np.asarray(data.keys[:n]), prefs=prefs, objectives=obj,
                        hv=np.asarray(hv), nd_size=np.asarray(nd_size))
    gi = {tuple(np.round(w, 6)): k for k, w in enumerate(prefs)}
    at = {lab: {k: float(obj[:, gi[tuple(np.round(w, 6))], j].mean()) for j, k in enumerate(MM.OBJECTIVES)}
          for lab, w in {**CORNERS, "w=(1/3,1/3,1/3)": tuple(EQUAL[0])}.items()}
    rec = {"model": name, "checkpoint_sha256": B.file_sha256(path), "update": meta.get("update"),
           "split": args.split, "size": args.size, "n": n, "grid_step": args.grid_step, "grid_size": len(grid),
           "hv_mean": float(np.mean(hv)), "nd_size_mean": float(np.mean(nd_size)),
           "igd_plus_mean": float(np.mean(igd)) if igd else None, "igd_plus_n": len(igd),
           "igd_plus_per_instance": dict(zip(igd_keys, igd)), "objectives_at": at, "seconds": seconds,
           "norm": str(args.norm), "norm_sha256": B.file_sha256(args.norm),
           "independent_rescore": "all schedules feasible; objectives identical to the environment"}
    (out_dir / f"{name}.json").write_text(json.dumps(rec, indent=1))
    print(json.dumps({k: v for k, v in rec.items() if k != "igd_plus_per_instance"}), flush=True)


if __name__ == "__main__":
    main()
