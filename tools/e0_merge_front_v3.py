"""Arm E0, zero-training merge front (docs/warmstart_protocol_v3.md): for each preference lambda of the simplex grid
(step 0.1, 66 vectors) plus equal weights, the interpolated specialist theta(lambda) = sum_x lambda_x theta_x
(published, unconditioned architecture; specialists of one seed) is rolled out greedily. No training. Output in the
layout of daniel/evaluate_mo_v3.py (objectives [instances, 67, 3], per-instance HV of the 66-point non-dominated set,
IGD+ on the B2 subset), so the analysis treats E0 like any arm. A Rewarded-Soups-style check (Rame et al., 2023).

    python tools/e0_merge_front_v3.py --size 10x5 --seed 111 [--split evaluation] [--first N] [--grid_step 0.1]
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
ap.add_argument("--size", required=True, choices=["10x5", "20x10"])
ap.add_argument("--seed", type=int, required=True)
ap.add_argument("--split", default="evaluation", choices=["validation", "development", "evaluation"])
ap.add_argument("--first", type=int, default=0)
ap.add_argument("--grid_step", type=float, default=0.1)
ap.add_argument("--spec_dir", default=str(REPO / "checkpoints" / "specialists_v3"))
ap.add_argument("--norm", nargs="+", default=[str(REPO / "results" / "baselines_v3" / "normalisation_points.json")])
ap.add_argument("--b2_root", default=str(REPO / "results" / "baselines_v3" / "cpsat" / "b2"))
ap.add_argument("--out", default=str(REPO / "results" / "warmstart_v3" / "eval"))
ap.add_argument("--device", default="cuda")
args = ap.parse_args()
sys.argv = [sys.argv[0], "--device", args.device, "--obs_version", "v3", "--reward_version", "v3", "--goal", "mct"]
sys.path.insert(0, str(REPO / "daniel"))

import numpy as np                                          # noqa: E402
import torch                                                # noqa: E402
from params import configs                                  # noqa: E402
import benchmark_v3 as B                                    # noqa: E402
import mo_metrics as MM                                     # noqa: E402
from checkpointing import file_sha256                       # noqa: E402
from common_utils import greedy_select_action, load_checkpoint_state_dict      # noqa: E402
from feature_schemas import SCHEMA_V3                       # noqa: E402
from fjsp_env_same_op_nums import FJSPEnvForSameOpNums      # noqa: E402
from model.main_model import DANIEL                         # noqa: E402
from warmstart import interpolate                           # noqa: E402

KEYS = ("fea_j", "op_mask", "candidate", "fea_m", "mch_mask", "comp_idx", "dynamic_pair_mask", "fea_pairs")


@torch.no_grad()
def rollout(net, data, idx):
    env = FJSPEnvForSameOpNums(*B.SIZES[data.size])
    st = env.set_initial_data(**data.env_kwargs(idx, "mct"))
    machine = np.full((len(idx), env.number_of_ops), -1)
    while True:
        pi, _ = net(**{k: getattr(st, f"{k}_tensor") for k in KEYS})
        a = greedy_select_action(pi).cpu().numpy()
        op = env.candidate[env.env_idxs, a // env.number_of_machines]
        machine[env.env_idxs, op] = a % env.number_of_machines
        st, _, done = env.step(a)
        if np.all(done):
            break
    pt = env.true_op_pt[env.env_idxs[:, None], np.arange(env.number_of_ops)[None, :], machine]
    comp = env.true_op_ct[env.env_job_idx, env.job_last_op_id]
    tard = np.maximum(0.0, comp - env.true_due_dates).sum(1)
    return np.stack([env.current_makespan, env.total_carbon, tard], 1), machine, env.true_op_ct - pt


def main():
    t0 = time.time()
    data = B.load_split(args.split, args.size)
    n = len(data) if args.first <= 0 else min(args.first, len(data))
    idx = list(range(n))
    paths = [Path(args.spec_dir) / f"{args.size}+mix+bv3_spec_{o}_s{args.seed}.pth" for o in "mct"]
    sds = [load_checkpoint_state_dict(p, map_location=configs.device, expected_schema=SCHEMA_V3) for p in paths]
    grid = MM.simplex_grid(args.grid_step)
    prefs = np.vstack([grid, [[1 / 3, 1 / 3, 1 / 3]]])
    obj = np.zeros((n, len(prefs), 3))
    net = DANIEL(configs).to(configs.device); net.eval()
    for g, lam in enumerate(prefs):
        net.load_state_dict(interpolate(sds, lam))
        o, machine, start = rollout(net, data, idx)
        for r, i in enumerate(idx):
            ind = B.schedule_objectives(data.job_length[i], data.op_pt[i], data.carbon[i], data.due[i],
                                        machine[r], start[r])
            if max(abs(ind["makespan"] - o[r, 0]), abs(ind["carbon"] - o[r, 1]), abs(ind["tardiness"] - o[r, 2])) > 1e-9:
                raise AssertionError(f"environment and independent scorer disagree on {data.keys[i]}")
        obj[:, g] = o
    norm = {}
    for f in args.norm:
        norm.update(json.loads(Path(f).read_text())["instances"])
    b2_dir = Path(args.b2_root) / f"{args.split}_{args.size}"
    hv, nd, igd = [], [], {}
    for i in idx:
        key = data.keys[i]
        ideal, nadir = MM.ideal_nadir(norm[key])
        pts = obj[i, :len(grid)]
        hv.append(MM.instance_hv(pts, ideal, nadir)); nd.append(int(MM.nondominated(pts).sum()))
        f = b2_dir / f"{Path(key).stem}.json"
        if f.exists():
            front = np.array([[p[k] for k in MM.OBJECTIVES] for p in json.loads(f.read_text())["front"]])
            igd[key] = MM.instance_igd_plus(pts, front, ideal, nadir)
    name = f"{args.size}+mix+bv3_ws_E0_s{args.seed}"
    out = Path(args.out) / f"{args.split}_{args.size}"; out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / f"{name}.npz", keys=np.asarray(data.keys[:n]), prefs=prefs, objectives=obj,
                        hv=np.asarray(hv), nd_size=np.asarray(nd))
    labels = {"w=(1,0,0)": 0, "w=(0,1,0)": None, "w=(0,0,1)": None, "w=(1/3,1/3,1/3)": len(prefs) - 1}
    gi = {tuple(np.round(w, 6)): k for k, w in enumerate(prefs)}
    for lab, w in (("w=(1,0,0)", (1.0, 0, 0)), ("w=(0,1,0)", (0, 1.0, 0)), ("w=(0,0,1)", (0, 0, 1.0))):
        labels[lab] = gi[tuple(np.round(w, 6))]
    rec = {"model": name, "arm": "E0", "split": args.split, "size": args.size, "n": n, "grid_step": args.grid_step,
           "grid_size": len(grid), "hv_mean": float(np.mean(hv)), "nd_size_mean": float(np.mean(nd)),
           "igd_plus_mean": float(np.mean(list(igd.values()))) if igd else None, "igd_plus_n": len(igd),
           "igd_plus_per_instance": igd,
           "objectives_at": {lab: {k: float(obj[:, j, c].mean()) for c, k in enumerate(MM.OBJECTIVES)}
                             for lab, j in labels.items()},
           "specialists": {p.name: file_sha256(p) for p in paths}, "seconds": time.time() - t0,
           "independent_rescore": "all schedules feasible; objectives identical to the environment"}
    (out / f"{name}.json").write_text(json.dumps(rec, indent=1))
    print(json.dumps({k: v for k, v in rec.items() if k != "igd_plus_per_instance"}))


if __name__ == "__main__":
    main()
