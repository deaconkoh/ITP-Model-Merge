"""World-model M3 Part A (docs/world_model_m3_protocol_v3.md): zero-training preference-weighted merge fronts of the four
specialist sets. For every preference w of the simplex grid (step 0.1, 66 vectors) plus equal weights, the merged network
theta(w) = sum_x w_x theta_x (warmstart.interpolate, the plain unconditioned architecture; equivalent to the
preference-conditioned mix with zero preference columns) is rolled out greedily. The code path is that of arm E0
(tools/e0_merge_front_v3.py), so the Original specialists reproduce E0 seed 111 exactly.

Per set and split, the E0 layout (npz: keys, prefs, objectives [n, 67, 3], hv, nd_size; json summary):
  evaluation  66-point grid + equal weights; HV (frozen evaluation normalisation), IGD+ on the CP-SAT B2 subset
  validation  the same, HV with the frozen validation normalisation (no B2 fronts: no IGD+), plus the 15-point training
              validation grid (step 0.25; file suffix _g025), whose mean HV is comparable with the training validation HV

    python tools/world_model_m3_part_a.py [--sets wm,ctrl,orig,old] [--splits validation,evaluation] [--first N]
                                          [--out DIR] [--device cuda]
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
ap.add_argument("--sets", default="wm,ctrl,orig,old")
ap.add_argument("--splits", default="validation,evaluation")
ap.add_argument("--first", type=int, default=0)
ap.add_argument("--out", default=str(REPO / "results" / "world_model_m3_v3" / "part_a"))
ap.add_argument("--tag", default="wm3A")
ap.add_argument("--device", default="cuda")
args = ap.parse_args()
sys.argv = [sys.argv[0], "--device", args.device, "--obs_version", "v3", "--reward_version", "v3", "--goal", "mct"]
sys.path.insert(0, str(REPO / "daniel")); sys.path.insert(0, str(REPO / "tools"))

import numpy as np                                          # noqa: E402
import torch                                                # noqa: E402
from params import configs                                  # noqa: E402
import benchmark_v3 as B                                    # noqa: E402
import mo_metrics as MM                                     # noqa: E402
import freeze_world_model_m3_v3 as F                        # noqa: E402
from checkpointing import file_sha256                       # noqa: E402
from common_utils import greedy_select_action, load_checkpoint_state_dict      # noqa: E402
from feature_schemas import SCHEMA_V3                       # noqa: E402
from fjsp_env_same_op_nums import FJSPEnvForSameOpNums      # noqa: E402
from model.main_model import DANIEL                         # noqa: E402
from warmstart import interpolate                           # noqa: E402

KEYS = ("fea_j", "op_mask", "candidate", "fea_m", "mch_mask", "comp_idx", "dynamic_pair_mask", "fea_pairs")


@torch.no_grad()
def rollout(net, data, idx):
    """Greedy schedules of instances idx (arm E0's rollout, unchanged)."""
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


def front(sds, data, idx, prefs):
    """objectives [n, len(prefs), 3] of the merged network for every preference (independently re-scored)."""
    net = DANIEL(configs).to(configs.device); net.eval()
    obj = np.zeros((len(idx), len(prefs), 3))
    for g, lam in enumerate(prefs):
        net.load_state_dict(interpolate(sds, lam))
        o, machine, start = rollout(net, data, idx)
        for r, i in enumerate(idx):
            ind = B.schedule_objectives(data.job_length[i], data.op_pt[i], data.carbon[i], data.due[i],
                                        machine[r], start[r])
            if max(abs(ind["makespan"] - o[r, 0]), abs(ind["carbon"] - o[r, 1]), abs(ind["tardiness"] - o[r, 2])) > 1e-9:
                raise AssertionError(f"environment and independent scorer disagree on {data.keys[i]}")
        obj[:, g] = o
    return obj


def metrics(obj, keys, norm, b2_dir, n_grid):
    hv, nd, igd = [], [], {}
    for r, key in enumerate(keys):
        ideal, nadir = MM.ideal_nadir(norm[key])
        pts = obj[r, :n_grid]
        hv.append(MM.instance_hv(pts, ideal, nadir)); nd.append(int(MM.nondominated(pts).sum()))
        f = b2_dir / f"{Path(key).stem}.json" if b2_dir is not None else None
        if f is not None and f.exists():
            fr = np.array([[p[k] for k in MM.OBJECTIVES] for p in json.loads(f.read_text())["front"]])
            igd[key] = MM.instance_igd_plus(pts, fr, ideal, nadir)
    return hv, nd, igd


def main():
    cfg = F.load()
    pa = cfg["part_a"]
    grid = MM.simplex_grid(pa["grid_step"])
    prefs = np.vstack([grid, [[1 / 3, 1 / 3, 1 / 3]]])           # as E0: the last row is equal weights
    vgrid = MM.simplex_grid(pa["val_grid_step"])
    for s in args.sets.split(","):
        st = cfg["sets"][s]
        paths = [REPO / st["specialists"][o] for o in "mct"]
        for o, p in zip("mct", paths):
            if file_sha256(p) != st["sha256"][o]:
                raise SystemExit(f"{p} does not match its frozen SHA-256")
        sds = [load_checkpoint_state_dict(p, map_location=configs.device, expected_schema=SCHEMA_V3) for p in paths]
        for split in args.splits.split(","):
            t0 = time.time()
            data = B.load_split(split, cfg["size"])
            n = len(data) if args.first <= 0 else min(args.first, len(data))
            idx, keys = list(range(n)), list(data.keys[:n])
            norm_path = REPO / (pa["eval_norm"] if split == "evaluation" else pa["val_norm"])
            norm = json.loads(norm_path.read_text())["instances"]
            b2 = REPO / pa["b2_root"] / f"{split}_{cfg['size']}"
            b2 = b2 if b2.exists() else None
            out = Path(args.out) / f"{split}_{cfg['size']}"; out.mkdir(parents=True, exist_ok=True)
            runs = [("", prefs, len(grid))] + ([("_g025", vgrid, len(vgrid))] if split == "validation" else [])
            for sfx, pr, n_grid in runs:
                obj = front(sds, data, idx, pr)
                hv, nd, igd = metrics(obj, keys, norm, b2 if not sfx else None, n_grid)
                name = f"{cfg['size']}+mix+bv3_{args.tag}_{s}_s{cfg['seed']}{sfx}"
                np.savez_compressed(out / f"{name}.npz", keys=np.asarray(keys), prefs=pr, objectives=obj,
                                    hv=np.asarray(hv), nd_size=np.asarray(nd))
                rec = {"model": name, "set": s, "label": st["label"], "split": split, "size": cfg["size"], "n": n,
                       "grid_size": n_grid, "hv_mean": float(np.mean(hv)), "nd_size_mean": float(np.mean(nd)),
                       "igd_plus_mean": float(np.mean(list(igd.values()))) if igd else None, "igd_plus_n": len(igd),
                       "igd_plus_per_instance": igd, "norm": str(norm_path.relative_to(REPO)),
                       "specialists": {Path(p).name: st["sha256"][o] for o, p in zip("mct", paths)},
                       "seconds": time.time() - t0,
                       "independent_rescore": "all schedules feasible; objectives identical to the environment"}
                (out / f"{name}.json").write_text(json.dumps(rec, indent=1))
                print(json.dumps({k: v for k, v in rec.items() if k not in ("igd_plus_per_instance", "specialists")}),
                      flush=True)


if __name__ == "__main__":
    main()
