"""Frozen per-instance ideal and nadir points for the MO-DANIEL VALIDATION metric (docs/evaluation_protocol_v3.md,
Addendum A, A2). results/baselines_v3/normalisation_points.json covers the evaluation split only, so the validation
split gets its own points by the section-6 definitions, restricted to what exists on that split:

  ideal  best-known makespan and tardiness over every CP-SAT B1 schedule of the instance (same B1 procedure, limits and
         seed as the baselines), and the analytic minimum carbon;
  nadir  the worst value of each objective over those CP-SAT schedules, the greedy schedules of the nine selected
         specialists (DANIEL-M/C/T x seeds 111/222/333) and the MWKR/ECT reference schedule. (There is no B2 front on
         the validation split.)

Writes results/mo_daniel_v3/validation_normalisation_points_<size>.json once and never overwrites it (an existing
file is verified and kept). CP-SAT results are cached per instance under results/mo_daniel_v3/cpsat/ (resume-safe).

    python tools/mo_v3_normalisation.py --size 10x5 [--split validation] [--first N] [--limit 60] [--lex_limit 20]
"""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
ap.add_argument("--size", required=True, choices=["10x5", "20x10"])
ap.add_argument("--split", default="validation", choices=["validation", "development"])
ap.add_argument("--first", type=int, default=0, help="only the first N instances (smoke tests)")
ap.add_argument("--limit", type=float, default=60.0, help="B1 primary solve limit (s)")
ap.add_argument("--lex_limit", type=float, default=20.0, help="B1 lexicographic stage limit (s)")
ap.add_argument("--workers", type=int, default=8)
ap.add_argument("--seeds", default="111 222 333", help="specialist seeds whose greedy schedules enter the nadir")
ap.add_argument("--out_dir", default=str(REPO / "results" / "mo_daniel_v3"))
ap.add_argument("--specialist_dir", default=os.environ.get("MO_SPECIALIST_DIR",
                                                           str(REPO / "daniel" / "trained_network" / "SD2")))
ap.add_argument("--device", default="cuda")
args = ap.parse_args()

sys.argv = [sys.argv[0], "--device", args.device, "--obs_version", "v3", "--reward_version", "v3", "--goal", "mct"]
sys.path.insert(0, str(REPO / "daniel")); sys.path.insert(0, str(REPO / "tools"))
import numpy as np                                          # noqa: E402
import torch                                                # noqa: E402
from params import configs                                  # noqa: E402
import benchmark_v3 as B                                    # noqa: E402
from common_utils import greedy_select_action, load_checkpoint_state_dict      # noqa: E402
from feature_schemas import SCHEMA_V3                       # noqa: E402
from fjsp_env_same_op_nums import FJSPEnvForSameOpNums      # noqa: E402
from model.main_model import DANIEL                         # noqa: E402
import cpsat_baselines_v3 as CP                             # noqa: E402
from analyze_baselines_v3 import cpsat_schedules, rule_baseline     # noqa: E402

OBJ = ("makespan", "carbon", "tardiness")


@torch.no_grad()
def greedy(path, data, idx):
    net = DANIEL(configs).to(configs.device)
    net.load_state_dict(load_checkpoint_state_dict(path, map_location=configs.device, expected_schema=SCHEMA_V3))
    net.eval()
    env = FJSPEnvForSameOpNums(*B.SIZES[data.size])
    state = env.set_initial_data(**data.env_kwargs(idx, "mct"))
    while True:
        pi, _ = net(fea_j=state.fea_j_tensor, op_mask=state.op_mask_tensor, candidate=state.candidate_tensor,
                    fea_m=state.fea_m_tensor, mch_mask=state.mch_mask_tensor, comp_idx=state.comp_idx_tensor,
                    dynamic_pair_mask=state.dynamic_pair_mask_tensor, fea_pairs=state.fea_pairs_tensor)
        state, _, done = env.step(greedy_select_action(pi).cpu().numpy())
        if np.all(done):
            break
    comp = env.true_op_ct[env.env_job_idx, env.job_last_op_id]
    tard = np.maximum(0.0, comp - env.true_due_dates).sum(1)
    return np.stack([env.current_makespan, env.total_carbon, tard], 1)


def main():
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    tag = "" if args.split == "validation" else f"_{args.split}"
    out = out_dir / f"validation_normalisation_points{tag}_{args.size}.json"
    data = B.load_split(args.split, args.size)
    n = len(data) if args.first <= 0 else min(args.first, len(data))
    idx = list(range(n))
    if out.exists():
        rec = json.loads(out.read_text())
        missing = [k for k in data.keys[:n] if k not in rec["instances"]]
        if missing:
            raise SystemExit(f"{out} exists but lacks {len(missing)} instances; it is never overwritten")
        print(f"{out} exists and covers the {n} instances: kept (frozen)")
        return
    cdir = out_dir / "cpsat" / "b1" / f"{args.split}_{args.size}"
    cdir.mkdir(parents=True, exist_ok=True)
    for i in idx:                                            # B1, exactly as tools/cpsat_baselines_v3.py b1
        inst = CP.Inst(data, i)
        f = cdir / f"{inst.name}.json"
        if f.exists():
            continue
        t0 = time.time()
        r = CP.b1_instance(inst, args.limit, args.lex_limit, args.workers)
        r.update(index=i, split=args.split, size=args.size, wall_total_s=time.time() - t0, workers=args.workers,
                 limits={"primary": args.limit, "lex": args.lex_limit}, manifest_sha256=B.file_sha256(B.MANIFEST))
        tmp = f.with_suffix(".tmp"); tmp.write_text(json.dumps(r)); tmp.replace(f)
        print(f"B1 {args.split} {args.size} {inst.name} {time.time() - t0:.0f}s", flush=True)
    spec = {}
    for o in "mct":
        for s in args.seeds.split():
            name = f"{args.size}+mix+bv3_spec_{o}_s{s}"
            path = Path(args.specialist_dir) / f"{name}.pth"
            if not path.exists():
                raise SystemExit(f"specialist checkpoint missing: {path}")
            spec[name] = {"sha256": B.file_sha256(path), "objectives": greedy(path, data, idx)}
    rule = rule_baseline(data, data.keys[:n])
    res = {"definition": "docs/evaluation_protocol_v3.md, Addendum A (A2): section-6 ideal/nadir on the validation "
                         "split; nadir over CP-SAT B1 schedules, the nine specialists' greedy schedules and MWKR/ECT "
                         "(no B2 front on this split)",
           "split": args.split, "size": args.size, "n": n,
           "cpsat_b1": {"limit_s": args.limit, "lex_limit_s": args.lex_limit, "workers": args.workers,
                        "seed": CP.SEED},
           "specialists": {k: v["sha256"] for k, v in spec.items()},
           "manifest_sha256": B.file_sha256(B.MANIFEST), "created": time.strftime("%Y-%m-%d %H:%M:%S"),
           "instances": {}}
    for r_i, i in enumerate(idx):
        key = data.keys[i]
        b1 = json.loads((cdir / f"{Path(key).stem}.json").read_text())
        sch = [s for s in cpsat_schedules(b1, None) if s]
        pts = list(sch) + [dict(zip(OBJ, map(float, v["objectives"][r_i]))) for v in spec.values()]
        pts.append({k: float(rule[k][r_i]) for k in OBJ})
        res["instances"][key] = {"ideal": {k: float(min(s[k] for s in sch)) for k in OBJ},
                                 "nadir": {k: float(max(p[k] for p in pts)) for k in OBJ}}
    tmp = out.with_suffix(".tmp"); tmp.write_text(json.dumps(res, indent=1)); tmp.replace(out)
    print(f"wrote {out} ({n} instances)")


if __name__ == "__main__":
    main()
