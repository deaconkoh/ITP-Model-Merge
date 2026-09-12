"""Step 6: does the initialisation advantage hold across preferences, or only at one point?

For each additional preference on the simplex this:
  1. derives the closed-form reward-weight anchors for that preference,
  2. runs the two core arms -- merge-init and from-scratch -- at several seeds.

Only the two core arms are run per preference. The single-specialist controls (Step 4)
answer "is it composition or just warm-starting", which is a question about the mechanism
and does not need repeating at every preference.

IMPORTANT: each preference has its OWN reward weights. Reusing one preference's weights at
another preference is the apples-to-oranges error that Step 2 exists to prevent, so the
anchors are recomputed per preference from the measured reference composition.

    python tools/run_preference_variation.py --plan --size 10x5 --ref-metrics 536.2 1958.2 210.0
    python tools/run_preference_variation.py --run  --size 10x5 --ref-metrics 536.2 1958.2 210.0 \
        --init-merge 10x5+carbon+priority+simplex_m333c333p334_s{seed}
"""
from __future__ import annotations
import argparse, subprocess, sys, time
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[1]
DAN = REPO / "daniel"
sys.path.insert(0, str(REPO / "tools"))
from preference3 import Preference, reward_anchors  # noqa: E402

BUDGETS = "0,50,100,250,500,1000"
# carbon-leaning, makespan-leaning, priority-leaning: spread around the centroid
DEFAULT_PREFS = [(0.5, 0.25, 0.25), (0.25, 0.5, 0.25)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", action="store_true"); ap.add_argument("--run", action="store_true")
    ap.add_argument("--size", default="10x5")
    ap.add_argument("--seeds", nargs="+", type=int, default=[111, 222, 333])
    ap.add_argument("--prefs", nargs="+",
                    help="preference triples as m,c,p e.g. 0.5,0.25,0.25")
    ap.add_argument("--ref-metrics", nargs=3, type=float, required=True,
                    help="pool-mean [makespan carbon priority] at the reference composition")
    ap.add_argument("--init-merge", help="composition checkpoint to warm-start from")
    ap.add_argument("--arms", nargs="+", default=["merge", "scratch"])
    ap.add_argument("--max-updates", type=int, default=1000)
    ap.add_argument("--log-dir", default=str(Path.home() / ".claude/jobs/programme"))
    args = ap.parse_args()

    prefs = ([Preference(*[float(x) for x in p.split(",")]) for p in args.prefs]
             if args.prefs else [Preference(*p) for p in DEFAULT_PREFS])
    ref = np.array(args.ref_metrics)

    print(f"reference composition metrics: makespan={ref[0]:.1f} carbon={ref[1]:.1f} "
          f"priority={ref[2]:.1f}")
    plan = []
    for pref in prefs:
        wc, wp = reward_anchors(pref, ref)
        print(f"  preference {pref.tag}: carbon_reward_weight={wc:.6f}  "
              f"priority_reward_weight={wp:.6f}")
        for arm in args.arms:
            for s in args.seeds:
                plan.append((pref, wc, wp, arm, s))

    if args.plan or not args.run:
        est = len(plan) * (14 if args.size == "10x5" else 72)
        print(f"\nPLAN: {len(prefs)} preferences x {len(args.arms)} arms x {len(args.seeds)} seeds "
              f"= {len(plan)} runs")
        print(f"  estimated wall-clock: {est} min ({est/60:.1f} hr); checkpoints: {len(plan)*6}")
        return

    if "merge" in args.arms and not args.init_merge:
        print("--init-merge is required when the merge arm is included"); return 1

    log = Path(args.log_dir); log.mkdir(parents=True, exist_ok=True)
    n_j, n_m = args.size.split("x")
    for pref, wc, wp, arm, seed in plan:
        tag = f"s6_{pref.tag}_{arm}_s{seed}"
        out = DAN / "trained_network/SD2" / f"{args.size}+carbon+priority+{tag}@u{args.max_updates}.pth"
        if out.exists():
            print(f"skip {tag} (complete)"); continue
        cmd = [sys.executable, "train.py", "--config", "../configs/canonical/mcp.json",
               "--train_data_path", f"./data/data_train/SD2/{args.size}+carbon+priority",
               "--validation_data_path", f"./data/data_validation/SD2/{args.size}+carbon+priority",
               "--test_data_path", f"./data/data_final_test/SD2/{args.size}+carbon+priority",
               "--n_j", n_j, "--n_m", n_m, "--data_suffix", "carbon+priority",
               "--carbon_reward_weight", str(wc), "--priority_reward_weight", str(wp),
               "--max_updates", str(args.max_updates), "--device", "cuda",
               "--budget_checkpoints", BUDGETS, "--seed_train", str(seed),
               "--model_suffix", tag]
        if arm == "merge":
            init = args.init_merge
            cmd += ["--init_from", init.format(seed=seed) if "{seed}" in init else init]
        print(f"=== {tag} start {time.strftime('%H:%M:%S')} ===", flush=True)
        t0 = time.time()
        with open(log / f"{tag}.log", "w") as fh:
            r = subprocess.run(cmd, cwd=DAN, stdout=fh, stderr=subprocess.STDOUT)
        print(f"    exit={r.returncode} elapsed={time.time()-t0:.0f}s", flush=True)
        if r.returncode != 0:
            print("ABORTING: run failed; prior runs intact, resumable"); return 1
    print("PREFERENCE_VARIATION_DONE")


if __name__ == "__main__":
    raise SystemExit(main() or 0)
