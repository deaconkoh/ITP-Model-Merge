"""Step 4/5/6: launch fine-tuning arms with budget checkpointing.

Arms (Step 4):
  merge     init from the best simplex composition        <- the hypothesis
  scratch   no init, three-objective reward from random   <- the expensive baseline
  spec_m    init from the makespan specialist             <- control
  spec_c    init from the carbon specialist               <- control
  spec_p    init from the priority specialist             <- control

The three single-specialist controls are what separate "composition helps" from
"any pretrained policy is an equally good start".

Sanity checks built into the downstream analysis (not here):
  merge   @ budget 0 must reproduce the merge exactly
  scratch @ budget 0 must sit at random-init quality

    python tools/run_arms.py --plan --size 10x5 --init-merge <name> --wc 0.0028 --wp 2.55
    python tools/run_arms.py --run  --size 10x5 --init-merge <name> --wc 0.0028 --wp 2.55
"""
from __future__ import annotations
import argparse, subprocess, sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DAN = REPO / "daniel"
BUDGETS = "0,50,100,250,500,1000"
SEEDS = [111, 222, 333, 444]


def arm_specs(size, init_merge):
    return {
        "merge":   init_merge,
        "scratch": None,
        "spec_m":  f"{size}+carbon+priority+m_s{{seed}}",
        "spec_c":  f"{size}+carbon+priority+c_s{{seed}}",
        "spec_p":  f"{size}+carbon+priority+p_s{{seed}}",
    }


def cmd_for(size, arm, init, seed, wc, wp, tag_prefix, max_updates):
    n_j, n_m = size.split("x")
    c = [sys.executable, "train.py",
         "--config", "../configs/canonical/mcp.json",
         "--train_data_path", f"./data/data_train/SD2/{size}+carbon+priority",
         "--validation_data_path", f"./data/data_validation/SD2/{size}+carbon+priority",
         "--test_data_path", f"./data/data_final_test/SD2/{size}+carbon+priority",
         "--n_j", n_j, "--n_m", n_m, "--data_suffix", "carbon+priority",
         "--carbon_reward_weight", str(wc), "--priority_reward_weight", str(wp),
         "--max_updates", str(max_updates), "--device", "cuda",
         "--budget_checkpoints", BUDGETS,
         "--seed_train", str(seed),
         "--model_suffix", f"{tag_prefix}_{arm}_s{seed}"]
    if init:
        c += ["--init_from", init.format(seed=seed) if "{seed}" in init else init]
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", action="store_true"); ap.add_argument("--run", action="store_true")
    ap.add_argument("--size", default="10x5")
    ap.add_argument("--arms", nargs="+",
                    default=["merge", "scratch", "spec_m", "spec_c", "spec_p"])
    ap.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    ap.add_argument("--init-merge", required=True, help="checkpoint name of the chosen composition")
    ap.add_argument("--wc", type=float, required=True, help="calibrated carbon_reward_weight")
    ap.add_argument("--wp", type=float, required=True, help="calibrated priority_reward_weight")
    ap.add_argument("--tag-prefix", default="s4")
    ap.add_argument("--max-updates", type=int, default=1000)
    ap.add_argument("--log-dir", default=str(Path.home() / ".claude/jobs/programme"))
    args = ap.parse_args()

    specs = arm_specs(args.size, args.init_merge)
    todo = [(a, s) for a in args.arms for s in args.seeds]
    if args.plan or not args.run:
        print(f"PLAN: {len(todo)} runs of {args.max_updates} updates, budgets [{BUDGETS}]")
        print(f"  size={args.size}  wc={args.wc}  wp={args.wp}  goal=mcp")
        for a in args.arms:
            init = specs[a]
            print(f"  arm {a:<8} init_from = {init if init else '(none: from scratch)'}")
        est = len(todo) * (14 if args.size == "10x5" else 72)
        print(f"  estimated wall-clock: {est} min ({est/60:.1f} hr)")
        print(f"  checkpoints produced: {len(todo)} x 6 budgets = {len(todo)*6}")
        return

    log = Path(args.log_dir); log.mkdir(parents=True, exist_ok=True)
    for arm, seed in todo:
        name = f"{args.size}+carbon+priority+{args.tag_prefix}_{arm}_s{seed}"
        if (DAN / "trained_network/SD2" / f"{name}@u{args.max_updates}.pth").exists():
            print(f"skip {arm} s{seed} (complete)"); continue
        c = cmd_for(args.size, arm, specs[arm], seed, args.wc, args.wp,
                    args.tag_prefix, args.max_updates)
        print(f"=== {arm} s{seed} start {time.strftime('%H:%M:%S')} ===", flush=True)
        t0 = time.time()
        with open(log / f"{args.tag_prefix}_{arm}_s{seed}.log", "w") as fh:
            r = subprocess.run(c, cwd=DAN, stdout=fh, stderr=subprocess.STDOUT)
        print(f"    exit={r.returncode} elapsed={time.time()-t0:.0f}s", flush=True)
        if r.returncode != 0:
            print("ABORTING: a run failed; prior runs are intact and this is resumable")
            return 1
    print("ARMS_DONE")


if __name__ == "__main__":
    raise SystemExit(main() or 0)
