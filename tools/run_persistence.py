"""Step 5 (RQ3): does the starting composition leave a lasting imprint?

Fine-tunes from several starting compositions spread across the simplex -- not just the
optimum -- and asks whether they converge to equivalent final performance or whether the
initialisation persists.

Expectation carried forward from Track C: per-seed spread shrank monotonically with budget
(0.0134 -> 0.0013), so CONVERGENCE is the current expectation and lasting path dependence
would be the surprise. The analysis therefore reports spread at every budget, not just the
endpoint.

Default starting set: the simplex optimum plus the three vertices' neighbourhoods and the
centroid, so the spread of starting points is wide enough for persistence to show if it
exists.

    python tools/run_persistence.py --plan --size 10x5 --wc W --wp W
    python tools/run_persistence.py --run  --size 10x5 --wc W --wp W
"""
from __future__ import annotations
import argparse, subprocess, sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DAN = REPO / "daniel"
sys.path.insert(0, str(REPO / "tools"))
from simplex_merge import name_for, build, SPECIALISTS  # noqa: E402

BUDGETS = "0,50,100,250,500,1000"
# spread across the simplex: centroid, three lopsided corners, one edge point
DEFAULT_STARTS = [(333, 333, 334), (600, 200, 200), (200, 600, 200),
                  (200, 200, 600), (500, 500, 0)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", action="store_true"); ap.add_argument("--run", action="store_true")
    ap.add_argument("--size", default="10x5")
    ap.add_argument("--seeds", nargs="+", type=int, default=[111, 222, 333])
    ap.add_argument("--starts", nargs="+", help="permille triples, e.g. 333,333,334")
    ap.add_argument("--wc", type=float, required=True)
    ap.add_argument("--wp", type=float, required=True)
    ap.add_argument("--tag-prefix", default="s5")
    ap.add_argument("--max-updates", type=int, default=1000)
    ap.add_argument("--log-dir", default=str(Path.home() / ".claude/jobs/programme"))
    args = ap.parse_args()

    starts = ([tuple(int(x) for x in s.split(",")) for s in args.starts]
              if args.starts else DEFAULT_STARTS)
    for s in starts:
        assert sum(s) == 1000, f"weights must sum to 1000 permille: {s}"

    todo = [(p, s) for p in starts for s in args.seeds]
    if args.plan or not args.run:
        est = len(todo) * (14 if args.size == "10x5" else 72)
        print(f"PLAN: {len(starts)} starting compositions x {len(args.seeds)} seeds "
              f"= {len(todo)} runs")
        for p in starts:
            print(f"  start m={p[0]/10:5.1f}% c={p[1]/10:5.1f}% p={p[2]/10:5.1f}%")
        print(f"  estimated wall-clock: {est} min ({est/60:.1f} hr); "
              f"checkpoints: {len(todo)*6}")
        return

    log = Path(args.log_dir); log.mkdir(parents=True, exist_ok=True)
    n_j, n_m = args.size.split("x")
    for pts, seed in todo:
        init = name_for(args.size, pts, seed)
        if not (DAN / "trained_network/SD2" / f"{init}.pth").exists():
            r = build(args.size, seed, pts, SPECIALISTS)
            if r not in ("ok", "exists"):
                print(f"  could not build {init}: {r}"); return 1
        tag = f"{args.tag_prefix}_m{pts[0]:03d}c{pts[1]:03d}p{pts[2]:03d}_s{seed}"
        out = DAN / "trained_network/SD2" / f"{args.size}+carbon+priority+{tag}@u{args.max_updates}.pth"
        if out.exists():
            print(f"skip {tag} (complete)"); continue
        cmd = [sys.executable, "train.py", "--config", "../configs/canonical/mcp.json",
               "--train_data_path", f"./data/data_train/SD2/{args.size}+carbon+priority",
               "--validation_data_path", f"./data/data_validation/SD2/{args.size}+carbon+priority",
               "--test_data_path", f"./data/data_final_test/SD2/{args.size}+carbon+priority",
               "--n_j", n_j, "--n_m", n_m, "--data_suffix", "carbon+priority",
               "--carbon_reward_weight", str(args.wc), "--priority_reward_weight", str(args.wp),
               "--max_updates", str(args.max_updates), "--device", "cuda",
               "--budget_checkpoints", BUDGETS, "--seed_train", str(seed),
               "--init_from", init, "--model_suffix", tag]
        print(f"=== {tag} start {time.strftime('%H:%M:%S')} ===", flush=True)
        t0 = time.time()
        with open(log / f"{tag}.log", "w") as fh:
            r = subprocess.run(cmd, cwd=DAN, stdout=fh, stderr=subprocess.STDOUT)
        print(f"    exit={r.returncode} elapsed={time.time()-t0:.0f}s", flush=True)
        if r.returncode != 0:
            print("ABORTING: run failed; prior runs intact, resumable"); return 1
    print("PERSISTENCE_DONE")


if __name__ == "__main__":
    raise SystemExit(main() or 0)
