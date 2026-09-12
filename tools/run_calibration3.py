"""Step 2: calibrate the three-objective reward weights. This is a correctness gate.

The fine-tuning arms must optimise the SAME trade-off the merge targets. With priority in
the objective there are two weights to set, so the sweep would scale quadratically if done
blind -- instead the closed-form anchors from preference3.py locate the answer and the sweep
only VERIFIES it, one dimension at a time.

Pipeline:
  1. build + evaluate the centroid composition -> per-instance reference metrics
  2. derive closed-form anchors (w_c, w_p) for the target preference
  3. verification sweep: vary each weight around its anchor, holding the other fixed
  4. report the mapping AND the achieved operating point, not just the objective score

    python tools/run_calibration3.py --size 10x5 --pref 0.3333 0.3333 0.3334
"""
from __future__ import annotations
import argparse, subprocess, sys, time
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[1]
DAN = REPO / "daniel"
sys.path.insert(0, str(REPO / "tools"))
from preference3 import Preference, reward_anchors, scalarise  # noqa: E402
from simplex_merge import build, name_for, SPECIALISTS  # noqa: E402

CENTROID = (333, 333, 334)
SEEDS = [111, 222, 333, 444]
CALIB_SEED = 111
CALIB_UPDATES = 250


def res_path(size, tag, pool_tag="trainvali"):
    return (REPO / "daniel/test_results/SD2" / f"{pool_tag}_{size}+carbon+priority" /
            f"Result_DANIELG+{size}+carbon+priority+{tag}_{pool_tag}_{size}+carbon+priority.npy")


def evaluate(size, names, log):
    cmd = [sys.executable, "../tools/eval_checkpoints.py", "--pool-root", "./data/data_train_vali",
           "--pool", f"{size}+carbon+priority", "--out-tag", "trainvali", "--models", *names]
    with open(log, "w") as fh:
        return subprocess.run(cmd, cwd=DAN, stdout=fh, stderr=subprocess.STDOUT).returncode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", default="10x5")
    ap.add_argument("--pref", nargs=3, type=float, default=[1 / 3, 1 / 3, 1 / 3])
    ap.add_argument("--factors", nargs="+", type=float, default=[0.5, 2.0])
    ap.add_argument("--log-dir", default=str(Path.home() / ".claude/jobs/programme"))
    args = ap.parse_args()
    pref = Preference(*args.pref)
    log = Path(args.log_dir); log.mkdir(parents=True, exist_ok=True)
    n_j, n_m = args.size.split("x")

    # ---- 1. reference composition -------------------------------------------------
    print("STEP 2.1  reference composition (simplex centroid)")
    need = []
    for s in SEEDS:
        nm = name_for(args.size, CENTROID, s)
        if not (DAN / "trained_network/SD2" / f"{nm}.pth").exists():
            r = build(args.size, s, CENTROID, SPECIALISTS)
            if r not in ("ok", "exists"):
                print(f"  could not build {nm}: {r}"); return 1
        if not res_path(args.size, f"simplex_m{CENTROID[0]}c{CENTROID[1]}p{CENTROID[2]}_s{s}").exists():
            need.append(nm)
    if need:
        print(f"  evaluating {len(need)} centroid checkpoints ...")
        evaluate(args.size, need, log / "eval_centroid.log")

    per_seed = [np.load(res_path(args.size,
                f"simplex_m{CENTROID[0]}c{CENTROID[1]}p{CENTROID[2]}_s{s}"))[:, :3] for s in SEEDS]
    ref_inst = {s: v for s, v in zip(SEEDS, per_seed)}
    ref_mean = np.stack([v.mean(0) for v in per_seed]).mean(0)
    print(f"  reference metrics (pool mean over {len(SEEDS)} seeds): "
          f"makespan={ref_mean[0]:.1f}  carbon={ref_mean[1]:.1f}  priority={ref_mean[2]:.1f}")

    # ---- 2. closed-form anchors ----------------------------------------------------
    wc0, wp0 = reward_anchors(pref, ref_mean)
    print(f"\nSTEP 2.2  closed-form anchors for preference {pref.tag}")
    print(f"  carbon_reward_weight   = {wc0:.6f}")
    print(f"  priority_reward_weight = {wp0:.6f}")

    # ---- 3. verification sweep -----------------------------------------------------
    cands = [("anchor", wc0, wp0)]
    for f in args.factors:
        cands.append((f"wc x{f:g}", wc0 * f, wp0))
    for f in args.factors:
        cands.append((f"wp x{f:g}", wc0, wp0 * f))
    print(f"\nSTEP 2.3  verification sweep: {len(cands)} runs of {CALIB_UPDATES} updates "
          f"(seed {CALIB_SEED}, init from centroid)")

    init = name_for(args.size, CENTROID, CALIB_SEED)
    tags = []
    for label, wc, wp in cands:
        tag = f"cal3_wc{wc:.6f}_wp{wp:.4f}".replace(".", "p")
        tags.append((label, wc, wp, tag))
        out = DAN / "trained_network/SD2" / f"{args.size}+carbon+priority+{tag}@u{CALIB_UPDATES}.pth"
        if out.exists():
            print(f"  skip {label} (done)"); continue
        cmd = [sys.executable, "train.py", "--config", "../configs/canonical/mcp.json",
               "--train_data_path", f"./data/data_train/SD2/{args.size}+carbon+priority",
               "--validation_data_path", f"./data/data_validation/SD2/{args.size}+carbon+priority",
               "--test_data_path", f"./data/data_final_test/SD2/{args.size}+carbon+priority",
               "--n_j", n_j, "--n_m", n_m, "--data_suffix", "carbon+priority",
               "--carbon_reward_weight", str(wc), "--priority_reward_weight", str(wp),
               "--max_updates", str(CALIB_UPDATES), "--device", "cuda",
               "--budget_checkpoints", str(CALIB_UPDATES), "--seed_train", str(CALIB_SEED),
               "--init_from", init, "--model_suffix", tag]
        print(f"  {label}: wc={wc:.6f} wp={wp:.4f}  start {time.strftime('%H:%M:%S')}", flush=True)
        t0 = time.time()
        with open(log / f"{tag}.log", "w") as fh:
            r = subprocess.run(cmd, cwd=DAN, stdout=fh, stderr=subprocess.STDOUT)
        print(f"    exit={r.returncode} elapsed={time.time()-t0:.0f}s", flush=True)
        if r.returncode != 0:
            return 1

    ev = [f"{args.size}+carbon+priority+{t}@u{CALIB_UPDATES}" for _, _, _, t in tags
          if not res_path(args.size, f"{t}@u{CALIB_UPDATES}").exists()]
    if ev:
        print(f"  evaluating {len(ev)} calibration checkpoints ...")
        evaluate(args.size, ev, log / "eval_calibration.log")

    # ---- 4. report mapping AND operating point --------------------------------------
    print(f"\nSTEP 2.4  results (preference {pref.tag}; lower objective is better)")
    print(f"  {'candidate':<12}{'w_c':>11}{'w_p':>9}{'makespan':>10}{'carbon':>10}"
          f"{'priority':>10}{'obj':>9}   operating point m:c:p vs reference")
    ref_i = ref_inst[CALIB_SEED]
    best = None
    for label, wc, wp, tag in tags:
        p = res_path(args.size, f"{tag}@u{CALIB_UPDATES}")
        if not p.exists():
            print(f"  {label:<12} (missing)"); continue
        M = np.load(p)[:, :3]
        obj = float(scalarise(M, ref_i, pref).mean())
        mu = M.mean(0); rel = mu / ref_mean
        print(f"  {label:<12}{wc:11.6f}{wp:9.4f}{mu[0]:10.1f}{mu[1]:10.1f}{mu[2]:10.1f}{obj:9.4f}"
              f"   {rel[0]:.3f} : {rel[1]:.3f} : {rel[2]:.3f}")
        if best is None or obj < best[0]:
            best = (obj, label, wc, wp, rel)

    if best:
        obj, label, wc, wp, rel = best
        print(f"\n  BEST: {label}  ->  carbon_reward_weight={wc:.6f}  "
              f"priority_reward_weight={wp:.6f}  (objective {obj:.4f})")
        spread = float(np.max(rel) - np.min(rel))
        print(f"  achieved operating point, relative to the reference composition: "
              f"m={rel[0]:.3f}  c={rel[1]:.3f}  p={rel[2]:.3f}")
        print(f"  balance check: max-min across the three axes = {spread:.3f} "
              f"({'balanced -- no single objective is being sacrificed' if spread < 0.15 else 'SKEWED -- one objective is being traded away; inspect before proceeding'})")
        if label == "anchor":
            print("  the closed-form anchor won the sweep: the derivation is validated empirically")
        else:
            print(f"  NOTE: the anchor did NOT win; the empirical optimum is {label}. "
                  f"Use the empirical weights and record the discrepancy.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
