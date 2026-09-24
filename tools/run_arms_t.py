"""STEP 4 (tardiness triple, 10x5): fine-tuning arms on the three-objective reward.

Arms (all on goal mct, preference (1/3,1/3,1/3), calibrated weights, 1000 updates, budget checkpoints
at 0/50/100/250/500/1000):
  merge    fine-tune from the VALIDATION-selected composition for that seed (Step A)
  scratch  train from scratch
  spec_m   fine-tune from the makespan specialist
  spec_c   fine-tune from the carbon specialist
  spec_t   fine-tune from the tardiness specialist          <- the key control

Names: 10x5+carbon+priority+s4t_<arm>_k125s060_s<seed>[@u<N>]. Due dates are READ from the frozen
manifest by train.py and by the evaluator.

Safety nets: hang guard (training log or evaluation silent for 7 min -> kill and STOP), GPU health
check before every run, a .done marker written only after training, evaluation of all six budget
checkpoints on the n=100 train_vali pool, and a commit -- so a resumed run skips only fully
finished (arm, seed) pairs. Seeds are the outer loop, so every finished seed is a complete comparison.
"""
from __future__ import annotations
import argparse, hashlib, json, subprocess, sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DAN = REPO / "daniel"
CKPT = DAN / "trained_network/SD2"
PROV = REPO / "results/provenance/step4_arms"
MANIFEST = DAN / "data/due_dates/due_date_manifest.json"
SEL = REPO / "results/step4/merge_init_selection.json"
TAG = "k125s060"
BUDGETS = [0, 50, 100, 250, 500, 1000]
ARMS = ["merge", "scratch", "spec_m", "spec_c", "spec_t"]
SEEDS = [111, 222, 333, 444]


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def gpu_ok():
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=temperature.gpu,utilization.gpu,memory.used",
                            "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=30)
    except subprocess.TimeoutExpired:
        return False, "nvidia-smi TIMED OUT"
    if r.returncode != 0:
        return False, f"nvidia-smi exit {r.returncode}"
    t, u, m = [x.strip() for x in r.stdout.strip().split(",")]
    return int(t) < 85, f"{t} C, {u}% util, {m} MiB"


def guarded(cmd, logfile, hang_sec, progress=None):
    """Run cmd; kill if the log (or its progress-line count) stops changing for hang_sec."""
    with open(logfile, "w") as fh:
        proc = subprocess.Popen(cmd, cwd=DAN, stdout=fh, stderr=subprocess.STDOUT)
        last, t_last = None, time.time()
        while proc.poll() is None:
            time.sleep(10)
            txt = Path(logfile).read_text(errors="replace")
            mark = txt.count(progress) if progress else len(txt)
            if mark != last:
                last, t_last = mark, time.time()
            elif time.time() - t_last > hang_sec:
                proc.kill()
                return "hang"
    return "ok" if proc.returncode == 0 else f"exit {proc.returncode}"


def init_for(arm, seed, sel):
    if arm == "merge":
        return sel[str(seed)]["checkpoint"]
    if arm == "scratch":
        return None
    return {"spec_m": f"10x5+carbon+priority+m_s{seed}", "spec_c": f"10x5+carbon+priority+c_s{seed}",
            "spec_t": f"10x5+carbon+priority+t_{TAG}_s{seed}"}[arm]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wc", type=float, required=True)
    ap.add_argument("--wt", type=float, required=True)
    ap.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    ap.add_argument("--arms", nargs="+", default=ARMS)
    ap.add_argument("--hang-sec", type=int, default=420)
    ap.add_argument("--log-dir", default=str(Path.home() / ".claude/jobs/programme"))
    args = ap.parse_args()
    sel = json.loads(SEL.read_text())
    logdir = Path(args.log_dir); logdir.mkdir(parents=True, exist_ok=True)
    PROV.mkdir(parents=True, exist_ok=True)
    for seed in args.seeds:
        for arm in args.arms:
            suffix = f"s4t_{arm}_{TAG}_s{seed}"
            name = f"10x5+carbon+priority+{suffix}"
            done = logdir / f"{name}.done"
            if done.exists():
                log(f"skip {suffix} (complete)"); continue
            ok, g = gpu_ok(); log(f"GPU before {suffix}: {g}")
            if not ok:
                log(f"STOPPED: GPU unhealthy before {suffix}"); return 1
            init = init_for(arm, seed, sel)
            cmd = [sys.executable, "train.py", "--config", "../configs/canonical/mct.json",
                   "--train_data_path", "./data/data_train/SD2/10x5+carbon+priority",
                   "--validation_data_path", "./data/data_validation/SD2/10x5+carbon+priority",
                   "--test_data_path", "./data/data_final_test/SD2/10x5+carbon+priority",
                   "--n_j", "10", "--n_m", "5", "--data_suffix", "carbon+priority",
                   "--carbon_reward_weight", str(args.wc), "--tardiness_reward_weight", str(args.wt),
                   "--max_updates", "1000", "--device", "cuda",
                   "--budget_checkpoints", ",".join(map(str, BUDGETS)),
                   "--seed_train", str(seed), "--model_suffix", suffix]
            if init:
                cmd += ["--init_from", init]
            t0 = time.time()
            r = guarded(cmd, logdir / f"train_{name}.log", args.hang_sec)
            if r != "ok":
                log(f"STOPPED: training {suffix}: {r}"); return 1
            t_train = time.time() - t0
            missing = [b for b in BUDGETS if not (CKPT / f"{name}@u{b}.pth").exists()]
            if missing:
                log(f"STOPPED: {suffix} missing budget checkpoints {missing}"); return 1
            ev = [sys.executable, "../tools/eval_checkpoints.py", "--pool-root", "./data/data_train_vali",
                  "--pool", "10x5+carbon+priority", "--out-tag", "trainvali", "--due-dates",
                  "--models", *[f"{name}@u{b}" for b in BUDGETS]]
            r = guarded(ev, logdir / f"eval_{name}.log", args.hang_sec, progress="] 10x5")
            if r != "ok" or "EVAL_CHECKPOINTS_DONE" not in (logdir / f"eval_{name}.log").read_text():
                log(f"STOPPED: evaluating {suffix}: {r}"); return 1
            rec = {"model_name": name, "arm": arm, "seed": seed, "init_from": init,
                   "carbon_reward_weight": args.wc, "tardiness_reward_weight": args.wt,
                   "budget_checkpoint_sha256": {b: sha(CKPT / f"{name}@u{b}.pth") for b in BUDGETS},
                   "due_date_manifest_sha256": sha(MANIFEST), "train_seconds": round(t_train, 1),
                   "finished": time.strftime("%Y-%m-%d %H:%M:%S"), "command": " ".join(cmd)}
            (PROV / f"{name}.json").write_text(json.dumps(rec, indent=2))
            res = sorted((DAN / f"test_results/SD2/trainvali-{TAG}_10x5+carbon+priority").glob(
                f"*{suffix}@u*"))
            subprocess.run(["git", "add", str(PROV / f"{name}.json"), *map(str, res)], cwd=REPO, check=True)
            subprocess.run(["git", "commit", "-q", "-m",
                            f"Step 4 arm {arm}, seed {seed}: train + evaluate budget checkpoints\n\n"
                            "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>\n"
                            "Claude-Session: https://claude.ai/code/session_01Rcu6f737di34xSocjiJkEU"],
                           cwd=REPO, check=True)
            done.write_text(json.dumps({"train_seconds": round(t_train, 1)}))
            log(f"DONE {suffix}: train {t_train/60:.1f} min, total {(time.time()-t0)/60:.1f} min; committed")
    log("STEP4_ARMS_DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
