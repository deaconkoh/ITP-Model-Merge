"""Step 3 driver: evaluate the coarse simplex one seed per block, with safety stops.

Written after a GPU driver hang ~10 hr into continuous evaluation. Each seed is a separate
block; between blocks the GPU is health-checked and the finished seed is committed, so an
interruption costs at most the block in progress.

Stops (and does NOT continue to the next seed) if:
  * hang     -- no evaluation completes within --hang-sec (default 420 s = 7 min)
  * drift    -- rolling mean of the last --drift-window evals exceeds --drift-factor x baseline,
                or any single eval exceeds --single-factor x baseline
  * GPU      -- nvidia-smi fails/times out, or temperature >= --max-temp
  * the eval process exits non-zero or without EVAL_CHECKPOINTS_DONE

Baseline 46.8 s/eval is the seed-111 test block (30 evals, max 67.8 s).

    python tools/run_step3_blocks.py --seeds 222 333 444
"""
from __future__ import annotations
import argparse, re, subprocess, sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DAN = REPO / "daniel"
sys.path.insert(0, str(REPO / "tools"))
from simplex_merge import simplex_points, name_for, build, SPECIALISTS_T  # noqa: E402

RES = DAN / "test_results/SD2/trainvali_10x5+carbon+priority"
THIRD, TAG = "p", None          # set from --third/--tag; tardiness runs use trainvali-<tag>
DONE_RE = re.compile(r"\[(\d+)/(\d+)\] (\S+): ([0-9.]+)s")


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def gpu_health(max_temp):
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=temperature.gpu,utilization.gpu,memory.used",
                            "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=30)
    except subprocess.TimeoutExpired:
        return False, "nvidia-smi TIMED OUT (driver unresponsive)"
    if r.returncode != 0:
        return False, f"nvidia-smi exit {r.returncode}: {r.stderr.strip()}"
    t, u, m = [x.strip() for x in r.stdout.strip().split(",")]
    ok = int(t) < max_temp
    return ok, f"{t} C, {u}% util, {m} MiB" + ("" if ok else f"  (>= {max_temp} C)")


def run_block(seed, args, log_dir):
    names = [name_for("10x5", p, seed, THIRD, TAG) for p in simplex_points(0.1)]
    if THIRD == "t":
        # build the compositions for this seed first (idempotent; existing ones are kept)
        for p in simplex_points(0.1):
            r = build("10x5", seed, p, SPECIALISTS_T, third="t", tag=TAG)
            if r not in ("ok", "exists"):
                return False, f"BUILD FAILED {name_for('10x5', p, seed, 't', TAG)}: {r}", []
    logf = log_dir / f"eval_simplex{'_' + TAG if TAG else ''}_s{seed}.log"
    cmd = [sys.executable, "../tools/eval_checkpoints.py", "--pool-root", "./data/data_train_vali",
           "--pool", "10x5+carbon+priority", "--out-tag", "trainvali", "--models", *names]
    if THIRD == "t":
        cmd.append("--due-dates")
    fh = open(logf, "w")
    proc = subprocess.Popen(cmd, cwd=DAN, stdout=fh, stderr=subprocess.STDOUT)
    t_block = time.time(); last_progress = time.time(); seen = 0; times = []
    base = args.baseline
    while True:
        time.sleep(5)
        text = logf.read_text(errors="replace")
        hits = DONE_RE.findall(text)
        if len(hits) > seen:
            for k, n, name, sec in hits[seen:]:
                times.append(float(sec))
                log(f"  s{seed} [{k}/{n}] {float(sec):5.1f}s")
            seen = len(hits); last_progress = time.time()
            if times[-1] > args.single_factor * base:
                proc.kill(); return False, f"DRIFT: single eval {times[-1]:.1f}s > {args.single_factor}x baseline {base}s", times
            w = times[-args.drift_window:]
            if len(w) == args.drift_window and sum(w) / len(w) > args.drift_factor * base:
                proc.kill(); return False, (f"DRIFT: rolling mean of last {len(w)} evals "
                                            f"{sum(w)/len(w):.1f}s > {args.drift_factor}x baseline {base}s"), times
        if proc.poll() is not None:
            fh.close()
            if proc.returncode != 0 or "EVAL_CHECKPOINTS_DONE" not in text:
                return False, f"eval process exited rc={proc.returncode} without EVAL_CHECKPOINTS_DONE", times
            return True, f"block done in {(time.time()-t_block)/60:.1f} min", times
        if time.time() - last_progress > args.hang_sec:
            proc.kill()
            ok, g = gpu_health(args.max_temp)
            return False, f"HANG: no eval completed in {args.hang_sec}s; GPU check: {g}", times


def commit_seed(seed):
    pat = f"*simplex_*{TAG}_s{seed}_trainvali-{TAG}_*.npy" if THIRD == "t" else f"*simplex_*_s{seed}_trainvali_*.npy"
    files = sorted(str(p.relative_to(REPO)) for p in RES.glob(pat))
    subprocess.run(["git", "add", *files], cwd=REPO, check=True)
    what = f"tardiness-triple ({TAG}) " if THIRD == "t" else ""
    msg = (f"Add {what}coarse simplex evaluations for seed {seed} ({len(files)} files)\n\n"
           "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>\n"
           "Claude-Session: https://claude.ai/code/session_01Rcu6f737di34xSocjiJkEU")
    r = subprocess.run(["git", "commit", "-q", "-m", msg], cwd=REPO)
    return r.returncode == 0, len(files)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=[222, 333, 444])
    ap.add_argument("--baseline", type=float, default=46.8)
    ap.add_argument("--hang-sec", type=int, default=420)
    ap.add_argument("--drift-window", type=int, default=10)
    ap.add_argument("--drift-factor", type=float, default=1.5)
    ap.add_argument("--single-factor", type=float, default=2.5)
    ap.add_argument("--max-temp", type=int, default=85)
    ap.add_argument("--gap-sec", type=int, default=120)
    ap.add_argument("--log-dir", default=str(Path.home() / ".claude/jobs/programme"))
    ap.add_argument("--third", choices=["p", "t"], default="p")
    ap.add_argument("--tag", help="due-date tag, required with --third t")
    args = ap.parse_args()
    global THIRD, TAG, RES
    THIRD, TAG = args.third, args.tag
    if THIRD == "t":
        if not TAG:
            raise SystemExit("--third t needs --tag")
        RES = DAN / f"test_results/SD2/trainvali-{TAG}_10x5+carbon+priority"
    log_dir = Path(args.log_dir); log_dir.mkdir(parents=True, exist_ok=True)
    n_grid = len(simplex_points(0.1))

    for i, seed in enumerate(args.seeds):
        if i:
            log(f"gap {args.gap_sec}s before next block"); time.sleep(args.gap_sec)
        ok, g = gpu_health(args.max_temp)
        log(f"GPU health before seed {seed}: {g}")
        if not ok:
            log(f"STOPPED_GPU_UNHEALTHY before seed {seed}"); return 1
        log(f"=== block seed {seed} start ===")
        ok, why, times = run_block(seed, args, log_dir)
        if times:
            log(f"seed {seed}: {len(times)} evals this run, mean {sum(times)/len(times):.1f}s, "
                f"max {max(times):.1f}s")
        if not ok:
            log(f"STOPPED seed {seed}: {why}"); return 1
        otag = f"trainvali-{TAG}" if THIRD == "t" else "trainvali"
        have = sum((RES / f"Result_DANIELG+{name_for('10x5', p, seed, THIRD, TAG)}_{otag}_10x5+carbon+priority.npy").exists()
                   for p in simplex_points(0.1))
        if have != n_grid:
            log(f"STOPPED seed {seed}: only {have}/{n_grid} result files present"); return 1
        cok, nf = commit_seed(seed)
        log(f"seed {seed}: {why}; {have}/{n_grid} present; committed {nf} files (ok={cok})")
        ok, g = gpu_health(args.max_temp)
        log(f"GPU health after seed {seed}: {g}")
        if not ok:
            log(f"STOPPED_GPU_UNHEALTHY after seed {seed}"); return 1
    log("STEP3_BLOCKS_DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
