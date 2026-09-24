"""STEP C: 20x10 tardiness SENSITIVITY arm at k = 1.20 (the loosest k in the pre-stated band).

Does NOT replace the pre-registered k = 0.95; the pre-registered 20x10 Gate 1 failure stands as the
primary result. These specialists are never used in any composition or in Step 4.

  1. train 4 tardiness specialists at 20x10 (canonical protocol and budget) reading the SEPARATE
     sensitivity manifest; names carry k120s060, so they cannot be confused with k095s060
  2. evaluate the makespan, carbon and new tardiness specialists on the 20x10 train_vali pool with
     the sensitivity due dates (out-tag trainvali-k120s060)
  3. commit the evaluations
Hang guard, GPU checks, resume and per-seed commits come from tools/run_tardiness_specialists.py.
"""
from __future__ import annotations
import subprocess, sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DAN = REPO / "daniel"
sys.path.insert(0, str(REPO / "tools"))
from run_arms_t import guarded, log  # noqa: E402

MAN = DAN / "data/due_dates/due_date_manifest_SENSITIVITY_20x10_k120s060.json"
TAG = "k120s060"
LOGD = Path.home() / ".claude/jobs/programme"


def main():
    r = subprocess.run([sys.executable, "-u", str(REPO / "tools/run_tardiness_specialists.py"),
                        "--sizes", "20x10", "--manifest", str(MAN)], cwd=REPO)
    if r.returncode != 0:
        log("STOPPED: sensitivity specialist training failed"); return 1
    names = [f"20x10+carbon+priority+{o}_s{s}" for o in ("m", "c", f"t_{TAG}") for s in (111, 222, 333, 444)]
    ev = [sys.executable, "../tools/eval_checkpoints.py", "--pool-root", "./data/data_train_vali",
          "--pool", "20x10+carbon+priority", "--out-tag", "trainvali", "--due-dates",
          "--manifest", str(MAN), "--models", *names]
    lf = LOGD / "eval_specialists_20x10_k120s060.log"
    r = guarded(ev, lf, 420, progress="] 20x10")
    if r != "ok" or "EVAL_CHECKPOINTS_DONE" not in lf.read_text():
        log(f"STOPPED: evaluating sensitivity specialists: {r}"); return 1
    res = DAN / f"test_results/SD2/trainvali-{TAG}_20x10+carbon+priority"
    subprocess.run(["git", "add", str(res)], cwd=REPO, check=True)
    subprocess.run(["git", "commit", "-q", "-m",
                    "Sensitivity arm (20x10, k = 1.20): evaluate makespan, carbon and tardiness specialists\n\n"
                    "SENSITIVITY ONLY -- the pre-registered k = 0.95 result stands.\n\n"
                    "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>\n"
                    "Claude-Session: https://claude.ai/code/session_01Rcu6f737di34xSocjiJkEU"],
                   cwd=REPO, check=True)
    log("STEP_C_DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
