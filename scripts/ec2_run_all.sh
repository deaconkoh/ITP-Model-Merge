#!/usr/bin/env bash
# Warm-start study on benchmark v3 (docs/warmstart_protocol_v3.md, PRE-REGISTERED), one command on a fresh GPU machine:
#
#   cd <repo> && bash scripts/ec2_run_all.sh [--sizes "10x5"] [--arms all] [--seeds "111 222 333"] [--jobs N]
#                                           [--smoke] [--with-shared-base] [--shutdown]
#
# Foreground: Python 3.14.4 venv (uv) + pinned requirements -> pre-flight (GPU, disk, every input hash, test suites)
# -> detaches into the background (survives SSH disconnects) and prints the log and status paths.
# Background: calibration (DANIEL-M, timed) -> validation normalisation points -> every arm (seed 111 first) ->
# evaluation -> E0 -> tools/analyze_warmstart_v3.py -> results/warmstart_v3.md -> local commits (never pushes)
# -> "DONE: git push when ready".
#
#   --sizes "10x5 20x10"   default 10x5          --arms "B CM BBM D E CC CT BBC BBT E0" (default all)
#   --jobs N               N arms in parallel on one GPU (record: timing comparisons use one N for all arms)
#   --smoke                tiny end-to-end check (~5-8 min): seed 111 unless --seeds, 40 updates, 3 validation
#                          instances, development split; separate names (bv3_wss_*) and outputs
#                          (results/warmstart_v3_smoke[.md]), so it never affects the full run
#   --with-shared-base     add the optional E-shared arm
#   --shutdown             sudo shutdown -h +2 after the final commit
#
# Contrastive fine-tuning arms (docs/contrastive_protocol_v3.md), their own mode and outputs:
#   bash scripts/ec2_run_all.sh --arms CL-B,CL-D,CL-E --seeds 111 --jobs 3      [--smoke: 5 updates per arm]
#   -> results/contrastive_v3/, results/contrastive_v3.md, logs/contrastive_v3_STATUS.txt
# Merge-category arms (docs/merge_categories_protocol_v3.md), contrastive fine-tuning, their own mode and outputs:
#   bash scripts/ec2_run_all.sh --arms SB-AVG,TA,TIES --seeds 111 --jobs 3       [--smoke: 20-update base/fine-tunes,
#   2 lambdas, 5 CL updates] -> results/merge_categories_v3/, results/merge_categories_v3.md,
#   checkpoints/merge_categories_v3/ (with SHA256SUMS.json), logs/merge_categories_v3_STATUS.txt
# Converged contrastive arms (docs/contrastive_converged_protocol_v3.md), the PPO stopping rule, cap 8,000 updates:
#   bash scripts/ec2_run_all.sh --arms CC-B,CC-D,CC-E,CC-SBAVG,CC-TA,CC-TIES --seeds 111 --jobs 3   [--smoke: 5 updates]
#   -> results/contrastive_converged_v3/, results/contrastive_converged_v3.md, checkpoints/contrastive_converged_v3/
# Merge-failure test (docs/merge_failure_test_protocol_v3.md): update-0 diagnostics + three one-change arms from the
# Model Soup start, the converged runs' stopping rule:
#   bash scripts/ec2_run_all.sh --arms MF-SP,MF-NA,MF-PR --seeds 111 --jobs 3                 [--smoke: 5 updates]
#   -> results/merge_failure_test_v3/, results/merge_failure_test_v3.md, checkpoints/merge_failure_test_v3/
# Resume (also after a spot interruption on the same volume): run the same command again.
# Progress: bash scripts/ec2_status.sh
set -u -o pipefail

SIZES="10x5"; ARMS="all"; SEEDS="111 222 333"; JOBS=1; SMOKE=0; SHARED=0; SHUTDOWN=0; WORKER=0; LOG=""
SEEDS_GIVEN=0
ARGS=("$@")
while [ $# -gt 0 ]; do
  case "$1" in
    --sizes) SIZES="$2"; shift 2 ;;
    --arms) ARMS="$2"; shift 2 ;;
    --seeds) SEEDS="$2"; SEEDS_GIVEN=1; shift 2 ;;
    --jobs) JOBS="$2"; shift 2 ;;
    --smoke) SMOKE=1; shift ;;
    --with-shared-base) SHARED=1; shift ;;
    --shutdown) SHUTDOWN=1; shift ;;
    --worker) WORKER=1; shift ;;
    --log) LOG="$2"; shift 2 ;;
    -h|--help) sed -n 2,21p "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

[ "$SMOKE" = 1 ] && [ "$SEEDS_GIVEN" = 0 ] && SEEDS="111"      # the ~5-minute smoke check uses one seed
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO" || exit 2
GITDIR="$(git rev-parse --absolute-git-dir)"
STATE="$GITDIR/warmstart_v3"; mkdir -p "$STATE" logs
STATUS_FILE="logs/warmstart_v3_STATUS.txt"
STUDY="warmstart_v3"                                            # contrastive_v3 for the CL arms
case "$ARMS" in *CL*) STUDY="contrastive_v3"; STATUS_FILE="logs/contrastive_v3_STATUS.txt" ;; esac
case "$ARMS" in *SB-AVG*|*SBAVG*|*TA*|*TIES*) STUDY="merge_categories_v3"; STATUS_FILE="logs/merge_categories_v3_STATUS.txt" ;; esac
case "$ARMS" in *CC-*|*CCB*|*CCD*|*CCE*) STUDY="contrastive_converged_v3"; STATUS_FILE="logs/contrastive_converged_v3_STATUS.txt" ;; esac
case "$ARMS" in *MF-*|*MFSP*|*MFNA*|*MFPR*) STUDY="merge_failure_test_v3"; STATUS_FILE="logs/merge_failure_test_v3_STATUS.txt" ;; esac
PY="$REPO/.venv/bin/python"
export WANDB_MODE=offline HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1

say() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

# ------------------------------------------------------------------ background worker
if [ "$WORKER" = 1 ]; then
  exec 8>"$STATE/run.lock"
  flock -n 8 || { say "another worker holds $STATE/run.lock; exiting"; exit 2; }
  RUNNER_PID=""
  on_signal() {
    trap - INT TERM
    say "INTERRUPTED: stopping the runner (no shutdown). Re-run the same command to resume."
    [ -n "$RUNNER_PID" ] && kill -TERM "$RUNNER_PID" 2>/dev/null && wait "$RUNNER_PID" 2>/dev/null
    exit 130
  }
  trap on_signal INT TERM
  R_ARGS=(--sizes "$SIZES" --arms "$ARMS" --seeds "$SEEDS" --jobs "$JOBS" --log "$LOG")
  [ "$SMOKE" = 1 ] && R_ARGS+=(--smoke)
  [ "$SHARED" = 1 ] && R_ARGS+=(--with_shared_base)
  say "worker started: ${R_ARGS[*]}"
  "$PY" tools/ec2_runner.py "${R_ARGS[@]}" & RUNNER_PID=$!
  wait "$RUNNER_PID"; rc=$?; RUNNER_PID=""
  say "runner finished (rc $rc)"
  A_ARGS=(); if [ "$SMOKE" = 1 ]; then               # smoke outputs are separate: results/<study>_smoke(.md)
    A_ARGS=(--eval_split development --results "results/${STUDY}_smoke" --out "results/${STUDY}_smoke.md"
            --norm results/${STUDY}_smoke/norm/validation_normalisation_points_development_*.json)
  fi
  ANALYSIS="tools/analyze_warmstart_v3.py"; [ "$STUDY" = contrastive_v3 ] && ANALYSIS="tools/analyze_contrastive_v3.py"
  [ "$STUDY" = merge_categories_v3 ] && ANALYSIS="tools/analyze_merge_categories_v3.py"
  [ "$STUDY" = contrastive_converged_v3 ] && ANALYSIS="tools/analyze_contrastive_converged_v3.py"
  [ "$STUDY" = merge_failure_test_v3 ] && ANALYSIS="tools/analyze_merge_failure_test_v3.py"
  say "analysis: $ANALYSIS"
  timeout 3600 "$PY" "$ANALYSIS" "${A_ARGS[@]}" || say "analysis FAILED (see above); results are committed"
  say "DONE: git push when ready"
  [ "$SHUTDOWN" = 1 ] && say "after this final commit: shutting down in 2 minutes"
  for p in "results/$STUDY" "results/$STUDY.md" "results/${STUDY}_smoke" "results/${STUDY}_smoke.md" \
           "checkpoints/$STUDY" "checkpoints/${STUDY}_smoke" "$STATUS_FILE" logs/${STUDY}_*.log; do
    [ -e "$p" ] && git add -A -- "$p"
  done
  git diff --cached --quiet || git commit -q -m "${STUDY//_/ }: analysis and final status" \
    -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
  [ "$SHUTDOWN" = 1 ] && ${WS_SHUTDOWN_CMD:-sudo shutdown -h +2}
  exit 0
fi

# ------------------------------------------------------------------ foreground: setup
exec 9>"$STATE/run.lock"
if ! flock -n 9; then
  say "a run is already in progress. Progress: bash scripts/ec2_status.sh"; exit 0
fi
exec 9>&-                                                       # release; the worker takes the lock
LOG="logs/${STUDY}_$(date +%Y%m%d_%H%M%S).log"
die() { say "PRE-FLIGHT FAILED: $*  (nothing started)" | tee -a "$LOG"; exit 2; }

if ! command -v uv >/dev/null 2>&1 && [ ! -x "$HOME/.local/bin/uv" ]; then
  say "installing uv"
  curl -LsSf https://astral.sh/uv/install.sh | sh >>"$LOG" 2>&1 || die "uv installation failed"
fi
UV="$(command -v uv || echo "$HOME/.local/bin/uv")"
if [ ! -x "$PY" ] || [ "$("$PY" -c 'import platform; print(platform.python_version())' 2>/dev/null)" != "3.14.4" ]; then
  say "creating .venv (Python 3.14.4)"
  "$UV" venv --python 3.14.4 .venv >>"$LOG" 2>&1 || die "uv venv failed"
fi
REQ_SHA="$(sha256sum requirements-ec2.txt | cut -d' ' -f1)"
if [ "$(cat .venv/.requirements_sha 2>/dev/null)" != "$REQ_SHA" ]; then
  say "installing requirements-ec2.txt"
  "$UV" pip install --python "$PY" --index-strategy unsafe-best-match -r requirements-ec2.txt >>"$LOG" 2>&1 \
    || die "pip install failed (see $LOG)"
  echo "$REQ_SHA" > .venv/.requirements_sha
fi
mkdir -p daniel/trained_network/SD2
cp -n checkpoints/step4_v1/*.pth daniel/trained_network/SD2/ 2>/dev/null   # git-ignored location the v1 test reads

# ------------------------------------------------------------------ pre-flight
say "pre-flight" | tee -a "$LOG"
"$PY" -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null || die "GPU not visible to torch"
NEED_GB=$([ "$SMOKE" = 1 ] && echo 2 || echo 30)
AVAIL=$(df -BG --output=avail "$REPO" | tail -1 | tr -dc 0-9)
[ "$AVAIL" -ge "$NEED_GB" ] || die "only ${AVAIL} GB free (need $NEED_GB)"
DIRTY=$(git status --porcelain -- . ':!logs/warmstart_v3_*' ':!results/warmstart_v3' ':!results/warmstart_v3.md' \
        ':!results/warmstart_v3_smoke' ':!results/warmstart_v3_smoke.md' \
        ':!logs/contrastive_v3_*' ':!results/contrastive_v3' ':!results/contrastive_v3.md' \
        ':!results/contrastive_v3_smoke' ':!results/contrastive_v3_smoke.md' \
        ':!logs/merge_categories_v3_*' ':!results/merge_categories_v3' ':!results/merge_categories_v3.md' \
        ':!results/merge_categories_v3_smoke' ':!results/merge_categories_v3_smoke.md' \
        ':!checkpoints/merge_categories_v3' ':!checkpoints/merge_categories_v3_smoke' \
        ':!logs/contrastive_converged_v3_*' ':!results/contrastive_converged_v3' ':!results/contrastive_converged_v3.md' \
        ':!results/contrastive_converged_v3_smoke' ':!results/contrastive_converged_v3_smoke.md' \
        ':!checkpoints/contrastive_converged_v3' ':!checkpoints/contrastive_converged_v3_smoke' \
        ':!logs/merge_failure_test_v3_*' ':!results/merge_failure_test_v3' ':!results/merge_failure_test_v3.md' \
        ':!results/merge_failure_test_v3_smoke' ':!results/merge_failure_test_v3_smoke.md' \
        ':!checkpoints/merge_failure_test_v3' ':!checkpoints/merge_failure_test_v3_smoke' \
        ':!logs/baselines_v3_console.out')
[ -z "$DIRTY" ] || die "git tree not clean: $(echo "$DIRTY" | head -5 | tr '\n' ' ')"
"$PY" tools/ec2_inputs.py verify >>"$LOG" 2>&1 || die "input hash verification failed (see $LOG)"
"$PY" tools/baselines_v3_status.py verify >>"$LOG" 2>&1 || die "v3 manifest verification failed"
if [ "$STUDY" = contrastive_v3 ]; then
  "$PY" tools/freeze_contrastive_v3.py verify >>"$LOG" 2>&1 || die "configs/contrastive_v3.json verification failed"
fi
if [ "$STUDY" = contrastive_converged_v3 ]; then
  "$PY" tools/freeze_contrastive_converged_v3.py verify >>"$LOG" 2>&1 || die "configs/contrastive_converged_v3.json verification failed"
fi
if [ "$STUDY" = merge_failure_test_v3 ]; then
  "$PY" tools/freeze_contrastive_converged_v3.py verify >>"$LOG" 2>&1 || die "configs/contrastive_converged_v3.json verification failed"
  "$PY" tools/freeze_merge_failure_test_v3.py verify >>"$LOG" 2>&1 || die "configs/merge_failure_test_v3.json verification failed"
fi
if [ "$STUDY" = merge_categories_v3 ]; then
  "$PY" tools/freeze_merge_categories_v3.py verify >>"$LOG" 2>&1 || die "configs/merge_categories_v3.json verification failed"
fi
OUT=$("$PY" -m unittest tests.test_benchmark_v3 tests.test_baselines_v3 tests.test_mo_daniel_v3 tests.test_warmstart_v3 \
      tests.test_contrastive_v3 tests.test_merge_categories_v3 tests.test_contrastive_converged_v3 \
      tests.test_merge_failure_test_v3 2>&1)
RC=$?; echo "$OUT" | tail -15 >>"$LOG"
[ $RC -eq 0 ] || die "v3 / MO / warm-start / contrastive / merge-category test suites failed (see $LOG)"
echo "$OUT" | grep -q "skipped" && die "a test was skipped"
V1=$("$PY" -m unittest tests.test_v1_reproduction 2>&1); V1RC=$?; echo "$V1" | tail -8 >>"$LOG"
if [ $V1RC -ne 0 ] || echo "$V1" | grep -q skipped; then
  if [ "${WS_V1_STRICT:-0}" = 1 ]; then die "v1 reproduction test failed or skipped"; fi
  say "WARNING: v1 reproduction test did not pass on this machine (GPU numerics differ from the reference machine?)." \
      "Recorded in $LOG; not blocking (pre-registered decision). v3/MO/warm-start tests passed." | tee -a "$LOG"
else
  say "v1 reproduction test passed" | tee -a "$LOG"
fi
{ echo "host: $(nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | head -1); $(nproc) logical CPUs; $(lscpu | sed -n 's/^Model name: *//p')"
  echo "versions: $("$PY" -c 'import sys,torch,numpy,ortools; print(sys.version.split()[0], "torch", torch.__version__, "cuda", torch.version.cuda, "numpy", numpy.__version__, "ortools", ortools.__version__)')"
  echo "free disk: ${AVAIL} GB; jobs $JOBS; sizes '$SIZES'; arms '$ARMS'; seeds '$SEEDS'; smoke $SMOKE; shared-base $SHARED"
} | tee -a "$LOG"
say "pre-flight OK" | tee -a "$LOG"

# ------------------------------------------------------------------ detach
W_ARGS=(--worker --log "$LOG" --sizes "$SIZES" --arms "$ARMS" --seeds "$SEEDS" --jobs "$JOBS")
[ "$SMOKE" = 1 ] && W_ARGS+=(--smoke)
[ "$SHARED" = 1 ] && W_ARGS+=(--with-shared-base)
[ "$SHUTDOWN" = 1 ] && W_ARGS+=(--shutdown)
setsid nohup bash "$REPO/scripts/ec2_run_all.sh" "${W_ARGS[@]}" >>"$LOG" 2>&1 < /dev/null &
say "running in the background (pid $!); safe to disconnect."
echo "  log:     $REPO/$LOG"
echo "  status:  $REPO/$STATUS_FILE   (bash scripts/ec2_status.sh)"
