#!/usr/bin/env bash
# Plateau diagnosis (docs/plateau_diagnosis_protocol_v3.md), 10x5, seed 111: what limits the ~0.54-0.59 validation HV
# plateau of every converged contrastive run?
#   default   Part B: four 1,500-update runs from the World-model base mix start (beta 0.5, lr 1e-3, K 16, Control),
#             snapshots every 250 updates and best checkpoint with SHA-256; greedy and best-of-16 evaluation; report
#   --part-a  Part A (no training): learning signals from the logs, checkpoints along training (SHA-256), greedy vs
#             best-of-16, noise check, report
#
# Usage: scripts/run_plateau_diagnosis_v3.sh [--part-a] [--jobs 3] [--smoke] [--preflight-only]
#   --smoke   5 updates per run, a few instances, separate outputs (testing only)
#
# Resume: re-run the same command. Commits locally; never pushes. Never reads the reserved final-test sets.
set -u -o pipefail

PART=b; JOBS=3; SMOKE=0; PREFLIGHT_ONLY=0
while [ $# -gt 0 ]; do
  case "$1" in
    --part-a) PART=a; shift ;;
    --jobs) JOBS="$2"; shift 2 ;;
    --smoke) SMOKE=1; shift ;;
    --preflight-only) PREFLIGHT_ONLY=1; shift ;;
    -h|--help) sed -n 2,13p "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO" || exit 2
PY="$REPO/.venv/bin/python"; [ -x "$PY" ] || PY="$(command -v python3)"
GITDIR="$(git rev-parse --absolute-git-dir)"
STATE="$GITDIR/plateau_diagnosis_v3"; mkdir -p "$STATE" logs
SFX=""; [ "$SMOKE" = 1 ] && SFX="_smoke"
LOG="logs/plateau_diagnosis_v3_part${PART}${SFX}_$(date +%Y%m%d_%H%M%S).log"
export WANDB_MODE=offline PYTHONUNBUFFERED=1

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }
die() { log "STOPPED: $*"; exit 2; }
exec 8>"$STATE/run.lock"
flock -n 8 || { echo "another run_plateau_diagnosis_v3.sh is running" >&2; exit 2; }

log "pre-flight: part $PART, $JOBS parallel jobs$([ $SMOKE = 1 ] && echo ' [SMOKE]')"
"$PY" -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null || die "GPU not visible to torch"
[ "$(nproc --all)" -ge 12 ] || die "the 3 slots are pinned to CPUs 0-11; found $(nproc --all) CPUs"
mem=$(awk '/MemAvailable/ {print int($2 / 1048576)}' /proc/meminfo); [ "$mem" -ge 6 ] || die "only ${mem} GB RAM available"
avail=$(df -BG --output=avail "$REPO" | tail -1 | tr -dc 0-9); [ "$avail" -ge 10 ] || die "only ${avail} GB disk free"
"$PY" tools/freeze_plateau_diagnosis_v3.py verify >>"$LOG" 2>&1 || die "frozen settings or a source do not verify"
dirty=$(git status --porcelain -- . ':!logs/plateau_diagnosis_v3_*' ":!results/plateau_diagnosis_v3$SFX" \
        ":!results/plateau_diagnosis_v3$SFX.md" ":!checkpoints/plateau_diagnosis_v3$SFX" ':!logs/baselines_v3_console.out')
[ -z "$dirty" ] || die "git tree not clean: $(echo "$dirty" | head -5 | tr '\n' ' ')"
# the v1 reproduction test needs v1 settings, so it runs in its own process
out=$( { WS_V1_STRICT=1 "$PY" -m unittest -v tests.test_v1_reproduction &&
         "$PY" -m unittest tests.test_benchmark_v3 tests.test_contrastive_v3 tests.test_world_model_m2_v3 \
           tests.test_world_model_m3_v3 tests.test_plateau_diagnosis_v3; } 2>&1 ); rc=$?
echo "$out" | tail -20 >>"$LOG"
[ $rc -eq 0 ] || die "unit tests failed (see $LOG)"
echo "$out" | grep -q "skipped" && die "a test was skipped"
log "pre-flight OK: GPU $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1), ${mem} GB RAM, settings verified, tests pass"
[ "$PREFLIGHT_ONLY" = 1 ] && exit 0

T0=$(date +%s)
"$PY" tools/plateau_diagnosis_v3_runner.py --part "$PART" --jobs "$JOBS" $([ $SMOKE = 1 ] && echo --smoke) 2>&1 | tee -a "$LOG"
rc=${PIPESTATUS[0]}
log "$([ $rc = 0 ] && echo DONE || echo "FINISHED WITH FAILURES (rc $rc)") ($(( ($(date +%s) - T0) / 60 )) min). Committed locally; push when ready."
git add -- "$LOG" && git commit -q -m "plateau diagnosis v3$([ $SMOKE = 1 ] && echo ' SMOKE'): part $PART run log" \
  -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01Rcu6f737di34xSocjiJkEU" 2>/dev/null || true
exit $rc
