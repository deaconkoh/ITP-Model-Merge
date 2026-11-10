#!/usr/bin/env bash
# World-model M2 (docs/world_model_m2_protocol_v3.md), 10x5: single-objective contrastive specialists (makespan, carbon,
# tardiness) from a shared random start vs the M1 world-model start; start and best checkpoints with SHA-256; evaluation
# on the evaluation split; results/world_model_m2_v3.md. No merging.
#
# Usage: scripts/run_world_model_m2_v3.sh [--seeds 111[,222,333]] [--jobs 3] [--smoke] [--preflight-only]
#   --seeds   default 111 (6 specialists); 111,222,333 adds the matched seeds (resumes, finished runs are kept)
#   --smoke   5 updates per run, evaluation on 3 development instances, separate outputs (testing only)
#
# Resume: re-run the same command. Commits locally; never pushes. Never reads the reserved final-test sets.
set -u -o pipefail

SEEDS=111; JOBS=3; SMOKE=0; PREFLIGHT_ONLY=0
while [ $# -gt 0 ]; do
  case "$1" in
    --seeds) SEEDS="$2"; shift 2 ;;
    --jobs) JOBS="$2"; shift 2 ;;
    --smoke) SMOKE=1; shift ;;
    --preflight-only) PREFLIGHT_ONLY=1; shift ;;
    -h|--help) sed -n 2,11p "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO" || exit 2
PY="$REPO/.venv/bin/python"; [ -x "$PY" ] || PY="$(command -v python3)"
GITDIR="$(git rev-parse --absolute-git-dir)"
STATE="$GITDIR/world_model_m2_v3"; mkdir -p "$STATE" logs
SFX=""; [ "$SMOKE" = 1 ] && SFX="_smoke"
LOG="logs/world_model_m2_v3_$(date +%Y%m%d_%H%M%S).log"
export WANDB_MODE=offline PYTHONUNBUFFERED=1

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }
die() { log "STOPPED: $*"; exit 2; }
exec 8>"$STATE/run.lock"
flock -n 8 || { echo "another run_world_model_m2_v3.sh is running" >&2; exit 2; }

log "pre-flight: seeds $SEEDS, $JOBS parallel jobs$([ $SMOKE = 1 ] && echo ' [SMOKE]')"
"$PY" -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null || die "GPU not visible to torch"
[ "$(nproc --all)" -ge 12 ] || die "the 3 slots are pinned to CPUs 0-11; found $(nproc --all) CPUs"
mem=$(awk '/MemAvailable/ {print int($2 / 1048576)}' /proc/meminfo); [ "$mem" -ge 6 ] || die "only ${mem} GB RAM available"
avail=$(df -BG --output=avail "$REPO" | tail -1 | tr -dc 0-9); [ "$avail" -ge 10 ] || die "only ${avail} GB disk free"
"$PY" tools/freeze_world_model_m2_v3.py verify >>"$LOG" 2>&1 || die "frozen settings or the M1 trunk do not verify"
dirty=$(git status --porcelain -- . ':!logs/world_model_m2_v3_*' ":!results/world_model_m2_v3$SFX" \
        ":!results/world_model_m2_v3$SFX.md" ":!checkpoints/world_model_m2_v3$SFX" ':!logs/baselines_v3_console.out')
[ -z "$dirty" ] || die "git tree not clean: $(echo "$dirty" | head -5 | tr '\n' ' ')"
# the v1 reproduction test needs v1 settings, so it runs in its own process
out=$( { WS_V1_STRICT=1 "$PY" -m unittest -v tests.test_v1_reproduction &&
         "$PY" -m unittest tests.test_benchmark_v3 tests.test_contrastive_v3 tests.test_world_model_m1_v3 \
           tests.test_world_model_m2_v3; } 2>&1 ); rc=$?
echo "$out" | tail -20 >>"$LOG"
[ $rc -eq 0 ] || die "unit tests failed (see $LOG)"
echo "$out" | grep -q "skipped" && die "a test was skipped"
log "pre-flight OK: GPU $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1), ${mem} GB RAM, settings and M1 trunk verified, tests pass"
[ "$PREFLIGHT_ONLY" = 1 ] && exit 0

T0=$(date +%s)
"$PY" tools/world_model_m2_v3_runner.py --seeds "$SEEDS" --jobs "$JOBS" $([ $SMOKE = 1 ] && echo --smoke) 2>&1 | tee -a "$LOG"
rc=${PIPESTATUS[0]}
log "$([ $rc = 0 ] && echo DONE || echo "FINISHED WITH FAILURES (rc $rc)") ($(( ($(date +%s) - T0) / 60 )) min). Committed locally; push when ready."
git add -- "$LOG" && git commit -q -m "world-model M2 v3$([ $SMOKE = 1 ] && echo ' SMOKE'): run log" \
  -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" 2>/dev/null || true
exit $rc
