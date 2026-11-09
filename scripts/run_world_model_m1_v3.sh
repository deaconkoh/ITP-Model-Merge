#!/usr/bin/env bash
# World-model trunk, Milestone 1 (docs/world_model_m1_protocol_v3.md), 10x5:
#   data generation (SD2 generator stream + the v3 validation split) -> pretraining -> Question A (prediction sanity)
#   -> Question B (frozen-trunk transfer probes) -> results/world_model_m1_v3.md, with checkpoints and SHA-256 hashes.
#
# Usage: scripts/run_world_model_m1_v3.sh [--smoke] [--preflight-only]
#   --smoke   tiny data, 2 epochs, 1 probe seed, separate outputs (testing only)
#
# Resume: re-run the same command; finished steps are skipped and data generation continues from its last shard.
# Commits locally (results and checkpoints; the data stay in data/, gitignored); never pushes. Never reads the
# evaluation split or the reserved final-test sets.
set -u -o pipefail

SMOKE=0; PREFLIGHT_ONLY=0
while [ $# -gt 0 ]; do
  case "$1" in
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
STATE="$GITDIR/world_model_m1_v3"; mkdir -p "$STATE" logs
SFX=""; S_ARG=(); PREFIX="world-model M1 v3"
[ "$SMOKE" = 1 ] && { SFX="_smoke"; S_ARG=(--smoke); PREFIX="world-model M1 v3 SMOKE"; }
RES="results/world_model_m1_v3$SFX"; CKPT="checkpoints/world_model_m1_v3$SFX"; DATA="data/world_model_m1_v3$SFX"
REPORT="results/world_model_m1_v3$SFX.md"
LOG="logs/world_model_m1_v3_$(date +%Y%m%d_%H%M%S).log"
export WANDB_MODE=offline PYTHONUNBUFFERED=1 CUBLAS_WORKSPACE_CONFIG=:4096:8

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }
die() { log "STOPPED: $*"; exit 2; }
exec 8>"$STATE/run.lock"
flock -n 8 || { echo "another run_world_model_m1_v3.sh is running" >&2; exit 2; }
CHILD=""
trap 'log "INTERRUPTED: re-run the same command to resume"; [ -n "$CHILD" ] && kill -TERM "$CHILD" 2>/dev/null; exit 130' INT TERM
step() {   # step <timeout seconds> <args of tools/world_model_m1_v3.py...>
  timeout --kill-after=60 "$1" "$PY" tools/world_model_m1_v3.py "${@:2}" "${S_ARG[@]}" >>"$LOG" 2>&1 & CHILD=$!
  wait "$CHILD"; local rc=$?; CHILD=""; return $rc
}
commit() {
  local msg="$1"; shift
  local p; for p in "$@"; do [ -e "$p" ] && git add -A -- "$p"; done
  git diff --cached --quiet || git commit -q -m "$msg" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
}

# ------------------------------------------------------------------ pre-flight
log "pre-flight$([ $SMOKE = 1 ] && echo ' [SMOKE]')"
"$PY" -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null || die "GPU not visible to torch"
avail=$(df -BG --output=avail "$REPO" | tail -1 | tr -dc 0-9); [ "$avail" -ge 10 ] || die "only ${avail} GB disk free"
mem=$(awk '/MemAvailable/ {print int($2 / 1048576)}' /proc/meminfo); [ "$mem" -ge 6 ] || die "only ${mem} GB RAM available (need 6)"
"$PY" tools/freeze_world_model_m1_v3.py verify >>"$LOG" 2>&1 || die "frozen settings do not verify"
dirty=$(git status --porcelain -- . ':!logs/world_model_m1_v3_*' ":!$RES" ":!$CKPT" ":!$REPORT" ':!logs/baselines_v3_console.out')
[ -z "$dirty" ] || die "git tree not clean: $(echo "$dirty" | head -5 | tr '\n' ' ')"
# the v1 reproduction test needs v1 settings, so it runs in its own process
out=$( { WS_V1_STRICT=1 "$PY" -m unittest -v tests.test_v1_reproduction &&
         "$PY" -m unittest tests.test_benchmark_v3 tests.test_tardiness_diagnosis_v3 tests.test_world_model_m1_v3; } 2>&1 ); rc=$?
echo "$out" | tail -20 >>"$LOG"
[ $rc -eq 0 ] || die "unit tests failed (see $LOG)"
echo "$out" | grep -q "skipped" && die "a test was skipped"
log "pre-flight OK: GPU $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1), ${mem} GB RAM, ${avail} GB disk, settings verified, tests pass"
[ "$PREFLIGHT_ONLY" = 1 ] && exit 0
T0=$(date +%s)

# ------------------------------------------------------------------ steps (each skipped when done)
if [ ! -f "$DATA/train/meta.json" ]; then
  log "1/5 data: SD2 generator stream (resumable)"; step 14400 generate --split train || die "data generation failed"
fi
log "data: $("$PY" -c "import json; m=json.load(open('$DATA/train/meta.json')); print(f\"{m['instances']} instances, {m['states']} states, {m['bytes']/1e9:.2f} GB, {m['seconds']/60:.1f} min\")")"
if [ ! -f "$DATA/val/meta.json" ]; then
  log "2/5 data: v3 validation split"; step 3600 generate --split val || die "validation data failed"
fi
if [ ! -f "$RES/pretrain.json" ]; then
  log "3/5 pretraining"; step 21600 pretrain || die "pretraining failed"
  commit "$PREFIX: pretraining" "$RES" "$CKPT"
fi
if [ ! -f "$RES/sanity.json" ]; then log "4/5 Question A: prediction sanity"; step 3600 sanity || die "sanity failed"; fi
if [ ! -f "$RES/probe.json" ]; then log "5/5 Question B: transfer probes"; step 14400 probe || die "probes failed"; fi
cp "$DATA/train/meta.json" "$RES/data_train_meta.json"; cp "$DATA/val/meta.json" "$RES/data_val_meta.json"
"$PY" tools/analyze_world_model_m1_v3.py "${S_ARG[@]}" >>"$LOG" 2>&1 || die "report failed"
log "outcome: $("$PY" -c "import json; print(json.load(open('$RES/outcome.json'))['outcome'])")"
log "DONE ($(( ($(date +%s) - T0) / 60 )) min). Committed locally; push when ready."
commit "$PREFIX: checks and report" "$REPORT" "$RES" "$CKPT" "$LOG"
