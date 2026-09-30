#!/usr/bin/env bash
# Overnight v2 re-run of the three-objective warm-start comparison, SEED 111 ONLY (EXPLORATORY).
# Pre-registration: results/v2_rerun_seed111.md. Fully offline and unattended; safe to re-run (resumes).
#
#   scripts/overnight_v2_rerun.sh               run (or resume) the study
#   scripts/overnight_v2_rerun.sh --shutdown    ... and shut Windows down at the very end
#   scripts/overnight_v2_rerun.sh --smoke       tiny end-to-end test; outputs under $V2RR_SMOKE_ROOT (or a temp dir),
#                                               never inside the repository, no commits
#
# Arms, in priority order: V2-M, V2-S, V1-CONT, V2-T. Each arm trains in chunks (500 updates) wrapped in a timeout; a
# failed or timed-out chunk is retried once from the last checkpoint; a second failure marks the arm FAILED and the
# run moves on. Local git commits after each arm (results and logs only; checkpoints are git-ignored, SHA-256 in
# results/v2_rerun_seed111/provenance/). Never pushes, fetches, installs or downloads anything.
set -u -o pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO" || exit 1
PY="$REPO/.venv/bin/python"
SHUTDOWN=0; SMOKE=0
for arg in "$@"; do
  case "$arg" in
    --shutdown) SHUTDOWN=1 ;;
    --smoke) SMOKE=1 ;;
    *) echo "usage: $0 [--shutdown] [--smoke]"; exit 2 ;;
  esac
done

# ---- offline, unattended ----------------------------------------------------------------------------------------
export WANDB_MODE=offline WANDB_DISABLED=true HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PIP_NO_INDEX=1 PIP_DISABLE_PIP_VERSION_CHECK=1 GIT_TERMINAL_PROMPT=0 GIT_OPTIONAL_LOCKS=0 PYTHONUNBUFFERED=1
export MPLBACKEND=Agg

SEEDS="111"
ARMS="V2-M V2-S V1-CONT V2-T"
CHUNK_TIMEOUT=2700      # s; a 500-update chunk takes ~7-8 min at the measured 0.77-0.97 s/update
EVAL_TIMEOUT=1200
TEST_TIMEOUT=900
ANALYSIS_TIMEOUT=1800
MIN_FREE_GB=1           # estimate: 38 checkpoints x 0.32 MB + evaluations + logs < 50 MB; 1 GB leaves ample margin

if [ "$SMOKE" = 1 ]; then
  OUT="${V2RR_SMOKE_ROOT:-$(mktemp -d /tmp/v2rr_smoke.XXXXXX)}"; mkdir -p "$OUT"
  case "$(cd "$OUT" && pwd)/" in "$REPO"/*) echo "smoke root must be outside the repository"; exit 2 ;; esac
  RUNARGS=(--smoke --out-root "$OUT"); RES="$OUT/results"; REPORT="$OUT/v2_rerun_seed111.md"; LOGDIR="$OUT/logs"
  [ -f "$REPORT" ] || cp results/v2_rerun_seed111.md "$REPORT"
  CHUNK_TIMEOUT=600
else
  RUNARGS=(); RES="results/v2_rerun_seed111"; REPORT="results/v2_rerun_seed111.md"; LOGDIR="logs"
fi
mkdir -p "$LOGDIR"
TS="$(date +%Y%m%d_%H%M%S)"
LOG="$LOGDIR/overnight_v2_rerun_$TS.log"
STATUS="$LOGDIR/overnight_v2_rerun_STATUS.txt"
exec > >(tee -a "$LOG") 2>&1
T0="$(date +%s)"
FAILURES=()

log() { echo "[$(date '+%F %T')] $*"; }

status() {   # $1: last event
  "$PY" tools/run_v2_rerun.py status --arms $ARMS --seeds $SEEDS --t0 "$T0" --status-file "$STATUS" "${RUNARGS[@]}" >/dev/null 2>&1 \
    || echo "(status summary unavailable)" > "$STATUS"
  { echo; echo "last event: $1"; echo "log: $LOG"
    [ "${#FAILURES[@]}" -gt 0 ] && printf 'logged failures:\n' && printf '  - %s\n' "${FAILURES[@]}"; } >> "$STATUS"
}

commit_outputs() {   # $1: message. Real mode only; commits results and logs of this study, nothing else.
  if [ "$SMOKE" = 1 ]; then log "SMOKE: would commit: $1"; return 0; fi
  git add -- "$RES" "$REPORT" "$LOGDIR"/overnight_v2_rerun_* 2>/dev/null
  if git diff --cached --quiet; then log "nothing new to commit ($1)"; return 0; fi
  if git commit -q -m "$1

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01Rcu6f737di34xSocjiJkEU"; then
    log "committed: $1 ($(git rev-parse --short HEAD))"; return 0
  fi
  log "COMMIT FAILED: $1"; FAILURES+=("commit failed: $1"); return 1
}

finish() {   # $1: exit code
  if [ "$SHUTDOWN" = 1 ]; then
    log "shutting Windows down in 60 s (/mnt/c/Windows/System32/shutdown.exe /s /t 60)"
    /mnt/c/Windows/System32/shutdown.exe /s /t 60 || log "Windows shutdown command FAILED"
  fi
  log "END (exit $1). Status: $STATUS"
  exit "$1"
}

die() {   # pre-flight or fatal failure: logged, committed, then (optional) shutdown
  log "STOPPED: $*"; FAILURES+=("$*"); status "STOPPED: $*"; commit_outputs "v2 re-run seed 111: stopped ($*)"; finish 1
}

run_step() {   # $1 name, $2 timeout, rest: command. Two attempts; the training subcommand resumes from its last checkpoint.
  local name="$1" t="$2"; shift 2
  for attempt in 1 2; do
    log "$name: attempt $attempt"
    timeout --signal=TERM --kill-after=60 "$t" "$@"
    local rc=$?
    if [ "$rc" -eq 0 ]; then return 0; fi
    if [ "$rc" -eq 124 ] || [ "$rc" -eq 137 ]; then log "$name: TIMED OUT after ${t}s (rc $rc)"; else log "$name: FAILED (rc $rc)"; fi
    nvidia-smi --query-gpu=temperature.gpu,utilization.gpu,memory.used --format=csv,noheader 2>/dev/null | sed 's/^/  GPU: /'
  done
  return 1
}

log "v2 re-run (seed 111, EXPLORATORY) starting; smoke=$SMOKE shutdown=$SHUTDOWN; log $LOG"
status "starting"

# ---- pre-flight -------------------------------------------------------------------------------------------------
[ -x "$PY" ] || die "python not found at $PY"
if [ "$SMOKE" = 0 ]; then   # a previous interrupted run may have left its own logs/results uncommitted
  commit_outputs "v2 re-run seed 111: recover outputs of an interrupted run" || die "could not commit leftover outputs"
fi
[ -z "$(git status --porcelain)" ] || die "git working tree not clean: $(git status --porcelain | head -5 | tr '\n' ' ')"
log "pre-flight: git working tree clean"
nvidia-smi -L >/dev/null 2>&1 && nvidia-smi -L | grep -q GPU || die "no GPU visible (nvidia-smi)"
log "pre-flight: GPU: $(nvidia-smi -L | head -1)"
FREE_GB=$(df -Pk "$REPO" | awk 'NR==2 {print int($4/1024/1024)}')
[ "$FREE_GB" -ge "$MIN_FREE_GB" ] || die "only ${FREE_GB} GB free; need >= ${MIN_FREE_GB} GB"
log "pre-flight: ${FREE_GB} GB free (need ~50 MB; threshold ${MIN_FREE_GB} GB)"
"$PY" - <<'EOF' || die "input checkpoint missing or SHA-256 mismatch"
import hashlib, json, sys
from pathlib import Path
rec = json.loads(Path("results/v2_rerun_seed111/input_hashes.json").read_text())
bad = []
for name, r in rec.items():
    p = Path("daniel/trained_network/SD2") / name
    if not p.exists():
        bad.append(f"missing {name}"); continue
    if hashlib.sha256(p.read_bytes()).hexdigest() != r["sha256"]:
        bad.append(f"hash mismatch {name}")
    else:
        print(f"  input OK: {name}")
if bad:
    print("\n".join(bad)); sys.exit(1)
EOF
log "pre-flight: v1 reproduction test"
timeout "$TEST_TIMEOUT" "$PY" -m unittest -v tests.test_v1_reproduction > "$LOGDIR/.v1test_$TS.txt" 2>&1
V1RC=$?; grep -v -i "warning\|pynvml" "$LOGDIR/.v1test_$TS.txt"
grep -q "test_c1_at_1000_held_out .* ok" "$LOGDIR/.v1test_$TS.txt" || V1RC=${V1RC/#0/99}   # skipped != passed
rm -f "$LOGDIR/.v1test_$TS.txt"
[ "$V1RC" -eq 0 ] || die "tests/test_v1_reproduction.py did not pass (rc $V1RC; a skip counts as a failure)"
log "pre-flight: all checks passed"
status "pre-flight passed"

# ---- arms ---------------------------------------------------------------------------------------------------------
for ARM in $ARMS; do
  for SEED in $SEEDS; do
    DONE="$RES/provenance/${ARM}_s${SEED}.done"; FAILED="$RES/provenance/${ARM}_s${SEED}.FAILED"
    if [ -f "$DONE" ]; then log "$ARM s$SEED: already complete, skipping"; continue; fi
    if [ -f "$FAILED" ]; then log "$ARM s$SEED: was FAILED in an earlier run; retrying from its last checkpoint"; rm -f "$FAILED"; fi
    mkdir -p "$RES/provenance"
    ARM_OK=1
    for END in $("$PY" tools/run_v2_rerun.py plan --arms "$ARM" "${RUNARGS[@]}"); do
      if ! run_step "$ARM s$SEED train -> u$END" "$CHUNK_TIMEOUT" "$PY" tools/run_v2_rerun.py train --arms "$ARM" --seeds "$SEED" --until "$END" "${RUNARGS[@]}"; then
        ARM_OK=0; break
      fi
      if ! run_step "$ARM s$SEED evaluate" "$EVAL_TIMEOUT" "$PY" tools/run_v2_rerun.py eval --arms "$ARM" --seeds "$SEED" "${RUNARGS[@]}"; then
        ARM_OK=0; break
      fi
      status "$ARM s$SEED reached update $END"
    done
    if [ "$ARM_OK" = 1 ]; then
      echo "ok $(date '+%F %T')" > "$DONE"; log "$ARM s$SEED: DONE"; status "$ARM s$SEED done"
      commit_outputs "v2 re-run seed $SEED: arm $ARM complete (train + HELD-OUT evaluation)"
    else
      echo "failed twice $(date '+%F %T'); see $LOG" > "$FAILED"; FAILURES+=("$ARM s$SEED FAILED")
      log "$ARM s$SEED: FAILED (moving on to the next arm)"; status "$ARM s$SEED FAILED"
      commit_outputs "v2 re-run seed $SEED: arm $ARM FAILED (partial results kept)"
    fi
  done
done

# ---- analysis, v1 check, final commit ------------------------------------------------------------------------------
ANALYSIS_ARGS=(--seeds $SEEDS); [ "$SMOKE" = 1 ] && ANALYSIS_ARGS+=(--smoke --out-root "$OUT")
if run_step "analysis" "$ANALYSIS_TIMEOUT" "$PY" tools/analyze_v2_rerun.py "${ANALYSIS_ARGS[@]}"; then
  log "analysis written to $REPORT"
else
  FAILURES+=("analysis failed"); log "analysis FAILED"
fi
log "final v1 reproduction test"
timeout "$TEST_TIMEOUT" "$PY" -m unittest -v tests.test_v1_reproduction > "$LOGDIR/.v1final_$TS.txt" 2>&1
V1RC=$?; grep -v -i "warning\|pynvml" "$LOGDIR/.v1final_$TS.txt"
if [ "$V1RC" -eq 0 ] && grep -q "test_c1_at_1000_held_out .* ok" "$LOGDIR/.v1final_$TS.txt"; then
  log "final v1 reproduction test: PASS"
else
  FAILURES+=("final v1 reproduction test FAILED"); log "final v1 reproduction test: FAILED (rc $V1RC)"
fi
rm -f "$LOGDIR/.v1final_$TS.txt"
if [ "${#FAILURES[@]}" -gt 0 ]; then log "failures: ${FAILURES[*]}"; else log "no failures"; fi
status "finished ($([ "${#FAILURES[@]}" -gt 0 ] && echo "${#FAILURES[@]} failure(s)" || echo "no failures"))"
commit_outputs "v2 re-run seed 111: final analysis and verdicts (exploratory)"
[ "$SMOKE" = 1 ] && log "SMOKE: outputs in $OUT"
finish "$([ "${#FAILURES[@]}" -gt 0 ] && echo 1 || echo 0)"
