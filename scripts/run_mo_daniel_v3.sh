#!/usr/bin/env bash
# MO-DANIEL on benchmark v3 (docs/evaluation_protocol_v3.md, Addendum A, PRE-REGISTERED): a preference-conditioned
# DANIEL trained from random initialisation on makespan, carbon and tardiness, to convergence, then evaluated once on
# the evaluation split.
#
# Usage: scripts/run_mo_daniel_v3.sh [--sizes "10x5 20x10"] [--seeds "111 222 333"] [--shutdown] [--preflight-only] [--smoke]
#   --shutdown     at the very end (after the analysis and the final commit) shut Windows down in 60 s
#   --smoke        tiny budgets, validation/evaluation on 3 instances, development split for evaluation (testing only)
#
# Steps: pre-flight -> frozen validation normalisation points per size (CP-SAT B1 + specialists + MWKR/ECT on the
# validation split; the 20x10 points are computed on CPUs 4-11 while 10x5 trains) -> per (size, seed): chunked
# training with a hang guard, evaluation (66 preferences + equal weights), commit -> analysis -> final commit.
# Resume: re-run the same command. Finished runs are skipped; training continues from its last chunk (optimiser, RNG
# and preference-sampler states saved). Progress: logs/mo_daniel_v3_STATUS.txt and the log file.
# Fully offline; commits locally (results, never checkpoints); never pushes. Never reads the reserved final-test sets.
set -u -o pipefail

SIZES="10x5 20x10"; SEEDS="111 222 333"; SHUTDOWN=0; PREFLIGHT_ONLY=0; SMOKE=0
while [ $# -gt 0 ]; do
  case "$1" in
    --sizes) SIZES="$2"; shift 2 ;;
    --seeds) SEEDS="$2"; shift 2 ;;
    --shutdown) SHUTDOWN=1; shift ;;
    --preflight-only) PREFLIGHT_ONLY=1; shift ;;
    --smoke) SMOKE=1; shift ;;
    -h|--help) sed -n 2,17p "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO" || exit 2
PY="$REPO/.venv/bin/python"; [ -x "$PY" ] || PY="$(command -v python3)"
GITDIR="$(git rev-parse --absolute-git-dir)"
STATE="$GITDIR/mo_daniel_v3"; mkdir -p "$STATE" logs results/mo_daniel_v3
TS="$(date +%Y%m%d_%H%M%S)"
LOG="logs/mo_daniel_v3_$TS.log"
STATUS_FILE="logs/mo_daniel_v3_STATUS.txt"
RES="results/mo_daniel_v3"

# ------------------------------------------------------------------ settings (Addendum A)
TRAIN_CPUS="0-3"; CPSAT_CPUS="4-11"; CPSAT_WORKERS=8; TRAIN_THREADS=4
VALIDATE=50; WINDOW=1000; TOL=0.005; MIN_UPDATES=1000; CAP=8000; MIN_STOP=2000
VAL_GRID=0.25; VAL_FIRST=0; EVAL_GRID=0.1
declare -A CHUNK=([10x5]=500 [20x10]=200)
declare -A SPU=([10x5]=0.85 [20x10]=4.5)                  # measured seconds per update (pinned, 2026-10-19), for timeouts/ETA
NORM_SPLIT=validation; NORM_TAG=""; NORM_FIRST=0; NORM_LIMIT=60; NORM_LEX=20
declare -A NORM_N=([10x5]=18 [20x10]=13)
EVAL_SPLIT=evaluation; EVAL_FIRST=0; EVAL_NORM="results/baselines_v3/normalisation_points.json"
MIN_DISK_GB=10
if [ "$SMOKE" = 1 ]; then
  VALIDATE=10; WINDOW=20; MIN_UPDATES=20; CAP=40; MIN_STOP=40; CHUNK=([10x5]=20 [20x10]=20)
  VAL_GRID=0.5; VAL_FIRST=3; EVAL_GRID=0.25
  NORM_FIRST=3; NORM_LIMIT=2; NORM_LEX=1; NORM_N=([10x5]=3 [20x10]=3)
  EVAL_SPLIT=development; EVAL_FIRST=3; MIN_DISK_GB=1
fi
export WANDB_MODE=offline HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTHONUNBUFFERED=1

exec 3>&1
log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG" >&3; }
die_preflight() { log "PRE-FLIGHT FAILED: $*  (nothing started; no shutdown)"; exit 2; }

exec 8>"$STATE/run.lock"
flock -n 8 || { echo "another run_mo_daniel_v3.sh is running (lock $STATE/run.lock)" >&2; exit 2; }

CHILD=""; NORM_PID=""; STATUS_PID=""
run_guarded() {   # run_guarded <timeout seconds> <cmd...>; the child is killed with us
  timeout --kill-after=60 "$1" "${@:2}" &
  CHILD=$!; wait "$CHILD"; local rc=$?; CHILD=""; return $rc
}
on_signal() {
  trap - INT TERM
  log "INTERRUPTED by signal: stopping children (no shutdown). Re-run the same command to resume."
  [ -n "$CHILD" ] && kill -TERM "$CHILD" 2>/dev/null
  [ -n "$NORM_PID" ] && kill -TERM "$NORM_PID" 2>/dev/null
  [ -n "$STATUS_PID" ] && { pkill -P "$STATUS_PID" 2>/dev/null; kill "$STATUS_PID" 2>/dev/null; }
  wait 2>/dev/null
  exit 130
}
trap on_signal INT TERM

commit() {        # commit <message> <paths...>: results only, serialised
  local msg="$1"; shift
  (
    flock 9
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] commit: $msg" >>"$LOG"
    local p; for p in "$@"; do [ -e "$p" ] && git add -A -- "$p"; done
    if ! git diff --cached --quiet; then
      git commit -q -m "$msg" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
    fi
  ) 9>"$STATE/commit.lock" >>"$STATE/git.out" 2>&1
}

norm_file() { echo "$RES/validation_normalisation_points${NORM_TAG}_$1.json"; }

# ------------------------------------------------------------------ pre-flight
preflight() {
  log "pre-flight: repo $REPO, sizes '$SIZES', seeds '$SEEDS'$([ $SMOKE = 1 ] && echo ' [SMOKE]')"
  local ncpu; ncpu=$(nproc --all)
  [ "$ncpu" -ge 12 ] || die_preflight "CPU allocation assumes 12 logical CPUs (training $TRAIN_CPUS, CP-SAT $CPSAT_CPUS); found $ncpu"
  "$PY" -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null || die_preflight "GPU not visible to torch"
  log "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -1)"
  local avail; avail=$(df -BG --output=avail "$REPO" | tail -1 | tr -dc 0-9)
  [ "$avail" -ge "$MIN_DISK_GB" ] || die_preflight "only ${avail} GB free (need $MIN_DISK_GB)"
  "$PY" tools/baselines_v3_status.py verify >>"$LOG" 2>&1 || die_preflight "v3 manifest verification failed (see log)"
  [ -f "$EVAL_NORM" ] || [ "$SMOKE" = 1 ] || die_preflight "frozen normalisation points missing: $EVAL_NORM"
  local size o s
  for size in $SIZES; do
    for o in m c t; do for s in 111 222 333; do
      local ck="${MO_SPECIALIST_DIR:-daniel/trained_network/SD2}/${size}+mix+bv3_spec_${o}_s${s}.pth"
      [ -f "$(norm_file "$size")" ] || [ -f "$ck" ] || die_preflight "specialist checkpoint missing (needed for the validation nadir): $ck"
    done; done
  done
  local dirty; dirty=$(git status --porcelain -- . ':!logs/mo_daniel_v3_*' ":!$RES" ':!results/mo_daniel_v3.md' ':!logs/baselines_v3_console.out')
  [ -z "$dirty" ] || die_preflight "git tree not clean: $(echo "$dirty" | head -5 | tr '\n' ' ')"
  local out rc
  out=$("$PY" -m unittest -v tests.test_v1_reproduction tests.test_benchmark_v3 tests.test_baselines_v3 tests.test_mo_daniel_v3 2>&1); rc=$?
  echo "$out" | tail -40 >>"$LOG"
  [ $rc -eq 0 ] || die_preflight "v1/v3/MO test suites failed"
  echo "$out" | grep -q "skipped" && die_preflight "a test was skipped (v1 reproduction needs CUDA and the Step 4 checkpoints)"
  log "pre-flight OK: $ncpu CPUs (training $TRAIN_CPUS, CP-SAT $CPSAT_CPUS x$CPSAT_WORKERS), ${avail} GB free, manifest verified, tree clean, v1+v3+MO tests pass"
}

# ------------------------------------------------------------------ plan + status
write_plan() {
  "$PY" - "$STATE/plan.json" <<EOF
import json, sys, time
runs = [dict(name=f"{sz}+mix+bv3_mo_s{s}", size=sz, seed=int(s)) for sz in "$SIZES".split() for s in "$SEEDS".split()]
json.dump(dict(started=time.time(), log="$LOG", state_dir="$STATE", runs=runs, sizes="$SIZES".split(), cap=$CAP,
               min_stop=$MIN_STOP, sec_per_update={"10x5": ${SPU[10x5]}, "20x10": ${SPU[20x10]}},
               eval_split="$EVAL_SPLIT", norm_split="$NORM_SPLIT", norm_tag="$NORM_TAG",
               norm_n={"10x5": ${NORM_N[10x5]}, "20x10": ${NORM_N[20x10]}}), open(sys.argv[1], "w"), indent=1)
EOF
}
status_now() { "$PY" tools/mo_daniel_v3_status.py status --plan "$STATE/plan.json" 2>>"$LOG"; }
status_loop() { while true; do status_now; sleep 60; done; }

# ------------------------------------------------------------------ validation normalisation points (Addendum A2)
norm_one() {      # norm_one <size>: compute and freeze; retried once; returns non-zero on failure
  local size=$1 f; f=$(norm_file "$size")
  [ -f "$f" ] && return 0
  local tmo=$(( ${NORM_N[$size]} * (2 * NORM_LIMIT + 6 * NORM_LEX) * 3 / 2 + 900 ))
  local attempt
  for attempt in 1 2; do
    log "N validation normalisation points $size (attempt $attempt, CPUs $CPSAT_CPUS, timeout ${tmo}s)"
    run_guarded "$tmo" taskset -c "$CPSAT_CPUS" "$PY" tools/mo_v3_normalisation.py --size "$size" \
      --split "$NORM_SPLIT" --first "$NORM_FIRST" --limit "$NORM_LIMIT" --lex_limit "$NORM_LEX" \
      --workers "$CPSAT_WORKERS" >>"$LOG" 2>&1 && [ -f "$f" ] && break
    log "N normalisation $size FAILED (attempt $attempt)"
  done
  commit "MO-DANIEL v3: frozen $NORM_SPLIT normalisation points $size" "$RES/cpsat" "$f" "$STATUS_FILE"
  [ -f "$f" ]
}
norm_background() {   # the subshell needs its own trap to stop its child
  trap '[ -n "$CHILD" ] && kill -TERM "$CHILD" 2>/dev/null; exit 130' INT TERM
  norm_one "$1"
}

# ------------------------------------------------------------------ one run
train_one() {     # train_one <size> <seed>
  local size=$1 seed=$2
  local nj=${size%x*} nm=${size#*x}
  local name="${size}+mix+bv3_mo_s${seed}"
  local rdir="$RES/runs/$name" edir="$RES/eval/${EVAL_SPLIT}_${size}"
  local live="daniel/train_log/SD2/$name.status.json" out="daniel/train_log/SD2/$name.stdout.log"
  if [ -f "$rdir/status.json" ] && [ -f "$edir/$name.json" ]; then log "skip $name (done)"; return 0; fi
  mkdir -p "$rdir" daniel/train_log/SD2; rm -f "$rdir/FAILED"
  if [ ! -f "$(norm_file "$size")" ]; then
    echo "validation normalisation points for $size are missing (see log)" > "$rdir/FAILED"
    log "$name FAILED: no validation normalisation points"; commit "MO-DANIEL v3: $name FAILED" "$rdir"; return 1
  fi
  echo "$name" > "$STATE/current"
  "$PY" tools/mo_daniel_v3_status.py meta --name "$name" --train_cpus "$TRAIN_CPUS" --omp_threads "$TRAIN_THREADS" \
    --gpu "$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)" \
    --cpu_model "$(lscpu | sed -n 's/^Model name: *//p')" --size "$size" --seed "$seed"
  local timeout_s; timeout_s=$("$PY" -c "print(int(${CHUNK[$size]} * ${SPU[$size]} * 3 + 600))")
  timeout_s=${MO_CHUNK_TIMEOUT:-$timeout_s}                 # testing override of the hang guard
  local fails=0 st
  while true; do
    st=$("$PY" -c "import json; print(json.load(open('$live'))['status'])" 2>/dev/null || echo NONE)
    case "$st" in CONVERGED|NOT_CONVERGED) break ;; esac
    log "train $name: chunk of ${CHUNK[$size]} updates (status $st, timeout ${timeout_s}s)"
    local t0; t0=$(date +%s.%N)
    run_guarded "$timeout_s" env -C daniel OMP_NUM_THREADS=$TRAIN_THREADS MKL_NUM_THREADS=$TRAIN_THREADS \
        taskset -c "$TRAIN_CPUS" "$PY" train.py --config ../configs/benchmark_v3/mct.json --n_j "$nj" --n_m "$nm" \
        --preference_dim 3 --log_ppo_stats True --mo_val_norm "../$(norm_file "$size")" --mo_val_grid_step "$VAL_GRID" \
        --mo_val_first "$VAL_FIRST" --model_suffix "bv3_mo_s${seed}" --seed_train "$seed" --v3_train_source generator \
        --validate_timestep "$VALIDATE" --converge_window "$WINDOW" --converge_tol "$TOL" --min_updates "$MIN_UPDATES" \
        --max_updates "$CAP" --chunk_updates "${CHUNK[$size]}" --resume True --device cuda >>"$out" 2>&1
    local rc=$?
    "$PY" tools/mo_daniel_v3_status.py meta --name "$name" \
      --add_process_seconds "$("$PY" -c "import time; print(time.time() - $t0)")" --append_chunk_rc "$rc"
    if [ $rc -ne 0 ]; then
      fails=$((fails + 1))
      log "train $name: chunk FAILED (rc $rc; $([ $rc = 124 ] && echo 'hang guard timeout' || echo error)), attempt $fails"
      if [ $fails -ge 2 ]; then
        echo "training failed twice in a row (last rc $rc) at $(date '+%F %T'); see $out" > "$rdir/FAILED"
        log "$name marked FAILED; continuing"; commit "MO-DANIEL v3: $name FAILED" "$rdir"; return 1
      fi
    else
      fails=0
    fi
    status_now
  done
  cp "daniel/train_log/SD2/valiquality_$name.json" "$rdir/valiquality.json"
  cp "daniel/train_log/SD2/reward_$name.json" "$rdir/reward.json"
  cp "daniel/train_log/SD2/$name.run.json" "$rdir/run.json"
  cp "$live" "$rdir/status.json"
  log "$name training $st: $("$PY" -c "import json; s=json.load(open('$live')); print(f\"update {s['update']}, best val HV {-s['best_value']:.4f} @ {s['best_update']}, train {s['train_seconds']/3600:.2f} h\")")"
  local efails=0
  while [ ! -f "$edir/$name.json" ]; do
    log "eval $name on $EVAL_SPLIT ($([ $EVAL_FIRST = 0 ] && echo all || echo first $EVAL_FIRST)): greedy, grid step $EVAL_GRID + equal weights"
    run_guarded 3600 env -C daniel taskset -c "$TRAIN_CPUS" "$PY" evaluate_mo_v3.py --model "$name" --split "$EVAL_SPLIT" \
        --size "$size" --first "$EVAL_FIRST" --grid_step "$EVAL_GRID" --norm "../$EVAL_NORM" --out "../$RES/eval" \
        --device cuda >>"$out" 2>&1
    if [ $? -ne 0 ]; then
      efails=$((efails + 1)); log "eval $name FAILED (attempt $efails)"
      if [ $efails -ge 2 ]; then
        echo "evaluation failed twice at $(date '+%F %T'); see $out" > "$rdir/FAILED"
        commit "MO-DANIEL v3: $name evaluation FAILED" "$rdir"; return 1
      fi
    fi
  done
  commit "MO-DANIEL v3: $name ($st)" "$rdir" "$edir" "$STATUS_FILE"
  return 0
}

# ------------------------------------------------------------------ main
preflight
[ "$PREFLIGHT_ONLY" = 1 ] && { log "pre-flight only: done"; exit 0; }
if [ "$SMOKE" = 1 ]; then                 # evaluation on the development split needs development points too
  NORM_SPLIT=development; NORM_TAG="_development"
  for size in $SIZES; do norm_one "$size"; done
  NORM_SPLIT=validation; NORM_TAG=""
fi
write_plan
status_loop & STATUS_PID=$!
log "START  log $LOG  status $STATUS_FILE"
first=1
for size in $SIZES; do
  if [ $first = 1 ]; then
    norm_one "$size" || log "validation normalisation points for $size could not be computed"
    first=0
    for later in $SIZES; do                  # the remaining sizes' points are computed alongside training
      [ "$later" = "$size" ] && continue
      [ -f "$(norm_file "$later")" ] && continue
      norm_background "$later" & NORM_PID=$!
      log "N $later normalisation points computing in the background (pid $NORM_PID)"
      break
    done
  else
    if [ -n "$NORM_PID" ]; then log "waiting for the $size normalisation points"; wait "$NORM_PID"; NORM_PID=""; fi
    norm_one "$size" || log "validation normalisation points for $size could not be computed"
  fi
  if [ "$SMOKE" = 1 ]; then EVAL_NORM="$RES/validation_normalisation_points_development_$size.json"; fi
  for seed in $SEEDS; do train_one "$size" "$seed"; done
done
rm -f "$STATE/current"
log "analysis: tools/analyze_mo_daniel_v3.py"
NORM_ARGS=(); if [ "$SMOKE" = 1 ]; then
  NORM_ARGS=(--norm); for size in $SIZES; do NORM_ARGS+=("$RES/validation_normalisation_points_development_$size.json"); done
fi
run_guarded 1800 "$PY" tools/analyze_mo_daniel_v3.py --eval_split "$EVAL_SPLIT" "${NORM_ARGS[@]}" \
  >>"$LOG" 2>&1 || log "analysis FAILED (see log); results are committed"
pkill -P "$STATUS_PID" 2>/dev/null; kill "$STATUS_PID" 2>/dev/null; wait "$STATUS_PID" 2>/dev/null; STATUS_PID=""
status_now
log "DONE"
[ "$SHUTDOWN" = 1 ] && log "after the final commit: shutting Windows down in 60 s"
commit "MO-DANIEL v3: analysis and final status" "$RES" results/mo_daniel_v3.md "$STATUS_FILE" logs/mo_daniel_v3_*.log
if [ "$SHUTDOWN" = 1 ]; then
  "${MO_SHUTDOWN_EXE:-/mnt/c/Windows/System32/shutdown.exe}" /s /t 60
fi
