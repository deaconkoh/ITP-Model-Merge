#!/usr/bin/env bash
# Benchmark v3 baselines (docs/evaluation_protocol_v3.md, PRE-REGISTERED).
#   Part A: DANIEL-M / -C / -T specialists trained to convergence (GPU), each evaluated on the evaluation split.
#   Part B: CP-SAT reference bounds (CPU): B1 single-objective + lexicographic points, B2 epsilon-constraint front.
#
# Usage: scripts/run_baselines_v3.sh [--part A|B|all] [--sizes "10x5 20x10"] [--objectives "m c t"]
#                                    [--seeds "111 222 333"] [--concurrent] [--shutdown] [--preflight-only] [--smoke]
#   --concurrent   run Part A and Part B at the same time (default: sequential, A then B)
#   --shutdown     at the very end (after the analysis and the final commit) shut Windows down in 60 s
#   --smoke        tiny budgets on the development split (testing only; never for results)
#
# Resume: re-run the same command. Finished units are skipped, training continues from its last chunk (optimiser
# and RNG states saved), CP-SAT continues per instance. Progress: logs/baselines_v3_STATUS.txt and the log file.
# Fully offline; commits locally (results, never checkpoints); never pushes. Never reads the reserved final-test sets.
set -u -o pipefail

PART=all; SIZES="10x5 20x10"; OBJECTIVES="m c t"; SEEDS="111 222 333"
CONCURRENT=0; SHUTDOWN=0; PREFLIGHT_ONLY=0; SMOKE=0
while [ $# -gt 0 ]; do
  case "$1" in
    --part) PART="$2"; shift 2 ;;
    --sizes) SIZES="$2"; shift 2 ;;
    --objectives) OBJECTIVES="$2"; shift 2 ;;
    --seeds) SEEDS="$2"; shift 2 ;;
    --concurrent) CONCURRENT=1; shift ;;
    --shutdown) SHUTDOWN=1; shift ;;
    --preflight-only) PREFLIGHT_ONLY=1; shift ;;
    --smoke) SMOKE=1; shift ;;
    -h|--help) sed -n 2,16p "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done
case "$PART" in A|B|all) ;; *) echo "--part must be A, B or all" >&2; exit 2 ;; esac

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO" || exit 2
PY="$REPO/.venv/bin/python"; [ -x "$PY" ] || PY="$(command -v python3)"
GITDIR="$(git rev-parse --absolute-git-dir)"
STATE="$GITDIR/baselines_v3"; mkdir -p "$STATE" logs results/baselines_v3
TS="$(date +%Y%m%d_%H%M%S)"
LOG="logs/baselines_v3_$TS.log"
STATUS_FILE="logs/baselines_v3_STATUS.txt"
RES="results/baselines_v3"

# ------------------------------------------------------------------ settings (protocol sections 2-4)
TRAIN_CPUS="0-3"; CPSAT_CPUS="4-11"; CPSAT_WORKERS=8; TRAIN_THREADS=4
VALIDATE=50; WINDOW=1000; TOL=0.005; MIN_UPDATES=1000; CAP=5000; MIN_STOP=2000
declare -A CHUNK=([10x5]=500 [20x10]=200)
declare -A SPU=([10x5]=0.88 [20x10]=4.4)                 # measured seconds per update (pinned), for timeouts/ETA
EVAL_SPLIT=evaluation; EVAL_FIRST=0; SAMPLE=100
CPSAT_SPLIT=evaluation; B1_N=100; B1_BATCH=10; B1_LIMIT=60; LEX_LIMIT=20
declare -A B2_N=([10x5]=20 [20x10]=10); declare -A B2_LIMIT=([10x5]=30 [20x10]=60); B2_BATCH=5; B2_GRID=5
MIN_DISK_GB=20
if [ "$SMOKE" = 1 ]; then
  VALIDATE=10; WINDOW=20; MIN_UPDATES=20; CAP=40; MIN_STOP=40; CHUNK=([10x5]=20 [20x10]=20)
  EVAL_SPLIT=development; EVAL_FIRST=3
  CPSAT_SPLIT=development; B1_N=2; B1_BATCH=1; B1_LIMIT=5; LEX_LIMIT=5
  B2_N=([10x5]=2 [20x10]=2); B2_LIMIT=([10x5]=5 [20x10]=5); B2_BATCH=1; B2_GRID=3; MIN_DISK_GB=1
fi
export WANDB_MODE=offline HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTHONUNBUFFERED=1

exec 3>&1                                                 # the terminal, also from inside redirected calls
log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG" >&3; }
die_preflight() { log "PRE-FLIGHT FAILED: $*  (nothing started; no shutdown)"; exit 2; }

# one instance at a time
exec 8>"$STATE/run.lock"
flock -n 8 || { echo "another run_baselines_v3.sh is running (lock $STATE/run.lock)" >&2; exit 2; }

CHILD=""; A_PID=""; STATUS_PID=""
run_guarded() {   # run_guarded <timeout seconds> <cmd...>; the child is killed with us
  timeout --kill-after=60 "$1" "${@:2}" &
  CHILD=$!; wait "$CHILD"; local rc=$?; CHILD=""; return $rc
}
on_signal() {
  trap - INT TERM
  log "INTERRUPTED by signal: stopping children (no shutdown). Re-run the same command to resume."
  [ -n "$CHILD" ] && kill -TERM "$CHILD" 2>/dev/null
  [ -n "$A_PID" ] && kill -TERM "$A_PID" 2>/dev/null
  [ -n "$STATUS_PID" ] && { pkill -P "$STATUS_PID" 2>/dev/null; kill "$STATUS_PID" 2>/dev/null; }
  wait 2>/dev/null
  exit 130
}
trap on_signal INT TERM

commit() {        # commit <message> <paths...>: results only, serialised between the two parts
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

has_part() { [ "$PART" = all ] || [ "$PART" = "$1" ]; }

# ------------------------------------------------------------------ pre-flight
preflight() {
  log "pre-flight: repo $REPO, mode $([ $CONCURRENT = 1 ] && echo concurrent || echo sequential), part $PART, sizes '$SIZES', objectives '$OBJECTIVES', seeds '$SEEDS'$([ $SMOKE = 1 ] && echo ' [SMOKE]')"
  local ncpu; ncpu=$(nproc --all)
  [ "$ncpu" -ge 12 ] || die_preflight "CPU allocation assumes 12 logical CPUs (training $TRAIN_CPUS, CP-SAT $CPSAT_CPUS); found $ncpu"
  if has_part A; then
    "$PY" -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null || die_preflight "GPU not visible to torch"
    log "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -1)"
  fi
  local avail; avail=$(df -BG --output=avail "$REPO" | tail -1 | tr -dc 0-9)
  [ "$avail" -ge "$MIN_DISK_GB" ] || die_preflight "only ${avail} GB free (need $MIN_DISK_GB)"
  "$PY" tools/baselines_v3_status.py verify >>"$LOG" 2>&1 || die_preflight "v3 manifest verification failed (see log)"
  local dirty; dirty=$(git status --porcelain -- . ':!logs/baselines_v3_*' ":!$RES" ':!results/baselines_v3.md')
  [ -z "$dirty" ] || die_preflight "git tree not clean: $(echo "$dirty" | head -5 | tr '\n' ' ')"
  local out rc
  out=$("$PY" -m unittest -v tests.test_v1_reproduction tests.test_benchmark_v3 tests.test_baselines_v3 2>&1); rc=$?
  echo "$out" | tail -40 >>"$LOG"
  [ $rc -eq 0 ] || die_preflight "v1/v3 test suites failed"
  echo "$out" | grep -q "skipped" && die_preflight "a test was skipped (v1 reproduction needs CUDA and the Step 4 checkpoints)"
  log "pre-flight OK: $ncpu CPUs (training $TRAIN_CPUS, CP-SAT $CPSAT_CPUS x$CPSAT_WORKERS workers), ${avail} GB free, manifest verified, tree clean, v1+v3 tests pass"
}

# ------------------------------------------------------------------ plan + status
write_plan() {
  "$PY" - "$STATE/plan.json" <<EOF
import json, sys, time
runs = [dict(name=f"{sz}+mix+bv3_spec_{o}_s{s}", size=sz, obj=o, seed=int(s))
        for sz in "$SIZES".split() for o in "$OBJECTIVES".split() for s in "$SEEDS".split()] if "$PART" in ("A", "all") else []
b2n = {"10x5": ${B2_N[10x5]}, "20x10": ${B2_N[20x10]}}
b2l = {"10x5": ${B2_LIMIT[10x5]}, "20x10": ${B2_LIMIT[20x10]}}
cps = []
if "$PART" in ("B", "all"):
    for sz in "$SIZES".split():
        cps.append(dict(part="b1", size=sz, n=$B1_N, worst_s=2 * $B1_LIMIT + 6 * $LEX_LIMIT))
        cps.append(dict(part="b2", size=sz, n=b2n[sz], worst_s=$B2_GRID ** 2 * b2l[sz]))
json.dump(dict(mode="$([ $CONCURRENT = 1 ] && echo concurrent || echo sequential)", started=time.time(), log="$LOG",
               state_dir="$STATE", runs=runs, cpsat=cps, cap=$CAP, min_stop=$MIN_STOP,
               sec_per_update={"10x5": ${SPU[10x5]}, "20x10": ${SPU[20x10]}},
               eval_split="$EVAL_SPLIT", cpsat_split="$CPSAT_SPLIT"), open(sys.argv[1], "w"), indent=1)
EOF
}
status_now() { "$PY" tools/baselines_v3_status.py status --plan "$STATE/plan.json" 2>>"$LOG"; }
status_loop() { while true; do status_now; sleep 60; done; }

# ------------------------------------------------------------------ Part A
train_one() {     # train_one <size> <objective> <seed>
  local size=$1 obj=$2 seed=$3
  local nj=${size%x*} nm=${size#*x}
  local name="${size}+mix+bv3_spec_${obj}_s${seed}"
  local rdir="$RES/runs/$name" edir="$RES/eval/${EVAL_SPLIT}_${size}"
  local live="daniel/train_log/SD2/$name.status.json" out="daniel/train_log/SD2/$name.stdout.log"
  if [ -f "$rdir/status.json" ] && [ -f "$edir/$name.json" ]; then log "A skip $name (done)"; return 0; fi
  mkdir -p "$rdir" daniel/train_log/SD2; rm -f "$rdir/FAILED"
  echo "$name" > "$STATE/current_A"
  "$PY" tools/baselines_v3_status.py meta --name "$name" --mode "$([ $CONCURRENT = 1 ] && echo concurrent || echo sequential)" \
    --train_cpus "$TRAIN_CPUS" --omp_threads "$TRAIN_THREADS" --gpu "$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)" \
    --cpu_model "$(lscpu | sed -n 's/^Model name: *//p')" --size "$size" --objective "$obj" --seed "$seed"
  local timeout_s; timeout_s=$("$PY" -c "print(int(${CHUNK[$size]} * ${SPU[$size]} * 3 + 600))")
  timeout_s=${BASELINES_CHUNK_TIMEOUT:-$timeout_s}          # testing override of the hang guard
  local fails=0 st
  while true; do
    st=$("$PY" -c "import json,sys; print(json.load(open('$live'))['status'])" 2>/dev/null || echo NONE)
    case "$st" in CONVERGED|NOT_CONVERGED) break ;; esac
    log "A train $name: chunk of ${CHUNK[$size]} updates (status $st, timeout ${timeout_s}s)"
    local t0; t0=$(date +%s.%N)
    run_guarded "$timeout_s" env -C daniel OMP_NUM_THREADS=$TRAIN_THREADS MKL_NUM_THREADS=$TRAIN_THREADS \
        taskset -c "$TRAIN_CPUS" "$PY" train.py --config "../configs/benchmark_v3/$obj.json" --n_j "$nj" --n_m "$nm" \
        --model_suffix "bv3_spec_${obj}_s${seed}" --seed_train "$seed" --v3_train_source generator \
        --validate_timestep "$VALIDATE" --converge_window "$WINDOW" --converge_tol "$TOL" --min_updates "$MIN_UPDATES" \
        --max_updates "$CAP" --chunk_updates "${CHUNK[$size]}" --resume True --device cuda >>"$out" 2>&1
    local rc=$?
    "$PY" tools/baselines_v3_status.py meta --name "$name" --add_process_seconds "$("$PY" -c "import time; print(time.time() - $t0)")" \
      --append_chunk_rc "$rc"
    if [ $rc -ne 0 ]; then
      fails=$((fails + 1))
      log "A train $name: chunk FAILED (rc $rc; $([ $rc = 124 ] && echo 'hang guard timeout' || echo error)), attempt $fails"
      if [ $fails -ge 2 ]; then
        echo "training failed twice in a row (last rc $rc) at $(date '+%F %T'); see $out" > "$rdir/FAILED"
        log "A $name marked FAILED; continuing"; commit "baselines v3: $name FAILED" "$rdir"; return 1
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
  log "A $name training $st: $("$PY" -c "import json; s=json.load(open('$live')); print(f\"update {s['update']}, best {s['best_value']:.5g} @ {s['best_update']}, train {s['train_seconds']/3600:.2f} h\")")"
  local efails=0
  while [ ! -f "$edir/$name.json" ]; do
    log "A eval $name on $EVAL_SPLIT ($([ $EVAL_FIRST = 0 ] && echo all || echo first $EVAL_FIRST)): greedy + best of $SAMPLE"
    run_guarded 3600 env -C daniel taskset -c "$TRAIN_CPUS" "$PY" evaluate_v3.py --models "$name" --split "$EVAL_SPLIT" \
        --size "$size" --first "$EVAL_FIRST" --sample "$SAMPLE" --out "../$RES/eval" --device cuda >>"$out" 2>&1
    if [ $? -ne 0 ]; then
      efails=$((efails + 1)); log "A eval $name FAILED (attempt $efails)"
      if [ $efails -ge 2 ]; then
        echo "evaluation failed twice at $(date '+%F %T'); see $out" > "$rdir/FAILED"
        commit "baselines v3: $name evaluation FAILED" "$rdir"; return 1
      fi
    fi
  done
  commit "baselines v3: $name ($st)" "$rdir" "$edir" "$STATUS_FILE"
  return 0
}

part_a_background() {   # concurrent mode: the subshell needs its own trap to stop its child
  trap '[ -n "$CHILD" ] && kill -TERM "$CHILD" 2>/dev/null; exit 130' INT TERM
  part_a
}

part_a() {
  for size in $SIZES; do for obj in $OBJECTIVES; do for seed in $SEEDS; do
    train_one "$size" "$obj" "$seed"
  done; done; done
  rm -f "$STATE/current_A"
  log "PART A finished"
}

# ------------------------------------------------------------------ Part B
cpsat_batches() {  # cpsat_batches <b1|b2> <size> <n> <batch> <timeout per batch> <extra args...>
  local part=$1 size=$2 n=$3 batch=$4 tmo=$5; shift 5
  local first_arg=(); [ "$part" = b2 ] && first_arg=(--first "$n")
  for ((start = 0; start < n; start += batch)); do
    if "$PY" tools/baselines_v3_status.py cpsat_done --part "$part" --size "$size" --split "$CPSAT_SPLIT" \
         --start "$start" --count "$batch" "${first_arg[@]}" 2>>"$LOG"; then continue; fi
    local tag="${part}_${CPSAT_SPLIT}_${size}_${start}"
    rm -f "$RES/cpsat/FAILED_$tag"
    local ok=0 attempt
    for attempt in 1 2; do
      echo "$part $size instances $start..$((start + batch - 1)) (attempt $attempt)" > "$STATE/current_B"
      log "B $part $size instances $start..$((start + batch - 1)) (attempt $attempt, timeout ${tmo}s)"
      run_guarded "$tmo" taskset -c "$CPSAT_CPUS" "$PY" tools/cpsat_baselines_v3.py "$part" --size "$size" \
        --split "$CPSAT_SPLIT" --start "$start" --count "$batch" --workers "$CPSAT_WORKERS" "${first_arg[@]}" "$@" >>"$LOG" 2>&1
      if [ $? -eq 0 ]; then ok=1; break; fi
      log "B $part $size batch $start FAILED (attempt $attempt)"
    done
    if [ $ok = 0 ]; then
      mkdir -p "$RES/cpsat"; echo "failed twice at $(date '+%F %T')" > "$RES/cpsat/FAILED_$tag"
    fi
    commit "baselines v3: CP-SAT $part $size instances $start-$((start + batch - 1))" "$RES/cpsat" "$STATUS_FILE"
    status_now
  done
}

part_b() {
  for size in $SIZES; do
    cpsat_batches b1 "$size" "$B1_N" "$B1_BATCH" $(( (B1_BATCH * (2 * B1_LIMIT + 6 * LEX_LIMIT) * 3) / 2 + 300 )) \
      --limit "$B1_LIMIT" --lex_limit "$LEX_LIMIT"
    cpsat_batches b2 "$size" "${B2_N[$size]}" "$B2_BATCH" $(( (B2_BATCH * B2_GRID * B2_GRID * ${B2_LIMIT[$size]} * 3) / 2 + 300 )) \
      --limit "${B2_LIMIT[$size]}" --grid "$B2_GRID"
  done
  rm -f "$STATE/current_B"
  log "PART B finished"
}

# ------------------------------------------------------------------ main
preflight
[ "$PREFLIGHT_ONLY" = 1 ] && { log "pre-flight only: done"; exit 0; }
write_plan
status_loop & STATUS_PID=$!
log "START  log $LOG  status $STATUS_FILE"
if [ "$CONCURRENT" = 1 ] && has_part A && has_part B; then
  log "concurrent: Part A pinned to CPUs $TRAIN_CPUS, Part B pinned to CPUs $CPSAT_CPUS"
  part_a_background & A_PID=$!
  part_b
  wait "$A_PID"; A_PID=""
else
  has_part A && part_a
  has_part B && part_b
fi
log "analysis: tools/analyze_baselines_v3.py"
run_guarded 1800 "$PY" tools/analyze_baselines_v3.py --eval_split "$EVAL_SPLIT" --cpsat_split "$CPSAT_SPLIT" >>"$LOG" 2>&1 \
  || log "analysis FAILED (see log); results are committed"
pkill -P "$STATUS_PID" 2>/dev/null; kill "$STATUS_PID" 2>/dev/null; wait "$STATUS_PID" 2>/dev/null; STATUS_PID=""
status_now
log "DONE"
[ "$SHUTDOWN" = 1 ] && log "after the final commit: shutting Windows down in 60 s"
commit "baselines v3: analysis and final status" "$RES" results/baselines_v3.md "$STATUS_FILE" logs/baselines_v3_*.log
if [ "$SHUTDOWN" = 1 ]; then
  "${BASELINES_SHUTDOWN_EXE:-/mnt/c/Windows/System32/shutdown.exe}" /s /t 60
fi
