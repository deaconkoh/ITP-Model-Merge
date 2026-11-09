#!/usr/bin/env bash
# Tardiness specialist with lateness inputs (docs/tardiness_features_protocol_v3.md).
#   Trains DANIEL-T from scratch with --lateness_inputs on (10x5, seed 111) with exactly the committed DANIEL-T s111
#   command and stopping rule (scripts/run_baselines_v3.sh, train_one), saves the start and best checkpoints with
#   SHA-256 hashes, evaluates on the evaluation split (greedy + best of 100), renders results/tardiness_features_v3.md.
#
# Usage: scripts/run_tardiness_features_v3.sh [--smoke] [--preflight-only]
#   --smoke   5 updates, evaluation on the first 3 development instances, separate outputs (testing only)
#
# Resume: re-run the same command (training continues from its last chunk). Commits locally; never pushes.
# Never reads the reserved final-test sets.
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
STATE="$GITDIR/tardiness_features_v3"; mkdir -p "$STATE" logs

# ------------------------------------------------------------------ settings: the committed DANIEL-T s111's
SIZE=10x5; NJ=10; NM=5; SEED=111; OBJ=t
TRAIN_CPUS="0-3"; TRAIN_THREADS=4
VALIDATE=50; WINDOW=1000; TOL=0.005; MIN_UPDATES=1000; CAP=5000; CHUNK=500; SPU=0.9
EVAL_SPLIT=evaluation; EVAL_FIRST=0; SAMPLE=100
SUFFIX="bv3_tlate_s$SEED"; RES="results/tardiness_features_v3"; CKPT="checkpoints/tardiness_features_v3"
REPORT="results/tardiness_features_v3.md"; PREFIX="tardiness features v3"
COMMITTED_T="checkpoints/specialists_v3/10x5+mix+bv3_spec_t_s111.pth"
COMMITTED_T_SHA=4f3fb611057ca84bb30ef99bfe3bc770f7dd5b42f77260136c771edf97fccecd
if [ "$SMOKE" = 1 ]; then
  VALIDATE=5; WINDOW=1000; MIN_UPDATES=1000; CAP=5; CHUNK=0          # one call of 5 updates (no chunking)
  EVAL_SPLIT=development; EVAL_FIRST=3; SAMPLE=4
  SUFFIX="bv3_tlates_s$SEED"; RES="results/tardiness_features_v3_smoke"; CKPT="checkpoints/tardiness_features_v3_smoke"
  REPORT="results/tardiness_features_v3_smoke.md"; PREFIX="tardiness features v3 SMOKE"
fi
NAME="${SIZE}+mix+${SUFFIX}"
TS="$(date +%Y%m%d_%H%M%S)"
LOG="logs/tardiness_features_v3_$TS.log"
export WANDB_MODE=offline HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTHONUNBUFFERED=1

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }
die() { log "STOPPED: $*"; exit 2; }

exec 8>"$STATE/run.lock"
flock -n 8 || { echo "another run_tardiness_features_v3.sh is running" >&2; exit 2; }

CHILD=""
run_guarded() { timeout --kill-after=60 "$1" "${@:2}" & CHILD=$!; wait "$CHILD"; local rc=$?; CHILD=""; return $rc; }
trap 'log "INTERRUPTED: re-run the same command to resume"; [ -n "$CHILD" ] && kill -TERM "$CHILD" 2>/dev/null; exit 130' INT TERM

commit() {
  local msg="$1"; shift
  local p; for p in "$@"; do [ -e "$p" ] && git add -A -- "$p"; done
  git diff --cached --quiet || git commit -q -m "$msg" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
}

# ------------------------------------------------------------------ pre-flight
log "pre-flight: $NAME$([ $SMOKE = 1 ] && echo ' [SMOKE]')"
"$PY" -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null || die "GPU not visible to torch"
[ "$(sha256sum "$COMMITTED_T" | cut -d' ' -f1)" = "$COMMITTED_T_SHA" ] || die "committed DANIEL-T s111 checkpoint hash mismatch"
dirty=$(git status --porcelain -- . ':!logs/tardiness_features_v3_*' ":!$RES" ":!$CKPT" ":!$REPORT" ':!logs/baselines_v3_console.out')
[ -z "$dirty" ] || die "git tree not clean: $(echo "$dirty" | head -5 | tr '\n' ' ')"
# the v1 reproduction test needs v1 settings, so it runs in its own process (the v3 test modules set v3 on import)
out=$( { WS_V1_STRICT=1 "$PY" -m unittest -v tests.test_v1_reproduction &&
         "$PY" -m unittest tests.test_benchmark_v3 tests.test_tardiness_diagnosis_v3 tests.test_tardiness_features_v3; } 2>&1); rc=$?
echo "$out" | tail -20 >>"$LOG"
[ $rc -eq 0 ] || die "unit tests failed (see $LOG)"
echo "$out" | grep -q "skipped" && die "a test was skipped"
log "pre-flight OK: GPU $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1), committed DANIEL-T verified, tree clean, tests pass"
[ "$PREFLIGHT_ONLY" = 1 ] && exit 0

# ------------------------------------------------------------------ train (chunks, resumable)
rdir="$RES/runs/$NAME"; edir="$RES/eval/${EVAL_SPLIT}_${SIZE}"
live="daniel/train_log/SD2/$NAME.status.json"; tout="daniel/train_log/SD2/$NAME.stdout.log"
mkdir -p "$rdir" "$CKPT" daniel/train_log/SD2
timeout_s=$("$PY" -c "print(int(max($CHUNK, $CAP if $CHUNK <= 0 else 0) * $SPU * 3 + 600))")
fails=0; t_start=$(date +%s)
while true; do
  st=$("$PY" -c "import json; print(json.load(open('$live'))['status'])" 2>/dev/null || echo NONE)
  case "$st" in CONVERGED|NOT_CONVERGED) break ;; esac
  extra=(); [ "$st" = NONE ] && extra=(--budget_checkpoints 0)     # the start network: first chunk only
  log "train $NAME: chunk of $CHUNK updates (status $st)"
  t0=$(date +%s.%N)
  run_guarded "$timeout_s" env -C daniel OMP_NUM_THREADS=$TRAIN_THREADS MKL_NUM_THREADS=$TRAIN_THREADS \
      taskset -c "$TRAIN_CPUS" "$PY" train.py --config "../configs/benchmark_v3/$OBJ.json" --n_j $NJ --n_m $NM \
      --model_suffix "$SUFFIX" --seed_train $SEED --v3_train_source generator \
      --validate_timestep $VALIDATE --converge_window $WINDOW --converge_tol $TOL --min_updates $MIN_UPDATES \
      --max_updates $CAP --chunk_updates $CHUNK --resume True --device cuda --lateness_inputs on "${extra[@]}" >>"$tout" 2>&1
  rc=$?
  "$PY" - "$rdir/process.json" "$t0" "$rc" <<'EOF'
import json, sys, time, os
f, t0, rc = sys.argv[1], float(sys.argv[2]), sys.argv[3]
d = json.load(open(f)) if os.path.exists(f) else {"process_seconds": 0.0, "chunk_rc": []}
d["process_seconds"] += time.time() - t0; d["chunk_rc"].append(rc)
json.dump(d, open(f, "w"), indent=1)
EOF
  if [ $rc -ne 0 ]; then
    fails=$((fails + 1)); log "chunk FAILED (rc $rc), attempt $fails"
    [ $fails -ge 2 ] && { echo "training failed twice (rc $rc); see $tout" > "$rdir/FAILED"; commit "$PREFIX: $NAME FAILED" "$rdir"; die "training failed"; }
  else
    fails=0
  fi
done
for f in valiquality reward; do cp "daniel/train_log/SD2/${f}_$NAME.json" "$rdir/$f.json"; done
cp "daniel/train_log/SD2/$NAME.run.json" "$rdir/run.json"; cp "$live" "$rdir/status.json"
log "training $st: $("$PY" -c "import json; s=json.load(open('$live')); print(f\"update {s['update']}, best {s['best_value']:.5g} @ {s['best_update']}, train {s['train_seconds']/3600:.2f} h\")")"

# ------------------------------------------------------------------ checkpoints + hashes
cp "daniel/trained_network/SD2/$NAME@u0.pth" "$CKPT/${NAME}_start.pth" || die "start checkpoint missing"
cp "daniel/trained_network/SD2/$NAME.pth" "$CKPT/$NAME.pth" || die "best checkpoint missing"
"$PY" - "$CKPT" <<'EOF'
import hashlib, json, sys
from pathlib import Path
d = Path(sys.argv[1])
sums = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(d.glob("*.pth"))}
(d / "SHA256SUMS.json").write_text(json.dumps(sums, indent=1) + "\n")
EOF
commit "$PREFIX: $NAME training ($st)" "$rdir" "$CKPT"

# ------------------------------------------------------------------ evaluation
efails=0
while [ ! -f "$edir/$NAME.json" ]; do
  log "eval $NAME on $EVAL_SPLIT ($([ $EVAL_FIRST = 0 ] && echo all || echo first $EVAL_FIRST)): greedy + best of $SAMPLE"
  run_guarded 3600 env -C daniel taskset -c "$TRAIN_CPUS" "$PY" evaluate_v3.py --models "../$CKPT/$NAME.pth" \
      --split "$EVAL_SPLIT" --size "$SIZE" --first "$EVAL_FIRST" --sample "$SAMPLE" --out "../$RES/eval" --device cuda \
      --lateness_inputs on >>"$tout" 2>&1 || { efails=$((efails + 1)); log "eval FAILED (attempt $efails)";
      [ $efails -ge 2 ] && die "evaluation failed twice"; }
done
commit "$PREFIX: $NAME evaluation" "$edir"

# ------------------------------------------------------------------ report
log "report: tools/analyze_tardiness_features_v3.py"
"$PY" tools/analyze_tardiness_features_v3.py $([ $SMOKE = 1 ] && echo --smoke) >>"$LOG" 2>&1 || log "report FAILED (see log)"
commit "$PREFIX: report" "$REPORT" "$RES" "$LOG"
log "DONE ($(( ($(date +%s) - t_start) / 60 )) min). Committed locally; push when ready."
