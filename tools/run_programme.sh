#!/bin/bash
# Initialisation-quality programme runbook.
#
#   ./tools/run_programme.sh <step>
#
# Steps run ONE AT A TIME and the two gates are hard stops -- the script will not carry on
# past a gate on its own. Reserved final-test sets are untouched until step 8.
#
#   1  train priority specialists (4 seeds x 2 sizes) + evaluate them
#   G1 GATE: specialist sanity check                      <-- STOP, inspect
#   2  reference composition eval -> closed-form anchors -> verification sweep
#   3  simplex grid: coarse (step 0.1) -> evaluate -> fine refinement -> evaluate
#   G2 GATE: edge-vs-interior go/no-go                    <-- STOP, inspect
#   4  fine-tuning arms: merge / scratch / spec_m / spec_c / spec_p
#   5  initialisation persistence (several starting compositions)
#   6  preference variation (2 extra preferences at 10x5)
#   7  analysis: budget curves, crossovers, Pareto fronts
#   8  reserved final-test -- ONCE, after everything is frozen
set -uo pipefail
REPO=/home/deaconkoh/projects/ITP-PROJECT-multi-objective-model-merging-for-fjsp
cd "$REPO"; source .venv/bin/activate
LOG=${LOG:-$HOME/.claude/jobs/programme}; mkdir -p "$LOG"
SEEDS="111 222 333 444"
PRIMARY=10x5
POOL=trainvali

step=${1:-help}

case "$step" in

1)  echo "=== STEP 1: priority specialists ==="
    for size in 10x5 20x10; do
      nj=${size%x*}; nm=${size#*x}
      for s in $SEEDS; do
        name="${size}+carbon+priority+p_s${s}"
        [ -f "daniel/trained_network/SD2/${name}.pth" ] && { echo "  skip $name"; continue; }
        echo "  training $name at $(date +%H:%M:%S)"
        ( cd daniel && python train.py --config ../configs/canonical/p.json \
            --train_data_path ./data/data_train/SD2/${size}+carbon+priority \
            --validation_data_path ./data/data_validation/SD2/${size}+carbon+priority \
            --test_data_path ./data/data_final_test/SD2/${size}+carbon+priority \
            --n_j $nj --n_m $nm --data_suffix carbon+priority \
            --seed_train $s --model_suffix p_s${s} --device cuda \
            > "$LOG/train_p_${size}_s${s}.log" 2>&1 )
        echo "    exit=$?"
      done
      # evaluate all three specialists so the gate has data
      NAMES=""; for o in m c p; do for s in $SEEDS; do NAMES="$NAMES ${size}+carbon+priority+${o}_s${s}"; done; done
      ( cd daniel && python ../tools/eval_checkpoints.py --pool-root ./data/data_train_vali \
          --pool "${size}+carbon+priority" --out-tag $POOL --models $NAMES \
          > "$LOG/eval_specialists_${size}.log" 2>&1 )
    done
    echo "STEP 1 COMPLETE -> now run: ./tools/run_programme.sh G1" ;;

G1) echo "=== GATE 1: do the priority specialists behave as specialists? ==="
    python tools/check_specialists.py --sizes 10x5 20x10 --pool-tag $POOL
    echo; echo ">>> HARD STOP. Inspect the result before running step 2." ;;

2)  echo "=== STEP 2: reference composition + reward-weight anchors ==="
    echo "  (a) build + evaluate the centroid composition, which defines the per-instance reference"
    python tools/simplex_merge.py --build --points 333,333,334 --sizes $PRIMARY --seeds $SEEDS
    NAMES=""; for s in $SEEDS; do NAMES="$NAMES ${PRIMARY}+carbon+priority+simplex_m333c333p334_s${s}"; done
    ( cd daniel && python ../tools/eval_checkpoints.py --pool-root ./data/data_train_vali \
        --pool "${PRIMARY}+carbon+priority" --out-tag $POOL --models $NAMES \
        > "$LOG/eval_centroid.log" 2>&1 )
    echo "  (b) derive closed-form anchors from the measured reference, then verify with a short sweep"
    echo "      -> tools/preference3.py provides reward_anchors(); the sweep runs 250-update jobs"
    echo "STEP 2a COMPLETE -- compute anchors, then run the verification sweep manually or via step 2b" ;;

3)  echo "=== STEP 3: simplex grid (coarse -> fine) ==="
    echo "  coarse: step 0.10 (66 points x 4 seeds)"
    python tools/simplex_merge.py --build --step 0.1 --sizes $PRIMARY --seeds $SEEDS
    NAMES=$(python - <<'PY'
import sys; sys.path.insert(0,'tools')
from simplex_merge import simplex_points, name_for
print(" ".join(name_for("10x5",p,s) for s in (111,222,333,444) for p in simplex_points(0.1)))
PY
)
    ( cd daniel && python ../tools/eval_checkpoints.py --pool-root ./data/data_train_vali \
        --pool "${PRIMARY}+carbon+priority" --out-tag $POOL --models $NAMES \
        > "$LOG/eval_simplex_coarse.log" 2>&1 )
    echo "  coarse evaluation done. Inspect, then build the fine refinement around the optimum:"
    echo "    python tools/simplex_merge.py --build --step 0.05 --points <neighbourhood>"
    echo "STEP 3 COMPLETE -> now run: ./tools/run_programme.sh G2" ;;

G2) echo "=== GATE 2: does the third specialist earn its place? ==="
    python tools/analyze_simplex.py --size $PRIMARY --pool-tag $POOL
    echo; echo ">>> HARD STOP. If the optimum is on an EDGE, do not run steps 4-6." ;;

4)  echo "=== STEP 4: fine-tuning arms ==="
    : "${INIT_MERGE:?set INIT_MERGE to the chosen composition checkpoint name}"
    : "${WC:?set WC to the calibrated carbon_reward_weight}"
    : "${WP:?set WP to the calibrated priority_reward_weight}"
    python tools/run_arms.py --run --size $PRIMARY --init-merge "$INIT_MERGE" \
        --wc "$WC" --wp "$WP" --seeds $SEEDS --tag-prefix s4
    echo "  evaluating all budget checkpoints"
    NAMES=""; for a in merge scratch spec_m spec_c spec_p; do for s in $SEEDS; do
      for b in 0 50 100 250 500 1000; do NAMES="$NAMES ${PRIMARY}+carbon+priority+s4_${a}_s${s}@u${b}"; done; done; done
    ( cd daniel && python ../tools/eval_checkpoints.py --pool-root ./data/data_train_vali \
        --pool "${PRIMARY}+carbon+priority" --out-tag $POOL --models $NAMES \
        > "$LOG/eval_arms.log" 2>&1 )
    echo "STEP 4 COMPLETE" ;;

7)  echo "=== STEP 7: analysis ==="
    : "${REF_TAG:?set REF_TAG to the reference composition tag, e.g. simplex_m333c333p334_s{seed}}"
    python tools/analyze_arms.py --size $PRIMARY --pool-tag $POOL --prefix s4 --ref "$REF_TAG" ;;

8)  echo "=== STEP 8: reserved final-test -- ONCE, after everything is frozen ==="
    echo "  deliberately not automated. Confirm the frozen model list, then evaluate with"
    echo "  --pool-root ./data/data_final_test"
    ;;

*)  sed -n '2,22p' "$0" ;;
esac
