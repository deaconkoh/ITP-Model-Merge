# Plateau diagnosis (10x5, seed 111), protocol fixed before the run

## Question

Every converged contrastive run reaches a validation HV of about 0.54–0.59 and then stays flat, whatever its start:
- From scratch;
- Distillation;
- the World-model base, Random shared base and Original specialists mixes (M3).

The mixes gain most within 250–500 updates. **What limits the plateau?** Three candidate causes:
- **a weak training signal:** the loss stuck near log 2, entropy collapsing, or fewer valid pairs;
- **greedy decoding:** the policy can produce better schedules, but its greedy choice does not;
- **something shared by all methods:** for example the no-waiting rule or the model size.

Settings are frozen in `configs/plateau_diagnosis_v3.json` (`tools/freeze_plateau_diagnosis_v3.py`). There are no changes to the benchmark, and the reserved final-test sets are never read.

## Part A: no training (`tools/plateau_diagnosis_v3_runner.py --part a`)

**1. Learning signals, read from the committed logs of the five runs.** Plotted over updates, smoothed over 50 updates:
- contrastive loss (log 2 = 0.693 means the policy separates better and worse schedules no more than the reference does);
- policy entropy (mean per step);
- distinct schedules per group of K;
- pairs per group and the share of groups with a pair;
- gradient norm before clipping, with the share of updates clipped at 1.0;
- distance of the weights from the start: ||θ − θ0|| / ||θ0|| over trunk and actor (all three networks of a mix). It is computed from the saved checkpoints, because the runs did not log it per update.

Early means the first 250 updates; late means the last 500. **Rules, per run:**
- **S1, loss stuck near log 2:** late mean loss > 0.68.
- **S2, entropy collapsing:** late entropy < 0.5 × early entropy.
- **S3, fewer valid pairs:** late pairs per group < 0.5 × early, or late distinct schedules < K/2.
- The signal **shrinks** if S2 or S3 holds.

**2. Greedy vs best-of-16 along training.** The checkpoints are the ones that exist; contrastive runs saved only the start, the best and the final state.

| Run | Checkpoints (update) | How obtained |
|---|---|---|
| From scratch | 0, 2450, 2600, 4150, 4550 | 0: re-created (`train.py --max_updates 0` with the run's own settings, no update), checked against the run's logged start hash. 2450 and 2600: the first segment's best and final state. 4150 and 4550: the converged run's best and final state. |
| World-model base mix | 0, 1650, 2250 | start, best and final state. Nothing was saved in between, so there are only 3 points. |
| Distillation, Random shared base mix, Original specialists mix | start, best, final (Distillation also 1000) | as above |

Each checkpoint is evaluated on the evaluation split (66 preferences plus equal weights, 100 instances):
- **greedy;**
- **best-of-16:** for each preference, the best of 16 sampled schedules by the preference-weighted score, torch seed 50. This is the existing `evaluate_mo_v3.py --sample` path.

**Rule:** best-of-16 "keeps rising while greedy stays flat" if, from the best to the final checkpoint, best-of-16 HV rises by ≥ 0.01 while greedy HV changes by < 0.01.

**3. Noise check.** For each mix, and for From scratch and Distillation:
- evaluation HV of the best and the final checkpoint, with the difference and a paired bootstrap 95% interval over the 100 instances (10,000 resamples, seed 2026);
- the same difference on validation, with the bootstrap over its 18 instances;
- the spread of the validation curve over the last 1,000 updates.

**Context for "shared" causes (no new solver runs).** On the 20 evaluation instances with a CP-SAT B2 front, three HVs are compared:
- the B2 front, which may wait;
- the union of every non-delay schedule produced in Part A (all checkpoints, greedy and best-of-16);
- each method alone.

## Part B: short training runs (set up here, run by the user)

**Start:** the World-model base mix start, the three M3 world-model-base specialists. Everything is identical to the M3 run except the one change and the fixed length:
- 1,500 updates, no stopping rule;
- validation every 50 updates;
- snapshots at 0, 250, …, 1500;
- the trunk and actor distance from the start logged every update (`--cl_log_drift`, logging only).

| Arm | Change |
|---|---|
| Stronger push | β 0.5 (instead of 0.1) |
| Faster learning | lr 1e-3 (instead of 3e-4) |
| More samples | K 16 (instead of 8) |
| Control | none (the M3 settings for 1,500 updates) |

**Reported per arm:**
- the validation HV curve;
- the final (update 1500) and best-validation checkpoints, each evaluated greedy and best-of-16;
- the loss and entropy curves;
- wall-clock time.

The Control should reproduce the M3 World-model base mix run's first 1,500 updates. This is checked and reported.

**Rule:** an arm beats the Control if its final greedy evaluation HV is ≥ the Control's + 0.02.

## Verdict rules (plain language in the report)

- **Weak training signal:**
  - "yes" if any Part B arm beats the Control;
  - "no" if none does;
  - before Part B, the Part A signal rules are reported and the answer is "pending".
- **Greedy decoding:**
  - "yes" if best-of-16 keeps rising while greedy stays flat for the World-model base mix or From scratch;
  - "partly" if the best-of-16 − greedy gap is ≥ 0.05 at every final checkpoint without that rise: greedy leaves HV unused, but this does not explain the flat curve;
  - otherwise "no".
- **Shared by all methods:**
  - "likely" if the final checkpoints of the five runs lie within 0.05 of each other on both greedy and best-of-16 evaluation HV;
  - otherwise "no".
  - The B2 comparison says how much of the remaining gap the no-waiting rule could explain.
  - Model size is not varied, so it is not tested.

Single seed: numbers only.

## Run

```
bash scripts/run_plateau_diagnosis_v3.sh --part-a     # Part A (no training)
bash scripts/run_plateau_diagnosis_v3.sh --jobs 3     # Part B (training; 4 runs on 3 slots)
```

Local commits only; never pushes.
