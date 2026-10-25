# Warm-start study on benchmark v3: pre-registered protocol

**Status: PRE-REGISTERED on 2026-10-25, before any run of these arms.** This document is not edited after the full run starts. Later changes are recorded, with date and reason, under **Deviations**.

- **Runner:** `bash scripts/ec2_run_all.sh` (orchestration: `tools/ec2_runner.py`).
- **Analysis:** `tools/analyze_warmstart_v3.py`, which writes `results/warmstart_v3.md` with an unchanged copy of this document.

## 1. Question and scope

Can the trained single-objective specialists (DANIEL-M, -C and -T; `results/baselines_v3.md`) make a preference-conditioned multi-objective DANIEL cheaper to train or better? The baseline is the same model trained from scratch.

- **Benchmark:** `sd2-mo-v3` (`configs/problem_spec_v3.yaml`, frozen).
- **Sizes:** 10x5 by default; 20x10 with `--sizes 20x10`.
- **Seeds:** 111, 222 and 333 for every arm. **Arm seed s uses the specialists of seed s.**
- **Unchanged:**
  - the spec, the manifest, the reward and score definitions;
  - DANIEL's action mask (non-delay; `results/diagnose_v3.md` Test 2; a known limitation, reported, not changed);
  - DANIEL's PPO algorithm and hyperparameters (no gradient clipping, lr 3e-4, 4 epochs, clip 0.2, γ 1, λ 0.98, entropy 0.01, 20 environments).
- **Additions:** only preference conditioning, weight transfer, distillation, merging and logging.
- **Data separation:** selection and convergence use only the validation split. The evaluation split is used once, at the end. The reserved final-test sets are never read.

## 2. The preference-conditioned model (all arms)

The model is the MO-DANIEL of `docs/evaluation_protocol_v3.md`, **Addendum A**: w = (w_m, w_c, w_t) concatenated to the actor's and the critic's inputs; the feature extractor does not see w.

- **WI-DAN.** It corresponds to WI-DAN, the concatenation baseline in Smit et al. (DCAN, ICLR 2026).
- **Reward.** The return is −(w_m·MK/MKref + w_c·C/Cref + w_t·T/Tref).
- **Training preferences.** w ~ Dirichlet(1,1,1) per training instance (environment) and update. Fresh SD2-generator instances; the instance stream is the same as the specialists' for the same seed.
- **Validation.** Mean normalised hypervolume over the 15-preference grid (step 0.25), every 50 updates and at update 0.
- **Validation normalisation points.** Computed on the run machine at the start of the run, by Addendum A2's definition (CP-SAT B1 procedure, the specialists' greedy schedules, MWKR/ECT), frozen in `results/warmstart_v3/norm/`. The file records the CPU model.
- **Convergence and cap.** Addendum A3: the best validation HV improves by < 0.5% over 1,000 updates, after a minimum of 1,000; cap 8,000 updates (NOT CONVERGED if reached). DEGRADED flag: the mean of the last 3 validation HVs is > 5% below the best.
- **Checkpoint:** the best validation HV.
- **Training mechanics.** Chunked training (500 updates at 10x5, 200 at 20x10) with bit-exact resume. Hang guard: a timeout per chunk, retried once, then FAILED, and the run continues.

## 3. Arms

| Arm | Initialisation (`train.py --init_from … --init_mode …`), then PPO exactly as B |
|---|---|
| **B** | Random initialisation (the baseline; MO-DANIEL / WI-DAN). |
| **C-x**, x ∈ {M, C, T} | `--init_mode full`: every tensor of specialist x, its critic included. The three preference-input columns of the first actor layer and the first critic layer are **zero**, so at update 0 the policy and the value equal specialist x's for every w (`tests/test_warmstart_v3.py`: max \|Δπ\| ≤ 1e-6, identical greedy actions along validation rollouts; bitwise equality is not guaranteed because the matrix multiply may sum in a different order). |
| **BB-x** | `--init_mode backbone`: only the dual-attention feature extractor (`feature_exact.*`) of x. The actor, critic and preference weights keep the fresh initialisation, which is identical to B's for the same seed. |
| **D** | Vertex distillation (below), then `--init_mode no_critic`: everything except the critic from the student. The critic is the fresh initialisation, identical to B's. |
| **E** | `--init_mode full` from the uniform average of the three specialists' full state dicts (extractor, actor, critic). Preference columns zero. Label in every output: **"naive merge of independently trained specialists (no shared base); expected to be weak"**. |
| **E0** | No training. θ(λ) = Σ λ_x θ_x (published unconditioned architecture), greedy at each of the 66 grid preferences plus equal weights, evaluated like the arms. Reported only; a Rewarded-Soups-style check. |
| **E-shared** (`--with-shared-base`, off by default) | An unconditioned equal-weight (`mct`) base trained for 500 updates; M, C and T fine-tuned from its best-validation checkpoint for 1,000 updates each (fixed budgets, best-validation checkpoints); merged uniformly; preference columns zero; PPO as B. Extra cost per seed: about 0.9 h at 10x5 (estimated from the measured specialist speed), about 5× at 20x10. Base and fine-tune training count toward its marginal cost. |

**Arm D: distillation** (`daniel/distill_v3.py`).

| Item | Value |
|---|---|
| States | Each specialist x samples its own rollouts on fresh generator instances (forbidden hashes excluded; global numpy stream seeded seed + 7001; torch sampling generator seeded seed + 7001). 400 instances per specialist at 10x5 and 100 at 20x10, about 20k states each, 60k in total. |
| Loss | KL(π_x(·\|s) ‖ π_student(·\|s, e_x)) over the allowed actions, with e_x the corner preference |
| Training | Adam, lr 3e-4, batch 256, **15 epochs** |
| Budget | A hard wall-clock cap of **20 GPU-minutes**, collection included. Measured at full budget (2026-10-26, RTX 5060 Ti): 2.1 min at 10x5 and 2.4 min at 20x10, far under the cap; the budget is kept. |
| Cost | Collection plus training time is reported separately and counted in D's marginal cost |
| Held-out check | KL and greedy-action agreement on the teachers' greedy validation trajectories |

**Run order:**
- Seed 111 for all arms first, in the order B, C-M, BB-M, D, E, C-C, C-T, BB-C, BB-T (then E-shared if enabled).
- Then seeds 222 and 333.
- E0 per size after that size's training runs.
- With `--jobs N`, up to N arms train at the same time on one GPU (CPU cores split evenly). **N is the same for every arm and recorded with each run**, so timing comparisons are within one N.

## 4. Hypotheses and verdict rules

**Notation:**
- **"Paired CI":** the bootstrap 95% CI (10,000 resamples, `default_rng(20261013)`, percentile) of the mean over instances of the seed-averaged per-instance difference, arm − B.
- **Instances:** evaluation instances for final quantities; validation instances for HV@500.

**C. Copying a whole specialist helps early but biases toward its corner.** Per x:
- (i) The paired CI of validation HV at 500 updates (C-x − B) has its lower bound > 0.
- (ii) Corner bias: per instance, mean regret in x's corner region (λ_x ≥ 0.6) minus mean regret elsewhere. The paired CI of (C-x − B) on this quantity has its upper bound < 0.
- **Verdict per x:** SUPPORTED if (i) and (ii) both hold; NOT SUPPORTED if neither holds; otherwise UNCLEAR.
- **Overall:** SUPPORTED or NOT SUPPORTED if at least 2 of the 3 x agree; otherwise UNCLEAR.

**BB. A general scheduling representation transfers.** Per x:
- SUPPORTED if the paired CI of final evaluation HV (BB-x − B) has its lower bound > 0, **or** BB-x reaches 95% of B's final validation HV in fewer updates than B on every seed (seed-paired; reaching it when B does not counts as fewer).
- NOT SUPPORTED if that CI's upper bound is ≤ 0 and it is not faster.
- UNCLEAR otherwise.
- **Overall:** as for C.

**D. Specialist behaviour transfers.** The BB rule applied to D.

**E. Weight merging of unrelated specialists fails.**
- SUPPORTED if all three hold:
  - E is not better than B at 500 updates (paired CI lower bound ≤ 0);
  - E is not better than B at the end (paired CI lower bound ≤ 0);
  - E0's evaluation HV is below the specialists' combined-set HV (seed-paired; paired CI upper bound < 0).
- NOT SUPPORTED if E beats B at either point (lower bound > 0), or E0 beats the specialists (lower bound > 0).
- UNCLEAR otherwise.

Any missing data gives UNCLEAR.

## 5. Metrics

**Evaluation** (the selected checkpoint; evaluation split, 100 instances per size; once):
- Greedy rollouts under the 66 preferences of the step-0.1 simplex grid, plus equal weights (1/3,1/3,1/3), which is reported but not part of the set. The same for every arm, E0 included.
- Every schedule is re-scored independently.
- **Hypervolume** of the per-instance non-dominated set, normalised with the frozen section-6 points (`results/baselines_v3/normalisation_points.json`), reference 1.1.
- **IGD+** against the CP-SAT B2 reference front, on the B2 subset only (20 instances at 10x5, 10 at 20x10; best-known at 20x10).
- The number of unique non-dominated solutions.
- **Each objective** at the three corner preferences and at equal weights.
- **Preference regret** R(λ) = λ·f̂(λ) − min over every arm's and reference's points of λ·f, in the frozen normalised space. "Every point" means every arm's points (all seeds and preferences), the specialists, the rules and CP-SAT B1/B2. Mean over instances and seeds, per λ, shown across the simplex.
- **Secondary:** best-of-16 sampled rollouts per preference (by the preference-weighted score; torch generator seed 50), with the same metrics.
  - At 10x5 on all 100 instances (measured about 3 min per run).
  - At 20x10 on the **first 20** evaluation instances, which contain the 10-instance B2 subset. On all 100 it would cost about 50 min per run.
  - Greedy remains the primary result.

**Efficiency** (validation):
- AUC of validation HV over [0, 2,000] updates (step function, last value carried forward), divided by 2,000.
- HV at 500, 1,000 and 2,000 updates.
- Updates and training wall-clock time until the validation HV first reaches 90% and 95% of B's final validation HV. The target is the mean over B's seeds of their best validation HV, fixed after B finishes.

**Cost, two views:**

| View | Definition |
|---|---|
| Marginal | The arm's own PPO training time, plus D's imitation time and E-shared's base and fine-tune time |
| Full lifecycle | Marginal plus the training time of the specialists the arm uses: x for C-x and BB-x; all three for D and E. Local `train_seconds` are converted to run-machine time with the calibration factor. |

**Calibration.** The calibration run is 200 PPO updates of DANIEL-M at each requested size, with the specialists' flags, on the run machine, timed. factor = run-machine s/update ÷ local s/update, where the local value is the mean over the three DANIEL-M specialists' `train_seconds/update`.

**Statistics.** Mean ± std (ddof 1) over seeds of the per-seed instance means; paired bootstrap CIs versus B.

**Logging per PPO update** (all arms): entropy, approximate KL, clip fraction, policy loss, value loss, explained variance, gradient norm (computed, never clipped) and training return. Observation only; tested to leave the update bit-identical.

## 6. Inputs (committed; hash-verified before every run)

- `checkpoints/specialists_v3/` holds the 18 selected specialist checkpoints. Each hash equals the `checkpoint_sha256` recorded by its reported evaluation.
- `checkpoints/step4_v1/` holds the 4 checkpoints `tests/test_v1_reproduction.py` needs.
- Also committed: the frozen spec and data manifest, the evaluation normalisation points, the CP-SAT B1/B2 results, and the specialists' evaluation schedules and run records.
- `checkpoints/INPUTS.sha256.json` lists them; `tools/ec2_inputs.py verify` is part of the pre-flight.

## 7. Machine, environment and pre-flight

**Environment:**
- Python 3.14.4 venv created by `uv`.
- `requirements-ec2.txt`, pinned from the local environment: torch 2.9.1+cu128, ortools 9.15.6755, numpy 2.5.3, and so on.
- An NVIDIA driver ≥ 570 (CUDA 12.8).

**Pre-flight checks:**
- GPU visible;
- free disk space;
- clean git tree;
- every input hash;
- the v3 manifest;
- the v3, baselines, MO and warm-start test suites (blocking).

**v1 reproduction test.** It is also run. It requires bit-identical objectives versus result files from the reference machine (RTX 5060 Ti), so on a different GPU model a mismatch is **recorded as a warning and does not block** (decision of 2026-10-25). It was verified on the reference machine before hand-over.

**Recorded with every run:** the GPU model, CPU model and count, versions, and `--jobs`.

## 8. Expected runtime (from local measurements; to be confirmed by the calibration run)

**Measured locally** (2026-10-26; RTX 5060 Ti, Ryzen 5 7500F with 12 logical CPUs; MO training with full-size validation): aggregate PPO updates per second with N concurrent arms, CPUs split as the runner does.

| `--jobs` N | 10x5 aggregate (per-run s/update) | 20x10 aggregate (per-run s/update) |
|---|---|---|
| 1 | 1.13 (0.87) | 0.216 (4.59) |
| 2 | 1.68 (1.10) | 0.308 (6.40) |
| 3 | 2.31 (1.21) | 0.383 (7.76) |
| 4 | 2.52 (1.53) | 0.418 (9.42) |

**Estimates.** Each of the 27 runs per size stops between about 2,000 updates (earliest realistic convergence) and 8,000 (the cap), so 54k–216k updates in total per size:

| Size | `--jobs 1` | `--jobs 3` | `--jobs 4` |
|---|---|---|---|
| 10x5 | 13–52 h | 6.5–26 h | 6.0–24 h |
| 20x10 | 69–275 h | 39–157 h | 36–144 h |

**Added overheads:**
- calibration: about 3 min at 10x5 and 15 min at 20x10;
- validation points: about 5 min at 10x5; up to about 1 h at 20x10, overlapping training;
- distillation: about 2.5 min per seed;
- evaluation: greedy about 15 s (10x5) and 3 min (20x10) per run; sampling about 3 min (10x5) and 10 min (20x10) per run;
- E0: a few minutes per seed.

The run machine's CPU sets the speed (the environment is CPU-bound), so the calibration factor in `results/warmstart_v3/calibration_<size>.json` scales these numbers.

## 9. Smoke-test evidence and measured throughput (before launch)

**From fresh clones of the committed repository** (2026-10-26), so nothing needed is git-ignored, using `bash scripts/ec2_run_all.sh` itself:
- **Default smoke** (`--smoke`: seed 111, all 9 arms and E0, `--jobs 1`):
  - setup: uv venv with Python 3.14.4 and `requirements-ec2.txt`;
  - pre-flight: GPU, disk, clean tree, 345 input hashes, the v3 manifest, the v3/baselines/MO/warm-start test suites; the v1 reproduction test passed;
  - self-detach, calibration, development and validation points, distillation, merge, 9 training runs (40 updates, 20-update chunks), evaluation (greedy and best-of-2), E0, analysis and local commits;
  - 8.6 min end to end, no failures, clean tree at the end.
- **Kill and resume** (`--smoke --jobs 2 --with-shared-base --shutdown`, three seeds, shutdown replaced by a stub):
  - SIGTERM to the detached worker during two parallel training chunks stopped every child, wrote no FAILED marker and did not shut down;
  - re-running the same command resumed B at update 20 from its resume state and finished all 45 units, E-shared included;
  - the analysis and final commit followed, and the shutdown stub was called only after the final commit.
- **Hang guard:** with a 5 s chunk timeout the chunk timed out twice, the run was marked FAILED and committed, and the runner continued and finished. A plain re-run retried and completed it.
- **Fixed during the smoke tests:**
  - distillation logs next to checkpoints made the tree dirty (now git-ignored);
  - the final commit now includes every run log;
  - the smoke check defaults to one seed;
  - E0 is selectable with `--arms`.
- **Equivalence and determinism:**
  - C-x at update 0 matches specialist x within 1e-6, with identical greedy actions on validation rollouts (`tests/test_warmstart_v3.py`);
  - default v1 and v3-specialist training are bit-identical to before (20 updates each);
  - chunked MO training resumes bit-exactly (Addendum A7).
- **Tests:** all 48 tests (v1, v3, baselines, MO, warm-start) pass locally in one process, with the v1 test hard.

## Deviations

(none)
