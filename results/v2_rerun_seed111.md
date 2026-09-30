# v2 re-run of the three-objective warm-start comparison, seed 111, to 3,000 updates (EXPLORATORY)

## Pre-registration

Written and committed BEFORE any training for this study (the smoke test and the loop-fidelity check below use
separate scratch outputs and are not results). Never edited afterwards; changes go under "Deviations".

**This is an EXPLORATORY single-seed run (seed 111 only).** Its verdicts guide decisions only and are PROVISIONAL
until a 4-seed confirmation (same code, `--seeds 111 222 333 444`). All confidence intervals below are paired
bootstrap intervals over the 49 HELD-OUT instances: they reflect instance variation only, NOT seed variation.

### Question
v2 (results/v2_fixes.md) makes due dates observable, scales carbon, and makes the training return exactly
-(Step 4 score) with balanced terms. This is the first run where both fixes act together on the three-objective
problem. Does the v1 plateau (0.866-0.90 at update 1,000; best C1 0.8655 pooled, 0.8556 at seed 111; CP-SAT best
0.614 pooled, 0.6144 at seed 111) move, and does the gap to CP-SAT shrink?

### Setup (identical to Step 4 unless stated)
* 10x5; seed 111 only; goal mct, preference (1/3, 1/3, 1/3); training generator = the 10x5 train split, 20
  environments resampled every 20 updates; PPO settings = `params.py` defaults as Step 4 (lr 3e-4, eps_clip 0.2,
  k_epochs 4, minibatch 1024, entloss_coef 0.01, vloss_coef 0.5, gamma 1, gae_lambda 0.98, tau 0); due dates
  k125s060 from the frozen manifest. No hyperparameter is tuned.
* v2 arms: `--obs_version v2 --reward_version v2` (reward references: frozen manifest, seed 111).
  V1-CONT: v1 flags, Step 4's reward weights (carbon_reward_weight 0.002384, tardiness_reward_weight 5.815362).
* Critic as Step 4: warm-start arms load the WHOLE checkpoint (critic included, no rescaling); scratch gets a fresh
  network from the seeded initialisation; no warm-up.
* Implementation: `tools/run_v2_rerun.py` drives Step 4's own `train.Trainer` (environment, instance sampling,
  rollouts from `policy_old`, `PPO.update`) with a copy of `Trainer.train`'s update loop that trains in resumable
  chunks and skips the per-10-update validation pass (validation only selects a "best" checkpoint, which this study
  does not use, and consumes no random numbers). Every checkpoint stores policy, policy_old, the Adam optimiser
  state and all random-number-generator states, so a run can be resumed or extended exactly. Loop fidelity is
  checked before launch: the chunked loop must reproduce `train.py` bit-for-bit (40 updates, v1 and v2), and a
  run interrupted and resumed must equal an uninterrupted one bit-for-bit.
* Snapshots and HELD-OUT evaluation at updates 0, 50, 100, 250, 500, 1000, 1500, 2000, 2500, 3000 (V1-CONT: 1000,
  1500, 2000, 2500, 3000). Evaluation: the 49 HELD-OUT instance IDs (`results/blend_zero_training/split.json`),
  greedy, batching only instances sharing the maximum processing time, unchanged Step 4 score (seed 111 centroid
  references), v2 checkpoints in the v2 environment and v1 in the v1 environment.
* Input checkpoints are hash-checked before training (`results/v2_rerun_seed111/input_hashes.json`).

### Arms (priority order V2-M, V2-S, V1-CONT, V2-T)
* **V2-M** makespan specialist converted to v2 (`m_s111_obsv2`, v2_fixes Part B), fine-tuned in v2, 3,000 updates.
* **V2-S** scratch in v2, 3,000 updates.
* **V1-CONT** control: v1 C1 (`s4t_spec_m_k125s060_s111@u1000`, Step 4 makespan arm) continued in v1 from update
  1,000 to 3,000. Step 4 saved no optimiser or random-number state at update 1,000: a fresh Adam optimiser is created
  and the generators are re-seeded with 111 at the continuation start (so its instance-sampling stream repeats the
  one Step 4 used from update 0). Recorded as a deviation from "exact continuation".
* **V2-T** the v2 tardiness specialist (`t_k125s060_s111_v2ft`, v2_fixes Part C), fine-tuned in v2, 3,000 updates.

### Metrics
* Step 4 score per checkpoint (mean over the 49 instances).
* Gap to CP-SAT per checkpoint: mean over instances of (score - CP-SAT 60 s best) / CP-SAT best, seed 111 solutions
  of ceiling_check, with a bootstrap 95% CI over instances.
* Raw objectives (makespan, carbon, tardiness) at 1,000 and 3,000.
* Due-date behaviour at 1,000 and 3,000, as input_reward_audit: Spearman(d_j/P_j, C_j/P_j) over the 490 HELD-OUT
  jobs (v1 C1 about -0.02; CP-SAT +0.74) and the share of tightest-quartile jobs late (the audit's Q1 job set).
* Comparisons: relative gain of X over Y = (mean score Y - mean score X) / mean score Y (positive = X better), paired
  bootstrap 95% CI over instances, 10,000 resamples, generator 20261001.
* Cost: GPU wall-clock hours per arm.

### Decision rules (computed automatically by `tools/analyze_v2_rerun.py`; exploratory, seed 111)
* **Q1 Does v2 lift the plateau?** LIFTED if V2-M@1000 beats v1 C1@1000 (Step 4 result file, seed 111, re-scored)
  by >= 2% with the CI excluding 0, AND V2-M@3000 beats V1-CONT@3000 by >= 2% with the CI excluding 0. Otherwise
  NOT LIFTED.
* **Q2 Is v2 still budget-limited?** BUDGET-LIMITED if V2-M@3000 beats V2-M@1000 by >= 1% with the CI excluding 0;
  otherwise PLATEAU. The same comparison is reported for V1-CONT (3000 vs 1000).
* **Q3 Is warm-starting still more efficient than scratch in v2?** YES if V2-M beats V2-S by >= 2% with the CI
  excluding 0 at both update 250 and update 500; otherwise NO. Also reported: the first checkpoint at which V2-S's
  mean score <= V2-M@1000's, and whether V2-S@3000 is within +-1% of V2-M@3000 with the CI including 0 ("caught up").
* **Q4 (secondary) Is the due-date-aware tardiness specialist the better start?** V2-T vs V2-M at 1,000 and 3,000:
  BETTER if gain >= 1% with the CI excluding 0; WORSE if gain <= -1% with the CI excluding 0; otherwise TIED.
* If an arm is FAILED or incomplete, every rule that depends on it is NOT DETERMINED; no verdict is computed from
  partial data.

## Deviations

* **V1-CONT is not an exact continuation.** Step 4 saved neither the Adam state nor the random-number-generator
  states at update 1,000, so V1-CONT starts from C1@1000's weights with a fresh Adam optimiser, and its generators
  are re-seeded with 111 (its instance-sampling stream repeats the one Step 4 used from update 0). Stated in the
  pre-registration; recorded here as required.
* **Loop-fidelity check (before launch; scratch outputs, not committed).** 40 updates from a warm start (v1: C1@1000
  with Step 4 weights; v2: `m_s111_obsv2` with v2 flags), seed 111: the chunked runner (2 chunks of 20, resumed from
  the saved checkpoint) and an uninterrupted runner (1 chunk of 40) gave bit-identical policy, policy_old and Adam
  state, and both were bit-identical to `daniel/train.py` run with the same flags (policy at update 40). Measured
  speed 0.77-0.97 s/update.
* (Anything that happens during the overnight run is appended below by hand in a later session.)

<!-- BEGIN GENERATED RESULTS (tools/analyze_v2_rerun.py) -->
_(filled in automatically by `tools/analyze_v2_rerun.py` at the end of `scripts/overnight_v2_rerun.sh`)_
<!-- END GENERATED RESULTS -->

## Plain-English summary

_(placeholder: 5-8 sentences, to be written in a later session after reading the results)_
