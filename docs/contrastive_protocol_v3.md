# Contrastive fine-tuning on benchmark v3: protocol (fixed before any run)

## Question

The warm-start study (`results/warmstart_v3.md`, `docs/warmstart_protocol_v3.md`) found that PPO fine-tuning lowered
the hypervolume (HV) of good starting points: D's best checkpoint was update 0 on every seed, and every run ended more
than 5% below its best validation HV. This study asks whether a contrastive objective behaves differently when it
replaces the PPO update and **nothing else changes**.

## What is identical to the PPO arms

- **The model:** the preference-conditioned DANIEL, schema `benchmark_v3_f12_m9_p10_w3`.
- **The environment and data:**
  - the environment, including the non-delay action mask;
  - the frozen v3 spec;
  - the training instances: the SD2 generator stream for seed 111, 20 instances resampled every 20 updates;
  - the training preferences: Dirichlet(1,1,1), the same `pref_rng` stream, 20 per update.
- **The normalisation used in training:** the per-instance generator references and the frozen T_ref.
- **Validation:** 15 preferences (grid step 0.25), at update 0 and every 50 updates, on the frozen validation
  normalisation points `results/warmstart_v3/norm/validation_normalisation_points_10x5.json`.
- **Checkpoint selection:** best validation HV.
- **Evaluation:** `daniel/evaluate_mo_v3.py` on the evaluation split, 66 preferences, greedy, plus best-of-16
  sampling. The metrics are HV, IGD+ against the CP-SAT B2 front (20 instances), unique non-dominated solutions and
  preference regret.

Only the update rule differs (`train.py --finetune_objective contrastive`; the default `ppo` path is unchanged, checked
bit-for-bit on 5-update MO and specialist runs).

## Arms (10x5, seed 111)

| Arm | Start | Frozen reference | Updates (= PPO counterpart's final update) |
|---|---|---|---|
| CL-B | random initialisation (same as B) | that initialisation | 2600 (B) |
| CL-D | D's update-0 network: the distilled student, `--init_mode no_critic` (regenerated with `distill_v3.py`, seed 111) | the student | 1000 (D) |
| CL-E | the naive merge of the s111 specialists, `--init_mode full` (as E) | the merge | 3350 (E) |

- **E0 is not included.** It is a family of interpolated, unconditioned networks, one per preference, not a single
  network that could be fine-tuned.
- **The budgets and constants are frozen** in `configs/contrastive_v3.json`, with hashes of their sources; the file is
  verified before every run.

## Method

On each update:

1. **Groups.** Each of the 20 instances, with its preference w, is repeated K = 8 times (160 environments). The policy
   samples one schedule per environment.
2. **Scoring.** Each schedule is scored exactly on makespan, carbon and tardiness. The normalised objectives are
   f̂ = (MK/mk_ref, C/cb_ref, T/T_ref).
3. **Pairs (better, worse) inside a group.** The pair set is the union of two sources:
   - **(a) Pareto dominance** on the raw objectives: ≤ on all three and < on at least one (tolerance 1e-9).
   - **(b) Tchebycheff score** g(x|w) = max_k w_k f̂_k with ideal point 0, used when g(worse) − g(better) > 0.01.

   Identical objective vectors never form a pair.
4. **Loss.** Over all pairs of the update:

   mean of −log σ(β[(ℓ_i − ℓ_i^ref) − (ℓ_j − ℓ_j^ref)]), with β = 0.1 and ℓ = Σ_t log π(a_t|s_t) over the full schedule.

   The reference is the starting network, frozen. Negatives come only from the policy's own samples.
5. **Gradient step.** One step, using PPO's Adam at learning rate 3e-4, with gradient-norm clipping at 1.0.
   - The gradient is exact and computed in two passes: trajectory log-probabilities first, then per-state
     log-probabilities in minibatches of 1,024, weighted by dL/dℓ.
   - The critic is unused and receives no gradient.
   - An update without any pair takes no step and is logged.

**Logged per update:**
- the loss and pair counts (total, dominance, Tchebycheff);
- the share of groups with at least one pair;
- distinct schedules (action sequences) and distinct objective vectors per group;
- policy entropy and the gradient norm before clipping;
- environment steps, both per update and cumulative;
- seconds per update and mean objectives.

Validation HV is logged at the usual checkpoints, including update 0.

**Stopping.** Each arm runs its fixed budget: `--converge_window 0`, status FINISHED, with no early stopping.

## Comparison and reporting (`results/contrastive_v3.md`, written by `tools/analyze_contrastive_v3.py`)

- **Per CL arm next to its PPO counterpart:**
  - HV at update 0, best HV, final HV;
  - the update and wall-clock time of the best checkpoint;
  - a plain statement of whether fine-tuning raised or lowered HV relative to update 0.
- **At equal updates and at equal environment steps.** A CL update uses K× PPO's environment steps on the same
  instances, so CL at update u is compared with PPO at update Ku.
- **Evaluation table:** HV, HV on the B2 subset, IGD+, unique non-dominated solutions, regret and best-of-16 HV.
- **Single seed:** numbers only, no significance claims. No hyperparameter is tuned, on any split.

## Run

```
bash scripts/ec2_run_all.sh --arms CL-B,CL-D,CL-E --seeds 111 --jobs 3
```

The run uses its own outputs: `results/contrastive_v3/`, `results/contrastive_v3.md` and
`logs/contrastive_v3_STATUS.txt`. It commits locally and never pushes. `--smoke` runs 5 updates per arm with the
development split and separate outputs.

## Known limitation

DANIEL's action mask admits only non-delay schedules (`results/diagnose_v3.md`), so part of every gap to CP-SAT is a
gap of the action space.
