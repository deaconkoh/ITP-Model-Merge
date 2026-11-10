# World-model M3: preference-weighted merging (10x5, seed 111), protocol fixed before the run

## Question

**H2:** do specialists trained from the world-model base merge better than specialists trained from a random shared base?
- The merge is applied identically to four specialist sets. The two earlier sets are references.
- Settings are frozen in `configs/world_model_m3_v3.json` (`tools/freeze_world_model_m3_v3.py`).

## Merge (identical for every set)

For preference w = (w_M, w_C, w_T), the policy is the network with weights θ(w) = w_M·θ_M + w_C·θ_C + w_T·θ_T. Each preference gets its own mix; in training, each sample does.
- **Trunk and actor:** mixed per preference. In Part B they are trainable, and each θ_x receives w_x·∂L/∂θ(w).
- **Critic:** mixed the same way and kept in the checkpoints. The contrastive loss never reads it, so it receives no gradient.
- **Preference columns:**
  - The specialists have no preference input. Each is copied into a preference-conditioned DANIEL with `warmstart.transfer(..., "full")`, which sets its three w-input columns (first actor and critic layers) to zero.
  - The mixed columns therefore start at zero, and the update-0 network is the plain weight mix for every w.
  - The columns are trainable in Part B. w is fed both as the mixing weights and as the preference input.

## Specialist sets (seed 111; makespan, carbon and tardiness specialists each)

| Set | Specialists | Parts |
|---|---|---|
| World-model base | `checkpoints/world_model_m2_v3/10x5+mix+bv3_wm2_wm_{m,c,t}_s111.pth` | A, B |
| Random shared base | `checkpoints/world_model_m2_v3/10x5+mix+bv3_wm2_ctrl_{m,c,t}_s111.pth` | A, B |
| Original specialists | `checkpoints/specialists_v3/10x5+mix+bv3_spec_{m,c,t}_s111.pth` (independent random initialisations) | A, B |
| Old shared-base specialists | `checkpoints/merge_categories_v3/10x5+mix+bv3_mcft_{m,c,t}_s111.pth` (PPO fine-tunes of one equal-weight PPO base) | A |

## Part A: no training (`tools/world_model_m3_part_a.py`)

**Rollouts:** greedy rollouts of θ(w) for the 66-point grid (step 0.1) plus equal weights. The code path is arm E0's (`tools/e0_merge_front_v3.py`), so the Original specialists must reproduce the committed E0 seed-111 front bit for bit.
- That front's HV is 0.5705. The often-quoted 0.555 is E0's mean over seeds 111, 222 and 333.

**Fronts:**
- **Evaluation split** (100 instances): HV (frozen evaluation normalisation), IGD+ on the CP-SAT B2 subset (20 instances), unique non-dominated count.
- **Validation split** (18 instances): the same, using the frozen validation normalisation. There are no B2 fronts on validation, so there is no IGD+.
- **Training validation grid:** HV on the 15-point training validation grid (step 0.25), which is comparable with the training validation HVs.

**Corner vs in-between:**
- **Corner:** any weight ≥ 0.8 (18 of 66 preferences). **In-between:** the other 48.
- **Mean regret:** λ·f − min over a fixed pool of λ·z, in normalised space.
  - **The pool, per split and instance:**
    - the `analyze_warmstart_v3.reference_points` references (specialists of all seeds, rules, CP-SAT B1);
    - the B2 front where it exists;
    - the converged From-scratch and Distillation fronts (evaluation split only);
    - the 66-point fronts of all four sets.
  - Part B arms are scored against the same pool plus their own points.
- **Share non-dominated:** for each preference, the share of instances whose schedule is not strictly dominated within the set's own 66-point front.

**Ridge test (validation):**
- **Path:** θ_t = t·θ_a + (1−t)·θ_b for t = 0, 0.1, …, 1, for each pair M-C, M-T and C-T. These are the simplex-edge grid points.
- **Score:** s(t) = t·f_a + (1−t)·f_b, with f the normalised objective averaged over instances.
- **Barrier:** max_t [s(t) − (t·s(1) + (1−t)·s(0))].
- Each objective's own dip is also reported.

## Part B: merge fine-tuning (sets 1–3; `train.py --merge_mix m,c,t`)

**Network:** three trainable specialist networks (`daniel/mixture.py`, `MixturePolicy`). For each sample with preference w, the policy is the mix for that w.
- **Implementation:** the per-sample mix is computed exactly by materialising per-row parameters. The layers in `daniel/model/` accept them; with ordinary parameters they compute exactly as before (pinned-weight tests).
- **Speed:** a grouped implementation (one forward per distinct w) is kept as the test reference. It was about 20× slower, at about 96 ms against 10 ms per rollout step.

**Training:**
- **Unchanged multi-objective contrastive settings** (`configs/contrastive_v3.json`):
  - Dirichlet(1,1,1) preferences, one per environment, from the same preference stream;
  - K 8, β 0.1, margin 0.01, lr 3e-4 (one Adam over all three networks), gradient clip 1.0 on the joint norm;
  - minibatch 1024, 20 instances resampled every 20 updates (generator stream, seed 111);
  - validation every 50 updates on 15 preferences.
- **Reference:** the frozen starting mix for the same w.
- **Stopping rule:** stop once the best validation HV has improved by less than 0.5% over the last 1,000 updates, after at least 1,000 updates. The cap is 8,000, reported as "capped". Training runs in chunks of 500 and is resumable.

**Saved and evaluated:**
- Start (update 0) and best bundles go to `checkpoints/world_model_m3_v3/` with `SHA256SUMS.json`.
- The best bundle is evaluated with `evaluate_mo_v3.py` on the evaluation split (66 preferences, 100 instances).

## Pre-registered criteria (World-model base vs Random shared base)

- **Mergeable:** before fine-tuning, both (a) and (b) must hold:
  - (a) in-between mean regret on the evaluation split is ≤ 0.90 × the Random shared base's;
  - (b) a smaller barrier on ≥ 2 of 3 pairs.
- **Better after merging:** the fine-tuned World-model base mix has evaluation HV ≥ 1.05 × the fine-tuned Random shared base mix's.
- **Faster after merging:** the first update at which validation HV ≥ 0.542 (From scratch's best) is ≤ 70% of the Random shared base mix's.
  - If the world-model mix never reaches 0.542, the answer is "no".
  - If only the control never reaches it, the answer is "yes (control not reached)".

## Report and cost

`results/world_model_m3_v3.md` (`tools/analyze_world_model_m3_v3.py`) contains:
- Part A tables for all four sets;
- Part B next to the converged From scratch (0.600) and Distillation (0.609) runs: HV at update 0, best validation HV and its update, stop update, evaluation HV, IGD+, unique non-dominated count, mean regret;
- costs three ways: merge fine-tuning; plus specialist training; plus world-model pretraining (1.27 h, world-model set only). Wall-clock is also given for specialists trained in parallel;
- a verdict per criterion, and `criteria.json`.

Single seed: numbers only.

## Run

```
bash scripts/run_world_model_m3_v3.sh --part-a     # Part A (no training)
bash scripts/run_world_model_m3_v3.sh --jobs 3     # Part B (training)
```

Local commits only; never pushes. The reserved final-test sets are never read.
