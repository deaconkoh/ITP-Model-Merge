# Specialist reuse vs from scratch, waiting allowed (benchmark v3, 10x5, seed 111): protocol

Fixed before any run. The frozen settings are in `configs/specialist_reuse_v3.json`
(`tools/freeze_specialist_reuse_v3.py`). Experiment name: `specialist_reuse_v3`; runs are tagged `bv3_sr_..._s111`.

## Question

Does reusing single-objective specialists improve multi-objective DANIEL over training from scratch? Four methods are
compared, all trained fresh:
- From scratch;
- preference-conditioned Distillation;
- Rewarded Soups;
- preference-weighted TIES-Merging.

## Shared setup

- **Waiting option.** `--dispatch_rule semi_active`, from the waiting experiment (`docs/waiting_training_protocol_v3.md`).
  - The environment drops the no-delay mask line.
  - Every eligible pair of each job's next operation is legal, and it starts at max(job ready, machine free).
  - Applied in:
    - specialist training and its validation (`train.py`);
    - Distillation stage 1: teacher rollouts and the held-out check (`distill_v3.py --dispatch_rule`);
    - all multi-objective training and its validation, and the TIES screen (`train.py`);
    - evaluation and timing (environments built by `tools/decoding_v3.py`).
- **Common to all methods:** the architecture, features, objectives, generator training data, validation and evaluation
  splits, preference vectors and frozen normalisations. The reserved final-test sets are never loaded.
- **Recipe:** `configs/contrastive_v3.json`.
- **Stopping rule:** stop when the best validation value improves by less than 0.5% over 1,000 updates; minimum
  1,000 updates, cap 8,000. A run that reaches the cap is reported as **capped**.
- **One validation decoder:** greedy on all 18 validation instances x 15 preferences (grid 0.25), with frozen norm
  `results/warmstart_v3/norm/validation_normalisation_points_10x5.json`, every 50 updates. It is used for tuning,
  checkpoint selection (the best checkpoint), stopping and time to target.

## Specialists

- **Shared base:** one random unconditioned DANIEL after `setup_seed(111)`.
- **Training:** makespan, carbon and tardiness specialists, each started from a copy of the base. Single-objective
  contrastive training with the recipe and the stopping rule.
- **Selection:** each uses its own validation objective (the benchmark score of its goal) for stopping and selection.
- **Reuse:** the three best checkpoints are reused by Distillation, Rewarded Soups and TIES.

## Methods (multi-objective contrastive, 3-dimensional preference input)

1. **From scratch:** a fresh preference-conditioned DANIEL, `setup_seed(111)`.
2. **Distillation:**
   - Stage 1: `distill_v3.py` (400 instances per teacher, 15 epochs, batch 256, lr 3e-4, 20-minute cap).
   - Stage 2: contrastive fine-tuning from the student (`--init_mode no_critic`).
   - Both stages count as its training time.
3. **Rewarded Soups:** `--merge_mix m,c,t`. For preference w the network is w_M θ_M + w_C θ_C + w_T θ_T.
   - The blend is formed per sample inside every forward pass, so it is kept during fine-tuning and generated before
     decoding.
   - All three specialists are trained.
4. **Preference-weighted TIES:** `--merge_mix m,c,t --merge_rule ties --merge_base base`, rule
   `daniel/mixture.py:pref_ties_flat`:
   - τ_x = θ_x − θ_base, over all tensors. Preference-input columns are zero in the base and the specialists.
   - **Trim:** keep the top `keep` fraction of each τ_x by |value| over the whole flattened vector. Masks are
     recomputed from the current weights in every forward pass, with no gradient through them.
   - **Weight before sign election:** v_x = w_x · trim(τ_x).
   - **Elect:** s = sign(Σ_x v_x) per entry.
   - **Merge:** the sum (not the mean) of the agreeing entries, merged = Σ_x v_x · [v_x ≠ 0 and sign(v_x) = s]; then
     θ(w) = θ_base + λ · merged.
   - **Zero weight:** v_x = 0, so that vector neither votes nor contributes. An entry where all v_x are zero stays at
     the base.
   - **Exact corner** w = e_x: θ(w) = θ_x itself, so trimming and λ cannot change a specialist at its own corner.
   - The base is frozen. The three specialists are trained, through the kept, agreeing entries.

**Checks before any multi-objective stage**, on the real specialists, for Rewarded Soups and for every screened TIES
setting:
- two interior preferences, (0.6, 0.2, 0.2) and (0.2, 0.2, 0.6), give different merged weights;
- each exact corner gives the specialist's weights, and the same action probabilities as the original specialist on
  validation states visited under the waiting rule (max difference < 1e-5).

The run stops if any check fails.

## Tuning: the same budget for every method

- **Trials:** 6 per method, each 1,000 updates with no stopping rule, over lr ∈ {1e-4, 3e-4, 1e-3} x
  β ∈ {0.1, 0.5}.
- **Score:** the best validation HV in the trial. Ties go to the setting closest to the recipe (lr 3e-4, β 0.1).
- **TIES, first:**
  - a screen of keep ∈ {0.2, 0.5, 0.8} x λ ∈ {0.8, 1.0, 1.5}, scored by validation HV of the merged start (update 0,
    no training);
  - the best setting is fixed, with ties going to λ closest to 1, then keep closest to 0.2;
  - then its 6 trials run.
- Distillation stage 1 runs once; the trials tune stage 2.
- **Full run:** the selected setting is trained fresh with the stopping rule.
- **Reported:** every setting tried, the selected settings and the tuning cost. The TIES screen counts in TIES's
  tuning cost.

## Evaluation

- **Instances:** 100 evaluation instances x 66 preferences. Equal weights are also decoded, so the batch layout matches
  `evaluate_mo_v3`; they are not in the metrics.
- **Decoders:**
  - greedy;
  - best-of-128 sampling: seed 51, one instance per batch, generator seeded once, picked by the preference-weighted
    benchmark score.
- **Saved:** one schedule per preference per decoder.
- **Metrics:**
  - HV with `results/baselines_v3/normalisation_points.json`, reference point 1.1, mean over instances;
  - ΔHV vs From scratch, per decoder;
  - IGD+ on the 20 instances with a CP-SAT B2 front.
- **Time to target:** the target is From scratch's best validation HV. Report the first update at or above it, and the
  cumulative training seconds at that point, or "not reached".
- **Costs:**
  - specialist training (the sum of the three, and the wall-clock of the parallel batch);
  - additional training (what a run costs when the specialists already exist);
  - total training;
  - inference seconds per instance-preference pair (GPU alone, same batching for every method);
  - tuning cost, separately.
- **Concurrency:** runs share 3 slots, which inflates wall-clock times. The slot layout is recorded.

Single seed: numbers only, no significance claims.
