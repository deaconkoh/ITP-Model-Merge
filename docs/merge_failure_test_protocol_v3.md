# Why merged starts end below random initialisation: merge-failure test on benchmark v3 (protocol, fixed before any run)

## Question

The merged starts begin contrastive fine-tuning with a higher validation HV than a random initialisation. They still end well below it: at convergence, Model Soup (shared base) reaches an evaluation HV of 0.313, against 0.600 for From scratch (`results/contrastive_converged_v3.md`).

This test asks why, using the Model Soup start as the test case.
- **Setting:** 10x5, seed 111.
- **Contrastive settings:** unchanged unless stated (`configs/contrastive_v3.json`: K 8, β 0.1, margin 0.01, gradient clip 1.0, learning rate 3e-4, 20 environments).
- **Stopping rule:** the converged runs' rule. Stop once the best validation HV has improved by less than 0.5% over the last 1,000 updates, after at least 1,000 updates. The hard cap is 8,000 updates, reported as "capped".

All inputs are frozen in `configs/merge_failure_test_v3.json` (`tools/freeze_merge_failure_test_v3.py`) and verified before every run.

## Part A: measurements at update 0 (no training)

**Networks measured:**
- the Model Soup start;
- the Distillation start (the distilled student, `--init_mode no_critic`);
- a random initialisation (seed 111, the From-scratch start);
- the starting networks of the three Part B arms.

**How each network is built:** by `train.py`'s own code with the flags of the run it starts (`daniel/merge_diagnostics.py`), so the measured network is exactly the update-0 network. Each is saved in `checkpoints/merge_failure_test_v3/starts/`.

**Data:** the validation split, every instance under each of the 15 validation preferences. The reserved final-test sets are never loaded.

**Measurements:**
- **Policy entropy and distinct schedules:** K = 8 schedules sampled per (instance, preference). Reported: the mean per-step policy entropy, and the number of distinct schedules out of 8. Both are averaged over the instances and preferences.
- **Near-inactive units per layer:** the validation batch is every state of the greedy validation rollouts, counting unfinished schedules only. A unit is near-inactive when |activation| ≤ 0.01 on at least 99% of the batch's entries.
  - Layers: the attention-block outputs (ELU) and the actor's hidden layers (tanh).
  - For tanh layers, the share of saturated units is also reported: |activation| ≥ 0.99 on at least 99% of the entries.
- **Weight norm per layer:** the Frobenius norm of all the layer's parameters. The norm of the preference-input columns is also reported.
- **Gradient norm of the contrastive loss on one batch:** the run's first contrastive batch, using the same instances and preferences for every network, with schedules sampled from the network. Reported before clipping, both total and per layer, together with the loss and the number of pairs.
- **Effect of the preference input on the actions:** on every state of the greedy validation rollouts, the policy is evaluated under each of the 15 preferences. Reported:
  - the share of states whose greedy action changes with the preference;
  - the mean total-variation distance between the action distributions of two preferences;
  - the number of distinct greedy schedules per instance, out of 15.

## Part B: three arms from the Model Soup start, each changing one thing

**What every arm shares:** each arm is the converged Model Soup run with one change. That means the same:
- start network, seed and contrastive settings;
- instance and preference streams;
- validation and checkpoint selection;
- stopping rule.

None of the changes draws random numbers, so every random stream is the Model Soup run's.

| Arm | Change | Flags |
|---|---|---|
| Shrink and perturb | Before fine-tuning, every weight becomes 0.3 × weight + 0.0001 × the same weight of the freshly initialised seed-111 network. The contrastive reference is this new start. | `--init_shrink 0.3 --init_perturb 0.0001` |
| No anchor | The same pairwise loss with the reference log-probabilities removed: −log σ(β (ℓ_better − ℓ_worse)) | `--cl_anchor none` |
| Random preference input | The preference-input columns of the first actor and critic layers keep the fresh seed-111 values, as in From scratch, instead of being set to zero. Every other weight comes from the merge. | `--init_pref_columns fresh` |

**Shrink and perturb values:** these come from Ash & Adams, "On Warm-Starting Neural Network Training", NeurIPS 2020, Section 4.
- **The procedure:** θ ← λθ + p, with p ~ N(0, σ²). The paper says: "in practice, we add noise by adding parameters from a scaled, randomly-initialized network". So the noise here is the scaled fresh network.
- **The values:** λ = 0.3 and noise scale 0.0001. These are the paper's values for the pre-training-then-fine-tuning case (Section 4.2, Figure 9), which is the case closest to fine-tuning a merged network.
- **Scope:** every parameter is shrunk, including the critic, which the contrastive objective never uses.

**The new options:** they default to the earlier behaviour, and the defaults leave runs and run identities bit-for-bit unchanged (tested).

**Checkpoints:** each arm's start network (`starts/`) and best-validation checkpoint are committed to `checkpoints/merge_failure_test_v3/`, with SHA-256 hashes in `SHA256SUMS.json`. Each run's contrastive-reference hash is checked against its saved start network.

## Report (`results/merge_failure_test_v3.md`, `tools/analyze_merge_failure_test_v3.py`)

**Part A:** the tables above, one row per network.

**Part B:** for each new arm, plus the existing converged Model Soup and From scratch runs (read-only):
- HV at update 0;
- the update of the best checkpoint, the best validation HV and the stop update (converged or capped);
- evaluation HV (100 instances, 66 preferences, greedy);
- IGD+ (CP-SAT B2, the same 20 instances as before);
- unique non-dominated count;
- wall-clock to the best checkpoint and to the stop.

All runs are on the same machine with `--jobs 3`. From scratch's wall-clock includes its fixed-budget part.

**Closing the gap, fixed before the run:**
- An arm **closes the gap** to From scratch if its evaluation HV is at least From scratch's converged evaluation HV.
- Otherwise, the report gives the share of the gap closed: (arm − Model Soup) / (From scratch − Model Soup), on evaluation HV.

**Single seed:** numbers only, no significance claims.

## Run

```
WS_V1_STRICT=1 bash scripts/ec2_run_all.sh --arms MF-SP,MF-NA,MF-PR --seeds 111 --jobs 3
```

- Runs locally and commits locally, never pushing.
- `--smoke` runs 5 updates per arm and 3 validation instances on the development split, with separate outputs.
