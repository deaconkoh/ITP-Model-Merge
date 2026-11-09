# World-model trunk, Milestone 1 (10x5): protocol, fixed before the run

## Question

Does action-conditioned predictive pretraining teach DANIEL's encoder (the "trunk") factory knowledge that transfers to an unseen task?

This is a feasibility study, not model selection:
- no specialists, merging or RL;
- 10x5 only;
- a single pretraining seed, so exploratory evidence only.

All settings are frozen in `configs/world_model_m1_v3.json` (`tools/freeze_world_model_m1_v3.py`).

## Data (`tools/world_model_m1_v3.py generate`)

**Pretraining instances:** the SD2 generator stream that every specialist trained on (`benchmark_v3_generator.GeneratorSource`, seed 111).
- Any instance that matches a validation, development or evaluation instance, or the reserved final-test stream, is discarded.
- The final-test stream is regenerated in memory; no file is opened.

**States:** each instance is run through the environment under one behaviour policy, drawn at random from:
- MWKR+ECT;
- MWKR+MINE (least energy);
- EDD+ECT;
- uniform random;
- random 30% / MWKR+ECT 70%.

At 6 random decision steps per episode, the environment's input tensors are stored.

**Actions:** the behaviour's choice plus up to 3 other allowed actions.

**Continuations:** each (state, action) is finished by every continuation rule c in {MWKR+ECT, MWKR+MINE, EDD+ECT}.
- All schedules follow DANIEL's own no-waiting rule.
- They use the tested copy of the environment's transition (`tardiness_diagnosis_v3.NonDelay`).

**Targets, exact:**
- each job's completion time;
- each machine's busy time and energy (kg CO2e).

**Normalisation, by fixed constants:**
- times / 990 (99 × N/M);
- energy / 133.65 (13.365 kg × N/M).

**Size: 100,000 instances,** about 600,000 states and 7.2 million (state, action, c) samples.
- An hour of generation, about 440,000 instances or 10 GB, would not fit in the PC's 11 GB of RAM for pretraining.
- About 15 minutes of generation therefore gives the largest dataset that does.
- **Measured peak memory,** from real runs at 10,000 and 20,000 instances: generation about 0.8 GB (one shard at a time); pretraining about 5.1 GB at 100,000 instances (2.6 GB of data, the normalised targets, and PyTorch/CUDA). The pre-flight requires 6 GB available.

**Disjointness by factory.** Each training state records its instance's structural SHA-256. Pretraining stops if any factory appears in both the training and hold-out parts, or matches a validation instance. Probing stops unless the 18 validation instances are 18 distinct factories and the probe fitting, early-stopping and test instances are disjoint.

**Check data:** all 18 v3 validation instances, under all five behaviours, at every decision step. Each continuation also includes the unseen least-slack + ECT rule. The evaluation split and the final-test sets are never used.

**Features:** the current 12 operation features. The `feature_set` setting can select the lateness features (17) instead; the comparison trunks need the current 12.

## Model and pretraining (`daniel/world_model.py`)

**Trunk:** DANIEL's dual-attention encoder, with the same architecture and size as the specialists. It sees the state only.

**Action readout:** the action is read out of the trunk's embeddings as DANIEL's actor reads it:

a = [h_op(chosen job's candidate operation), h_mch(chosen machine), mean operation embedding, mean machine embedding], 32 values.

**Heads.** c enters only the heads, as a one-hot of 3:
- **Job head:** [h_op(job j's candidate operation), a, c] → job j's completion time.
- **Machine head:** [h_mch(m), a, c] → machine m's busy time and energy.

**Training:**
- Huber loss (δ = 1) on the normalised targets;
- Adam 1e-3, 256 states per batch, seed 111;
- 10% of instances held out by instance; early stop after 5 epochs without improvement on that hold-out; cap 100 epochs.

**Saved** (`checkpoints/world_model_m1_v3/`, SHA-256 in `SHA256SUMS.json`):
- the start trunk;
- the best trunk, as a full DANIEL bundle that loads with `--init_mode exact` or `backbone`;
- the heads.

## Question A: prediction sanity

On the validation data, under the three pretraining continuations, MAE per outcome group (job completion, machine busy time, machine energy) for:
- **World model.**
- **Mean predictor:** the training mean per output and continuation.
- **Shortcut:**
  - *Jobs:* DANIEL's earliest-possible-completion estimate after the action.
  - *Machines:* the committed amount plus every unscheduled operation on its fastest or lowest-energy machine.

The world model **loses** a group if it is not better than both baselines.

## Question B: frozen-trunk transfer

**Split:** validation instances by index; even indices are probe-train (the last fifth of them for early stopping), odd indices are probe-test.

**Trunks, frozen:**
- the world-model trunk;
- random initialisation (seed 111, which is also the world model's own start);
- the equal-weight shared base (`checkpoints/merge_categories_v3/10x5+mix+bv3_mcbase_s111.pth`);
- the makespan specialist (`checkpoints/specialists_v3/10x5+mix+bv3_spec_m_s111.pth`), context only.

**Probes:**
- input: the standardised 32-value embedding *a*;
- a linear probe and an MLP probe (one hidden layer of 64 units);
- 5 seeds each; MSE training with early stopping.

**Targets, under least slack + ECT:**
- final makespan / 990;
- carbon / 668.25;
- tardiness / Tref.

**Reported:** test MAE per objective, mean ± std over the seeds, and pairwise accuracy (same-state candidate pairs ordered correctly).

## Outcome rule (MLP probe, mean over seeds)

- **FAIL:** the world model loses 2 or more groups in Question A, OR its probe MAE is not lower than the random trunk's on any objective.
- **PASS:** it loses at most 1 group, AND its probe MAE is at least 5% lower than the shared base's on at least 2 of 3 objectives, with none more than 5% worse.
- **INCONCLUSIVE:** anything else.

The makespan-specialist trunk does not enter the rule.

## Run

`bash scripts/run_world_model_m1_v3.sh` writes `results/world_model_m1_v3.md`.
- It commits locally and never pushes.
- It can be resumed by re-running it.
- `--smoke` is for testing only.
