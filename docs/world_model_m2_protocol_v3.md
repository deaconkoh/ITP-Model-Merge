# World-model M2: objective-specific specialisation from the world-model trunk (10x5), protocol fixed before the run

## Question

Does the M1 world-model trunk (`checkpoints/world_model_m1_v3/world_model_trunk.pth`) make objective-specific DANIEL learning faster or better than a matched random start, when the specialisation method is held fixed?
- **Research direction:** objective-agnostic world model → branch → independently specialise → merge.
- **M2 covers** branching and specialisation only. There is no merging and no change to the world model.

All settings are frozen in `configs/world_model_m2_v3.json` (`tools/freeze_world_model_m2_v3.py`).

## Arms and starting networks (`tools/world_model_m2_bases.py`)

For seed s:
- **Random start (control):** DANIEL (single-objective, schema `benchmark_v3_f12_m9_p10`) after `setup_seed(s)`.
- **World-model start:** the same network with its trunk (`feature_exact`) replaced by the M1 trunk. The actor and critic are identical to the random start.

Both arms start every objective from copies of their arm's file, loaded with `--init_mode exact`. For seed 111, the two bases are exactly M1's start and trunk bundles (tested).

## Specialisation: single-objective contrastive (the existing implementation, one new mode)

- **Groups:** per training instance, K = 8 schedules are sampled from the current policy.
- **Pairs:** (i, j) is a pair when f_j − f_i > 0.01, where f is the specialist's own objective normalised per instance (MK/MKref, C/Cref or T/Tref). There is no preference input and no dominance rule.
- **Loss:** the DPO-style loss against the frozen start network, as before.
- **Unchanged constants:**
  - β 0.1, gradient clip 1.0;
  - learning rate 3e-4 for every parameter, so the trunk is trainable in both arms;
  - minibatch 1024;
  - 20 instances per update, resampled every 20 updates from the SD2 generator stream, the same stream as the committed PPO specialists.
- **The multi-objective contrastive path is unchanged:** a 2-update run gives weights identical to the pre-change commit, and a test pins them.
- **Validation:** every 50 updates and at update 0, greedy, on the 18 v3 validation instances. The value is the specialist's mean normalised objective (lower is better), the value the PPO specialists were selected on. Each validation also logs the trunk's and the actor's relative distance from their starting weights.
- **Stopping rule:** stop once the best validation value has improved by less than 0.5% over the last 1,000 updates, after at least 1,000 updates. The hard cap is 8,000 updates, reported as "capped".

## Evaluation

The best checkpoint of each run is evaluated on the evaluation split (100 instances):
- greedy (primary), and best of 100 samples (sample seed 50), exactly as for the committed specialists;
- all three raw objectives and the share of jobs late are recorded.

**References:**
- the committed PPO specialists, seed 111 and the 3-seed mean;
- CP-SAT B1 single-objective;
- the MWKR+ECT reference rule.

## Pre-registered criteria (validation for training curves, evaluation for final values)

- **Faster:** the world-model specialist first reaches the random start's best validation value within 70% of the random start's updates to its best, on at least 2 of 3 objectives. "Not reached" counts as no.
- **Better:** the world-model specialist's final evaluation value (greedy, own objective) is at least 5% lower than the random start's, on at least 2 of 3 objectives.

Each criterion is also reported per objective, which answers whether any benefit depends on the objective. With more than one seed, the criteria use seed means and the report shows mean ± std.

## Outputs

- **Report:** `results/world_model_m2_v3.md`. It covers:
  - training curves against updates and against hours, and trunk drift;
  - updates and hours to reach the other arm's best and the PPO specialist's best;
  - the evaluation tables;
  - training costs, with the world-model pretraining cost (M1: 0.22 h data + 1.05 h GPU) reported separately.
- **Checkpoints:** `checkpoints/world_model_m2_v3/` with `SHA256SUMS.json`. They include the bases and the start (update 0) and best checkpoints of all six specialists, kept for later merging.

## Run

`bash scripts/run_world_model_m2_v3.sh` runs seed 111 (6 specialists, 3 in parallel).
- `--seeds 111,222,333` adds the matched seeds; finished runs are kept.
- It commits locally and never pushes.

Single seed: numbers only, no significance claims.
