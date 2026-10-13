# Benchmark v3 (`sd2-mo-v3`): SD2 multi-objective FJSP, frozen 2026-10-13

This is the frozen benchmark for baseline establishment and specialist convergence validation. Its specification
is `configs/problem_spec_v3.yaml` (status FROZEN). The validation behind it is in
`results/objective_validation.md`. v3 is opt-in: v1 stays the default, and v1 and v2 runs reproduce exactly as
before.

## Objectives

| Objective | Definition (raw units, 1 SD2 time unit = 1 minute) |
|---|---|
| Makespan | max_j C_j |
| Carbon | C = 0.540 × Σ over chosen (o, m) of P_m × p_om / 60 kg CO2e. Processing energy only, P_m ~ U(4, 15) kW per machine (CEA-FJSP, Wang et al. 2022). |
| Total tardiness | T = Σ_j max(0, C_j − d_j), with d_j = k_j W_j (TWK) and r_j = 0. W_j sums each operation's mean processing time over its eligible machines. k_j = k(1 + s(v_j − 0.5)) with v_j ~ U(0, 1). (k, s) = (1.3, 0.8) at 10x5 and (0.9, 0.2) at 20x10. |

- **Seeding.** P_m and v_j are drawn from `default_rng(int(sha256('<structural sha256>|power' or '|kdraw')[:16], 16))`. Each parameter is therefore a function of the instance content alone.
- **Carbon decision.** The carbon model is adopted as proposed. Its operation-level disagreement C1 is 14.4% at 10x5 and 16.5% at 20x10. That is below the 20% threshold of our own diagnostic, which is not a literature requirement. The result is documented and was not tuned.

## Normalised score and reward v3

For instance i in a given size, with preference weights w (goal g: 1/|g| on each objective named by g):

```text
score_i = w_m * MK_i / MKref_i  +  w_c * C_i / Cref_i  +  w_t * T_i / Tref
```

- **References.** MKref_i and Cref_i are the makespan and carbon of a fixed reference heuristic on instance i: MWKR job rule, ECT machine rule, active-style dispatching (`tools/objective_validation/schedules.py`).
- **Tardiness normaliser.** Tref is that heuristic's mean tardiness over the size's train pool: 557.42 at 10x5 and 1361.86 at 20x10. Tardiness is pooled because it can be 0 on a single instance.
- **No training needed.** None of the references depends on a trained policy.
- **Reward v3.** It uses the potential-based form of reward v2 (makespan and tardiness through completion-time lower bounds, carbon per step). The undiscounted episode return is exactly −score_i; the tests check this to 1e-9 for every goal.
- **Model selection.** Validation selects checkpoints on the mean `score_v3`.

## Observations (feature schema `benchmark_v3_f12_m9_p10`)

| Group | Features |
|---|---|
| Operation (12) | The 10 DANIEL features, plus job slack and time to due date. Priorities are retired, so there is no priority slot. |
| Machine (9) | The 8 DANIEL features, plus P_m / 15. Power is the only carbon input that is per machine rather than per pair. |
| Pair (10) | The 8 DANIEL features, plus carbon_om / 13.365 kg (the largest possible operation carbon), plus the lateness estimate. |

- **Fixed time scale.** Time is divided by a fixed 99 (the SD2 maximum processing time) instead of the batch maximum.
- **Batch-independent observations.** An instance's observations and rewards therefore do not depend on the other instances in its batch; this is tested. Whole splits can be evaluated in one batch.

## Data (`daniel/data/benchmark_v3/<split>/SD2/<size>+mix/`, plain SD2 format)

| Split | 10x5 | 20x10 | Source |
|---|---|---|---|
| train | 63 | 73 | `data_train/SD2/<size>+mix` (byte copies of the canonical grouped split) |
| validation | 18 | 13 | `data_validation/SD2/<size>+mix` |
| development | 19 | 14 | `data_development/SD2/<size>+mix` |
| evaluation | 100 | 100 | `data_train_vali/SD2/<size>+carbon+priority`, with the carbon and priority columns stripped |

- **The evaluation pool** is the pool every earlier evaluation used. Its 10x5 HELD-OUT subset of 49 instances is recorded in the manifest as `evaluation/10x5/heldout_49`.
- **The reserved final-test sets were not read** and are not part of v3.
- **Disjointness.** The builder checks every copy against its source, and checks that the splits are disjoint (exact and structural hashes) and disjoint from the objective-validation DEV instances.
- **The manifest** `daniel/data/benchmark_v3/manifest_benchmark_v3.json` has SHA-256 `6a8d6dab143156a9c737479e92c033c40b163a4ea42dd08e72182c3d201c33d0`.
  - Per instance it stores: file and structural SHA-256, source, P_m, v_j, k_j, W_j, d_j and the reference objectives.
  - It also stores: Tref per size, definitions, and the SHA-256 of the generating code.
- **Loading.** The loader (`benchmark_v3.load_directory`) checks every file hash. Carbon is computed from the frozen P_m by the definition above. Due dates and references are read from the manifest, never re-derived.

## Commands

```bash
# build (already done; refuses to overwrite)
python tools/benchmark_v3/build.py

# train a specialist (goal m, c or t; the configs set obs/reward v3 and the dimensions)
cd daniel
python train.py --config ../configs/benchmark_v3/m.json --n_j 10 --n_m 5 \
  --model_suffix bv3_m_s1 --seed_train 1
#   default data: data/benchmark_v3/{train,validation}/SD2/10x5+mix; 20x10: --n_j 20 --n_m 10
#   the checkpoint suffix must contain "bv3"; run.json records the manifest hash

# evaluate (greedy, whole split in one batch, every schedule re-scored independently)
python evaluate_v3.py --models 10x5+mix+bv3_m_s1 --split validation --size 10x5
python evaluate_v3.py --models ... --split evaluation --size 10x5 --subset heldout_49
```

## Validation (2026-10-13)

- **`tests/test_benchmark_v3.py`: 22 tests, all pass.** They check:
  - every manifest parameter, reference value and Tref regenerates exactly from the files and the definitions;
  - parameter ranges, split disjointness, the absence of final-test data, and the spec recording this manifest's hash;
  - that the loader rejects hash mismatches and missing instances;
  - the carbon and TWK formulas on hand-worked cases, and the independent scorer rejecting infeasible schedules;
  - for random-policy rollouts at both sizes and all seven goals, that the environment's makespan, carbon and tardiness equal the independent scorer and the return equals −score;
  - the observation dimensions, the carbon pair feature and the power machine feature;
  - batch invariance, reset reproducibility, the guards, model forward and the schema and configs.
- **The existing tests are unchanged and pass:**
  - `test_contracts`, `test_data_manifest`, `test_environment_semantics` and `test_objectives`;
  - the three audit tests;
  - `test_v1_reproduction`, where C1@1000 reproduces the Step 4 files exactly;
  - `tools/v2_verify.py` (run into a scratch directory), whose output is identical to the committed `results/v2_fixes/checkpoint_verification.json`.
- **Pipeline smoke test.** I ran `train.py` for 4 PPO updates at each size and `evaluate_v3.py` on the development split, both in a scratch directory with nothing kept. This confirms that loading, rollouts, validation selection on `score_v3`, checkpoint provenance and independent re-scoring work end to end. It is not training.

## Limitations

- **Carbon tracks workload.** Its operation-level conflict with makespan is weaker than our diagnostic asked for (C1 15–17%); carbon correlates strongly with total processing workload (Spearman 0.91). The schedule-level makespan–carbon trade-off is large.
- **Due dates are loose for strong schedulers.** They were calibrated on dispatching-rule schedules, but CP-SAT reaches zero tardiness on 8 of 10 DEV 10x5 instances.
- **Tardiness behaves like total completion time** across rule-based schedules (Spearman 0.99–1.00).
- **The score's anchor is a rule.** A score of 1 means "as good as MWKR/ECT on each objective"; it is not a bound.
- **Final test is not frozen yet.** The reserved final-test sets have no v3 parameters. When final evaluation is authorised they must be converted and parameterised with the same code, written to a separate manifest so that this one stays frozen.
- **Legacy tools stay on v1/v2.** `test_trained_model.py`, `tools/ceiling_scorer.py`, the v2 tools and SD1 / the various-operations environment are unchanged and do not support v3.
- **Inherited quirks, not changed:**
  - `data_manifest.structural_fjsp_sha256` tries the plain layout first, so an annotated row can be misread as plain. It happens for `data_train_vali/SD2/10x5+carbon+priority/10x5+mix_050.fjs`: the "99/100 overlap" between the train_vali `+mix` and `+carbon+priority` pools is an artefact, and the pools are structurally identical. v3 uses plain files only and compares matrices when converting.
  - `reset()` restores the state object but not the attribute `env.dynamic_pair_mask`; the policy only reads the state.
  - The `priority` validation metric is reported but is meaningless under v3 (all weights 1).
