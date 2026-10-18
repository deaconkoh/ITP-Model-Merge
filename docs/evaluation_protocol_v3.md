# Evaluation protocol for the benchmark v3 baselines (PRE-REGISTERED)

**Status: PRE-REGISTERED on 2026-10-13, before any baseline run.** This document is never edited after that.
Any later change is recorded, with its date and reason, in **Deviations** at the end.

- **Scope.** Benchmark `sd2-mo-v3` (`configs/problem_spec_v3.yaml`, `docs/benchmark_v3.md`).
- **Part A:** single-objective DANIEL specialists, trained to convergence.
- **Part B:** CP-SAT reference bounds.
- **Runner:** `scripts/run_baselines_v3.sh`. Analysis: `tools/analyze_baselines_v3.py`, which writes `results/baselines_v3.md`.
- **Never changed:** the frozen spec, the manifest, and the reward and score definitions.
- **Never read:** the reserved final-test sets.

## 1. Training-data decision

- **Finding.** Under the earlier v3 code, training reused the fixed v3 train pool. `daniel/train.py:245-248` (before this change) drew `random.randrange(len(self.v3_train))` indices every `--reset_env_timestep` updates: 63 instances at 10x5, 73 at 20x10.
- **Original DANIEL.** The original DANIEL (github.com/wrqccc/FJSP-DRL, the `train.py` training loop; diffed in `results/code_audit.md`, finding A3) instead draws a fresh batch from the SD2 generator every `reset_env_timestep = 20` updates. The batch has `num_envs = 20` instances (`daniel/params.py`: `--num_envs`, `--reset_env_timestep`; defaults unchanged from DANIEL).
- **Decision.** The specialists use DANIEL's scheme: `train.py --v3_train_source generator` (`daniel/benchmark_v3_generator.py`).
  - Fresh SD2 instances come from `data_utils.SD2_instance_generator` on the global numpy RNG seeded by `--seed_train`: 20 instances every 20 updates.
  - Their v3 parameters come from the same frozen code as the manifest:
    - P_m and v_j are keyed on the structural hash. The matrix-based hash is tested to equal the file-based one on all 400 manifest instances.
    - Due dates and carbon follow the `benchmark_v3` definitions.
    - The MWKR/ECT reference schedule comes from the manifest builder, and Tref is the size's frozen value.
- **Disjointness.** Every fresh instance's structural hash is checked against:
  - the v3 validation, development and evaluation pools (hashes recorded in the manifest);
  - the reserved final-test pool, regenerated in memory from its generator seed (`seed_final_test_datagen` = 400, 100 instances).

  A colliding instance is discarded and redrawn. The count is recorded in every run's status file.
- **Limitation of the final-test check.** The final-test files are not opened, as the standing rule requires. That check is therefore against the generator stream that produced them, not against the files.
- **Default unchanged.** `--v3_train_source pool` remains the default, so earlier behaviour is unchanged.

## 2. Part A: specialists

- **Models.** DANIEL-M, DANIEL-C and DANIEL-T, trained from random initialisation on 10x5 and 20x10.
  - Configs: `configs/benchmark_v3/{m,c,t}.json`, i.e. obs/reward v3 with goal m, c or t.
  - The reward is the v3 normalised score for that single objective: MK/MKref, C/Cref or T/Tref.
- **Fixed hyperparameters.** DANIEL's defaults, not tuned: 20 environments, lr 3e-4, 4 PPO epochs, clip 0.2, gamma 1, GAE lambda 0.98, entropy 0.01, value coefficient 0.5, minibatch 1024.
- **Seeds:** 111, 222 and 333 per (objective, size). Configurable; the default gives 18 runs. Checkpoint names are `<size>+mix+bv3_spec_<m|c|t>_s<seed>`.

### Convergence rule

| Item | Value |
|---|---|
| Validation | Greedy, on the v3 validation split only (18 instances at 10x5, 13 at 20x10), every 50 PPO updates and at update 0. The value is the mean v3 score for the run's own objective (lower is better). |
| CONVERGED | At a validation step u ≥ 1,000, when (best value at updates ≤ u − 1,000 − best value at updates ≤ u) / best value at updates ≤ u − 1,000 < 0.5% (`daniel/convergence.py`). |
| Hard cap | 5,000 updates. A run that reaches it without converging is reported as **NOT CONVERGED** in every table, never silently. |
| Checkpoint | The best-validation checkpoint (`train.py` `save_model`), which is the only one evaluated. |

**Why these values:**
- **Validation cadence.** A validation every 50 updates gives 20 points per 1,000-update window.
- **Minimum of 1,000 updates.** This is DANIEL's own full training budget (1,000 updates), so no specialist stops before the original budget.
- **0.5% threshold.** It is far below the differences that matter between objectives and well below seed-to-seed spread.
- **Cap of 5,000.** It is five times DANIEL's budget.
- **Known risk.** The validation sets are small (13–18 instances). Noise can create small "improvements" in the best value and delay convergence; the cap bounds this.

**Measured cost:**
- **Update speed** (RTX 5060 Ti, training pinned to 4 logical CPUs, 2026-10-13): 0.88 s per update at 10x5 and 4.4 s at 20x10, validation included.
- **Earliest stop.** A run cannot realistically converge before about 2,000 updates: at u = 1,000 the comparison is against the untrained policy.
- **Time per run:**

  | Size | Earliest stop (~2,000 updates) | Cap (5,000 updates) |
  |---|---|---|
  | 10x5 | 0.5 h | 1.2 h |
  | 20x10 | 2.4 h | 6.1 h |

- **Totals for 9 runs per size:**

  | Size | Range |
  |---|---|
  | 10x5 | 4.4–11 h |
  | 20x10 | 22–55 h |
  | Part A overall | 27–66 h |

  Evaluation adds under 1 hour in total.
- **Cap kept.** It is practical: all 10x5 runs go first and finish within 11 h, and results are committed after every run. The alternative (a 3,000-update cap at 20x10, at most 33 h) was considered and not adopted.

### Evaluation

After each run its best-validation checkpoint is evaluated on the v3 evaluation split: 100 instances per size, used for final reporting only and never for any selection. `daniel/evaluate_v3.py` handles it.

| Item | Definition |
|---|---|
| Primary | Greedy decoding |
| Secondary | Best of 100 sampled rollouts per instance (torch generator seed 50), selected by the model's own objective (ties: first sample) |
| Objectives | All three are recorded for every model, whatever it was trained on |
| Independent check | Every reported schedule is re-scored independently (feasibility and objectives) and must equal the environment's values |

## 3. Part B: CP-SAT reference bounds

Tool: `tools/cpsat_baselines_v3.py`. OR-Tools 9.15, 8 workers, `random_seed` 20261013, CPU only.

### B1: single-objective bounds (full evaluation split, 100 instances per size)

- **Three points per instance:**

  | Point | Primary solve | Lexicographic stages (20 s each, hinted) |
  |---|---|---|
  | MK | Minimise makespan, 60 s | Minimise carbon with makespan ≤ MK*, then tardiness with makespan ≤ MK* and carbon ≤ C* |
  | T | Minimise total tardiness, 60 s | Minimise carbon with T ≤ T*, then makespan |
  | C | Minimum carbon, analytic: every operation on its lowest-carbon eligible machine (OPTIMAL by construction) | With that assignment fixed: minimise makespan, then tardiness |

- **Warm start.** Each primary solve starts from the better of the MWKR/ECT and EDD/ECT dispatching schedules.
- **Recorded per instance:** status, bound and wall time of every solve, and all three objectives of every solution.
- **Integer scaling:**
  - carbon in units of 10⁻⁴ kg CO2e, with c_om rounded;
  - tardiness T_j ≥ 1000·C_j − ⌈1000·d_j⌉, in units of 10⁻³ time units;
  - makespan exact.

  Lexicographic bounds and budgets use the incumbent's integer values, so the incumbent stays feasible.
- **Exact re-scoring.** Every schedule is re-scored exactly by `benchmark_v3.schedule_objectives`. "Optimal" refers to the scaled model; the scaling error is at most 0.01 kg carbon per 20x10 schedule and under 0.001 time units of tardiness per job.

### B2: reference Pareto front

- **Subset:** the first 20 evaluation instances at 10x5 and the first 10 at 20x10.
- **Epsilon-constraint method.** Minimise makespan subject to carbon ≤ ε_c and tardiness ≤ ε_t.
  - Grid: 5 × 5, with ε = ideal + q·(nadir − ideal) and q ∈ {0, 0.25, 0.5, 0.75, 1}.
  - Ideal and nadir per objective are the best and worst values over the instance's three B1 points.
  - Limits: 30 s per solve at 10x5 and 60 s at 20x10, warm-started from the best feasible B1 point.
- **Recorded:** the status of every grid point.
- **The front** is the non-dominated set of the feasible grid points and the three B1 points, using exact objectives.
- **Labelling.** The 20x10 front is labelled **best-known**, not optimal.
- **Cost estimate, as solver wall-clock with 8 workers:**

  | Part | Worst case | Expected |
  |---|---|---|
  | B1 | 13.3 h: 100 × (60 + 60 + 6 × 20) s per size | About 7–8 h: 10x5 solves mostly prove optimality within seconds (DEV subset: 2.7 s makespan, 6.3 s tardiness); 20x10 solves use their full limits |
  | B2 | 8.3 h: 25 × (20 × 30 + 10 × 60) s | Less at 10x5 |

- **CPU-hours.** Interpreting "CPU-hours" as solver hours, B2's worst case is at the ~8 h threshold, so the grid and subset are not reduced. In core-hours it is 8× that, about 67.

## 4. Hardware and CPU allocation

| Item | Value |
|---|---|
| GPU | NVIDIA RTX 5060 Ti, 16 GB |
| CPU | AMD Ryzen 5 7500F, 6 cores / 12 threads |
| Memory | 11 GB visible to WSL2 |
| Training | Pinned to logical CPUs 0–3 (physical cores 0–1), `OMP_NUM_THREADS=4` |
| CP-SAT | Pinned to logical CPUs 4–11 (physical cores 2–5), 8 workers |

- **Same pinning in both modes.** Training times measured in sequential and concurrent mode differ only through shared cache and memory bandwidth, not core sharing.
- **Mode recorded per run.** Training-time comparisons are made within one mode.
- **Run times:**

  | Mode | Total |
  |---|---|
  | Sequential | 35–88 h (Part A + Part B) |
  | Concurrent | About Part A alone (27–66 h), since Part B ≤ 22 h runs alongside |

## 5. Reporting

### Per model, on the evaluation split, greedy primary

- **Main table, each objective separately.** No combined score in it:
  - mean makespan, mean carbon (kg CO2e) and mean total tardiness;
  - mean tardy-job share (fraction of jobs with C_j > d_j).
- **Spread:** mean ± std (ddof 1) across seeds of the per-seed instance means.
- **Bootstrap 95% CI.** Average each instance's value over seeds, then resample the instances 10,000 times (`numpy default_rng(20261013)`, percentile interval).
- **Context rows:** the MWKR/ECT rule baseline (from the manifest), and CP-SAT's three B1 points.
- **Secondary table:** best of 100 sampling, same columns.

### Training cost

| Measure | Definition |
|---|---|
| Training time | Wall-clock inside the update loop, validation included, summed over resumed chunks (`train_seconds`). Process wall-clock is recorded separately in `run_meta.json`: every chunk attempt that finished or hit the hang-guard timeout, start-up included, but not a chunk stopped by an interrupt. |
| PPO updates to convergence | The update at which CONVERGED was declared (or the cap), plus the update of the best checkpoint. |
| Time to 5%, 2% and 1% | Updates and training time until the validation value first lies within that percentage of the run's final converged validation value (the best validation value). |
| Hardware | Hardware, CPU allocation and mode, as in section 4. |
| Curves | Convergence curves of validation value against PPO updates and against training wall-clock time: one panel per (objective, size), one line per seed. |

### Gap to bound

Each specialist's own objective is compared with CP-SAT per instance.

- **Best-known value:** the best value of that objective over all CP-SAT schedules of the instance (B1 primary, lexicographic and final schedules, and B2 points where available).
- **Gap per objective:**

  | Objective | Gap |
  |---|---|
  | Makespan, carbon | Relative: (model − best) / best |
  | Tardiness | (model − best) / Tref, because tardiness is often 0 at the optimum and a relative gap is undefined; the absolute gap is also reported |

- **Labelling.** A gap is labelled **gap to bound** only on instances whose B1 primary solve for that objective is OPTIMAL (carbon: always), and **gap to best-known** otherwise. The share of proven-optimal instances is reported next to every gap.

## 6. Definitions for the later multi-objective stage (not used in this task)

| Item | Definition |
|---|---|
| Normalisation | Per instance, f̃ = (f − ideal) / (nadir − ideal) for each objective |
| Ideal point | The CP-SAT single-objective results: best-known makespan, analytic minimum carbon, best-known tardiness |
| Nadir point | The worst value of each objective across the reference and baseline set: CP-SAT B1 points, the B2 front, all specialists' greedy evaluation schedules (all seeds) and MWKR/ECT |
| Storage | Ideal and nadir are computed once, when Parts A and B are complete, and frozen in `results/baselines_v3/normalisation_points.json`. If nadir equals ideal for an objective on an instance, that objective is dropped from the normalised metrics for the instance. |
| Hypervolume | In normalised space, reference point (1.1, 1.1, 1.1) per instance, averaged over instances |
| IGD+ | Against the normalised CP-SAT reference front of B2, on the B2 subset |
| Preference grid | All weight vectors on the 3-simplex with step 0.1: 66 vectors |
| Convergence curves | Plotted against PPO updates and against wall-clock time |

## 7. Smoke-test evidence (before launch)

`scripts/run_baselines_v3.sh --smoke` ran on 2026-10-13 in a scratch git clone, which was deleted afterwards. It used tiny budgets and the development split, so the evaluation split was not touched.
- **Pass 1** (sequential, 10x5, M/C/T):
  - pre-flight (tests, manifest, clean tree, GPU, disk);
  - killed with SIGTERM during a training chunk: all children stopped, no shutdown;
  - re-run: finished units skipped, the interrupted run completed;
  - then evaluation (greedy and best of 100), B1, B2, the analysis, per-unit commits and a clean tree at the end.
- **Pass 2** (concurrent, 20x10):
  - `taskset` affinity checked at runtime: training on CPUs 0–3, CP-SAT on 4–11;
  - killed during the second chunk, with a resume state saved;
  - re-run resumed at update 20 and completed;
  - the shutdown stub was called only after the final commit.
- **Pass 3** (hang guard): with an 8 s chunk timeout the chunk timed out, was retried once, was marked FAILED, and the script continued and finished. A plain re-run retried the FAILED run and completed it.
- **Determinism check.** Chunked-and-resumed training (3 × 20 updates) was separately verified to be bit-identical to uninterrupted training (60 updates): identical validation log, rewards and best checkpoint.

## Deviations

(none)

## Addendum A: MO-DANIEL from random initialisation (PRE-REGISTERED)

**Status: PRE-REGISTERED on 2026-10-19, before any MO-DANIEL run.** Sections 1–7 and Deviations above are unchanged. Later changes to this addendum are recorded, with date and reason, under **Deviations (Addendum A)** at its end.

- **Scope.** One preference-conditioned DANIEL per (size, seed), trained from random initialisation on makespan, carbon and tardiness together, on benchmark `sd2-mo-v3`.
- **Runner:** `scripts/run_mo_daniel_v3.sh`. Analysis: `tools/analyze_mo_daniel_v3.py`, which writes `results/mo_daniel_v3.md`.
- **Unchanged:**
  - the frozen spec, the manifest, and the reward and score definitions;
  - DANIEL's action mask;
  - the PPO algorithm and all its hyperparameters (section 2: no gradient clipping, lr 3e-4);
  - the fresh-instance training scheme (section 1).
- **Additions:** only preference conditioning and logging.
- **Known limitation, reported and not changed:**
  - DANIEL's mask admits only non-delay schedules (`results/diagnose_v3.md`, Test 2). The carbon optimum and the CP-SAT/EDD tardiness schedules are unreachable.
  - The 20x10 DANIEL-T specialist degraded during training for an undiagnosed reason. This run logs the PPO statistics that diagnosis lacked.

### A1. Model, reward and training preferences

**Conditioning.** The preference w = (w_m, w_c, w_t) on the simplex is concatenated to the inputs of the actor and the critic (`daniel/model/main_model.py`, `--preference_dim 3`):
- **Actor:** every candidate pair's input (4 pooled and candidate embeddings plus 10 pair features) gets the 3 preference values appended, so the first actor layer is 42 → 45 inputs.
- **Critic:** the pooled operation and machine embeddings get w appended, so the first critic layer is 16 → 19 inputs.
- **Feature extractor:** the dual-attention extractor does **not** receive w. Its embeddings describe the scheduling state, which is the same under every preference. The preference changes only the decision and the value.
- **Unchanged:** every other layer, width, initialisation and the action mask.
- **Checkpoint schema:** `benchmark_v3_f12_m9_p10_w3`. Names: `<size>+mix+bv3_mo_s<seed>`.

**Why this design.** Feeding the preference as an input to the policy and value networks is the established "conditioned network" approach of multi-objective deep RL: Abels et al., *Dynamic Weights in Multi-Objective Deep Reinforcement Learning*, ICML 2019; Yang et al., *A Generalized Algorithm for Multi-Objective Reinforcement Learning and Policy Adaptation*, NeurIPS 2019. It is the smallest change that makes one network cover the whole preference simplex.
- The established alternative in neural combinatorial optimisation is PMOCO (Lin, Yang and Zhang, *Pareto Set Learning for Neural Multi-Objective Combinatorial Optimization*, ICLR 2022). There, a hypernetwork maps w to the decoder's parameters.
- That would replace DANIEL's actor with generated weights, a much larger architectural change. It is not used here.

**Reward.** For an episode with preference w, the per-step terms are the specialists' terms (section 2) weighted by w (`objectives.RewardV2.step_reward_weighted`). The undiscounted return is therefore exactly −(w_m·MK/MKref + w_c·C/Cref + w_t·T/Tref), with the frozen v3 references.
- `tests/test_mo_daniel_v3.py` checks this to within 1e-9.
- At a corner preference the reward equals the corresponding specialist's reward exactly.

**Training preferences:**
- w ~ Dirichlet(1, 1, 1), drawn independently for each of the 20 training environments at every PPO update. Each training episode is one (instance, preference) pair. A fresh batch of 20 instances is drawn every 20 updates (section 1), so each instance is seen under 20 different preferences.
- The draws use a dedicated generator, `numpy default_rng([seed, 20261019])`. Its state is saved in the resume state.
- The instance stream (global numpy RNG seeded by `--seed_train`) is therefore the same as the specialists' for the same seed.

**Seeds and sizes:**
- Seeds 111, 222 and 333.
- Sizes: 10x5 first, then 20x10, so six runs by default.
- `--goal mct` (the preference sets the weights), `--v3_train_source generator`.

### A2. Validation metric

**Definition.**
- Every 50 PPO updates and at update 0, the current policy is run greedily on every validation instance (18 at 10x5, 13 at 20x10) under each of the 15 preferences of the simplex grid with step 0.25.
- Per instance, the non-dominated set of its 15 schedules is normalised with the instance's frozen ideal and nadir points (section 6 rule, including dropping an objective whose nadir equals its ideal). Its hypervolume is computed with reference point 1.1 in every objective; only points that strictly dominate the reference contribute.
- **The validation metric is the mean over instances.** Higher is better.
- Implementation: `daniel/mo_metrics.py`, an exact 3-D hypervolume tested against Monte Carlo.

**Validation normalisation points: a necessary extension.** `results/baselines_v3/normalisation_points.json` (section 6) is frozen for the 200 *evaluation* instances only, and ideal and nadir are per instance. The validation split therefore gets its own points by the same section-6 definitions, restricted to what exists on that split (`tools/mo_v3_normalisation.py`):
- **ideal:**
  - the best makespan and best tardiness over every CP-SAT B1 schedule of the instance (exactly B1's procedure: 60 s primary solves, 20 s lexicographic stages, 8 workers, seed 20261013, warm starts);
  - the analytic minimum carbon.
- **nadir:** the worst value of each objective over:
  - those CP-SAT schedules;
  - the greedy schedules of the nine selected specialists (M/C/T × 111/222/333);
  - the MWKR/ECT reference schedule.

  There is no B2 front on the validation split, so B2 is not included.
- **Storage.** The points are computed once per size before that size's first training chunk, written to `results/mo_daniel_v3/validation_normalisation_points_<size>.json` with the specialists' checkpoint hashes, committed, and never overwritten.
- **Cost.** About 5 min at 10x5. Up to about 1 h at 20x10, on CPUs 4–11 while 10x5 trains.
- **Separation.** Only validation data are used. The evaluation split is not touched by any of this.

### A3. Convergence, cap, checkpoint, degradation flag

| Item | Value |
|---|---|
| CONVERGED | The specialists' rule (section 2) applied to the validation hypervolume: at a validation step u ≥ 1,000, the best validation HV at updates ≤ u improved by less than 0.5% relative to the best at updates ≤ u − 1,000. Implemented by selecting on −HV with `daniel/convergence.py` unchanged; its denominator is \|best\|. If the older best is 0 (no point inside the reference box), the improvement counts as infinite. |
| Hard cap | **8,000 updates.** A run that reaches it is reported as **NOT CONVERGED** in every table. |
| Checkpoint | The best validation hypervolume (`train.py` `save_model`), the only checkpoint evaluated |
| DEGRADED flag | Reported separately from convergence. A run is flagged when the mean of its last three validation HVs is more than 5% below its best validation HV (`tools/mo_daniel_v3_status.py`). The flag does not change the checkpoint. |
| Chunks | 500 updates (10x5) and 200 (20x10), with bit-exact resume. The preference sampler's state is included and verified bit-exact. |
| Hang guard | Timeout per chunk = chunk × measured s/update × 3 + 600 s. Retried once, then FAILED, and the runner continues. |

**Measured cost (2026-10-19).** RTX 5060 Ti, training pinned to CPUs 0–3 with `OMP_NUM_THREADS=4`, full-size validation (all validation instances × 15 preferences every 50 updates):
- 0.84 s per update at 10x5;
- 4.49 s per update at 20x10.

| Size | Per run, earliest stop (~2,000 updates) | Per run, cap (8,000) | 3 seeds |
|---|---|---|---|
| 10x5 | 0.47 h | 1.9 h | 1.4–5.6 h |
| 20x10 | 2.5 h | 10.0 h | 7.5–30 h |
| **Total** | | | **about 9–36 h** |

- **On top of training:** about 5 min for the 10x5 validation points before training. The 20x10 points (≤ 1 h) run in parallel with 10x5 training. Evaluation is under 5 min per run.
- **Cap kept at 8,000.** That is 1.6× the specialists' cap, because one network must cover the whole simplex. The worst case is about 1.5 days, all 10x5 runs finish within about 6 h, and results are committed after every run.

### A4. Logging only

`--log_ppo_stats True` records, per PPO update, in `train_log/SD2/reward_<run>.json` and copied to `results/mo_daniel_v3/runs/<run>/reward.json`:
- policy entropy;
- approximate KL (the k3 estimator, (r − 1) − log r): the mean over all minibatches and its last-epoch mean;
- clip fraction (|r − 1| > 0.2);
- policy loss and value loss;
- critic explained variance (1 − Var(target − V_old)/Var(target));
- gradient L2 norm before the optimiser step (computed, never clipped): mean and maximum;
- value-target mean and std;
- the training return;
- the sampled preferences (mean, minimum and maximum per objective);
- the batch's mean makespan, carbon and tardiness.

The statistics are computed from detached tensors after each backward pass. `tests/test_mo_daniel_v3.py` verifies that an update with logging is bit-identical to one without.

### A5. Final evaluation (evaluation split, once)

After each run, its best-validation checkpoint is evaluated on the 100 evaluation instances per size (`daniel/evaluate_mo_v3.py`):
- **Rollouts:** greedy, under all 66 preferences of the simplex grid with step 0.1.
- **The equal-weight preference (1/3, 1/3, 1/3)** is not on that grid. It is rolled out as a 67th preference for reporting and is not part of the non-dominated set.
- **Re-scoring:** every schedule is re-scored independently and must equal the environment.
- **Per instance:**
  - the non-dominated set of the 66 schedules;
  - its hypervolume, normalised with the frozen section-6 points, reference 1.1;
  - IGD+ against the normalised CP-SAT B2 front, on the B2 subset only (20 instances at 10x5, 10 at 20x10; the 20x10 front is best-known);
  - makespan, carbon and tardiness at the three corner preferences and at equal weights.
- **Aggregation:**
  - mean ± std (ddof 1) across seeds of the per-seed instance means;
  - bootstrap 95% CI over instances of the seed-averaged per-instance values (10,000 resamples, `numpy default_rng(20261013)`, percentile).

### A6. Reporting (`results/mo_daniel_v3.md`)

**Main table per size.** HV (all instances), HV CI, HV (B2 subset), IGD+ (B2 subset) and IGD+ CI, for these rows:
- MO-DANIEL;
- the three specialists as a combined non-dominated set {M_s, C_s, T_s} per seed (from their committed greedy evaluation schedules), mean ± std over seeds;
- each specialist as a single point;
- the rules MWKR+ECT, EDD+ECT and minimum energy (MWKR job rule with minimum-energy machine), each as a point and as a combined set;
- the CP-SAT B1 extreme points (three per instance);
- the CP-SAT B2 reference front on the subset.

**Further tables and figures:**
- **Objectives table:** MO-DANIEL at the corner and equal-weight preferences, next to the specialists and the rules.
- **Convergence curves:** validation HV against PPO updates and against training wall-clock time, one line per seed.
- **PPO-statistics plots.**
- **Example 3-D fronts:** MO-DANIEL, CP-SAT and specialists, for the first three B2-subset instances per size.
- **Per run:** status (CONVERGED, NOT CONVERGED or FAILED), the DEGRADED flag, best update and training time.
- **Plain-English summary:** a placeholder, written after the runs complete.

### A7. Smoke-test evidence (before launch)

`scripts/run_mo_daniel_v3.sh --smoke` ran on 2026-10-19 in a scratch git clone of the committed code, which was deleted afterwards.
- **Smoke settings:**
  - validation on 3 instances with a 6-preference grid (step 0.5);
  - evaluation on the first 3 *development* instances with a 15-preference grid plus equal weights;
  - 40 updates in 20-update chunks;
  - CP-SAT limits of 2 s and 1 s.
- **Pass 1** (10x5 and 20x10, seed 111, `--shutdown` with a stub):
  - pre-flight passed: tests, manifest, clean tree, GPU and specialist checkpoints;
  - development and validation points were frozen and committed;
  - the 20x10 validation points started on CPUs 4–11 in the background while 10x5 trained;
  - killed with SIGTERM during the 10x5 run's second chunk: all children stopped, including the background CP-SAT job; no shutdown; resume state kept.
- **Pass 2** (same command):
  - resumed at update 20 and completed;
  - the 20x10 points finished and were committed;
  - 20x10 trained;
  - both runs were evaluated (independent re-scoring passed) and committed;
  - the analysis wrote `results/mo_daniel_v3.md` and all six figures;
  - the shutdown stub was called only after the final commit;
  - the tree was clean.
- **Pass 3** (hang guard, 5 s chunk timeout): the chunk timed out, was retried once and was marked FAILED. The runner committed and continued.
- **Pass 4** (plain re-run): the FAILED run was retried and completed; the analysis succeeded.
- **Determinism:**
  - Chunked-and-resumed MO training (2 × 20 updates) is bit-identical to uninterrupted training (40 updates): best checkpoint, per-update reward/PPO-statistics/preference log and validation log, including the preference sampler.
  - With the new code, default v1 training and v3 specialist training (20 updates each) are bit-identical to the previous commit: weights, rewards and validation.
  - An update with PPO-statistics logging is bit-identical to one without (`tests/test_mo_daniel_v3.py`).
- **Report rows that need the evaluation split's references** (B2 subset, IGD+, specialists, CP-SAT B1) were exercised once with a synthetic MO result made only of the committed seed-111 specialist schedules. No MO checkpoint was rolled out on the evaluation split for this.
- **Disclosure.** During development, before this addendum, an untrained 10-update scratch checkpoint was evaluated on 5 evaluation instances to test `evaluate_mo_v3.py`. The output was deleted, and nothing was selected or tuned from it.
- **Fixed during smoke:**
  - the runner scripts lacked the executable bit in git (`run_baselines_v3.sh` too);
  - `--mo_val_norm` is resolved against the working directory;
  - the analysis skips a size whose normalisation points are missing.
- **Measured speed:** 0.84 s per update at 10x5 and 4.49 s at 20x10 (A3). Evaluation takes about 16 s and 3 min per run at 10x5 and 20x10.

### Deviations (Addendum A)

(none)
