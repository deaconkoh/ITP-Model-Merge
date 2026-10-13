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
