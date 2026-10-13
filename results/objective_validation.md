# Objective-generation validation: carbon and due dates for SD2 10x5 and 20x10 (Step 1)

## Pre-registration

Written and committed BEFORE any instance is generated or any diagnostic computed. Never edited afterwards; later
changes go under "Deviations". No DANIEL training. No existing training/environment/evaluation code path is changed.
The reserved final-test sets are never loaded; the 49 HELD-OUT instances are never used (they are only hashed, for
the disjointness check). New code: `tools/objective_validation/`. Proposed spec: `configs/problem_spec_v3_PROPOSED.yaml`.

### Development instances
* Generator: the existing `daniel/data_utils.SD2_instance_generator` (op_per_job = n_m, processing times U{1..99},
  eligible machines per operation U{1..n_m}, data_suffix "mix"), driven exactly like `generate_data_to_files`:
  `np.random.seed(seed)` once, then instances generated in sequence, written with `matrix_to_text` to
  `daniel/data/objective_dev/{cal,check}/SD2/<size>+mix/`.
* Seeds (dedicated, not used by any existing split; existing generator seeds are 100, 200, 400):
  DEV-CAL 10x5 = 20261013 (100 instances), DEV-CAL 20x10 = 20261014 (100), DEV-CHECK 10x5 = 20261015 (50),
  DEV-CHECK 20x10 = 20261016 (50). DEV-CAL is used for calibration; DEV-CHECK only for the final confirmation.
* Disjointness: structural SHA-256 (jobs, eligibility, processing times; `daniel/data_manifest.structural_fjsp_sha256`
  semantics, computed in memory) of every DEV instance vs every instance file in `daniel/data/{data_train,
  data_validation, data_train_vali, data_development, SD2}` (both sizes, mix and carbon+priority variants; the
  HELD-OUT instances are hashed only) and vs the hashes recorded in `daniel/data/manifests/`; DEV-CAL vs DEV-CHECK.
  `data_final_test` is never opened: its disjointness is argued from its generator seed (400) and recorded in the
  report as such.
* Per-instance random parameters use `numpy.random.default_rng(int(sha256("<instance key>|<purpose>")[:16], 16))`
  so they do not depend on generation order: purpose "power" (P_m), "kdraw" (v_j), "oldcarbon" (old i.i.d.
  carbon, comparison only), "random_schedules". All are stored in a manifest with each instance's SHA-256.

### Objectives (as decided)
* Time unit assumption: **1 SD2 time unit = 1 minute** (SD2 processing times are unitless). Carbon scales linearly with
  this choice; every comparison and ranking below is invariant to it.
* Carbon (processing energy only): P_m ~ U(4, 15) kW per machine (continuous, seeded, stored);
  E_om = P_m x p_om / 60 kWh; C = 0.540 x sum of E over the chosen (operation, machine) assignments, kg CO2e
  (gamma = 0.540 kg CO2e/kWh, CEA-FJSP, Wang et al., Frontiers in Environmental Science 2022).
* Old carbon (comparison only): i.i.d. integer U{1..100} per eligible (operation, machine), as `dataset_modifier.py`.
* Tardiness: T = sum_j max(0, C_j - d_j), r_j = 0, d_j = k_j x W_j, W_j = sum over the job's operations of the MEAN
  processing time over its eligible machines; k_j = k x (1 + s x (v_j - 0.5)), v_j ~ U(0, 1) per job, seeded and
  stored (common random numbers across all (k, s), i.e. k_j ~ U[k(1 - s/2), k(1 + s/2)]).

### Schedule set (no learning), independent builder `tools/objective_validation/schedules.py`
* Active-style dispatching: with r_j = completion of job j's last scheduled operation and F_m = machine free time,
  c* = min over candidate pairs (j, eligible m) of max(r_j, F_m) + p; conflict set = jobs whose next operation could
  start before c* on some eligible machine; the JOB rule picks a job from the conflict set (ties: lowest job index);
  the MACHINE rule picks one of its eligible machines; start = max(r_j, F_m).
* Job rules: SPT (smallest min-eligible processing time of the next operation), LPT (largest), MOR (most operations
  remaining), MWKR (most remaining work, mean-eligible processing times), EDD (smallest d_j), FIFO (smallest r_j).
  Machine rules: ECT (earliest completion), SPT (shortest processing time; ties ECT), MINE (minimum energy P_m x p;
  ties ECT). For the old-carbon comparison only: OLDC (minimum old carbon; ties ECT).
* Random dispatching: 50 schedules per instance (job uniform over the conflict set, machine uniform over eligible).
* Every schedule is validated (eligible machine, precedence, no machine overlap, every operation exactly once).
* Carbon diagnostics (Part 3) use the due-date-free set: 5 job rules (no EDD) x 3 machine rules + 50 random = 65
  schedules (old carbon: OLDC replaces MINE). Tardiness and joint diagnostics add EDD x 3 machine rules at the
  configuration being evaluated (68 schedules).
* CP-SAT subset: the first 10 DEV-CAL instances per size, 60 s, 8 workers, fixed seed: (a) minimum makespan,
  (b) minimum total tardiness at the chosen due-date configuration, (c) minimum makespan with every operation fixed to
  its minimum-energy machine. Status (OPTIMAL / FEASIBLE) reported; these schedules are reported separately and do not
  enter the pass/fail diagnostics.

### Part 3: carbon diagnostics and pass rule
* C1: share of operations whose fastest eligible machine (ties: lower energy) is not the minimum-energy eligible
  machine (ties: shorter time).
* C2 per instance over the carbon schedule set: Spearman(makespan, carbon); size of the 2-D non-dominated set
  (makespan, carbon); trade-off magnitudes TM_mk = (makespan of the best-makespan schedule among those attaining the
  set's minimum carbon - set minimum makespan) / set minimum makespan, and TM_c = (carbon of the lowest-carbon schedule
  among those attaining the set's minimum makespan - set minimum carbon) / set minimum carbon (median and IQR);
  Spearman(carbon, total processing workload sum of chosen p).
* **Pass**: C1 >= 20% at both sizes AND median TM_mk >= 5% AND median TM_c >= 5% at both sizes. The same diagnostics
  are reported for the old i.i.d. carbon. If the check fails it is reported and the carbon part stops there (no
  switch to per-operation power, no tuning of the power range).

### Part 4: tardiness calibration (per size, DEV-CAL)
Grid k in {0.6, 0.7, ..., 2.0} x s in {0.2, 0.4, 0.6, 0.8}. Rule schedules for the T statistics use ECT machines
unless stated.
* T1: mean (over instances) share of tardy jobs under EDD + ECT. Target 20-50%.
* T2: median relative reduction of total tardiness of EDD vs SPT and vs MWKR, (T_X - T_EDD) / T_X (0 if T_X = 0);
  share of instances where the best of the 18 rule schedules has zero tardiness (< 10%); share of instances where
  every job is late in every rule schedule (< 10%); median coefficient of variation of total tardiness over the
  68-schedule set.
* T3: median per-instance Spearman(T, makespan) and Spearman(T, sum_j C_j); among all schedule pairs whose makespans
  are within 1% (of the smaller), the share whose T differs by > 10% (of the larger; pairs with both T = 0 do not
  differ), pooled; median size of the 2-D non-dominated set (makespan, T).
* **Selection** (per size): keep configurations with T1 in [20%, 50%], zero-tardiness share < 10%, all-late share
  < 10% and median Spearman(T, makespan) < 0.9; among them choose the EDD tardy share closest to 35%; ties by the
  larger median EDD-vs-SPT reduction. If none passes, report the closest and stop.
* **Confirmation** on DEV-CHECK for the chosen configuration: passes if T1 is in [20%, 50%] and the zero-tardiness,
  all-late and Spearman(T, makespan) conditions hold. Carbon diagnostics are also re-reported on DEV-CHECK.

### Part 5: joint check (chosen carbon + chosen due dates, 68-schedule set)
Median per-instance Spearman matrix of {makespan, carbon, T, sum_j C_j, workload}; share of schedules in each
instance's 3-D non-dominated set (makespan, carbon, T). **Three meaningfully distinct objectives** at a size if every
pairwise median |Spearman| among makespan, carbon and T is < 0.9 and the median instance's 3-D non-dominated set
holds >= 3 schedules.
