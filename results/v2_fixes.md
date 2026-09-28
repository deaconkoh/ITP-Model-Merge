# v2 observations and reward: making due dates visible, and does the tardiness specialist then use them?

## Pre-registration

Written and committed BEFORE any code change for v2 is verified and before any training in this task. Never edited
afterwards; changes go under "Deviations". Reserved final-test sets (10x5, 20x10) are never loaded or evaluated.
No Step 4, blend or ceiling experiment is re-run.

### Versioning (v1 stays the default)
* New flags: `--obs_version {v1,v2}` (default v1), `--reward_version {v1,v2}` (default v1),
  `--reward_ref_manifest` (v2 reward references), `--reward_ref_seed` (default `--seed_train`).
* `--obs_version v2` sets operation dim 13, pair dim 10 and the new feature schema `canonical_f13_p10_obsv2`
  (machine dim 8 unchanged); schema validation rejects any mix of v1 schema with v2 observations and vice versa.
  Every v1 code path is unchanged.
* v1 must still reproduce earlier results: (i) a test (`tests/test_v1_reproduction.py`) re-evaluates C1@1000
  (`s4t_spec_m_k125s060_s<seed>@u1000`, 4 seeds) on the 49 HELD-OUT instances with default flags and requires
  results identical to the Step 4 result files; (ii) two 10-update v1 trainings (goal mct from C1@1000 with Step 4
  weights; goal t from scratch; seed 111), recorded BEFORE any code change (and verified bit-deterministic by a
  repeat), must give bit-identical checkpoints and validation logs after the change.

### A1. Due-date observations (v2). All quantities in the environment's scaled time units
Every time quantity is divided by U = the maximum processing time over the environment batch (`pt_upper_bound`),
exactly the divisor already applied to processing times, completion-time lower bounds, job-ready and machine-free
times, and (since the tardiness work) the due dates themselves (`due_dates = d_j / U`). Notation at a decision step:
r_j = job ready time = completion of j's last scheduled operation (`candidate_free_time`, 0 if none);
F_m = machine free time (`mch_free_time`); o_j = j's next unscheduled operation (`candidate`); minpt_o = minimum
processing time of o over its eligible machines (`op_min_pt`).
* **Current earliest start of j's next operation** (adapting to the environment's own start rule
  start = max(job ready, machine free), `pair_free_time`): EST_j = min over machines m eligible for o_j of
  max(r_j, F_m). For a finished job, EST_j = r_j = its completion time C_j.
* **Remaining-work lower bound** W_j = sum of minpt_o over j's unscheduled operations, o_j included (0 if finished).
* **Operation feature 12, job slack**: slack_j = d_j - EST_j - W_j.
* **Operation feature 13, time to due date**: ttd_j = d_j - EST_j.
  Both are given to every operation of job j (as the existing job-level features are), recomputed at every step,
  and then standardised exactly like every other operation feature (per instance, per feature, over the
  not-yet-deleted operations; `norm_op_features`).
* **Pair feature 10, projected lateness**: for candidate o_j on machine m,
  late_jm = max(0, max(r_j, F_m) + pt(o_j, m) + (W_j - minpt_{o_j}) - d_j), i.e. the completion on this machine under
  the start rule plus the lower bound on the job's work after it, minus the due date. For a finished job (all its
  pairs are masked) late_jm = max(0, C_j - d_j). Not further normalised, like the other pair features.
* Feature order: operation features 1-11 and pair features 1-9 keep their v1 positions (the retired priority
  feature stays at operation position 11); the new ones are appended.

### A2. Carbon scaling (v2)
Pair feature 9 = carbon(o, m) / 100. CARBON_SCALE = 100 = the generator's maximum (checked: max carbon = 100, min 1
in the 10x5 train, validation and train_vali pools).

### A3. Reward v2 (`--reward_version v2`)
Preference weights w_k = 1/|goal| for each objective in the goal (mct: 1/3 each, as Step 4; t: w_t = 1).
References from a frozen, hashed manifest (`daniel/data/reward_refs/`, built by `tools/build_reward_refs.py`,
hash-checked on load, fail-loud; per the due-date protocol): for seed s, MKref_i and CBref_i = the Step 4 centroid
composition `simplex_m333c333t334_k125s060_s<s>` evaluated greedily on instance i (canonical per-instance
evaluation; for the train_vali pool these ARE the Step 4 values and the builder checks they reproduce the Step 4
files exactly), and Tbar_s = that composition's mean total tardiness over the 100 train_vali instances (Step 4's
tardiness normaliser). Per step t on instance i:

    r_t = w_m * U * (PhiM_{t-1} - PhiM_t) / MKref_i  -  w_c * carbon_t / CBref_i  +  w_t * U * (PhiT_{t-1} - PhiT_t) / Tbar_s
    PhiM_t = max over operations of the completion-time lower bound (scaled), PhiM_0 := 0
    PhiT_t = v1 tardiness potential (tardiness of the lower-bound job completions, scaled), PhiT_0 := 0

Undiscounted return = -(w_m MK/MKref_i + w_c CB/CBref_i + w_t T/Tbar_s) = **-(Step 4 score of the episode), with no
additive constant** (the makespan potential starts at 0 like the tardiness one, so the initial lower bound is
charged at the first step instead of being dropped). Validation selection metric under v2 = mean of that score over
the validation instances. Verification: on 100 training-generator episodes (the audit's 5 x 20 draws, rng
20260929), C1@1000 converted to v2, goal mct, v2 obs + v2 reward: |sum of per-step rewards + score| < 1e-9 for every
episode.
A4: the reward-balance table exactly as the audit (mean |term| and between-episode sd per term, sampled and greedy
actions, 4 seeds x 100 episodes), v1 next to v2.

### B. Function-preserving conversion (`tools/convert_ckpt_v1_to_v2.py`)
* First operation-attention block: each head's W [11 x 32] -> [13 x 32], new rows exactly 0.
* First machine-attention block: each head's W_edge [11 x 32] (operation features as edge features) -> [13 x 32],
  new rows exactly 0.
* Actor first Linear: weight [64 x 41] -> [64 x 42], new column exactly 0; the carbon column (index 40 = 32
  embedding + 8) multiplied by 100.
* Everything else (including the critic) copied unchanged; metadata records schema `canonical_f13_p10_obsv2` and
  the parent's SHA-256.
* Nonlinearity check (from the code): operation features reach W only through the environment's per-feature
  standardisation (acts on each feature separately, so it cannot mix new features into old ones), input dropout
  (p = 0; identity in eval) and, for the machine block, `comp_idx @ fea_j` (linear); pair features enter the actor's
  first Linear directly. Zero rows/columns therefore contribute exactly 0. Carbon x100 against /100 is exact up to
  float rounding.
* Checkpoints: makespan `m_s<s>`, carbon `c_s<s>`, tardiness `t_k125s060_s<s>` specialists and C1@1000, 4 seeds (16).
* **Verification (B3)**: v1 checkpoint in v1 env vs v2 checkpoint in v2 env, lock-step on all 49 HELD-OUT instances
  (grouped batching, due dates from the manifest): identical greedy action at every step, max |difference in action
  probability| < 1e-5, |difference in Step 4 score| < 1e-6 per instance. **STOP 1** if any checkpoint fails.

### C. Retrain the 10x5 tardiness specialist on v2 (`tools/run_v2_tardiness.py`)
* Original v1 tardiness specialist protocol (Step 5, `tools/run_tardiness_specialists.py`): `configs/canonical/t.json`
  (goal t, tardiness_reward_weight 1.0), all `params.py` defaults (lr 3e-4, eps_clip 0.2, k_epochs 4, minibatch 1024,
  entloss_coef 0.01, vloss_coef 0.5, gamma 1, gae_lambda 0.98, num_envs 20, reset_env_timestep 20,
  validate_timestep 10), **1000 updates**, train/validation splits 10x5, due dates k125s060, seed_train = seed; the
  specialist is the best-validation checkpoint (saved on every validation improvement).
* **V2 arm**: identical protocol with `--obs_version v2 --reward_version v2`, `--init_from` the seed's converted
  tardiness specialist. Value re-scaling: v1's goal-t return is -T/U and v2's is -T/Tbar_s, so the critic's final
  Linear weight and bias are multiplied by U/Tbar_s (U = 99, the batch normaliser of every 10x5 training batch: 50/63
  training instances have max pt 99) so that the carried-over critic predicts the v2 return scale. (Advantages are
  normalised per environment in `PPO.update`, so the policy gradient is scale-free; the rescale only matters for the
  value loss.) 4 seeds.
* **V1-CONT control** (secondary; not part of the decision rule): the v1 tardiness specialist trained 1000 further
  updates with the identical protocol in v1 (obs v1, reward v1, critic as is). It separates "v2" from "1000 more
  updates".
* Evaluation on the 49 HELD-OUT instances (greedy, grouped batching, manifest due dates) of: v1 specialist, V2 arm,
  V1-CONT. Per job: d_j, P_j (due-date work content), C_j, T_j. Metrics per seed and pooled: mean total tardiness;
  share of jobs late; share of tightest-quartile jobs late (Q1 = jobs with d_j/P_j at or below the 25th percentile
  of all HELD-OUT jobs; a fixed job set); Spearman(d_j/P_j, C_j/P_j) pooled over all jobs of the 4 seeds, exactly as
  in input_reward_audit (v1 policy there: -0.02; CP-SAT +0.74); mean makespan and carbon.
* **Decision rule**:
  * per seed: reduction = (mean T of v1 - mean T of V2) / mean T of v1 over the 49 instances, 95% bootstrap CI
    (10,000 instance resamples, paired, generator 20260930). A seed counts if reduction >= 20% AND the CI excludes 0.
  * **SUCCESS**: >= 3 of 4 seeds count AND the pooled V2 tightness correlation >= +0.3.
  * **PARTIAL**: exactly one of the two holds. **NO EFFECT**: neither holds.
  * **STOP 2** if not SUCCESS (Part D skipped).

### D. 20x10 check (only if C is SUCCESS)
Cost estimated first from the v1 20x10 tardiness specialists' recorded wall clock; stop if > 8 GPU-hours.
Otherwise: 20x10 tardiness specialist from scratch in v2 (obs v2, reward v2), same setup as the failed v1 20x10
specialist (k095s060 manifest, t.json, defaults, same budget), 4 seeds; v2 reward normaliser for 20x10 (no Step 4
centroid exists there): Tbar = mean tardiness of the seed's v1 20x10 makespan specialist on the 20x10 validation
pool (a constant scale only). Evaluated on the 20x10 validation instances of the original Gate 1 check (never the
final-test set) with the same Gate 1 criterion, pass/fail per seed, plus the tightness correlation.

## Implementation (as pre-registered; code behind flags, v1 default)

* Flags `--obs_version`, `--reward_version`, `--reward_ref_manifest`, `--reward_ref_seed` (`daniel/params.py`);
  schema `canonical_f13_p10_obsv2` (`daniel/feature_schemas.py`, rejects mixing v1/v2); v2 features and reward in
  `daniel/fjsp_env_same_op_nums.py` (`initialize_v2`, `v2_job_state`, v2 branches of `construct_op_features`,
  `construct_pair_features`, `step`); `RewardV2` in `daniel/objectives.py`; frozen references
  `daniel/data/reward_refs/reward_ref_manifest.json` (built by `tools/build_reward_refs.py`, 181 instances of the
  train / validation / train_vali pools, SHA-256 per file; the centroid rollouts reproduced the Step 4 train_vali
  files exactly); `daniel/reward_refs.py` (hash-checked loader); `daniel/train.py` wiring.
  `FJSPEnvForVariousOpNums` (SD1 only) refuses v2.
* Feature definitions exactly as A1/A2 (op features 12 slack, 13 time-to-due; pair feature 10 projected lateness;
  carbon / 100). "Current earliest start" = min over the next operation's eligible machines of
  max(job ready, machine free), the environment's own start rule. Independent check: the three features recomputed
  with plain per-job loops from raw instance data and the schedule state match the environment to 7.1e-15
  (`tools/v2_reward_balance.py`, 2 batches x 4 seeds, every 7th step), and carbon x 100 equals the raw value.
* Reward v2 exactly as A3. Tbar (Step 4 centroid mean tardiness, 100 train_vali): 768.61 / 729.39 / 685.52 / 493.27
  for seeds 111 / 222 / 333 / 444.

## v1 reproduction (PASS)

* `tests/test_v1_reproduction.py`: C1@1000, 4 seeds, 49 HELD-OUT, default flags -> makespan, carbon, tardiness and
  Step 4 score **identical** to the Step 4 result files (runs, not skipped). The 25 existing unit tests pass before
  and after the change.
* Training regression: the two 10-update v1 trainings (mct from C1@1000 with Step 4 weights; t from scratch; seed
  111), recorded before any code change and confirmed bit-deterministic by a repeat, give **bit-identical**
  checkpoints (final and @u10) and validation logs after the change.
* B3's v1 side re-evaluated all 16 v1 checkpoints on HELD-OUT: identical to the Step 4 result files in every case.

## A3/A4: reward v2 check and balance

**A3 PASS**: over 800 training-generator episodes (the audit's 5 x 20 draws, 4 seeds, sampled and greedy),
max |sum of per-step v2 rewards + Step 4 score| = **5.6e-16** (criterion 1e-9). No additive constant.

Balance (C1@1000, training generator, 4 seeds x 100 episodes; v1 = input_reward_audit, v2 = converted C1@1000 in v2,
goal mct). Under v2 the makespan term is the full -w_m MK/MKref (potential from 0), not relative to the initial bound.

| term | v1 mean abs (sampled) | v1 sd | v2 mean abs (sampled) | v2 sd | v2 mean abs (greedy) | v2 sd (greedy) |
|---|---|---|---|---|---|---|
| makespan | 2.10 | 0.44 | 0.307 | 0.035 | 0.301 | 0.034 |
| carbon | 5.67 | 0.50 | 0.394 | 0.039 | 0.395 | 0.038 |
| tardiness | 16.06 | 10.06 | 0.140 | 0.091 | 0.129 | 0.081 |
| total | 23.83 | 10.30 | 0.841 | 0.107 | 0.826 | 0.094 |

v2 ratios (sampled): mean abs tardiness/makespan **0.45x** (v1 7.7x), tardiness/carbon 0.35x, carbon/makespan
1.28x; between-episode sd tardiness/makespan **2.6x** (v1 23x), tardiness/carbon 2.4x (v1 20x). **No term is out of
scale by more than 5x on either measure.** (The remaining 2.6x in spread is the score's own: tardiness genuinely
varies more between episodes than makespan and carbon do, and the v2 return is the score.)

## B3: checkpoint verification (PASS, 16/16)

v1 checkpoint in v1 env vs converted checkpoint in v2 env, lock-step, 49 HELD-OUT instances (2,450 decisions each):

| checkpoint | seeds | differing greedy actions | max abs probability diff | max abs score diff | v1 side = Step 4 files |
|---|---|---|---|---|---|
| makespan specialist `m_s<s>` | 111/222/333/444 | 0 / 0 / 0 / 0 | 8.9e-7 / 7.7e-7 / 6.6e-7 / 1.0e-6 | 0 / 0 / 0 / 0 | yes x4 |
| carbon specialist `c_s<s>` | 111/222/333/444 | 0 / 0 / 0 / 0 | 4.9e-7 / 5.4e-7 / 7.7e-7 / 7.7e-7 | 0 / 0 / 0 / 0 | yes x4 |
| tardiness specialist `t_k125s060_s<s>` | 111/222/333/444 | 0 / 0 / 0 / 0 | 4.5e-7 / 8.3e-7 / 2.4e-7 / 6.6e-7 | 0 / 0 / 0 / 0 | yes x4 |
| C1@1000 | 111/222/333/444 | 0 / 0 / 0 / 0 | 1.2e-6 / 1.4e-6 / 1.3e-6 / 7.2e-7 | 0 / 0 / 0 / 0 | yes x4 |

Nonlinearity check (as pre-registered, confirmed in the code): the new operation features reach the first
attention weights only through the per-feature standardisation (no cross-feature mixing), zero-probability input
dropout and the linear `comp_idx @ fea_j`; the new pair feature enters the actor's first Linear directly. The
remaining probability differences (<= 1.4e-6) are float32 rounding from carbon x 100 / 100. Converted checkpoints
are git-ignored like all checkpoints; SHA-256 in `v2_fixes/converted_checkpoints_sha256.json`.

## Part C: tardiness specialist retrained on v2

Protocol as pre-registered (t.json, defaults, 1000 updates, best-validation checkpoint, 4 seeds). Wall clock
12.5-13.5 min per run (8 runs, 1.8 GPU-h). Best-validation updates: V2 470 / 910 / 840 / 500; V1-CONT 560 / 1000 /
520 / 890. Provenance: `v2_fixes/provenance/`.

### Part C: tardiness specialists on the 49 HELD-OUT instances (means over instances; pooled = over 4 seeds)

| arm | seed | mean total tardiness | jobs late | tightest-quartile jobs late | Spearman(d/P, C/P) | makespan | carbon | Step 4 score |
|---|---|---|---|---|---|---|---|---|
| v1 tardiness specialist | 111 | 336.1 | 48.4% | 87.8% | -0.069 | 449.7 | 2503.2 | 0.8706 |
| v1 tardiness specialist | 222 | 347.4 | 47.1% | 75.6% | -0.041 | 454.2 | 2558.3 | 0.8863 |
| v1 tardiness specialist | 333 | 380.8 | 48.4% | 82.1% | -0.029 | 466.0 | 2493.8 | 0.9073 |
| v1 tardiness specialist | 444 | 330.5 | 45.1% | 83.7% | -0.092 | 443.0 | 2502.5 | 0.9401 |
| v1 tardiness specialist | pooled | 348.7 | 47.2% | 82.3% | -0.057 | 453.2 | 2514.4 | 0.9011 |
| V2: retrained on v2 | 111 | 305.8 | 48.4% | 81.3% | +0.150 | 446.4 | 2476.2 | 0.8504 |
| V2: retrained on v2 | 222 | 299.2 | 56.3% | 74.8% | +0.450 | 462.4 | 2472.0 | 0.8556 |
| V2: retrained on v2 | 333 | 278.8 | 48.2% | 75.6% | +0.283 | 455.4 | 2511.0 | 0.8541 |
| V2: retrained on v2 | 444 | 244.9 | 43.3% | 74.8% | +0.224 | 444.4 | 2520.8 | 0.8862 |
| V2: retrained on v2 | pooled | 282.2 | 49.0% | 76.6% | +0.275 | 452.1 | 2495.0 | 0.8616 |
| V1-CONT: v1 + 1000 updates (control) | 111 | 346.7 | 47.6% | 80.5% | -0.078 | 453.4 | 2560.5 | 0.8883 |
| V1-CONT: v1 + 1000 updates (control) | 222 | 350.9 | 46.9% | 85.4% | -0.060 | 471.4 | 2450.4 | 0.8817 |
| V1-CONT: v1 + 1000 updates (control) | 333 | 336.0 | 45.9% | 86.2% | -0.044 | 450.5 | 2502.0 | 0.8759 |
| V1-CONT: v1 + 1000 updates (control) | 444 | 347.8 | 46.3% | 81.3% | -0.040 | 458.8 | 2546.8 | 0.9706 |
| V1-CONT: v1 + 1000 updates (control) | pooled | 345.4 | 46.7% | 83.3% | -0.055 | 458.5 | 2514.9 | 0.9041 |
| CP-SAT 60 s best (ceiling_check) | pooled | 50.6 | 19.3% | - | +0.738 | - | - | 0.6140 |

**Tardiness reduction of V2: retrained on v2 vs v1 tardiness specialist** (paired over instances; 95% bootstrap CI, 10,000 resamples)

| seed | reduction | 95% CI | counts (>= 20% and CI excludes 0) |
|---|---|---|---|
| 111 | +9.0% | [-2.0, +19.0] | no |
| 222 | +13.9% | [+2.5, +24.1] | no |
| 333 | +26.8% | [+16.3, +37.0] | yes |
| 444 | +25.9% | [+15.7, +36.0] | yes |

**Tardiness reduction of V1-CONT: v1 + 1000 updates (control) vs v1 tardiness specialist** (paired over instances; 95% bootstrap CI, 10,000 resamples)

| seed | reduction | 95% CI | counts (>= 20% and CI excludes 0) |
|---|---|---|---|
| 111 | -3.2% | [-15.7, +7.6] | no |
| 222 | -1.0% | [-12.3, +9.4] | no |
| 333 | +11.8% | [+1.9, +21.3] | no |
| 444 | -5.2% | [-18.3, +6.0] | no |

**Tardiness reduction of V2: retrained on v2 vs V1-CONT: v1 + 1000 updates (control)** (paired over instances; 95% bootstrap CI, 10,000 resamples)

| seed | reduction | 95% CI | counts (>= 20% and CI excludes 0) |
|---|---|---|---|
| 111 | +11.8% | [+1.1, +22.2] | no |
| 222 | +14.7% | [+2.5, +25.5] | no |
| 333 | +17.0% | [+6.3, +27.2] | no |
| 444 | +29.6% | [+20.5, +38.8] | yes |

### Verdict: **NO EFFECT** (tardiness criterion 2/4 seeds not met; pooled tightness correlation +0.275 < +0.3)

Pooled over seeds (paired bootstrap over instances, not part of the rule): V2 vs v1 **-19.1%** tardiness
[CI -12.4, -26.1]; V2 vs V1-CONT -18.3% [-11.9, -24.6]; V1-CONT vs v1 -1.0% [+5.5, -7.2] (no change).

### Verdict: **NO EFFECT** (pre-registered rule)

* Tardiness criterion: **2 / 4** seeds (333, 444) show a >= 20% reduction with the CI excluding 0; seeds 111 (-9.0%)
  and 222 (-13.9%, CI excludes 0) fall short. Needed 3.
* Tightness correlation: pooled **+0.275**, below the pre-registered +0.3 (per seed +0.15 / +0.45 / +0.28 / +0.22).
* Neither criterion holds, so the rule gives NO EFFECT, and by STOP 2 **Part D (20x10) was not run**.

Reading the numbers beyond the label: both criteria are narrow misses, not nulls. Against v1 the V2 specialists'
tightness correlation moved from -0.06 to +0.28 (every seed positive; CP-SAT +0.74), and pooled tardiness fell 19%.
The V1-CONT control rules out "just 1000 more updates": it stays at -0.06 and changes tardiness by 1%. So the v2
observations change the policy in the intended direction (it starts finishing urgent jobs relatively earlier), but
by less than the pre-registered bar and far from the solver (282 vs 51 mean total tardiness). The share of jobs late
did not fall (49% vs 47%): the gain comes from shorter lateness, concentrated on the tightest jobs (82% -> 77% late).
Makespan and carbon are unchanged (452 vs 453; 2495 vs 2514), so there is no trade-off cost.

## Part D: NOT RUN (STOP 2: Part C is not SUCCESS)

## Deviations

* Extra helper files beyond the named deliverables: `tools/v2_common.py` (v1/v2 loading, rollouts, lock-step),
  `tools/v2_verify.py` (B3), `tools/v2_reward_balance.py` (A3/A4 and the independent feature check),
  `tools/build_reward_refs.py` + `daniel/reward_refs.py` (reference manifest). `tools/analyze_v2.py` also evaluates
  the original v1 tardiness specialists on HELD-OUT once (cached in `raw/eval_V1_s*.npz`).
* v1 checkpoint and run-manifest METADATA now also lists the new flags (e.g. `obs_version: "v1"`), because the
  config is serialised whole; weights, rewards and validation logs are bit-identical (training regression above).
* A 20-update v2 smoke training (seed 111, scratchpad output, not committed) was run before Part C to check the
  pipeline; it wrote nothing to the repository. The seed-111 v2 init checkpoint it used is the same file Part C used.
* The independent feature check (tolerance 1e-9) was added to A3/A4; not in the pre-registration.
* Pre-registration wording "committed BEFORE any code change for v2 is verified": it was committed before any v2
  code was written.

## Plain-English summary

The audit found that the scheduling policy was being graded on lateness without ever seeing the due dates. We
added three due-date inputs (how much slack each job has, how long until it is due, and how late a job would end if
placed on a given machine), put carbon on the same 0-1 scale as the other inputs, and changed the training reward
so that it equals the evaluation score exactly and no objective drowns out the others. Old trained models were
converted to the new format without changing a single decision, and the old setup still reproduces every earlier
result bit for bit. Retraining the lateness specialist with the new inputs cut its total lateness by about 19% and,
for the first time, made it finish urgent jobs relatively earlier (correlation up from -0.06 to +0.28). An equally
long retraining without the new inputs changed nothing, so the improvement comes from the inputs. It still fell just
short of both pre-registered bars (20% lateness cut in 3 of 4 seeds, correlation of +0.3), so the formal verdict is
"no effect" and the 20x10 check was not run. The policy remains far from the solver's schedules (lateness 282 vs 51),
so seeing due dates is necessary but not sufficient.
