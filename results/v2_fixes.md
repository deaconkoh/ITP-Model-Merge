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
