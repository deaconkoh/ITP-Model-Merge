# Input and reward audit: what the policy can observe, and how the reward is composed

READ-ONLY audit (no environment, feature, reward or model code changed; no training; reserved final-test sets never
loaded). Code audited: `daniel/fjsp_env_same_op_nums.py` (the environment every 10x5 run since Step 4 uses),
`daniel/model/main_model.py`, `daniel/objectives.py`, `daniel/params.py`, `configs/canonical/mct.json` (Step 4's
config; schema `canonical_f11_p9_v2`). Measurements: `tools/audit_input_reward.py` -> `results/input_reward_audit/`
(`reward_terms.json`, `jobs.csv` = every job of every HELD-OUT instance x seed x schedule).

## Verdict: **FAIL**

1. **Due dates are invisible to the policy.** No operation, machine or pair feature contains d_j or any function of it
   (slack, time-to-due, tardiness estimate). The policy is trained and evaluated on total tardiness it cannot see
   the cause of. Behaviourally confirmed: C1@1000 finishes jobs at the same multiple of their work content whatever
   their due-date tightness (Spearman(d_j/P_j, C_j/P_j) = **-0.02**), whereas CP-SAT sequences by due date (**+0.74**).
2. **The tardiness reward term is out of scale.** Under Step 4's weights its mean absolute per-episode return (16.1)
   is **7.7x** the makespan term's (2.1); by episode-to-episode spread (the part that carries learning signal) it is
   23x the makespan term and 20x the carbon term.
3. (Secondary, not by itself a FAIL criterion) **Carbon is visible but unnormalised**: the carbon pair feature is the
   raw integer 1-100 (mean 51), fed straight into the actor MLP next to eight features that live in [0, ~4].

Per the stop condition, Part 2 (longer training) was **not run**. Proposed fixes are at the end; none implemented.

## 1. Raw inputs at each decision step

All time quantities are first divided by `pt_upper_bound` = the maximum processing time over the WHOLE environment
batch (`environment_utils.normalize_processing_times`; 99 for every training batch measured). Operation and machine
features are then standardised per instance, per feature, across the not-yet-deleted operations / machines
(`norm_op_features`, `norm_machine_features`: (x - mean) / std). Pair features get NO further normalisation and are
concatenated raw into the actor input (`main_model.py:165-168`).

### Operation features (`fea_j`, 11 = DANIEL's 10 + 1)

| # | name | formula (per operation o of job j) | dynamic? | vs original DANIEL |
|---|---|---|---|---|
| 1 | scheduled flag | 1 if o scheduled | yes | same |
| 2 | completion-time lower bound | op_ct_lb: actual (scaled) completion if scheduled, else job-ready + cumulative min pt | yes | same |
| 3 | min pt | min over eligible machines | no | same |
| 4 | pt span | max - min eligible pt | no | same |
| 5 | mean pt | mean over eligible machines | no | same |
| 6 | waiting time | max(0, next schedule time - job-ready time) for candidates | yes | same |
| 7 | remaining work | max(0, op_ct - next schedule time) | yes | same |
| 8 | job ops left | # unscheduled ops of j | yes | same |
| 9 | job remaining work | sum of mean pt of j's unscheduled ops | yes | same |
| 10 | available machines | # eligible machines | no | same |
| 11 | **operation priority** | q_o from the instance file (retired priority objective) | no | **ADDED** (`enable_priority or enable_carbon`) |

Feature 11 is unrelated to due dates: Spearman(job mean priority, d_j/P_j) = 0.008 (p = 0.80, 1,000 pool jobs).

### Machine features (`fea_m`, 8 = DANIEL's 8, unchanged)

| # | name | formula (per machine m) | dynamic? |
|---|---|---|---|
| 1 | available candidate jobs | # candidate ops m can process now | yes |
| 2 | available ops | # unscheduled ops m can process | yes |
| 3 | min pt | min pt over ops m can process | no |
| 4 | mean pt | mean pt over ops m can process | no |
| 5 | waiting (idle) time | next schedule time - machine free time, if idle | yes |
| 6 | remaining work | max(0, machine free time - next schedule time) | yes |
| 7 | free time | time m becomes free | yes |
| 8 | working flag | 1 if busy at next schedule time | yes |

### Pair features (`fea_pairs`, 9 = DANIEL's 8 + 1), per (candidate op of job j, machine m); not standardised

| # | name | formula | dynamic? | observed range (train, C1@1000) | vs original |
|---|---|---|---|---|---|
| 1 | pt | pt[o, m] (scaled) | yes (candidate changes) | 0.01-1.00 | same |
| 2 | pt / op max pt | | yes | 0.01-1.00 | same |
| 3 | pt / machine max candidate pt | | yes | 0.01-1.00 | same |
| 4 | pt / max remaining pt | | yes | 0.01-1.00 | same |
| 5 | pt / machine max remaining pt | | yes | 0.01-1.00 | same |
| 6 | pt / max candidate pt | | yes | 0.01-1.00 | same |
| 7 | pt / job remaining work | | yes | 0.003-3.05 | same |
| 8 | wait time | op waiting + machine waiting | yes | 0-3.84 | same |
| 9 | **carbon** | carbon[o, m], **raw integer** | yes (candidate changes; static per pair) | **1-100, mean 50.9, sd 28.9** | **ADDED** (`carbon_feature`) |

Original DANIEL: fea_j_input_dim 10, fea_m_input_dim 8, pair 8. Ours: 11 / 8 / 9.

## 2. Parameter difference, exactly

Instantiating `DANIEL` with each input configuration (everything else as ours):

| (op, machine, pair) dims | total | feature extractor | actor | critic |
|---|---|---|---|---|
| (10, 8, 8) published default | 28,834 | 16,672 | 6,849 | 5,313 |
| (11, 8, 8) | 29,090 | 16,928 | 6,849 | 5,313 |
| (10, 8, 9) | 28,898 | 16,672 | 6,913 | 5,313 |
| **(11, 8, 9) ours** | **29,154** | **16,928** | **6,913** | **5,313** |

* **+256 = ONE extra operation feature, used twice.** Operation features enter the first operation-attention block
  (4 heads x W[11 x 32] -> +4 x 32 = 128) AND the first machine-attention block as edge features
  (4 heads x W_edge[11 x 32] -> +128). Not two extra operation features.
* **+64 = one extra pair feature** (carbon): the actor's first layer has hidden size 64.

## 3. Explicit answers

* **(a) Is d_j, or any function of it, visible at every step? NO.** `initialize_due_dates` stores the due dates only
  for the reward (`compute_tardiness_score`); no feature builder reads `due_dates` / `true_due_dates`. The only
  job-level time information is remaining work (op feature 9) and the completion-time lower bound (op feature 2).
  Due dates are d_j = k * u_j * P_j with u_j ~ U[0.7, 1.3] drawn per job: the policy can see P_j (remaining work) but
  the per-job tightness factor u_j, which decides which jobs are urgent, is nowhere in the state.
* **(b) Is the carbon of each (operation, machine) pair visible? YES**, for every currently eligible candidate pair,
  as pair feature 9; but raw (1-100), unlike every other input (see verdict item 3). Carbon of FUTURE operations is
  not visible (same as processing times of future operations in the pair features; op features 3-5 give only their
  processing-time summaries).
* **(c) Is the due date scaled consistently with the processing-time scaling? In the REWARD yes; as an input it does
  not exist.** `due_dates = true_due_dates / pt_upper_bound`, the same batch-max divisor as `op_ct_lb`, so the
  shaping potential compares like with like (verified below: exact to 2e-14). Caveat: the divisor is the batch
  maximum, so the reward scale of an instance depends on its batch mates (moot here: every 10x5 training batch had
  max pt 99).

## 4. Reward composition

Per step (`objectives.ObjectiveSpec.step_reward`, goal `mct`, Step 4 weights):

    r_t = [max LB_{t-1} - max LB_t]            (makespan, scaled time; LB = op_ct_lb / batch-max pt)
        - 0.002384 * carbon(chosen o, m)       (carbon, RAW units)
        + 5.815362 * [Phi_{t-1} - Phi_t]      (tardiness, Phi = sum_j max(0, LB_(last op of j) - d_j/UB), Phi_0 = 0)

Episode sums: makespan term = (LB_0 - MK)/UB (relative to the initial lower bound, so part of it is a
policy-independent constant); carbon term = -0.002384 * total carbon; tardiness term = -5.815362 * T/UB. The weights
come from the Step 4 closed-form anchors at preference (1/3, 1/3, 1/3) with the tardiness anchor multiplied by the
calibrated factor (`results/step4_and_20x10_sensitivity.md`).

Measured on the 10x5 TRAIN pool (63 instances; 5 batches of 20 drawn with replacement as training does,
default_rng(20260929); 4 seeds x 100 episodes = 400 per mode), C1@1000:

| per-episode term (after weighting) | sampled actions: mean abs value | sd across episodes | greedy: mean abs value | sd |
|---|---|---|---|---|
| makespan (LB_0 - MK)/UB | 2.10 | 0.44 | 2.01 | 0.42 |
| (for reference: full -MK/UB) | 4.51 | 0.60 | 4.42 | 0.57 |
| carbon -w_c * CB | 5.67 | 0.50 | 5.69 | 0.51 |
| tardiness -w_t * T/UB | **16.06** | **10.06** | 14.83 | 8.92 |
| total | 23.83 | 10.30 | 22.54 | 9.10 |

Ratios (sampled, mean abs value): tardiness / makespan = **7.7x** (**> 5x: FLAG**); tardiness / carbon = 2.8x;
carbon / makespan = 2.7x. Even against the full -MK/UB, tardiness is 3.6x. By spread across episodes, tardiness is 23x
makespan and 20x carbon: the return's variation, and so the advantage signal PPO learns from, is almost entirely
tardiness. (Raw means on these rollouts: makespan 446, carbon 2378, tardiness 273; 0.5% of episodes had T = 0.)

**Shaping identity: exact.** Summing the per-step tardiness terms gives -w_t * T/UB with max |error| **2.1e-14** over
all 800 episodes, and the three recorded terms sum to the environment's own reward to 1.4e-14. The potential-based
shaping sums to -T (scaled) as intended.

## 5. Behavioural check: which jobs are late? (49 HELD-OUT instances x 4 seeds, 1,960 jobs per schedule)

Full per-job table (seed, instance, job, d_j, P_j, completion, tardiness, for C1@1000 greedy and CP-SAT 60 s best):
`results/input_reward_audit/jobs.csv`. Tightness = d_j / P_j = k * u_j (1.25 x U[0.7, 1.3]; range 0.875-1.62).

| | C1@1000 greedy | CP-SAT 60 s best |
|---|---|---|
| share of jobs late | 46.0% | 19.3% |
| mean tardiness per job | 32.7 | 5.1 |
| Spearman(tightness, tardiness), pooled | -0.60 | -0.25 |
| same, per instance: median (share < 0) | -0.59 (98%) | -0.29 (75%) |
| **Spearman(tightness, completion / work content)** | **-0.02** | **+0.74** |
| mean tightness of late / on-time jobs | 1.14 / 1.38 | 1.16 / 1.29 |

By tightness quartile (pooled):

| quartile of d_j/P_j | policy: late | policy: mean T_j | policy: C_j/P_j | CP-SAT: late | CP-SAT: mean T_j | CP-SAT: C_j/P_j |
|---|---|---|---|---|---|---|
| Q1 tightest | 82% | 73.6 | 1.23 | 28% | 8.7 | 0.94 |
| Q2 | 61% | 41.9 | 1.26 | 30% | 9.4 | 1.14 |
| Q3 | 29% | 10.6 | 1.19 | 11% | 1.3 | 1.26 |
| Q4 loosest | 12% | 4.6 | 1.23 | 8% | 0.9 | 1.36 |

**Reading.** Yes, the policy's late jobs are the tightest ones (rho = -0.60), but for the wrong reason: it finishes
every job at about 1.2x its work content regardless of due date (rho = -0.02), so lateness falls mechanically on the
jobs whose due dates happen to be tight. CP-SAT does what a due-date-aware scheduler does: it pulls tight jobs forward
(0.94x work content) and lets loose ones finish later (1.36x), which is why its tardiness correlation is weaker and
its tardiness six times lower. This is exactly the behaviour a policy blind to d_j cannot learn.

Example (seed 111, pool instance 0, jobs sorted by tightness):

| job | P_j | d_j | d_j/P_j | C_j policy | T_j policy | C_j CP-SAT | T_j CP-SAT |
|---|---|---|---|---|---|---|---|
| 5 | 238.3 | 221.1 | 0.928 | 189 | 0.0 | 186 | 0.0 |
| 2 | 254.2 | 241.8 | 0.951 | 362 | 120.2 | 210 | 0.0 |
| 7 | 277.2 | 304.3 | 1.098 | 378 | 73.7 | 329 | 24.7 |
| 8 | 240.9 | 285.8 | 1.187 | 435 | 149.2 | 245 | 0.0 |
| 1 | 259.7 | 324.0 | 1.247 | 194 | 0.0 | 324 | 0.0 |
| 6 | 248.0 | 334.3 | 1.348 | 364 | 29.7 | 343 | 8.7 |
| 0 | 315.8 | 429.9 | 1.361 | 462 | 32.1 | 397 | 0.0 |
| 4 | 246.6 | 390.8 | 1.585 | 287 | 0.0 | 396 | 5.2 |
| 9 | 259.3 | 416.7 | 1.607 | 503 | 86.3 | 420 | 3.3 |
| 3 | 295.6 | 474.9 | 1.607 | 288 | 0.0 | 403 | 0.0 |

## Proposed fixes (NOT implemented; each changes the feature schema or reward and so needs retraining of every
specialist and arm under a new schema tag, e.g. `canonical_f13_p10_v3`)

1. **Make due dates observable** (primary). Add per-operation job-level features, scaled by the same
   `pt_upper_bound` and then standardised like the other operation features:
   * slack_j = (d_j - LB_(last op of j)) / UB, i.e. due date minus the job's current completion-time lower bound
     (dynamic; negative = already projected late);
   * time-to-due_j = (d_j - next schedule time) / UB (dynamic), optionally divided by the job's remaining work
     (critical ratio).
   And one pair feature: the projected tardiness increase if this pair is chosen,
   max(0, start + pt[o, m] + remaining min-pt work of j - d_j) / UB. The retired priority feature (op feature 11)
   could be dropped at the same time, since no objective uses it any more.
2. **Rescale the reward to the evaluation score.** Replace the anchor-based weights with per-term normalisation by
   the same reference values the Step 4 score uses (makespan and carbon by the centroid reference, tardiness by its
   pool mean), each multiplied by the preference weight, and keep the potential-based form. The training return
   then equals -(score) up to a constant, and no term can dominate by construction. At minimum, pick w_t so that the
   episode-to-episode spread of the tardiness term is within 5x of the others.
3. **Normalise the carbon pair feature**, e.g. carbon[o, m] / max carbon in the instance (or batch, mirroring the pt
   scaling), so it lives in [0, 1] like its neighbours.
4. Optional, cheap diagnostic before any retraining: an EDD-style dispatching rule (earliest due date / minimum
   slack with the policy's machine choice) on HELD-OUT, to size how much of the 41% ceiling gap due-date awareness
   alone recovers.

## Part 2 (longer training): NOT RUN

The verdict is FAIL, so per the stop condition no `results/longtrain.md`, no pre-registration and no GPU training
were produced. Training C1 and scratch to 3,000 updates on inputs that cannot see due dates would measure a budget
effect on a policy that is structurally unable to learn due-date sequencing, the source of 56% of the ceiling gap.
