# Independent implementation audit (v1 and v2)

Read-only for every existing code path: nothing in `daniel/` or the existing `tools/` was changed. New, evaluation-only
code: `tests/audit/` (tests) and `tools/audit/` (diagnostics); raw outputs in `results/code_audit/`. No training. The
reserved final-test sets were never loaded; in addition, the 19 paper-test instances that our split manifest labels
`final_test` (`daniel/data/manifests/split_seed_20260831_10x5+mix.json`) were deleted from the scratch clone of the
original repository before anything ran, and the original repository's saved per-instance result and OR-Tools files
(which cover those instances) were deleted unread.

## Verdict: NO MATERIAL IMPLEMENTATION BUGS FOUND, but two method-level findings change how the results read

The code does what it claims. The environment produces only feasible schedules; the rewards sum exactly to the
intended objectives; PPO's stored and recomputed log-probabilities agree; warm starts load every parameter exactly;
evaluation is deterministic and identical across arms. Our environment and model reproduce the original DANIEL code
**bit-for-bit (0 of 130 instances differ)** once one deliberate feature correction is undone, and our own v1 makespan
specialists reach the paper's SD2 10x5 numbers (greedy 411.5 vs the original model's 405.7 on the same instances;
sampling 364.7 vs the paper's 366.7).

What the audit did find is not a coding error but two properties that bear directly on the plateau and on the
ceiling-check conclusions:

1. **DANIEL's action mask only allows non-delay schedules, and none of the CP-SAT best solutions is non-delay**
   (A1). An operation can only be started at the earliest possible start time; deliberately waiting for a cleaner or
   better-placed machine is impossible. 0 of 49 CP-SAT schedules are reachable. The best of 3,000 schedules we could
   find INSIDE the action space scores 0.702 against CP-SAT's 0.614, so up to ~14% of the gap is the action space,
   and at least ~22% (greedy C1 0.856 vs 0.702) is the policy. `results/ceiling_check.md`'s statement that the
   CP-SAT schedules "lie inside DANIEL's action space" is **wrong** (its replay ignored the mask).
2. **In v1, the critic's value loss swamps the policy gradient in the shared feature extractor** (A2): 280-740 times
   larger at Step 4's reward scale, against about 0.9 in DANIEL's own makespan-only regime. Every v1 three-objective
   training result (Step 4, blend fine-tuning, C1) was trained in this regime. v2's reward
   rescaling removes it (ratio 0.02-0.10, critic explained variance about 0.8), and v2 still plateaus, so this is not
   what caps performance, but it is a confound for any v1 comparison.

The plateau (0.81-0.88) is therefore most likely DANIEL's genuine limit in this setting: a non-delay action space
that excludes the solver's schedules, plus a greedy policy that does not find the best schedules even within that
space (its own best-of-500 samples reach 0.72-0.74; DANIEL's paper shows the same pattern for makespan alone:
greedy 25% and sampling 12% above OR-Tools). Must-do before more runs: correct the ceiling-check reachability claim
(A1), and decide whether the study should keep DANIEL's non-delay action space; recommended: log training health
(A8), evaluate a validation-selected checkpoint rather than the last iterate (A7), and use a fixed time normaliser
(A4). Details below.

## Part 1: reproducing the published baseline

Data provenance (established in this audit): our `daniel/data/SD2/10x5+mix` is the paper's own SD2 10x5 test set
(identical instances; whitespace differs), our train (63) and validation (18) splits are taken from it, and our
`data_train_vali` pool (incl. the 49 HELD-OUT) is the paper's validation set (`FJSP-DRL/data/data_train_vali`).

| model / code | instances | greedy makespan | sampling (best of 100) |
|---|---|---|---|
| Paper, Table I (DANIEL) | SD2 10x5 test, 100 | 408.40 | 366.74 |
| Original checkpoint, original code | paper test minus the 19 manifest final_test = 81 | **407.68** | **365.60** |
| Original checkpoint, original code | HELD-OUT 49 (paper validation set) | 405.71 | - |
| Original checkpoint, OUR code (legacy 10/8 schema) | 81 / HELD-OUT 49 | 404.94 / 404.06 (22/81 and 13/49 instances differ) | - |
| Original checkpoint, OUR code with the original machine feature emulated | 81 / HELD-OUT 49 | 407.68 / 405.71 (**0/81 and 0/49 differ**) | - |
| OUR v1 makespan specialists m_s111/222/333/444 | HELD-OUT 49 | 404.69 / 412.00 / 426.96 / 402.47 (mean **411.5**, +1.4% vs original model) | 364.24 / 365.41 / 367.67 / 361.65 (mean **364.7**) |
| OUR v1 makespan specialists | 81 paper-test instances (63 of them are our TRAINING instances) | 405.38 / 413.26 / 416.27 / 403.64 (mean 409.6) | 365.11 / 367.37 / 370.21 / 364.04 |

Within 5% of the paper: **yes** (greedy +0.8% vs paper, +1.4% vs the original model on identical instances; sampling
-0.5%). Seed 333 is the weakest (+5.2% on HELD-OUT). The 22-instance divergence of the original checkpoint in our code
is fully explained by one deliberate change (A5). OR-Tools 326.24 was not re-run (it would require the paper's full
test set, which contains reserved instances).

### Settings: ours vs DANIEL's defaults

| setting | DANIEL (FJSP-DRL params.py / train.py) | ours (Step 4, v2 runs) |
|---|---|---|
| PPO updates | 1000 | 1000 (Step 4), 3000 (v2 re-run) |
| environments per update | 20 | 20 |
| training instances | **fresh SD2 instances generated every 20 updates (~1,000 distinct per run)** | **fixed pool of 63 files, 20 drawn with replacement every 20 updates** (`params.py:127`, `train.py:357-363`) |
| validation set (checkpoint selection) | 100 instances (`data_train_vali`) | 18 instances (our validation split); fixed-budget snapshots in Step 4 / v2 re-run, no selection |
| learning rate / schedule | 3e-4, constant | 3e-4, constant |
| gamma / GAE lambda | 1 / 0.98 | 1 / 0.98 |
| PPO clip / epochs / minibatch | 0.2 / 4 / 1024 (one full batch of 1,000 transitions, no shuffling) | identical |
| entropy coef / value-loss coef | 0.01 / 0.5 | 0.01 / 0.5 |
| gradient clipping, value clipping | none / none | none / none |
| advantage normalisation | per environment over time (`PPO.py:121`) | identical |
| reward / value-target normalisation | none (makespan reward in batch-max time units) | none in v1 (tardiness weight 5.8, carbon 0.0024); v2: reward = -score |
| policy_old update | hard copy (tau 0) | identical |

### Diff against the original DANIEL code (github.com/wrqccc/FJSP-DRL, commit 2cf81b1)

| file | differences | intentional? |
|---|---|---|
| `model/PPO.py`, `model/attention_layer.py`, `model/sub_layers.py` | none (byte-identical apart from whitespace) | - |
| `model/main_model.py` | `forward` split into `forward_logits` + softmax; configurable pair dimension | intentional refactor; behaviour identical (0/130 test above) |
| `fjsp_env_same_op_nums.py` | priority, carbon, tardiness, v2 features and rewards; processing-time normalisation moved to `environment_utils.normalize_processing_times` (divides by the max; the original's min-max subtracts np.min over a matrix with zeros, i.e. 0, plus 1e-8); `candidate_incompatibility` helper (equivalent); **machine feature 3 = true minimum processing time, the original computes `np.max` under the name `mch_min_pt`** | additions intentional; the machine-feature correction is deliberate but undocumented in the results (A5) |
| `train.py` | **file-based training pool** (A3), due dates and v2 reward references, budget snapshots, per-objective selection metric, run manifests; validation on our 18-instance split | due dates/refs/manifests intentional; the file pool is a deliberate leakage-avoidance choice but a protocol change |
| `params.py` | new flags (objectives, carbon, priority, tardiness, v2, schemas, `train_from_files`); every shared PPO/model default identical | intentional |
| `data_utils.py`, `common_utils.py` | carbon/priority parser and generator, split utilities; versioned checkpoint loading with schema checks | intentional |

## Findings

| ID | component | file:line | what is wrong | evidence | severity | affects | proposed fix (not implemented) |
|---|---|---|---|---|---|---|---|
| A1 | environment, action masking | `daniel/fjsp_env_same_op_nums.py:588-592` (original `:424-426`) | Pairs whose start max(job ready, machine free) exceeds the earliest start over all candidates are masked: only NON-DELAY schedules are reachable. Optimal multi-objective schedules need delays (wait for a low-carbon machine). Inherited from DANIEL. | `tools/audit/action_space_reachability.py`: 0/49 CP-SAT schedules reachable. `tools/audit/nondelay_search.py`: best of 3,000 masked rollouts per instance 0.702 vs CP-SAT 0.614 (+14.2%); C1 greedy 0.856. | MAJOR | ceiling_check's reachability claim and its "policy-or-budget" reading; any claim about closing the gap to CP-SAT | Correct ceiling_check (see A10). If CP-SAT-level schedules are the goal, add delay actions (e.g. allow pairs starting up to a window after the earliest start, or an explicit "wait" action), and report a non-delay-restricted ceiling. |
| A2 | PPO x reward scale (v1) | `daniel/model/main_model.py:200-201` (critic on shared embeddings, no stop-gradient), `daniel/model/PPO.py:199,207` (raw-return MSE, vloss 0.5), Step 4 weights | With Step 4's v1 rewards, the value-loss gradient on the shared feature extractor is 280x (C1@0-1000) to 740-8,000x (scratch) the policy-loss gradient; critic explained variance -4 to +0.62. DANIEL's makespan-only regime: ratio 0.88, EV 0.74. | `tools/audit/training_health.py` (table below) | MAJOR | all v1 three-objective training: Step 4 arms, blend fine-tune, C1 (ceiling_check); the v1 tardiness specialists (reward -T/U) were not measured | Already removed by v2's reward scale. For any v1 re-use: normalise value targets (running return scale) or detach the critic input / give the critic its own encoder. |
| A3 | training data | `daniel/params.py:127`, `daniel/train.py:357-363` | Training draws from a fixed pool of 63 instances (63 of the paper's own test set) instead of DANIEL's fresh generator. | `tools/audit/overfit_check.py`: TRAIN and HELD-OUT scores move together; HELD-OUT minus TRAIN only +0.0 to +3.0% (V2-M: +0.07% at u0, +2.2% at u3000). Specialists still match the paper. | MINOR (measured); protocol deviation | all trained arms; any comparison with the paper's published test numbers (our models trained on 63 of those instances) | Train from the SD2 generator (`train_from_files False`) with the generator seeded, keeping the file pools for validation/evaluation only. |
| A4 | environment scaling, train vs eval | `daniel/environment_utils.py:16`, `daniel/fjsp_env_same_op_nums.py:163` | Times are scaled by the BATCH maximum processing time. Every training batch of 20 has max 99 (100% of 10,000 simulated draws), so training never sees another scale; canonical evaluation scales each instance by its own max (96-99). | `tests/audit` t4 and `tools/audit/scale_mismatch.py`: 8/49 HELD-OUT instances affected; 0-4 of them change decisions; HELD-OUT mean changes by <= 0.5% (largest V2-M@3000 0.8733 vs 0.8691 "as trained"). | MINOR | all evaluations (tiny) | Use a fixed normaliser (the generator maximum, 99) in training and evaluation; then features never depend on batch companions. |
| A5 | environment, machine feature 3 | `daniel/fjsp_env_same_op_nums.py:199` vs original `:178` | We compute the true minimum processing time per machine; the original computes `np.max` under the name `mch_min_pt`. Deliberate correction, consistent in our training and evaluation, but undocumented in the results. | original checkpoint in our code: 22/81 instances differ (404.94 vs 407.68); with the original's max emulated: 0/130 differ. | MINOR | only cross-use of original DANIEL checkpoints | Document it; keep it (our checkpoints were trained with the corrected feature). |
| A6 | PPO, GAE | `daniel/model/PPO.py:105-112` | GAE ignores the stored done flags (bootstraps through any episode end). Correct here only because every environment ends at the same final step and the buffer holds one episode. Inherited. | `tests/audit/test_ppo_reward.py` t3: matches a reference to 2e-7 when episodes end at the buffer end; differs by 2.2 when an episode ends mid-buffer (never produced; t1 asserts all environments finish together). | MINOR (latent) | none at present; would break the SD1/various-op-count environment | Use the done mask in the recursion. |
| A7 | evaluation protocol | `tools/run_v2_rerun.py`, `tools/run_arms_t.py` (fixed-budget snapshots) | Fixed-budget LAST-ITERATE evaluation of a policy that keeps moving (KL 0.3-1.3 nats/decision between snapshots, constant lr, no annealing), whereas DANIEL keeps the best-validation checkpoint. "Gets worse with more updates" partly reflects which oscillation phase the snapshot lands on. | `eval_noise.json`: V2-M consecutive HELD-OUT changes of -5.5% to +3.6%, 4/9 beyond 2 SE, correlated +0.95 with the same changes on the TRAIN pool (real policy changes, not evaluation noise); V2-T mostly noise (1/9, corr -0.18). | MAJOR for the interpretation of v2 re-run Q2 ("worse at 3000") | v2 re-run Q2/Q3; Step 4 budget curves | Report a validation-selected checkpoint (never HELD-OUT) and/or the mean of the last k snapshots; consider lr annealing. |
| A8 | logging | `daniel/train.py:278`; `tools/run_v2_rerun.py` | Only the mean training return (and validation metrics) is logged; entropy, KL, clip fraction, value loss, explained variance and gradient norms are never logged; the v2 re-run logs nothing per update. | Part 4 had to reconstruct them from snapshots; clip fraction cannot be reconstructed. | MINOR | diagnosability of every run | Log these per update. |
| A9 | PPO advantage normalisation | `daniel/model/PPO.py:121` | Advantages are standardised per environment over time, removing each episode's mean advantage, so the policy learns relative credit within an episode rather than "this episode beat the critic". Inherited from DANIEL. | code reading; DANIEL reaches its published numbers with it | MINOR (design note) | all training | Optional: normalise over the whole batch. |
| A10 | ceiling_check analysis | `tools/analyze_ceiling.py:29` (`replay`), `results/ceiling_check.md:135` | The reachability check replays CP-SAT schedules with the append rule only, ignoring the non-delay mask, and concludes the schedules "lie inside DANIEL's action space". False (A1). | 0/49 reachable under the real mask | MAJOR (one supporting claim invalid; the pre-registered gap numbers stand) | ceiling_check interpretation; input_reward_audit's framing of the gap | Add a correction under Deviations in ceiling_check; report the non-delay search result (0.702) next to CP-SAT. |
| A11 | v1 reward weights | Step 4 `--carbon_reward_weight 0.002384 --tardiness_reward_weight 5.815362` | v1 three-objective weights do not match the Step 4 score's (tardiness about 8x relative); known from input_reward_audit, fixed in v2. Listed for completeness. | `tests/audit` t5 (v1 sums are exact for the stated weights) | MINOR (known) | v1 three-objective results | Done in v2. |

Checked and found correct (no finding): schedule feasibility and the append start rule; episode end exactly after
N_ops steps (`fjsp_env_same_op_nums.py:647`); invalid actions get probability exactly 0 and greedy argmax never picks
them; masked/finished operations are excluded from the per-instance standardisation; training and HELD-OUT instances
come from the same SD2 10x5 distribution (pt 1-99, carbon 1-100, same due-date manifest k125s060), with **0
structural duplicates** between train/validation and the 100-instance evaluation pool; seeding (training bit-
deterministic, earlier verified); every v1 and v2 reward sums exactly to its objective; log-probabilities, ratios,
entropy over valid actions, value targets; `load_state_dict` is strict, warm starts load all 52 tensors into policy
and policy_old including the critic, the optimiser starts empty (as intended; V1-CONT's recreation is documented);
the same 49 HELD-OUT IDs and per-seed Step 4 references everywhere; no checkpoint selection on HELD-OUT (merge
initialisations and specialists were selected on the 18-instance validation split, blend weights on the FIT half);
the merge is the exact weighted average of all 52 float tensors (max error 0.0, critic included); the blend
renormalises over valid actions only, F1's specialists are trainable, F2's are frozen (its optimiser holds only the
residual and the fresh critic; `tools/run_blend_finetune.py:149-170`), the residual's last layer is zero-initialised
(`:128-130`), and the warm-up freezes every non-critic parameter (`:191-196`).

## Part 3: test results (`tests/audit/`, run as `python tests/audit/<file>.py`)

| # | test | result |
|---|---|---|
| 1 | feasibility: 1,000 random-policy episodes (v1 and v2 envs) + greedy schedules of 4 checkpoints (m_s111, C1@1000, V2-M@3000, V2-S@3000) on HELD-OUT; independent validator incl. eligibility, precedence, machine overlap, append start rule; objectives re-computed | **PASS**: 0 violations in 1,196 schedules |
| 2 | log-prob consistency, rollout path vs update path (1,000 transitions each, v1 and v2) | **PASS**: max abs diff 6.0e-6 (v1) / 3.7e-6 (v2); max abs ratio-1 6e-6; probability on masked actions 0.0 |
| 3 | GAE vs a reference implementation | **PASS** for the case training produces (2e-7); documents that done flags are ignored (A6) |
| 4 | batch invariance, 10 HELD-OUT instances, v1 (C1@1000) and v2 (V2-M@3000), eval path and training batching | **PASS** with same-max companions (identical actions); **FAILS by design** with max-99 companions for instances whose own max < 99 (v1: 2/5, v2: 4/5 change), which is exactly what every training batch does (A4) |
| 5 | reward sums, 100 training-generator episodes each: v1 mct (Step 4 weights), v1 m, v1 t, v2 mct, v2 t | **PASS**: max abs error 1.4e-14 / 1.3e-15 / 3.6e-15 / 5.6e-16 / 5.6e-16 |
| 6 | warm-start load (m, c, t specialists, Step 4 merge init, C1@1000, v2 m and v2 tardiness specialist) through `train.Trainer` | **PASS**: 52/52 tensors equal in policy and policy_old, no missing/unexpected keys, critic loaded |
| 7 | determinism, 4 checkpoints, two runs | **PASS**: max abs difference 0.0 |

Also: the existing suite (`python -m unittest discover -s tests`, 26 tests incl. the v1 reproduction test) still
passes; no existing file was modified.

## Part 4: training health (reconstructed from snapshots, seed 111; `tools/audit/training_health.py`)

Not logged by any run: entropy, KL, clip fraction, value loss, explained variance, gradient norm (A8). Reconstructed
on 100 training-generator episodes per snapshot under each arm's own reward, without optimiser steps:

| arm, update | entropy (nats) | critic EV | mean abs step reward | sd of value targets | grad norm on shared extractor: value / policy (ratio) |
|---|---|---|---|---|---|
| Step 4 makespan arm (v1), u0 | 0.21 | -2.81 | 0.64 | 13.5 | 37.8 / 0.134 (**282x**) |
| Step 4 makespan arm (v1), u1000 | 0.19 | +0.62 | 0.53 | 11.3 | 58.3 / 0.190 (**307x**) |
| Step 4 scratch (v1), u1000 | 0.28 | +0.52 | 0.59 | 12.4 | 96.6 / 0.131 (**735x**) |
| m_s111 under DANIEL's makespan-only reward | 0.21 | +0.74 | 0.035 | 0.62 | 0.144 / 0.164 (0.88x) |
| V2-M u1000 / u3000 | 0.17 / 0.15 | +0.82 / +0.82 | 0.017 | 0.22 | 0.009 / 0.159 (0.05x) |
| V2-S u1000 / u3000 | 0.26 / 0.17 | +0.79 / +0.82 | 0.017 | 0.22 | 0.015 / 0.147 (0.10x) |
| V2-T u1000 / u3000 | 0.20 / 0.12 | +0.87 / +0.85 | 0.017 | 0.22 | 0.006 / 0.227 (0.03x) |

* **Entropy**: low from the start for warm-started policies (0.15-0.26 nats per decision, i.e. near-deterministic)
  and slowly falling in v2 (V2-T 0.26 -> 0.12, V2-M 0.21 -> 0.15). Scratch falls from 1.0 to 0.17 within 1,500
  updates. Low but not a sudden collapse.
* **KL between consecutive snapshots** (on the earlier snapshot's states): 0.2-1.3 nats per decision in v2 at every
  stage, i.e. the policy keeps moving substantially while its training return stays flat (V2-M -0.85 +- 0.02 from u50
  to u3000): it wanders among equally good policies.
* **Explained variance**: healthy in v2 (about 0.8); poor in v1 (negative for the first ~250 updates, 0.5-0.6 by 1000).
* **Training return vs HELD-OUT**: no case of training return improving while HELD-OUT worsens. In v2 both are flat,
  and TRAIN-pool and HELD-OUT greedy scores move together (`overfit_check.json`).
* **Reward scale**: v1 per-step rewards average 0.5-1.8 with value-target sd 11-32; v2 0.017 with sd 0.22. The v2
  rescaling does not stall the critic (EV 0.8 within 50 updates) or shrink the policy gradient (policy gradients
  0.15-0.35 on the extractor, similar to DANIEL's own regime; advantages are normalised anyway). It removes the v1
  value-loss dominance (A2).
* **Eval noise (V2-M)**: consecutive-snapshot HELD-OUT changes -5.5% to +3.6%, paired SE 1.2-1.8%; 4 of 9 exceed 2 SE
  and they track the TRAIN-pool changes (correlation +0.95), so the ~3% swings are real policy oscillation, not noise
  (A7). V2-T's swings are mostly within noise.

## Plain-English summary

We checked the code the way an outside reviewer would, and we did not find a programming error that explains the
plateau. When we load the original authors' trained model into our code, it makes exactly the same decisions as in
their code, apart from one feature we deliberately corrected, and our own makespan-only models score within about 1%
of the paper's published numbers. The schedules our system produces are always valid, the rewards add up exactly to
the objectives they are meant to measure, and the learning algorithm's internal bookkeeping is consistent. The audit
did find two things that matter. First, DANIEL only ever starts a job at the earliest moment something can start, so
it can never choose to wait for a cleaner or better machine; none of the solver's best schedules can be built that
way, which means part of the gap to the solver is built into the method, and our earlier claim that the solver's
schedules were reachable was wrong. Second, in our first-generation (v1) setup the critic's training signal drowned
out the scheduling signal inside the shared network; the v2 reward fixed that, yet v2 still plateaus. Overall, the
plateau looks like a genuine limit of DANIEL in this three-objective setting rather than a bug, and closing the gap
would need a change to the method itself, such as allowing deliberate waiting.
