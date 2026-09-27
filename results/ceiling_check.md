# Performance ceiling on 10x5: how far is the ~0.87 plateau from the best achievable?

## Pre-registration

Written and committed BEFORE Part A (CP-SAT on HELD-OUT) and Part B (sampling) were run. Never edited afterwards;
changes go under "Deviations". No training of any kind. Reserved final-test sets (10x5, 20x10) are never loaded.

### Question
Every arm so far ends between 0.866 and 0.90 (Step 4 score, HELD-OUT, update 1000; best C1 = Step 4 makespan
specialist fine-tune, 0.8655). Is that near the best achievable on these instances (Part A, exact solver), and can
the C1 policy itself reach better schedules than its greedy choice (Part B, best-of-N sampling)?

### Fixed inputs (identical to Step 4 / blend_finetune)
* Instances: the HELD-OUT instances of blend_zero_training (`results/blend_zero_training/split.json`, same IDs).
  NOTE: that file lists **49** held-out instances (pool of 100 split per maximum-processing-time group into fit
  1+1+7+42 = 51 and held-out 49); earlier reports called it "the 50 HELD-OUT instances". Same IDs, n = 49.
* Score: Step 4, preference (1/3, 1/3, 1/3); per seed, makespan and carbon relative to that seed's centroid
  composition `simplex_m333c333t334_k125s060_s<seed>` per instance, tardiness relative to the centroid's mean over
  all 100 pool instances. Due dates: frozen manifest k125s060 (hash-checked). Carbon: the instance files' per
  operation-machine carbon values. Raw time units (no processing-time scaling enters any reported objective).
* Policy: C1@1000 = `s4t_spec_m_k125s060_s<seed>@u1000`, seeds 111/222/333/444.

### Step 0 (done before this pre-registration; must pass, else stop)
`tools/ceiling_scorer.py` re-computes makespan, carbon, tardiness and the Step 4 score from a complete schedule
(machine + start per operation) with its own .fjs parser, verifying feasibility (eligibility, precedence, machine
no-overlap). On C1@1000's greedy schedules (all 49 HELD-OUT x 4 seeds) it must match the evaluator's score to
< 1e-9 for every instance. Result: PASS, max |difference| = 2.2e-16 (details below). The greedy rollout also
reproduced the Step 4 result files bit-for-bit.

### Design probe (done before this pre-registration, FIT instances only, no hint)
CP-SAT at 60 s / 8 workers on 3 FIT instances (pool 1, 6, 8; seed 111 objective): all FEASIBLE, none proven
optimal, scores 0.574 / 0.650 / 0.577, proven bounds 0.487 / 0.498 / 0.470. This sets the 600 s check's size.

### Part A: exact solver (CPU), `tools/ceiling_cpsat.py`
* Model: FJSP with job precedence, exactly one eligible machine per operation (optional intervals), NoOverlap per
  machine; makespan = max of job completions; tardiness_j >= Q*C_j - ceil(Q*d_j), >= 0, Q = 1000; carbon = sum
  over chosen operation-machine pairs. Objective = the Step 4 score with integer coefficients
  floor(S * w / normaliser), S = 1e12. Coefficients rounded down and due dates rounded up, so the integer
  objective never exceeds S x the exact score: the proven bound / S is a valid lower bound on the exact score.
* The score normalisers are seed-specific, so the objective is solved **per (seed, instance)**: 4 x 49 = 196 solves,
  60 s each, 8 workers, fixed random_seed 20260928, warm-started with that seed's C1@1000 greedy schedule as a
  complete hint. Recorded: best solution, proven bound, status (OPTIMAL / FEASIBLE), wall time.
* Every solution is re-scored with the Step 0 scorer; that exact score is reported (not the integer objective).
* Hint check: the re-scored solution must never be worse than the hint's score (tolerance 1e-6, the coefficient
  rounding); every violation is reported.
* 600 s check: 10 HELD-OUT instances drawn with numpy default_rng(20260928).choice(held, 10, replace=False),
  instance k of the draw paired with seed [111, 222, 333, 444][k mod 4] (CPU cost ~100 min instead of ~6.7 h for
  all seeds); same model, hint and parameters, 600 s. Reported: change in best score and in proven bound vs 60 s.
  The reference "best CP-SAT score" for the gap metrics is the 60 s solution for every (seed, instance), so all
  pairs are treated alike; the 600 s movement is reported alongside.
* Reachability check (secondary): each CP-SAT schedule is replayed through DANIEL's append rule (operations in
  order of CP-SAT start time; start = max(job ready, machine free)). All objectives are regular, so the replayed
  score must be <= the CP-SAT score, i.e. the ceiling lies inside DANIEL's action space. Reported.

### Part B: best-of-N sampling (GPU), `tools/ceiling_sampling.py`
* C1@1000 per seed: 100 stochastic rollouts per HELD-OUT instance, sampling from the policy's own softmax (no
  temperature change), torch generator seeded 20260928 + seed. Best schedule per instance by the Step 4 score
  (re-scored with the Step 0 scorer; scorer vs environment must agree). Best-of-10 and best-of-30 = the first 10 / 30
  of the same samples.

### Metrics (per seed and pooled over the 196 seed x instance pairs)
* Gap to the solver = (policy score - CP-SAT 60 s score) / CP-SAT 60 s score, per pair, for C1@1000 greedy and
  best-of-10/30/100. Pooled mean, median, and a 95% bootstrap CI (10,000 resamples of the 49 instances, each
  resampled instance carrying its 4 seeds; generator seed 20260928).
* Solver gap = (CP-SAT best - proven bound) / CP-SAT best per pair; share of pairs with status OPTIMAL.
* Raw objectives (makespan, carbon, tardiness): CP-SAT best vs C1@1000 greedy (and best-of-100), pooled means.

### Decision rule (applied to the pooled greedy and best-of-100 gaps)
* **SATURATED**: greedy pooled mean gap <= 2% AND its CI upper end <= 3% -> 10x5 cannot separate methods; further
  method comparisons need harder instances.
* **DECODING-LIMITED**: greedy gap > 2% but best-of-100 within 2% (pooled mean gap <= 2%, CI upper <= 3%) -> the
  policy can reach near-optimal schedules but its greedy choice misses them.
* **POLICY-OR-BUDGET-LIMITED**: greedy and best-of-100 gaps both > 2% -> the limit is the DANIEL policy class or the
  1,000-update training budget; separating the two needs the longer-training run, which is not part of this task.
* (Cases not listed above, e.g. greedy mean <= 2% but CI upper > 3%, are reported as "not decided" with the numbers.)
* If fewer than 80% of pairs are OPTIMAL, the ceiling is a best-known solution, not a proven optimum; the
  proven-bound gap is reported alongside every solver gap.
