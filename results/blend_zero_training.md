# Zero-training three-specialist blend, scored on the Step 4 metric

## Pre-registration

Written and committed BEFORE any evaluation in this study. Never edited afterwards; changes go under
"Deviations".

### Question
Does recombining the frozen specialists' decisions, with NO training, match or beat fine-tuning from a single
specialist (Step 4)?

### Setup (identical to Step 4, results/step4_and_20x10_sensitivity.md)
* 10x5 only; seeds 111/222/333/444; each seed uses ONLY its own specialists: makespan `m_s<seed>`, carbon
  `c_s<seed>`, tardiness `t_k125s060_s<seed>`.
* Instances: Step 4's evaluation pool, the 100 train_vali instances. Due dates from the frozen manifest
  (tag k125s060), read and hash-checked, never re-derived. The reserved final-test sets are never loaded.
* Score (Step 4's, unchanged): preference w = (1/3, 1/3, 1/3); per seed, makespan and carbon are divided by
  that instance's value under the seed's centroid composition `simplex_m333c333t334_k125s060_s<seed>`, and
  tardiness by the centroid's pool mean over all 100 instances; score = mean of the three ratios weighted by w,
  LOWER is better. The reference values are Step 4's, computed on all 100 instances; HELD-OUT scores are those
  per-instance scores restricted to HELD-OUT instances.
* Greedy deterministic decoding. Batching only among instances sharing the same maximum processing time.
* **Equivalence check before any comparison:** every single-checkpoint rollout here (the vertices in B4, and
  all of R) must reproduce the existing Step 4 evaluation files exactly on all 100 instances (makespan, carbon,
  tardiness). If any differs, the study stops.

### Split
The 100 instances are split 50/50 into FIT and HELD-OUT with RNG seed 20260927, stratified by maximum-
processing-time group (seeded shuffle within each group; odd groups give the extra instance to FIT). IDs in
`results/blend_zero_training/split.json`. All headline comparisons are on HELD-OUT.

### Arms (experts M, C, T; weights sum to 1)
* **B1** Logit mixture at the preference: log p = (1/3) log p_M + (1/3) log p_C + (1/3) log p_T over valid
  actions (invalid pairs -inf in every expert, set explicitly, never 0 x -inf), renormalised with log_softmax,
  argmax. No tuning.
* **B2** Probability mixture at the preference: p = (1/3)(p_M + p_C + p_T), argmax.
* **B3** Logit mixture, tuned alpha: alpha from the 66-point simplex grid (step 0.1), chosen per seed by the
  lowest mean score on FIT (ties -> first grid point in lexicographic (m, c, t) order), evaluated on HELD-OUT.
  The chosen alpha is reported.
* **B4** Each specialist alone (the three grid vertices), reference points.
* **R** Re-score the saved Step 4 checkpoints for all five arms (merge, scratch, makespan spec, carbon spec,
  tardiness spec) at update 0 and update 1000, on HELD-OUT, with this study's evaluator. A missing checkpoint
  is reported as missing, never substituted.

### Metrics (per seed and pooled)
* Step 4 score of every arm on HELD-OUT, and mean raw makespan, carbon and tardiness per arm.
* Paired comparisons on HELD-OUT, each as a relative gain of arm X over reference Y:
  gain = (mean score of Y - mean score of X) / mean score of Y over the 50 held-out instances (positive = X
  better), with a paired bootstrap 95% CI from 10,000 resamples of held-out INSTANCES (seed 20260927). Pooled:
  the same resampled instances are applied to all four seeds and the means taken over seeds and instances.
  Mean paired per-instance score differences are reported alongside.
  Comparisons: B1 vs R[makespan spec @1000], B1 vs R[merge @0], B1 vs R[scratch @1000], the same three for B3,
  and B1 vs B2.

### Confidence / compromise analysis (from the B1 rollouts only)
For every decision: each expert's top-1 action and entropy (bits, over valid actions), and the action B1 takes.
Categories: all three experts share the top-1 (unanimous); the blend takes a top-1 shared by exactly two
experts; the blend takes the top-1 of exactly one expert (M, C or T); compromise (none of the experts' top-1s).
Reported overall and by schedule fifth (decisions 1-10, ..., 41-50).
* Among decisions where the blend follows exactly one expert whose top-1 is unique: the share where that
  expert is the most confident (lowest entropy), against a permutation null that shuffles the three entropies
  across experts within each decision (200 shuffles; 95th percentile reported).
* Among decisions where the experts' top-1s are not unanimous: the share resolved by compromise.

### Decision rule (pre-registered; effect size required)
Each of B1 and B3 is assessed on its own; either one can meet a criterion. "Beats Y at a seed" means a relative
gain >= 1% with the lower end of its 95% CI > 0 at that seed.
* **STRONG:** B1 or B3 beats R[makespan spec @1000] at >= 3 of 4 seeds.
  -> Zero-training recombination matches or beats 1000 PPO updates of fine-tuning. Proceed to fine-tuning the
  blend.
* **MEANINGFUL:** not STRONG, but B1 or B3 beats BOTH R[merge @0] and R[scratch @1000] at >= 3 of 4 seeds
  (both comparisons met at the same seed).
  -> The blend is a better starting point than any Step 4 initialisation. Proceed to fine-tuning the blend.
* **NOT MEANINGFUL:** neither holds. -> Stop and report; do not propose fine-tuning.

### Cost
Estimated before starting from the measured rollout cost: about 7,800 environment-rollouts per seed at about
2 ms each (three policies per step), roughly 3 GPU-minutes per seed and 12 in total. Actual GPU-minutes are
recorded.

## Deviations

*(none yet -- filled in when the study runs)*
