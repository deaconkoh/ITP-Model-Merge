# Soft-weight segment oracle -- does changing the blend weight during a schedule beat the best fixed weight?

## Pre-registration

Written and committed BEFORE any experiment in this study was run. Not edited after results exist; later
changes go under "Deviations".

### Question
Is there value in a time- or state-dependent blend weight alpha_k over the schedule, beyond the best fixed
alpha? If not, a learned router alpha_t = g(s_t, w, pi_M, pi_C) has nothing to learn and the static logit
mixture is the answer.

### Setup (identical to the complementarity pilot, commit 8f51f80)
* 10x5 only; two frozen experts, the canonical makespan (M) and carbon (C) specialists; seeds 111/222/333/444,
  each seed using its own pair.
* Instances: the pilot's evaluation pool (the 100 train_vali instances). The reserved final-test sets are
  never loaded, evaluated or inspected.
* Preferences: w_c in {0, 0.05, ..., 1} (21 values), w_m = 1 - w_c.
* Normalisation and score exactly as in the pilot: per instance, each objective min-max normalised by the
  ideal and nadir observed across BOTH specialists and ALL 4 seeds on that instance (pool). Score =
  w_m * norm_makespan + w_c * norm_carbon, LOWER is better. Hypervolume reference point (1.1, 1.1),
  computed on pool-mean normalised points.
* Segments: 5 equal segments of the 50 decisions (steps 0-9, 10-19, 20-29, 30-39, 40-49).
* Greedy deterministic evaluation.
* Batching: only instances sharing the same maximum processing time are batched together (4 groups).
* Logit mixture: log p = alpha * log p_M + (1 - alpha) * log p_C over valid actions, renormalised
  (log_softmax); alpha is the weight on the MAKESPAN expert. The pilot parametrised the same family by the
  carbon weight w_c-mix = 1 - alpha; the arithmetic is kept identical to the pilot's, so fixed-alpha runs
  must reproduce the pilot's logit-mixture results exactly (checked before any arm is reported).

### Split
The 100 instances are split 50/50 into FIT and HELD-OUT with RNG seed 20260926, stratified by
maximum-processing-time group (within each group, half after a seeded shuffle; odd groups give the extra
instance to FIT). Instance IDs (file names) are recorded in `results/segment_oracle/split.json`.

### Arms (logit mixture throughout; G21 = {0, 0.05, ..., 1}, G101 = {0, 0.01, ..., 1})
* **A1** Best fixed alpha per (instance, w) from G21. The fair oracle control.
* **A2** Best fixed alpha per (instance, w) from G101. Matched-budget fixed control: more tries for the fixed
  arm, to separate "more tries" from "time-varying helps".
* **A3** Segment oracle per (instance, w): alpha_k per segment from G21, by coordinate descent. Start from A1's
  alpha; sweep segments 1 -> 5, try all 21 values for the current segment holding the others, keep the best
  (strict improvement only; ties keep the current value). Repeat passes until a full pass improves nothing or
  4 passes are done. Two restarts, from alpha = w_m (the matched mixture) and from a random schedule (seeded),
  keep the best. Evaluations used per (instance, w) are logged. Per-instance schedules are supported inside a
  batch.
* **A4** Deployable fixed alpha*: one alpha per (w, seed) from G21, chosen on FIT to minimise the mean score,
  evaluated on HELD-OUT.
* **A5** Deployable segment schedule: one 5-segment schedule per (w, seed), chosen on FIT by the same
  coordinate descent on the mean FIT score (start from A4's alpha*, same restarts), evaluated on HELD-OUT.
* **A6** Cross-seed transfer: the A5 schedule fitted on seed s applied to the specialists of seed s' != s,
  compared with seed s''s own A4 on HELD-OUT; all 12 ordered pairs.

### Metrics
* Score per (instance, w, seed), lower is better.
* HV across w per seed for A1 and A3 (all 100 instances, and on HELD-OUT) and for A4 and A5 (HELD-OUT).
* Per-instance headroom (A2 - A3) / A2 per (instance, w, seed). Operational definition: the mean over all
  (instance, w, seed) triples with A2 >= 0.02; triples with A2 < 0.02 (near the ideal point, where a relative
  gain is undefined or explosive) are excluded and their count reported. The ratio-of-means version
  (mean A2 - mean A3) / mean A2 is reported alongside. Also: % of (instance, w, seed) where A3 beats A2 by
  more than 0.5%.
* Deployable gain (A4 - A5) / A4 on HELD-OUT, as a ratio of means over held-out instances and all 21
  preferences: (mean A4 - mean A5) / mean A4. Paired bootstrap 95% CI resampling held-out INSTANCES (all
  preferences of an instance move together), 10,000 resamples, seed 20260926; per seed, and pooled (the same
  resampled instances applied to all four seeds). The mid-preference range w_c 0.3-0.7 is reported alongside.
* A6 gain vs the target seed's A4, the same way, per ordered pair and pooled.
* Schedule shape: mean alpha_k by segment for A3 (over instances and seeds) and A5 (over seeds), per w.
* Also: whether A2 closes most of the A3 - A1 gap (share = (A1 - A2) / (A1 - A3) on means).

### State-dependence check (uses A3 outputs only)
Target: the per-(instance, w, seed) A3 alpha_k. Features, measured at the start of segment k along the A3
rollout of that (instance, w, seed): expert top-choice disagreement rate and mean entropies of both experts
over the previous segment (segment 1: the first decision only); fraction of operations scheduled; remaining
work ratio (unscheduled mean-processing-time work / total); partial makespan relative to its initial lower
bound; partial carbon relative to the minimum-carbon cost of the operations scheduled so far; and w. No critic
values. Ridge regression (standardised features, penalty 1.0), 5-fold cross-validated R^2, per segment and
pooled; permutation null of 200 target shuffles, 95th percentile reported.

Operational definition for the decision rule (fixed now, because w alone determines most of alpha): the R^2
that counts is the one for the WITHIN-PREFERENCE part of alpha_k -- the target minus its mean over instances at
the same (w, seed), i.e. the part a fixed per-preference schedule cannot capture -- with the permutation null
shuffling targets within (w, seed) strata. The R^2 on the raw target with w included is reported as specified,
but it is not used for the decision.

### Decision rule
* **Outcome A** (time-varying blend helps and transfers): A5 beats A4 on HELD-OUT with a bootstrap CI above 0
  AND a mean gain >= 2%, in at least 3 of 4 seeds, AND A3 beats A2 by >= 3% mean per-instance.
  -> A learned router is justified; A5's fixed time schedule becomes an extra baseline the router must beat.
* **Outcome B** (headroom exists but only per instance): A3 beats A2 by >= 3%, but A5 fails the transfer
  criterion. -> Any value must come from state, not time. Proceed to a router only if the state-dependence
  R^2 beats the permutation-null 95th percentile in at least 2 segments; otherwise treat as Outcome C.
* **Outcome C** (no useful headroom): A3 beats A2 by < 3%. -> The static logit mixture is the answer; do not
  build a router.
* Also reported: whether A2 closes most of the A3 - A1 gap. If it does, the apparent gain was extra search,
  not time-variation.

### Cost rule
Estimate GPU time from the measured rollout cost before running; if over 6 GPU-hours, reduce the A3 restarts
to 1 and passes to 3 and log that as a deviation. Record actual GPU-minutes.

## Deviations

*(none yet -- this section is filled in when the study runs)*
