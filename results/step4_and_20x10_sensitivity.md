# Step A, Step 4 (10x5) and the 20x10 k = 1.2 sensitivity arm

Programme run 2026-09-24. Tardiness triple (makespan, carbon, total tardiness), due dates from the frozen
manifest (10x5 tag `k125s060`). Everything is evaluated on the n = 100 train_vali evaluation pool unless
stated. Objective = (1/3, 1/3, 1/3) scalarisation per seed, relative to that seed's centroid composition
(makespan and carbon per instance; tardiness by the centroid's pool mean); lower is better. Reserved
final-test sets untouched throughout.

**Status: A, B and C all COMPLETE.** No step stopped early; all sanity checks passed; every run finished
with no hangs or failures. The machine was shut down afterwards as instructed.

---

## Headline

1. **Merge vs scratch (H1): supported at low and middle budgets, converged by 1000 updates.** The merge
   arm beats training from scratch at every budget up to 500 updates at all four seeds (p <= 0.009,
   seeds as the unit), by 60% at budget 0, 41% at 100, 18% at 250 and 3% at 500. At 1000 updates the gap
   is +1.3% and no longer consistent (2/4 seeds, p = 0.35). Scratch does not reach the merge's
   zero-cost quality within 1000 updates when seeds are pooled.
2. **Merge vs the tardiness specialist alone, the key control (e): NO merge advantage.** Warm-starting
   from the tardiness specialist matches the merge at every budget (largest gap +1.9%, never significant
   with seeds as the unit, p = 0.09 to 0.93). **The finding is "warm-starting from a relevant specialist
   helps", not "merging helps".**
3. **The makespan specialist is the best starting point at longer budgets.** The merge is better at
   budget 0 (+2.1%, 4/4 seeds, p = 0.024), but fine-tuning from the makespan specialist overtakes it at
   all four seeds (pooled crossover at 100 updates) and ends significantly better at 1000 (-3.1% for the
   merge, 4/4 seeds, p = 0.001). It reaches the best final objective of any arm (0.863).
4. **Fine-tuning does not improve the merge at all under this reward.** The merge arm's objective is
   0.888 at budget 0 and 0.889 at 1000; it trades about 10 makespan for about 28 tardiness.

5. **20x10 sensitivity at k = 1.2: loosening the due dates does NOT rescue tardiness; it makes it
   worse.** Trained at the loosest k the pre-stated band allowed, the tardiness specialists lose to
   their makespan counterparts on tardiness by 10.4% on average (2/4 seeds significantly worse, 2 tied).
   The CPU screen had predicted the opposite (+30.6%). The smallest-k rule is therefore NOT what caused
   the pre-registered 20x10 Gate 1 failure, which stands as the primary result.

Caveat that applies to 2 and 3: two of the four validation-selected merges are 90% one specialist
(seed 333: 90% makespan; seed 444: 90% tardiness), so at those seeds the merge sits close to a single
specialist by construction.

---

## A -- merge-init composition selected on the VALIDATION split

268 evaluations (66 grid compositions + centroid, 4 seeds) on the 18-instance 10x5 validation split.
Rule: lowest mean objective on validation at (1/3, 1/3, 1/3). Tool: `tools/select_merge_init.py`;
selection: `results/step4/merge_init_selection.json`; output: `results/step4/stepA_selection.txt`.

**Deviation, recorded:** the literal rule (all 66 grid points) picked a VERTEX -- a single specialist, not
a merge -- at seeds 333 (100/0/0, the makespan specialist) and 444 (0/0/100, the tardiness specialist).
That would have made arm (a) the same checkpoint and seed as arm (c) or (e), so the merge-vs-specialist
comparison would be zero by construction. Arm (a) therefore uses the merge-only candidate set (63 points,
at least two specialists at non-zero weight). This is a definitional restriction, not a switch after an
unlucky draw. Recorded as evidence in its own right: **on validation, no merge beat the best single
specialist at 2 of 4 seeds.**

| seed | literal pick (66) | **merge used for arm (a)** | pool optimum | same? | pool gap | pick's rank on pool | not separable on validation (p > 0.05) |
|---|---|---|---|---|---|---|---|
| 111 | 50/10/40 | **50/10/40** | 0/10/90 | no | +0.67% | 4 of 63 | 18 of 62 |
| 222 | 10/10/80 | **10/10/80** | 20/10/70 | no | +1.28% | 11 of 63 | 20 of 62 |
| 333 | 100/0/0 (vertex) | **90/10/0** | 80/20/0 | no | +0.83% | 3 of 63 | 6 of 62 |
| 444 | 0/0/100 (vertex) | **10/0/90** | 10/10/80 | no | +2.25% | 19 of 63 | 28 of 62 |

(makespan / carbon / tardiness weights, %.) The validation choice never matched the pool optimum; the
mean cost on the pool is +1.26%. With 18 instances the choice is noisy (6 to 28 compositions cannot be
separated from each pick), as expected. The rule was not changed because of this.

---

## B -- Step 4 at 10x5: five arms, 4 seeds, budgets 0/50/100/250/500/1000

Reward: goal `mct` at (1/3, 1/3, 1/3), `carbon_reward_weight = 0.002384`, `tardiness_reward_weight =
5.815362`. The calibration left 2x, 4x and 8x the closed-form tardiness anchor statistically tied; the
calibration script's standing rule (empirical best) selects 8x, the same rule that set the priority
programme's weights. Tools: `tools/run_arms_t.py`, `tools/analyze_arms_t.py`; full output
`results/step4/step4_analysis.txt`; provenance per run under `results/provenance/step4_arms/`.
All 20 runs completed with no hangs or failures (12-15 min training + about 4 min evaluation each).

### Sanity checks -- PASS
* **Arm (a) at budget 0 reproduces its merge exactly:** maximum difference 0 on every metric, all 4 seeds.
* **Arm (b) at budget 0 is a random initialisation:** its weights are bit-identical to a freshly seeded,
  untrained network at all 4 seeds (`tools/verify_scratch_init.py`,
  `results/step4/scratch_init_verification.txt`). On performance it scores 1.29-1.51 against the
  centroid's 1.00 and the merge's 0.86-0.93. (A first performance screen also required scratch to be
  worse than EVERY warm start; it flagged seeds 111 and 444 only because the carbon specialist starts
  even worse on this objective (1.30 and 1.60). That comparison does not test random initialisation and
  was replaced by the weight test.)

### Budget curves (objective, mean +- sd across seeds; lower is better)

| updates | merge (a) | scratch (b) | makespan spec (c) | carbon spec (d) | tardiness spec (e) |
|---|---|---|---|---|---|
| 0 | **0.888** +- 0.031 | 1.421 +- 0.112 | 0.907 +- 0.038 | 1.401 +- 0.132 | 0.895 +- 0.026 |
| 50 | **0.900** +- 0.041 | 1.290 +- 0.048 | 0.927 +- 0.047 | 1.316 +- 0.079 | 0.917 +- 0.028 |
| 100 | 0.915 +- 0.090 | 1.286 +- 0.052 | **0.910** +- 0.031 | 1.220 +- 0.163 | 0.918 +- 0.054 |
| 250 | 0.897 +- 0.041 | 1.056 +- 0.040 | **0.881** +- 0.044 | 0.910 +- 0.016 | 0.894 +- 0.023 |
| 500 | 0.887 +- 0.031 | 0.914 +- 0.040 | **0.870** +- 0.040 | 0.910 +- 0.039 | 0.892 +- 0.033 |
| 1000 | 0.889 +- 0.030 | 0.902 +- 0.044 | **0.863** +- 0.032 | 0.899 +- 0.025 | 0.885 +- 0.043 |

Per-seed curves are in `results/step4/step4_analysis.txt`. Seed 444 is the noisiest (merge 0.928 -> 1.049
at 100 updates -> 0.933).

### Crossovers (first budget at which an arm is at least as good as the merge at the SAME budget)

| arm | pooled | per seed (111, 222, 333, 444) | verdict |
|---|---|---|---|
| scratch (b) | never | 1000, 1000, never, never | **not a crossover** (2/4 seeds) |
| makespan spec (c) | 100 | 250, 500, 50, 100 | crossover, 4/4 seeds |
| carbon spec (d) | never | 1000, never, never, 250 | **not a crossover** (2/4 seeds) |
| tardiness spec (e) | 250 | 100, 250, 250, 50 | crossover, 4/4 seeds |

### Merge vs scratch at matched budget, and what the merge is "worth"

| budget | scratch worse than merge by | seeds | p (seeds as unit) | scratch updates needed to match the merge at this budget: pooled (per seed) |
|---|---|---|---|---|
| 0 | +60.0% | 4/4 | 0.0015 | >1000 (>1000, 884, 475, >1000) |
| 50 | +43.6% | 4/4 | 0.0015 | >1000 (599, 866, >1000, 701) |
| 100 | +41.2% | 4/4 | 0.0017 | 498 (>1000, 972, 494, 332) |
| 250 | +17.9% | 4/4 | 0.0042 | >1000 (>1000, 766, 472, >1000) |
| 500 | +3.0% | 4/4 | 0.0094 | >1000 (>1000, 724, >1000, >1000) |
| 1000 | +1.3% | 2/4 | 0.35 | >1000 (754, 971, 497, >1000) |

The zero-cost merge is worth roughly 500 to more than 1000 updates of training from scratch, depending
on the seed. "Needed" is interpolated on a log-budget scale between checkpoints; the merge curve is
nearly flat, so these numbers are sensitive to small changes in the target.

### Merge vs each single specialist at matched budget (positive = merge better)

| budget | vs makespan spec (c) | vs carbon spec (d) | vs tardiness spec (e) |
|---|---|---|---|
| 0 | **+2.1%**, 4/4, p = 0.024 | **+36.4%**, 4/4, p = 0.002 | +0.8%, 4/4, p = 0.086 |
| 50 | +2.8%, 3/4, p = 0.19 | **+31.6%**, 4/4, p = 0.0002 | +1.9%, 3/4, p = 0.15 |
| 100 | -0.5%, 2/4, p = 0.89 | **+24.7%**, 4/4, p = 0.005 | +0.4%, 2/4, p = 0.93 |
| 250 | -1.8%, 1/4, p = 0.34 | +1.6%, 3/4, p = 0.35 | -0.3%, 1/4, p = 0.85 |
| 500 | -2.0%, 0/4, p = 0.14 | +2.5%, 4/4, p = 0.053 | +0.5%, 3/4, p = 0.69 |
| 1000 | **-3.1%**, 0/4, p = 0.001 | +1.1%, 2/4, p = 0.33 | -0.6%, 1/4, p = 0.55 |

p values use seeds as the unit (n = 4). The paired instance-level tests (n = 400, not independent) are in
the analysis output and agree in direction.

**Arm (e), stated plainly:** warm-starting from the tardiness specialist alone is statistically
indistinguishable from warm-starting from the merge at every budget. On this evidence the benefit over
scratch comes from starting at a relevant specialist, not from merging several.

### Operating points: where each arm starts (0) and ends (1000) -- raw pool means, mean +- sd over seeds

| arm | budget | makespan | carbon | tardiness | objective |
|---|---|---|---|---|---|
| merge (a) | 0 | 434.8 +- 13.7 | 2434.2 +- 74.2 | 380.1 +- 42.8 | 0.888 |
| merge (a) | 1000 | 444.7 +- 6.4 | 2471.0 +- 35.7 | 352.5 +- 9.8 | 0.889 |
| scratch (b) | 0 | 626.1 +- 46.0 | 2517.6 +- 351.0 | 1120.7 +- 169.9 | 1.421 |
| scratch (b) | 1000 | 452.8 +- 11.5 | 2486.2 +- 62.9 | 359.0 +- 31.1 | 0.902 |
| makespan spec (c) | 0 | 412.5 +- 9.9 | 2494.5 +- 14.6 | 422.2 +- 11.7 | 0.907 |
| makespan spec (c) | 1000 | 442.8 +- 5.9 | 2400.7 +- 24.9 | 326.6 +- 7.2 | **0.863** |
| carbon spec (d) | 0 | 605.8 +- 6.8 | 1882.9 +- 15.8 | 1322.8 +- 24.4 | 1.401 |
| carbon spec (d) | 1000 | 452.4 +- 13.6 | 2436.0 +- 92.8 | 371.7 +- 12.3 | 0.899 |
| tardiness spec (e) | 0 | 451.6 +- 7.5 | 2504.5 +- 31.8 | 345.6 +- 21.0 | 0.895 |
| tardiness spec (e) | 1000 | 447.4 +- 5.4 | 2423.6 +- 55.1 | 353.8 +- 12.3 | 0.885 |

Every arm ends in the same region (makespan 443-453, carbon 2400-2490, tardiness 327-372): the reward
pulls all starting points to a common operating point within 1000 updates, consistent with H3
(convergence over persistence). The carbon specialist gives up almost all of its carbon advantage
(1883 -> 2436). The makespan-initialised arm reaches the best point on all three objectives at once.

### Reading
* H1 (initialisation efficiency): **supported** against scratch at 0-500 updates; converged at 1000.
* H2 (composition specificity): **not supported.** The tardiness specialist alone matches the merge, and
  the makespan specialist overtakes it from about 100-250 updates. On validation, at 2 of 4 seeds the best
  composition was a single specialist.
* The merge does not improve under fine-tuning (0.888 -> 0.889), while every single-specialist start
  does. One possible reason is the reward weight: the tardiness weight is 8x the closed-form anchor
  (chosen from a statistical plateau, 2x-8x), and a reward that leans this hard on tardiness may not score
  the merge's starting point as improvable. Untested here.

---

## C -- 20x10 tardiness sensitivity arm at k = 1.2

**SENSITIVITY ARM.** It does not replace the pre-registered k = 0.95; the pre-registered 20x10 Gate 1
failure stands as the primary result. These specialists are not used in any composition or in Step 4.

* Separate manifest `daniel/data/due_dates/due_date_manifest_SENSITIVITY_20x10_k120s060.json`: 286
  instances (all 20x10 splits), tag `k120s060`, same s = 0.6, formula and per-job draws, due dates
  exactly 1.2/0.95 of the frozen ones, per-file SHA-256, tampered hash raises. The pre-registered
  manifest is byte-identical to before.
* Four tardiness specialists `20x10+carbon+priority+t_k120s060_s{111,222,333,444}`, canonical
  protocol and budget (1000 updates), 64.7-68.7 min each, provenance records under
  `results/provenance/tardiness_specialists/`. Makespan and carbon specialists unchanged, re-evaluated
  with the k = 1.2 due dates (pool `trainvali-k120s060`).

### Per-specialist Gate 1, side by side (tardiness specialist vs the SAME seed's makespan specialist)

| seed | k = 0.95 (pre-registered): gain | p | makespan cost | k = 1.2 (sensitivity): gain | p | makespan cost |
|---|---|---|---|---|---|---|
| 111 | +6.2% | 0.011 | +5.7% | **-34.3%** | 4.3e-06 | +15.9% |
| 222 | -4.6% | 0.075 | +9.4% | +1.5% | 0.80 | +13.0% |
| 333 | +7.7% | 0.0006 | +7.0% | **-13.8%** | 0.008 | +12.9% |
| 444 | -6.4% | 0.023 | +6.2% | +4.9% | 0.40 | +9.1% |
| **across seeds** | **+0.7%, 2/4 wins, p = 0.86** | | | **-10.4%, 2/4 wins, p = 0.33** | | |

(gain = reduction in total tardiness relative to the makespan specialist; paired over 100 instances.)
Both fail the per-specialist rule. At k = 1.2 the makespan specialist already keeps tardiness low (355
on average, with 24% of jobs late), and the tardiness specialists pay 9-16% in makespan without beating
it. Gate-1 distance: makespan<->tardiness 0.157 at k = 1.2 (10.9x) vs 0.085 at k = 0.95 (20.4x); the
larger distance comes from the tardiness specialist being worse on makespan, not better on tardiness.

### Reading (sensitivity evidence about the smallest-k rule, not the headline 20x10 result)
* The heuristic screen predicted that loosening k would quadruple tardiness's separation at 20x10
  (+7.3% -> +30.6%). Trained policies show the opposite. The screen's k-sensitivity does not transfer to
  trained policies at this size.
* So the smallest-k rule is not the explanation for the 20x10 failure. What remains is that at 20x10 the
  makespan specialist is already about as good on tardiness as a tardiness-trained policy can get within
  the budget. One untested possibility for the looser k: with only about a quarter of jobs late, the
  tardiness reward is sparse, which could make it harder to learn from.

---

## What is where

* Report: this file. Notes: `research_direction.md` (Gate 2 role change; Step 4 and sensitivity findings).
* A: `results/step4/merge_init_selection.json`, `results/step4/stepA_selection.txt`, validation
  evaluations `daniel/test_results/SD2/vali-k125s060_10x5+carbon+priority/`.
* B: `results/step4/step4_analysis.txt`, `results/step4/scratch_init_verification.txt`, evaluations
  `daniel/test_results/SD2/trainvali-k125s060_10x5+carbon+priority/*s4t_*`, provenance
  `results/provenance/step4_arms/`. Checkpoints (gitignored, on disk):
  `daniel/trained_network/SD2/10x5+carbon+priority+s4t_*@u*.pth`.
* C: `results/tardiness_programme/gate1_20x10_k120s060_SENSITIVITY.txt`, evaluations
  `daniel/test_results/SD2/trainvali-k120s060_20x10+carbon+priority/`, provenance
  `results/provenance/tardiness_specialists/20x10+carbon+priority+t_k120s060_*.json`.
* Not run, as instructed: Steps 5-6 of the arms programme, anything on the reserved final-test sets.
