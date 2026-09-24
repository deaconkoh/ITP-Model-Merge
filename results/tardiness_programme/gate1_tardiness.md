# Gate 1 -- tardiness specialists (Step 5), both sizes

Frozen due dates (manifest 800424d): 10x5 k125s060, 20x10 k095s060. Evaluations on the 100-instance
train_vali pool, greedy decoding, makespan/carbon specialists re-evaluated with the same due dates.

## 10x5 -- PASS, decisively
| seed | makespan-spec T | tardiness-spec T | gain | instances where t-spec better | p (paired) | makespan cost |
|---|---|---|---|---|---|---|
| 111 | 418.1 | 339.2 | +18.9% | 73% | 8.6e-08 | +12.0% |
| 222 | 427.2 | 347.9 | +18.6% | 73% | 1.4e-06 | +9.5% |
| 333 | 435.3 | 372.7 | +14.4% | 62% | 7.0e-04 | +7.8% |
| 444 | 408.3 | 322.5 | +21.0% | 68% | 9.2e-07 | +8.6% |
Across seeds: +18.2%, 4/4 seeds, t = 13.1, p = 0.001.

## 20x10 -- FAILS on the per-specialist criterion
| seed | makespan-spec T | tardiness-spec T | gain | instances where t-spec better | p (paired) | makespan cost |
|---|---|---|---|---|---|---|
| 111 | 1127.9 | 1057.8 | +6.2% | 62% | 0.011 | +5.7% |
| 222 | 1254.8 | 1312.5 | **-4.6%** | 43% | 0.075 | +9.4% |
| 333 | 1200.1 | 1108.2 | +7.7% | 64% | 5.6e-04 | +7.0% |
| 444 | 1125.5 | 1197.7 | **-6.4%** | 40% | 0.023 | +6.2% |
Across seeds: +0.7%, 2/4 seeds, t = 0.20, p = 0.86. Two of four tardiness specialists are beaten on
tardiness by the makespan specialist trained on the same seed (seed 444 significantly), while every
one of them pays 6-9% in makespan. The seed-mean comparison the original Gate 1 used prints PASS
(1169.1 vs 1177.1); the per-specialist criterion added to tools/check_specialists_t.py prints FAIL.

## Objective-space distances (Gate 1 method)
| size | triple | m<->c | c<->X | m<->X | ratio |
|---|---|---|---|---|---|
| 10x5 | makespan, carbon, priority (retired) | 1.665 | 1.651 | 0.169 | 9.8x |
| 10x5 | makespan, carbon, tardiness | 1.679 | 1.624 | 0.217 | 7.7x |
| 20x10 | makespan, carbon, priority (retired) | 1.701 | 1.666 | 0.128 | 13.3x |
| 20x10 | makespan, carbon, tardiness | 1.731 | 1.681 | **0.085** | **20.4x** |

At 20x10 tardiness is closer to makespan than the retired priority axis was.

## Per-specialist rule applied to all four objectives (same rule for the write-up)
Each specialist must beat the strongest OTHER specialist from its own seed on its own objective;
paired t-test over the 100 instances. 11 of 12 specialist sets pass; only tardiness at 20x10 fails.

| size | triple | makespan | carbon | third objective |
|---|---|---|---|---|
| 10x5 | m, c, priority | 4/4 (+2.3 to +7.3%) | 4/4 (+23.0 to +24.8%) | priority 4/4 (+4.9 to +6.8%) |
| 10x5 | m, c, tardiness | 4/4 (+7.3 to +10.7%) | 4/4 (+23.9 to +24.8%) | tardiness 4/4 (+14.4 to +21.0%) |
| 20x10 | m, c, priority | 4/4 (+5.5 to +12.6%) | 4/4 (+40.2 to +41.1%) | priority 4/4 (+3.0 to +6.3%) |
| 20x10 | m, c, tardiness | 4/4 (+5.4 to +8.6%) | 4/4 (+40.2 to +41.1%) | **tardiness 2/4 (-6.4 to +7.7%) FAIL** |

The retired priority specialists pass at both sizes (all p < 1e-8). Priority's problem was never that
its specialists failed to win; their wins were real but small and pointed the same way as makespan
(the collinearity finding).

## Had the 20x10 specialists stopped improving by update 1000? (saved validation curves)
Best-validation update per seed, and the last-quarter trend as % change per 100 updates (negative =
still improving). Validation uses only 13 (20x10) / 18 (10x5) instances, so trends are noisy.

| size | specialist | best update per seed | bests in last 20% | last-quarter trend (%/100 updates) |
|---|---|---|---|---|
| 20x10 | makespan | 690, 870, 690, 830 | 2/4 | -0.72, +0.17, +0.17, +0.06 |
| 20x10 | carbon | 970, 810, 720, 830 | 3/4 | -0.35, -0.15, -0.79, -0.93 |
| 20x10 | priority | 680, 950, 700, 910 | 2/4 | +0.58, -0.72, +0.15, -0.96 |
| 20x10 | tardiness | 830, 920, 1000, 380 | 3/4 | +4.59, -3.46, -2.86, -2.99 |
| 10x5 | tardiness | 580, 530, 230, 380 | 0/4 | +1.00, -4.07, -1.59, +4.25 |

Reading: makespan at 20x10 had levelled off (trends within +-0.7%). Carbon was still improving,
but slowly (all four negative, under 1% per 100 updates). Tardiness at 20x10 was still falling at
3-3.5% per 100 updates in three of four seeds, several times faster than any other specialist. So
the budget explanation, if it holds, is largely SPECIFIC TO TARDINESS, with a mild undertraining
signal for carbon. Caveat: tardiness validation is noisy even at 10x5 (trends of +-4% in both
directions there), so the diagnostic continuation is the real test.

## Caveat on the choice of k (recorded only; k is NOT changed)
The pre-stated rule selects the SMALLEST k in the qualifying band, i.e. the tightest deadlines and
the most late jobs the band allows (20x10: k = 0.95 at the lower edge of 0.95-1.20, 57% of jobs late
under the trained makespan specialist). With more than half the jobs late, total tardiness moves
closer to total completion time, which is time-like. The rule may therefore have favoured the
structural (collinear-with-makespan) explanation at 20x10. A larger k inside the band would leave
fewer jobs late and might separate better, but choosing k for separation after the fact is exactly
what the rule exists to prevent.

## 20x10 diagnostic: continue the two losing tardiness seeds for another 1000 updates
DIAGNOSTIC ONLY. The extended checkpoints (`..._ext1000`) are not used in any composition and break the
matched-budget protocol on purpose. Each run resumes from the seed's best-validation checkpoint (seed 444's
was at update 380), same seed, same frozen due dates (k095s060).

| seed | tardiness-spec T | makespan-spec T | gain | instances better | p (paired) | makespan cost |
|---|---|---|---|---|---|---|
| 222, original | 1312.5 | 1254.8 | -4.6% | 43% | 0.075 | +9.4% |
| 222, +1000 updates | 1247.1 | 1254.8 | +0.6% | 50% | 0.79 | +8.0% |
| 444, original | 1197.7 | 1125.5 | -6.4% | 40% | 0.023 | +6.2% |
| 444, +1000 updates | 1165.0 | 1125.5 | -3.5% | 46% | 0.22 | +10.9% |

Doubling the budget improved both specialists' tardiness by about 3-5%, but neither now beats its
makespan counterpart: seed 222 draws level, seed 444 still loses. Validation tardiness in the extension
flattened after its first quarter (seed 222 quarters 1311, 1223, 1235, 1244; seed 444 1282, 1250, 1237,
1251). So the extra budget helped a little and then plateaued.

Reading: the budget explanation is largely NOT supported. Extra training closes part of the gap but does
not produce a specialist that wins its own objective at 20x10, while it keeps paying 8-11% in makespan.
That leans toward the structural explanation, subject to the recorded k caveat (the smallest-k rule gave
the tightest deadlines). Two seeds, one extension length: suggestive, not conclusive.


## Sensitivity arm: trained 20x10 tardiness specialists at k = 1.2 (loosest in the band)
SENSITIVITY ONLY; the pre-registered k = 0.95 result above stands. Separate manifest, tag k120s060.

| seed | k = 0.95 gain (p) | k = 1.2 gain (p) | k = 1.2 makespan cost |
|---|---|---|---|
| 111 | +6.2% (0.011) | -34.3% (4.3e-06) | +15.9% |
| 222 | -4.6% (0.075) | +1.5% (0.80) | +13.0% |
| 333 | +7.7% (0.0006) | -13.8% (0.008) | +12.9% |
| 444 | -6.4% (0.023) | +4.9% (0.40) | +9.1% |
| across seeds | +0.7%, 2/4, p = 0.86 | -10.4%, 2/4, p = 0.33 | |

Loosening k made the trained tardiness specialists worse relative to the makespan specialist, the
opposite of the CPU screen's prediction (+7.3% -> +30.6%). The smallest-k caveat recorded above is
therefore NOT supported by trained policies: it is not what caused the 20x10 failure.
