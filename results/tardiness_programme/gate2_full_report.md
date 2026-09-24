# Gate 2 full report, 10x5 -- tardiness triple next to the retired priority triple

Same format for both triples (tools/gate2_full_report.py; raw output in gate2_full_report_tardiness.txt and
gate2_full_report_priority.txt). 66 grid compositions (step 0.1) per seed, 4 seeds, reference = centroid.
Optima are chosen from the grid only (the centroid is the reference, not a candidate), so a few interior
numbers differ slightly from the first Gate 2 record, which let the centroid compete.

## Pooled optimum per seed (m / c / third, %), face, distance to nearest edge

| preference | triple | seed 111 | seed 222 | seed 333 | seed 444 |
|---|---|---|---|---|---|
| 1/3 each | tardiness | 0/10/90 c-t edge, 0pp | 20/10/70 interior, 10pp | 80/20/0 **m-c edge**, 0pp | 10/10/80 interior, 10pp |
| 0.25/0.5/0.25 | tardiness | 10/20/70 interior, 10pp | 0/30/70 c-t edge, 0pp | 70/30/0 **m-c edge**, 0pp | 0/20/80 c-t edge, 0pp |
| 1/3 each | priority | 20/10/70 interior, 10pp | 60/20/20 interior, 20pp | 50/10/40 interior, 10pp | 40/20/40 interior, 20pp |
| 0.25/0.5/0.25 | priority | 50/20/30 interior, 20pp | 50/30/20 interior, 20pp | 70/30/0 m-c edge, 0pp | 60/30/10 interior, 10pp |

Mean weight at the optimum (sd across seeds):
* tardiness triple, 1/3 each: makespan 27.5% (35.9), carbon 12.5% (5.0), tardiness 60.0% (40.8)
* tardiness triple, 0.25/0.5/0.25: makespan 20.0% (33.7), carbon 25.0% (5.8), tardiness 55.0% (37.0)
* priority triple, 1/3 each: makespan 42.5% (17.1), carbon 15.0% (5.8), priority 42.5% (20.6)
* priority triple, 0.25/0.5/0.25: makespan 57.5% (9.6), carbon 27.5% (5.0), priority 15.0% (12.9)

**Is the tardiness optimum on the tardiness-carbon edge at every seed and both preferences? No.** It is ON
that edge in 3 of 8 (seed, preference) cases, interior but with makespan <= 20% in 3 more, and on the
MAKESPAN-carbon edge for seed 333 at both preferences. Six of eight optima put <= 20% on makespan and
>= 70% on tardiness.

## Interior vs best edge (paired across instances; seeds as the unit)

| preference | triple | per seed (interior advantage, paired p) | seeds as the unit |
|---|---|---|---|
| 1/3 each | tardiness | -0.20% (0.82), +0.45% (0.64), -1.36% (0.19), +0.45% (0.65) | -0.17%, 2/4, p = 0.72 |
| 0.25/0.5/0.25 | tardiness | +0.91% (0.22), -0.24% (0.75), **-1.93% (0.018)**, -0.12% (0.89) | -0.35%, 1/4, p = 0.60 |
| 1/3 each | priority | +0.43% (0.33), +0.62% (0.20), +0.08% (0.86), +0.41% (0.37) | +0.38%, 4/4, p = 0.04 |
| 0.25/0.5/0.25 | priority | +0.22% (0.55), +0.14% (0.74), -0.48% (0.32), +0.52% (0.25) | +0.10%, 3/4, p = 0.67 |

Both comparisons are biased toward the interior (both compositions are selected maxima). The only
significant per-seed result for tardiness goes the other way: seed 333's EDGE beats its interior.

## How much does adding makespan weight cost? (tardiness triple)
Best composition at each fixed makespan weight, relative to the best with makespan at zero:

| preference | w = 10% | w = 20% | w = 30% | w = 50% | w = 70% |
|---|---|---|---|---|---|
| 1/3 each | +0.17% (p = 0.59) | -0.04% (p = 0.85) | +0.53% | +1.03% | +0.89% |
| 0.25/0.5/0.25 | +0.23% (p = 0.59) | +0.24% (p = 0.33) | +0.24% | +0.86% | +0.35% |

Adding makespan costs essentially nothing up to 20-30% and about 1% at 50%. Makespan is REDUNDANT for
composition, not harmful: the surface is flat along the makespan direction near the tardiness-carbon edge.

For comparison, adding PRIORITY to the priority triple: -0.23% / -0.36% at 10% / 20% (equal preference;
p = 0.25, 0.07) and +0.05% / +0.06% (carbon-leaning) -- also close to flat.

## Cost of dropping each specialist entirely (from gate2_tardiness.md)

| triple, preference | drop makespan | drop carbon | drop third |
|---|---|---|---|
| priority, 1/3 each | +0.78% | +1.00% | +0.52% |
| priority, 0.25/0.5/0.25 | +1.08% | +3.36% | +0.25% |
| tardiness, 1/3 each | +0.54% | +0.59% | +2.48% |
| tardiness, 0.25/0.5/0.25 | +0.83% | +2.40% | +1.72% |

## Stability across seeds (one seed's optimum applied to the other seeds)

| preference | tardiness triple | priority triple |
|---|---|---|
| 1/3 each | mean 2.10%, max 5.39% worse | mean 0.95%, max 3.66% |
| 0.25/0.5/0.25 | mean 2.50%, max 6.23% | mean 0.71%, max 1.50% |

Seed 333's optimum transfers worst in the tardiness triple (3.2-5.4% worse on the other seeds). The
tardiness surface has more at stake (optima beat the centroid by 3.6-15.0%, against 0.0-4.1% for priority),
so choosing the wrong composition also costs more.

## Do the two preferences agree? (seed by seed)

| seed | tardiness triple | priority triple |
|---|---|---|
| 111 | c-t edge -> interior (moved 20pp) | interior -> interior (40pp) |
| 222 | interior -> c-t edge (20pp) | interior -> interior (10pp) |
| 333 | m-c edge -> m-c edge (10pp) | interior -> m-c edge (40pp) |
| 444 | interior -> c-t edge (10pp) | interior -> interior (30pp) |

Tardiness: the face changes at 3 of 4 seeds, but the optimum moves only 10-20pp and stays in the same
region (low makespan, high tardiness) everywhere except seed 333. The two preferences agree on the
verdict (EDGE) and broadly on the region, not on the exact face.
