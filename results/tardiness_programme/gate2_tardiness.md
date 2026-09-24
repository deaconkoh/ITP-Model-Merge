# Gate 2 -- (makespan, carbon, tardiness) coarse simplex, 10x5

66 compositions (step 0.1) + the centroid, x 4 seeds = 268 evaluations on the 100-instance train_vali
pool, frozen due dates k125s060. Reference = the centroid composition. Makespan and carbon are
normalised per instance; tardiness by its pool-mean reference (fixed in advance).

## Pooled optimum per seed

| preference | seed 111 (m/c/t) | seed 222 | seed 333 | seed 444 | tardiness weight (mean) | verdict |
|---|---|---|---|---|---|---|
| (1/3, 1/3, 1/3) | **0**/10/90 | 20/10/70 | 80/20/**0** | 10/10/80 | 60.0% | EDGE at 111, 333 |
| (0.25, 0.5, 0.25) | 10/20/70 | **0**/30/70 | 70/30/**0** | **0**/20/80 | 55.0% | EDGE at 222, 333, 444 |

Distance to the nearest edge: 0-10 pp (mean 5.0 pp) at equal preference, 0-10 pp (mean 2.5 pp) at
the carbon-leaning one. Both preferences give EDGE verdicts, so they agree on the verdict, but not
seed by seed.

## Which specialist drops out

At five of the seven edge optima, the specialist at zero weight is MAKESPAN, not tardiness. Only seed
333 drops tardiness, at both preferences. This is the opposite of priority's failure, where the third
specialist was the one that contributed least.

## Interior vs best edge (paired over 100 instances; biased toward the interior, both are maxima)

| preference | per seed | mean | across seeds |
|---|---|---|---|
| (1/3, 1/3, 1/3) | -0.20, +0.45, -1.36, +0.45% | -0.17% | t = -0.39, p = 0.73 |
| (0.25, 0.5, 0.25) | +0.91, -0.24, -1.93, -0.12% | -0.35% | t = -0.59, p = 0.60 |

No seed shows a significant interior advantage (per-seed paired p = 0.19 to 0.89, apart from seed 333
at the carbon-leaning preference, where the EDGE is better, p = 0.018).

## Cost of dropping each specialist entirely (best composition on that face vs the best overall)

| triple, preference | drop makespan | drop carbon | drop third |
|---|---|---|---|
| m, c, priority (retired), 1/3 each | +0.78% | +1.00% | **+0.52%** |
| m, c, priority (retired), 0.25/0.5/0.25 | +1.08% | +3.36% | **+0.25%** |
| m, c, tardiness, 1/3 each | +0.54% | +0.59% | **+2.48%** (2.3, 2.2, 0.0, 5.4) |
| m, c, tardiness, 0.25/0.5/0.25 | +0.83% | +2.40% | **+1.72%** (1.2, 2.5, 0.0, 3.2) |

Priority was the least valuable of its three specialists. Tardiness is the MOST valuable at the equal
preference (about 5x priority's value) and second only to carbon at the carbon-leaning one. Seed 333 is
the exception at both preferences.

## Optimum vs the centroid composition
+9.3 to +14.8% at equal preference, +3.5 to +7.3% at the carbon-leaning one (priority triple: +1.0 to
+4.1% and +0.0 to +1.1%).

## Reading
By its pre-stated letter, Gate 2 FAILS at both preferences: optima sit on an edge. But the gate was
written to ask whether the third specialist earns its place, and here it does, more than either of
the other two at equal preference. What does not earn its place is having all THREE specialists: the
tardiness specialist largely covers the makespan direction (it is only +9.5% worse on makespan), so
the best compositions are carbon + tardiness with little or no makespan. The interior advantage the
gate tests for is absent (-0.17%, p = 0.73). Seed 333 disagrees with the other three at both
preferences, the seed-level instability seen throughout this project.
