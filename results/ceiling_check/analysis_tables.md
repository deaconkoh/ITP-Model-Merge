### Gap to CP-SAT (60 s best solution), HELD-OUT, 4 seeds x 49 instances

Gap = (policy score - CP-SAT score) / CP-SAT score per (seed, instance); pooled mean, median, 95% bootstrap CI over instances (each carrying its seeds). Mean scores are pooled means over seeds and instances.

| method | mean score | gap mean | gap median | 95% CI | seed 111 | seed 222 | seed 333 | seed 444 |
|---|---|---|---|---|---|---|---|---|
| CP-SAT 60 s best | 0.6140 | - | - | - | 0.6144 | 0.6077 | 0.6088 | 0.6252 |
| CP-SAT proven bound | 0.5001 | - | - | - | 0.5021 | 0.4965 | 0.4951 | 0.5065 |
| C1@1000 greedy | 0.8655 | +40.99% | +40.00% | [+39.01, +43.01] | +39.42% | +38.00% | +41.42% | +45.13% |
| C1@1000 best-of-10 | 0.7986 | +30.07% | +29.29% | [+28.63, +31.59] | +29.05% | +27.67% | +31.27% | +32.27% |
| C1@1000 best-of-30 | 0.7754 | +26.31% | +25.47% | [+24.91, +27.81] | +25.01% | +24.30% | +27.65% | +28.29% |
| C1@1000 best-of-100 | 0.7563 | +23.23% | +22.78% | [+21.90, +24.63] | +22.25% | +21.47% | +24.17% | +25.01% |

Same methods against the PROVEN BOUND (upper limit on the gap to the true optimum): (policy - bound) / bound

| method | gap-to-bound mean | 95% CI |
|---|---|---|
| CP-SAT 60 s best | +22.96% | [+19.98, +26.05] |
| C1@1000 greedy | +73.37% | [+68.46, +78.42] |
| C1@1000 best-of-10 | +59.97% | [+55.64, +64.43] |
| C1@1000 best-of-30 | +55.34% | [+51.12, +59.67] |
| C1@1000 best-of-100 | +51.50% | [+47.55, +55.58] |

Share of the greedy-to-CP-SAT score difference closed by best-of-N: best-of-10 27%, best-of-30 36%, best-of-100 43%.

### CP-SAT status (60 s, 8 workers, warm-started with C1@1000 greedy)

* OPTIMAL: **5 / 196** pairs (3%); the rest FEASIBLE ({'FEASIBLE': 191, 'OPTIMAL': 5}).
* Solver gap (best - bound) / best: mean 18.0%, median 17.5%, range [0.0%, 41.0%]; per seed 17.8%, 17.9%, 18.2%, 18.3%.
* Hint check: solution worse than the C1 hint in **0** pairs (tolerance 1e-6); equal to the hint in 0. Mean wall time 58.9 s.
* Reachability: replaying every CP-SAT schedule through DANIEL's append rule gives a score worse than CP-SAT's in **0** pairs (strictly better in 0; mean 0.6140). The CP-SAT schedules lie inside DANIEL's action space.

### 60 s vs 600 s (10 pre-registered pairs)

| seed | instance | status 60 / 600 | score 60 | score 600 | change | bound 60 | bound 600 | change | solver gap 60 / 600 |
|---|---|---|---|---|---|---|---|---|---|
| 111 | 35 | FEASIBLE / FEASIBLE | 0.5736 | 0.5736 | +0.00% | 0.4554 | 0.5042 | +10.71% | 20.6% / 12.1% |
| 111 | 75 | FEASIBLE / FEASIBLE | 0.5768 | 0.5768 | +0.00% | 0.4914 | 0.5450 | +10.89% | 14.8% / 5.5% |
| 111 | 84 | FEASIBLE / FEASIBLE | 0.7983 | 0.7992 | +0.11% | 0.5463 | 0.5477 | +0.27% | 31.6% / 31.5% |
| 222 | 3 | FEASIBLE / FEASIBLE | 0.5637 | 0.5637 | +0.00% | 0.4929 | 0.5288 | +7.28% | 12.6% / 6.2% |
| 222 | 41 | FEASIBLE / FEASIBLE | 0.5550 | 0.5550 | +0.00% | 0.4714 | 0.5194 | +10.17% | 15.1% / 6.4% |
| 222 | 53 | FEASIBLE / FEASIBLE | 0.6237 | 0.6223 | -0.22% | 0.4755 | 0.5218 | +9.72% | 23.8% / 16.2% |
| 333 | 12 | FEASIBLE / FEASIBLE | 0.5834 | 0.5834 | +0.00% | 0.5246 | 0.5543 | +5.66% | 10.1% / 5.0% |
| 333 | 29 | FEASIBLE / FEASIBLE | 0.7728 | 0.7362 | -4.72% | 0.4931 | 0.5369 | +8.90% | 36.2% / 27.1% |
| 444 | 39 | FEASIBLE / FEASIBLE | 0.6296 | 0.6192 | -1.66% | 0.4792 | 0.5285 | +10.29% | 23.9% / 14.6% |
| 444 | 74 | FEASIBLE / FEASIBLE | 0.7595 | 0.7418 | -2.33% | 0.5144 | 0.5490 | +6.72% | 32.3% / 26.0% |

Mean change 60 -> 600 s: best score -0.88% (largest improvement -4.72%), proven bound +8.06% (largest rise +10.89%).

### Raw objectives, HELD-OUT (mean over instances, then seeds)

| schedule | makespan | carbon | tardiness | score |
|---|---|---|---|---|
| CP-SAT 60 s best | 376.8 | 2023.3 | 50.6 | 0.6140 |
| C1@1000 greedy | 444.5 | 2403.1 | 327.1 | 0.8655 |
| C1@1000 best-of-100 | 403.8 | 2326.7 | 194.7 | 0.7563 |

Weighted score terms (w x objective / normaliser; they sum to the score):

| schedule | makespan term | carbon term | tardiness term |
|---|---|---|---|
| CP-SAT 60 s best | 0.2597 | 0.3287 | 0.0256 |
| C1@1000 greedy | 0.3065 | 0.3923 | 0.1668 |
| C1@1000 best-of-100 | 0.2783 | 0.3791 | 0.0989 |

Greedy minus CP-SAT, per term: makespan +0.0467, carbon +0.0636, tardiness +0.1412 (share of the total difference: 19%, 25%, 56%).

### Verdict: **POLICY-OR-BUDGET-LIMITED** (greedy gap +40.99% [+39.01, +43.01], best-of-100 gap +23.23% [+21.90, +24.63]; best-known solution, NOT a proven optimum).
