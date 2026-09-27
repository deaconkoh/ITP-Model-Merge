### Step 4 score on HELD-OUT (lower is better), per seed and pooled

| arm | seed 111 | seed 222 | seed 333 | seed 444 | pooled |
|---|---|---|---|---|---|
| B1 logit, alpha = w | 0.8451 | 0.8838 | 0.9831 | 0.9616 | **0.9184** |
| B2 prob, alpha = w | 0.8703 | 0.8966 | 0.9173 | 0.9550 | **0.9098** |
| B3 logit, tuned alpha | 0.8584 | 0.8513 | 0.8894 | 0.9420 | **0.8853** |
| B4 makespan spec alone | 0.8756 | 0.8786 | 0.9198 | 0.9644 | **0.9096** |
| B4 carbon spec alone | 1.2970 | 1.3395 | 1.3641 | 1.6040 | **1.4012** |
| B4 tardiness spec alone | 0.8706 | 0.8863 | 0.9073 | 0.9401 | **0.9011** |
| R merge @0 | 0.8594 | 0.8825 | 0.8953 | 0.9333 | **0.8926** |
| R merge @1000 | 0.8907 | 0.8703 | 0.8972 | 0.9538 | **0.9030** |
| R scratch @0 | 1.2920 | 1.5361 | 1.3675 | 1.5282 | **1.4309** |
| R scratch @1000 | 0.8648 | 0.8737 | 0.9248 | 0.9853 | **0.9121** |
| R makespan spec @0 | 0.8756 | 0.8786 | 0.9198 | 0.9644 | **0.9096** |
| R makespan spec @1000 | 0.8556 | 0.8395 | 0.8609 | 0.9060 | **0.8655** |
| R carbon spec @0 | 1.2970 | 1.3395 | 1.3641 | 1.6040 | **1.4012** |
| R carbon spec @1000 | 0.8656 | 0.8974 | 0.9179 | 0.9220 | **0.9007** |
| R tardiness spec @0 | 0.8706 | 0.8863 | 0.9073 | 0.9401 | **0.9011** |
| R tardiness spec @1000 | 0.8761 | 0.8522 | 0.8899 | 0.9513 | **0.8924** |

B3's alpha chosen on FIT (permille m/c/t): seed 111: 100/100/800, seed 222: 300/200/500, seed 333: 600/100/300, seed 444: 200/0/800

### Raw objectives on HELD-OUT (mean over held-out instances, then over seeds)

| arm | makespan | carbon | tardiness |
|---|---|---|---|
| B1 logit, alpha = w | 446.2 | 2165.2 | 506.0 |
| B2 prob, alpha = w | 437.0 | 2341.5 | 445.5 |
| B3 logit, tuned alpha | 431.3 | 2430.3 | 375.9 |
| B4 makespan spec alone | 411.5 | 2500.5 | 424.1 |
| B4 carbon spec alone | 604.7 | 1887.7 | 1320.8 |
| B4 tardiness spec alone | 453.2 | 2514.4 | 348.7 |
| R merge @0 | 434.4 | 2441.2 | 385.1 |
| R merge @1000 | 448.4 | 2486.1 | 366.0 |
| R scratch @0 | 632.1 | 2527.1 | 1126.9 |
| R scratch @1000 | 453.0 | 2501.3 | 369.3 |
| R makespan spec @0 | 411.5 | 2500.5 | 424.1 |
| R makespan spec @1000 | 444.5 | 2403.1 | 327.1 |
| R carbon spec @0 | 604.7 | 1887.7 | 1320.8 |
| R carbon spec @1000 | 453.6 | 2444.8 | 368.4 |
| R tardiness spec @0 | 453.2 | 2514.4 | 348.7 |
| R tardiness spec @1000 | 446.8 | 2441.6 | 361.2 |

### Paired comparisons on HELD-OUT: relative gain of X over Y = (mean Y - mean X)/mean Y (positive = X better), 95% CI from 10,000 instance resamples

| X vs Y | seed 111 | seed 222 | seed 333 | seed 444 | pooled | seeds beating Y (gain >= 1%, CI > 0) |
|---|---|---|---|---|---|---|
| B1 logit vs R makespan spec @1000 | +1.22% [-1.49, +4.03] | -5.29% [-8.32, -2.42] | -14.20% [-18.89, -9.79] | -6.14% [-9.62, -2.68] | -6.12% [-8.48, -3.81] | 0/4 |
| B1 logit vs R merge @0 | +1.66% [-0.48, +3.76] | -0.15% [-3.49, +3.11] | -9.82% [-14.59, -5.38] | -3.04% [-6.69, +0.51] | -2.89% [-4.86, -0.98] | 0/4 |
| B1 logit vs R scratch @1000 | +2.27% [-0.75, +5.15] | -1.16% [-4.39, +2.05] | -6.31% [-11.11, -2.01] | +2.40% [-1.62, +6.43] | -0.69% [-3.50, +1.98] | 0/4 |
| B3 logit vs R makespan spec @1000 | -0.34% [-2.99, +2.33] | -1.41% [-4.66, +1.73] | -3.31% [-6.35, -0.20] | -3.98% [-7.71, -0.41] | -2.29% [-3.98, -0.63] | 0/4 |
| B3 logit vs R merge @0 | +0.11% [-2.07, +2.21] | +3.53% [+0.48, +6.34] * | +0.66% [-2.28, +3.55] | -0.94% [-3.69, +1.69] | +0.82% [-0.46, +2.05] | 1/4 |
| B3 logit vs R scratch @1000 | +0.73% [-2.17, +3.60] | +2.57% [-0.51, +5.44] | +3.83% [-0.51, +7.95] | +4.39% [+0.64, +7.85] * | +2.95% [+0.72, +5.04] | 1/4 |
| B1 logit vs B2 prob, alpha = w | +2.90% [+0.19, +5.45] * | +1.42% [-1.26, +3.95] | -7.18% [-11.19, -3.46] | -0.69% [-3.95, +2.55] | -0.95% [-2.48, +0.54] | 1/4 |

(* = gain >= 1% with the CI above 0 at that seed. Mean per-instance score differences Y - X are in summary.json.)

### Confidence / compromise analysis (B1 rollouts, all 100 instances x 4 seeds = 20,000 decisions)

| category | overall | fifth 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|---|
| unanimous (all three top-1s equal, blend takes it) | 33.0% | 15.4% | 26.1% | 28.8% | 35.5% | 59.0% |
| majority (blend takes a top-1 shared by exactly two) | 38.2% | 34.4% | 45.1% | 43.8% | 40.1% | 27.7% |
| follows M only | 6.5% | 10.3% | 7.4% | 6.4% | 5.5% | 2.6% |
| follows C only | 12.8% | 16.9% | 13.6% | 12.7% | 12.6% | 8.3% |
| follows T only | 4.7% | 8.6% | 4.5% | 5.1% | 3.9% | 1.4% |
| compromise (none of the top-1s) | 4.8% | 14.3% | 3.3% | 3.2% | 2.4% | 1.0% |

* Decisions where the blend follows exactly one expert: 24.0%. Among them, the followed expert is the MOST CONFIDENT (lowest entropy) in **65.0%**; permutation null (entropies shuffled across experts within each decision, 200x): mean 33.3%, 95th percentile 34.4% -> the blend follows the most confident expert MORE than chance.
* Among decisions where the experts' top-1s are not unanimous (67.0% of all decisions), **7.2%** are resolved by compromise.
* Mean entropy (bits): M 0.32, C 0.29, T 0.49.

### Verdict inputs

* Seeds where the arm beats R[makespan spec @1000] (gain >= 1%, CI > 0): B1 0/4, B3 0/4 (STRONG needs 3)
* Seeds where the arm beats BOTH R[merge @0] and R[scratch @1000]: B1 0/4, B3 0/4 (MEANINGFUL needs 3)
* **Outcome: NOT MEANINGFUL**
