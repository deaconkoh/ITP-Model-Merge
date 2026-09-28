### Part C: tardiness specialists on the 49 HELD-OUT instances (means over instances; pooled = over 4 seeds)

| arm | seed | mean total tardiness | jobs late | tightest-quartile jobs late | Spearman(d/P, C/P) | makespan | carbon | Step 4 score |
|---|---|---|---|---|---|---|---|---|
| v1 tardiness specialist | 111 | 336.1 | 48.4% | 87.8% | -0.069 | 449.7 | 2503.2 | 0.8706 |
| v1 tardiness specialist | 222 | 347.4 | 47.1% | 75.6% | -0.041 | 454.2 | 2558.3 | 0.8863 |
| v1 tardiness specialist | 333 | 380.8 | 48.4% | 82.1% | -0.029 | 466.0 | 2493.8 | 0.9073 |
| v1 tardiness specialist | 444 | 330.5 | 45.1% | 83.7% | -0.092 | 443.0 | 2502.5 | 0.9401 |
| v1 tardiness specialist | pooled | 348.7 | 47.2% | 82.3% | -0.057 | 453.2 | 2514.4 | 0.9011 |
| V2: retrained on v2 | 111 | 305.8 | 48.4% | 81.3% | +0.150 | 446.4 | 2476.2 | 0.8504 |
| V2: retrained on v2 | 222 | 299.2 | 56.3% | 74.8% | +0.450 | 462.4 | 2472.0 | 0.8556 |
| V2: retrained on v2 | 333 | 278.8 | 48.2% | 75.6% | +0.283 | 455.4 | 2511.0 | 0.8541 |
| V2: retrained on v2 | 444 | 244.9 | 43.3% | 74.8% | +0.224 | 444.4 | 2520.8 | 0.8862 |
| V2: retrained on v2 | pooled | 282.2 | 49.0% | 76.6% | +0.275 | 452.1 | 2495.0 | 0.8616 |
| V1-CONT: v1 + 1000 updates (control) | 111 | 346.7 | 47.6% | 80.5% | -0.078 | 453.4 | 2560.5 | 0.8883 |
| V1-CONT: v1 + 1000 updates (control) | 222 | 350.9 | 46.9% | 85.4% | -0.060 | 471.4 | 2450.4 | 0.8817 |
| V1-CONT: v1 + 1000 updates (control) | 333 | 336.0 | 45.9% | 86.2% | -0.044 | 450.5 | 2502.0 | 0.8759 |
| V1-CONT: v1 + 1000 updates (control) | 444 | 347.8 | 46.3% | 81.3% | -0.040 | 458.8 | 2546.8 | 0.9706 |
| V1-CONT: v1 + 1000 updates (control) | pooled | 345.4 | 46.7% | 83.3% | -0.055 | 458.5 | 2514.9 | 0.9041 |
| CP-SAT 60 s best (ceiling_check) | pooled | 50.6 | 19.3% | - | +0.738 | - | - | 0.6140 |

**Tardiness reduction of V2: retrained on v2 vs v1 tardiness specialist** (paired over instances; 95% bootstrap CI, 10,000 resamples)

| seed | reduction | 95% CI | counts (>= 20% and CI excludes 0) |
|---|---|---|---|
| 111 | +9.0% | [-2.0, +19.0] | no |
| 222 | +13.9% | [+2.5, +24.1] | no |
| 333 | +26.8% | [+16.3, +37.0] | yes |
| 444 | +25.9% | [+15.7, +36.0] | yes |

**Tardiness reduction of V1-CONT: v1 + 1000 updates (control) vs v1 tardiness specialist** (paired over instances; 95% bootstrap CI, 10,000 resamples)

| seed | reduction | 95% CI | counts (>= 20% and CI excludes 0) |
|---|---|---|---|
| 111 | -3.2% | [-15.7, +7.6] | no |
| 222 | -1.0% | [-12.3, +9.4] | no |
| 333 | +11.8% | [+1.9, +21.3] | no |
| 444 | -5.2% | [-18.3, +6.0] | no |

**Tardiness reduction of V2: retrained on v2 vs V1-CONT: v1 + 1000 updates (control)** (paired over instances; 95% bootstrap CI, 10,000 resamples)

| seed | reduction | 95% CI | counts (>= 20% and CI excludes 0) |
|---|---|---|---|
| 111 | +11.8% | [+1.1, +22.2] | no |
| 222 | +14.7% | [+2.5, +25.5] | no |
| 333 | +17.0% | [+6.3, +27.2] | no |
| 444 | +29.6% | [+20.5, +38.8] | yes |

### Verdict: **NO EFFECT** (tardiness criterion 2/4 seeds not met; pooled tightness correlation +0.275 < +0.3)
