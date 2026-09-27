### Step 4 score on HELD-OUT at every snapshot (lower is better): pooled mean over seeds (sd across seeds)

| arm | u0 | u50 | u100 | u250 | u500 | u1000 |
|---|---|---|---|---|---|---|
| C1 makespan spec (Step 4) | 0.9096 (0.042) | 0.9255 (0.046) | 0.9175 (0.040) | 0.8847 (0.047) | 0.8804 (0.042) | 0.8655 (0.029) |
| C2 makespan spec + fresh critic | 0.9096 (0.042) | 0.9096 (0.042) | 0.9377 (0.052) | 0.8879 (0.024) | 0.8720 (0.028) | 0.8845 (0.030) |
| C3 merge + fresh critic | 0.8926 (0.031) | 0.8926 (0.031) | 0.8926 (0.043) | 0.9059 (0.043) | 0.8884 (0.028) | 0.8937 (0.042) |
| F1 blend, joint fine-tune | 0.8853 (0.041) | 0.8853 (0.041) | 0.9224 (0.034) | 0.8884 (0.032) | 0.8712 (0.022) | 0.8668 (0.031) |
| F2 blend + residual | 0.8853 (0.041) | 0.8853 (0.041) | 0.8855 (0.044) | 0.8812 (0.040) | 0.8743 (0.038) | 0.8705 (0.036) |
| Step 4 merge | 0.8926 (0.031) | 0.9043 (0.039) | 0.9215 (0.099) | 0.9035 (0.044) | 0.8956 (0.046) | 0.9030 (0.036) |
| Step 4 scratch | 1.4309 (0.121) | 1.3013 (0.049) | 1.3031 (0.051) | 1.0619 (0.037) | 0.9198 (0.043) | 0.9121 (0.055) |

Per seed:

| arm | seed | u0 | u50 | u100 | u250 | u500 | u1000 |
|---|---|---|---|---|---|---|---|
| C1 makespan spec (Step 4) | 111 | 0.8756 | 0.9391 | 0.9222 | 0.8551 | 0.8554 | 0.8556 |
| C1 makespan spec (Step 4) | 222 | 0.8786 | 0.9075 | 0.8896 | 0.8918 | 0.8574 | 0.8395 |
| C1 makespan spec (Step 4) | 333 | 0.9198 | 0.8740 | 0.8856 | 0.8434 | 0.8655 | 0.8609 |
| C1 makespan spec (Step 4) | 444 | 0.9644 | 0.9814 | 0.9728 | 0.9486 | 0.9436 | 0.9060 |
| C2 makespan spec + fresh critic | 111 | 0.8756 | 0.8756 | 0.9530 | 0.8901 | 0.8447 | 0.8648 |
| C2 makespan spec + fresh critic | 222 | 0.8786 | 0.8786 | 0.9181 | 0.8663 | 0.8648 | 0.8603 |
| C2 makespan spec + fresh critic | 333 | 0.9198 | 0.9198 | 0.8791 | 0.8743 | 0.8682 | 0.8876 |
| C2 makespan spec + fresh critic | 444 | 0.9644 | 0.9644 | 1.0006 | 0.9211 | 0.9105 | 0.9255 |
| C3 merge + fresh critic | 111 | 0.8594 | 0.8594 | 0.8543 | 0.8689 | 0.8664 | 0.8697 |
| C3 merge + fresh critic | 222 | 0.8825 | 0.8825 | 0.8812 | 0.9324 | 0.8924 | 0.8821 |
| C3 merge + fresh critic | 333 | 0.8953 | 0.8953 | 0.8807 | 0.8695 | 0.8677 | 0.8671 |
| C3 merge + fresh critic | 444 | 0.9333 | 0.9333 | 0.9540 | 0.9530 | 0.9270 | 0.9558 |
| F1 blend, joint fine-tune | 111 | 0.8584 | 0.8584 | 0.9636 | 0.8530 | 0.8596 | 0.8485 |
| F1 blend, joint fine-tune | 222 | 0.8513 | 0.8513 | 0.8841 | 0.8801 | 0.8571 | 0.8567 |
| F1 blend, joint fine-tune | 333 | 0.8894 | 0.8894 | 0.9105 | 0.8897 | 0.8644 | 0.8488 |
| F1 blend, joint fine-tune | 444 | 0.9420 | 0.9420 | 0.9315 | 0.9309 | 0.9036 | 0.9134 |
| F2 blend + residual | 111 | 0.8584 | 0.8584 | 0.8560 | 0.8664 | 0.8568 | 0.8569 |
| F2 blend + residual | 222 | 0.8513 | 0.8513 | 0.8513 | 0.8417 | 0.8595 | 0.8465 |
| F2 blend + residual | 333 | 0.8894 | 0.8894 | 0.8888 | 0.8816 | 0.8494 | 0.8542 |
| F2 blend + residual | 444 | 0.9420 | 0.9420 | 0.9460 | 0.9353 | 0.9317 | 0.9243 |
| Step 4 merge | 111 | 0.8594 | 0.8866 | 0.8691 | 0.8619 | 0.8622 | 0.8907 |
| Step 4 merge | 222 | 0.8825 | 0.8840 | 0.8684 | 0.8929 | 0.8888 | 0.8703 |
| Step 4 merge | 333 | 0.8953 | 0.8839 | 0.8788 | 0.8941 | 0.8693 | 0.8972 |
| Step 4 merge | 444 | 0.9333 | 0.9625 | 1.0696 | 0.9650 | 0.9620 | 0.9538 |
| Step 4 scratch | 111 | 1.2920 | 1.2499 | 1.2578 | 1.0639 | 0.8869 | 0.8648 |
| Step 4 scratch | 222 | 1.5361 | 1.3681 | 1.2601 | 1.0413 | 0.9336 | 0.8737 |
| Step 4 scratch | 333 | 1.3675 | 1.2972 | 1.3488 | 1.0294 | 0.8834 | 0.9248 |
| Step 4 scratch | 444 | 1.5282 | 1.2901 | 1.3457 | 1.1129 | 0.9752 | 0.9853 |

### Raw objectives at update 1000 on HELD-OUT (mean over instances, then seeds)

| arm | makespan | carbon | tardiness | score |
|---|---|---|---|---|
| C1 makespan spec (Step 4) | 444.5 | 2403.1 | 327.1 | 0.8655 |
| C2 makespan spec + fresh critic | 451.1 | 2414.2 | 355.0 | 0.8845 |
| C3 merge + fresh critic | 453.1 | 2435.1 | 358.0 | 0.8937 |
| F1 blend, joint fine-tune | 435.1 | 2433.5 | 330.3 | 0.8668 |
| F2 blend + residual | 430.8 | 2432.4 | 343.4 | 0.8705 |
| Step 4 merge | 448.4 | 2486.1 | 366.0 | 0.9030 |
| Step 4 scratch | 453.0 | 2501.3 | 369.3 | 0.9121 |

### Paired comparisons at update 1000 on HELD-OUT

Best single-specialist control (lower pooled score at update 1000 among C1 and C2): **C1 makespan spec (Step 4)** (0.8655; C1 makespan spec (Step 4): 0.8655, C2 makespan spec + fresh critic: 0.8845).

Relative gain of X over Y = (mean Y - mean X)/mean Y, positive = X better; 95% CI from 10,000 held-out instance resamples. * = gain >= 1% with CI above 0; ! = gain <= -1% with CI below 0 (Y beats X).

| X vs Y | seed 111 | seed 222 | seed 333 | seed 444 | pooled | X beats Y | Y beats X |
|---|---|---|---|---|---|---|---|
| F1 blend, joint fine-tune vs C1 makespan spec (Step 4) | +0.82% [-1.61, +3.17] | -2.05% [-5.57, +1.46] | +1.41% [-1.14, +3.80] | -0.81% [-4.00, +2.31] | -0.16% [-1.80, +1.49] | 0/4 | 0/4 |
| F2 blend + residual vs C1 makespan spec (Step 4) | -0.16% [-2.48, +2.25] | -0.84% [-3.46, +1.86] | +0.78% [-2.05, +3.55] | -2.02% [-4.79, +0.79] | -0.58% [-1.95, +0.72] | 0/4 | 0/4 |
| C3 merge + fresh critic vs Step 4 merge | +2.35% [-0.32, +4.99] | -1.35% [-4.44, +1.55] | +3.35% [+0.84, +5.90] * | -0.21% [-3.78, +3.37] | +1.03% [-0.57, +2.63] | 1/4 | 0/4 |

### Updates-to-threshold: first snapshot at which the arm's score is <= C1 makespan spec (Step 4) at update 1000

| arm | pooled | seed 111 | seed 222 | seed 333 | seed 444 | seeds reaching it by update 500 |
|---|---|---|---|---|---|---|
| C1 makespan spec (Step 4) | 1000 | 250 | 1000 | 250 | 1000 | 2 |
| C2 makespan spec + fresh critic | never | 500 | never | never | never | 1 |
| C3 merge + fresh critic | never | 100 | never | never | never | 1 |
| F1 blend, joint fine-tune | never | 250 | never | 1000 | 500 | 2 |
| F2 blend + residual | never | never | never | 500 | never | 1 |
| Step 4 merge | never | never | never | never | never | 0 |
| Step 4 scratch | never | never | never | never | never | 0 |

### Cost (training wall-clock on one RTX 5060 Ti; GPU-bound, so wall-clock ~ GPU time)

| arm | trainable parameters | training min per seed (mean, range) | GPU-hours, all seeds | x single-network cost |
|---|---|---|---|---|
| F1 blend, joint fine-tune | 94,823 | 32.2 (31.1-32.7) | 2.14 | 2.5x |
| F2 blend + residual | 19,714 | 25.0 (24.9-25.1) | 1.67 | 1.9x |
| C2 makespan spec + fresh critic | 29,154 | 13.0 (12.9-13.1) | 0.87 | 1.0x |
| C3 merge + fresh critic | 29,154 | 13.2 (13.1-13.3) | 0.88 | 1.0x |
| C1 makespan spec (Step 4) | 29,154 | reused (Step 4 took 11.8-15 min per run) | 0 (reused) | 1.0x |

### Verdict inputs

* F1 blend, joint fine-tune: beats C1 makespan spec (Step 4) at update 1000 (gain >= 1%, CI > 0) at 0/4 seeds; reaches its update-1000 score by update 500 at 2/4 seeds (WIN needs 3 on either)
* F2 blend + residual: beats C1 makespan spec (Step 4) at update 1000 (gain >= 1%, CI > 0) at 0/4 seeds; reaches its update-1000 score by update 500 at 1/4 seeds (WIN needs 3 on either)
* C1 makespan spec (Step 4) beats BOTH F1 and F2 at the same seed at 0/4 seeds (LOSE needs 3)
* **Primary outcome: TIE**
* C3 beats Step 4 merge at update 1000 at 1/4 seeds (needs 3) -> **merge stall: not supported**
