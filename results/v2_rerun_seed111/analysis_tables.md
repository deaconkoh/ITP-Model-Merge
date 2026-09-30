Seeds: [111]; 49 HELD-OUT instances. EXPLORATORY: confidence intervals are paired bootstrap intervals over instances (10,000 resamples) and reflect instance variation only, not seed variation.

Arm status: V2-M makespan spec, v2: **complete**, V2-S scratch, v2: **complete**, V1-CONT C1 continued, v1: **complete**, V2-T v2 tardiness spec, v2: **complete**.

### Step 4 score per checkpoint (mean over instances; lower is better)

| arm | u0 | u50 | u100 | u250 | u500 | u1000 | u1500 | u2000 | u2500 | u3000 |
|---|---|---|---|---|---|---|---|---|---|---|
| V2-M makespan spec, v2 | 0.8756 | 0.8572 | 0.8325 | 0.8621 | 0.8853 | 0.8370 | 0.8436 | 0.8421 | 0.8462 | 0.8733 |
| V2-S scratch, v2 | 1.1935 | 1.0279 | 0.8596 | 0.8611 | 0.8175 | 0.8397 | 0.8332 | 0.8088 | 0.8430 | 0.8390 |
| V1-CONT C1 continued, v1 | - | - | - | - | - | 0.8556 | 0.8642 | 0.8671 | 0.8593 | 0.8534 |
| V2-T v2 tardiness spec, v2 | 0.8504 | 0.8384 | 0.8640 | 0.8435 | 0.8492 | 0.8393 | 0.8590 | 0.8844 | 0.8729 | 0.8766 |
| v1 Step 4 C1 (makespan spec) (reference) | 0.8756 | 0.9391 | 0.9222 | 0.8551 | 0.8554 | 0.8556 | - | - | - | - |
| v1 Step 4 scratch (reference) | 1.2920 | 1.2499 | 1.2578 | 1.0639 | 0.8869 | 0.8648 | - | - | - | - |
| CP-SAT 60 s best (ceiling_check) | 0.6144 |  |  |  |  |  |  |  |  |  |

### Gap to the CP-SAT best per checkpoint: mean of (score - CP-SAT) / CP-SAT, 95% bootstrap CI

| arm | checkpoint | gap | 95% CI |
|---|---|---|---|
| V2-M makespan spec, v2 | u0 | +42.8% | [+39.2, +46.4] |
| V2-M makespan spec, v2 | u50 | +39.5% | [+35.9, +43.0] |
| V2-M makespan spec, v2 | u100 | +35.6% | [+33.3, +37.8] |
| V2-M makespan spec, v2 | u250 | +40.3% | [+37.1, +43.5] |
| V2-M makespan spec, v2 | u500 | +44.0% | [+40.2, +48.1] |
| V2-M makespan spec, v2 | u1000 | +36.1% | [+32.8, +39.4] |
| V2-M makespan spec, v2 | u1500 | +37.3% | [+33.7, +40.9] |
| V2-M makespan spec, v2 | u2000 | +37.1% | [+33.8, +40.6] |
| V2-M makespan spec, v2 | u2500 | +37.9% | [+34.3, +41.5] |
| V2-M makespan spec, v2 | u3000 | +42.3% | [+38.8, +45.9] |
| V2-S scratch, v2 | u0 | +95.4% | [+89.1, +101.9] |
| V2-S scratch, v2 | u50 | +67.7% | [+63.9, +71.7] |
| V2-S scratch, v2 | u100 | +39.8% | [+36.7, +43.1] |
| V2-S scratch, v2 | u250 | +40.1% | [+37.4, +43.0] |
| V2-S scratch, v2 | u500 | +32.9% | [+30.5, +35.4] |
| V2-S scratch, v2 | u1000 | +36.5% | [+33.2, +40.0] |
| V2-S scratch, v2 | u1500 | +35.5% | [+32.4, +38.7] |
| V2-S scratch, v2 | u2000 | +31.8% | [+28.7, +34.9] |
| V2-S scratch, v2 | u2500 | +37.2% | [+34.2, +40.4] |
| V2-S scratch, v2 | u3000 | +36.7% | [+33.8, +39.7] |
| V1-CONT C1 continued, v1 | u1000 | +39.4% | [+36.7, +42.4] |
| V1-CONT C1 continued, v1 | u1500 | +41.1% | [+38.1, +44.3] |
| V1-CONT C1 continued, v1 | u2000 | +41.4% | [+38.4, +44.6] |
| V1-CONT C1 continued, v1 | u2500 | +40.0% | [+37.0, +43.1] |
| V1-CONT C1 continued, v1 | u3000 | +39.0% | [+35.8, +42.2] |
| V2-T v2 tardiness spec, v2 | u0 | +38.4% | [+35.3, +41.5] |
| V2-T v2 tardiness spec, v2 | u50 | +36.6% | [+33.6, +39.5] |
| V2-T v2 tardiness spec, v2 | u100 | +40.6% | [+37.5, +43.9] |
| V2-T v2 tardiness spec, v2 | u250 | +37.2% | [+35.0, +39.3] |
| V2-T v2 tardiness spec, v2 | u500 | +38.1% | [+35.3, +40.9] |
| V2-T v2 tardiness spec, v2 | u1000 | +36.6% | [+33.6, +39.7] |
| V2-T v2 tardiness spec, v2 | u1500 | +39.9% | [+36.5, +43.4] |
| V2-T v2 tardiness spec, v2 | u2000 | +43.9% | [+40.5, +47.2] |
| V2-T v2 tardiness spec, v2 | u2500 | +42.3% | [+38.5, +46.2] |
| V2-T v2 tardiness spec, v2 | u3000 | +42.7% | [+39.7, +45.5] |
| v1 Step 4 C1 (reference) | u1000 | +39.4% | [+36.7, +42.4] |

### Raw objectives and due-date behaviour at updates 1,000 and 3,000

Spearman = Spearman(d_j/P_j, C_j/P_j) over the HELD-OUT jobs, as input_reward_audit (v1 C1 about -0.02, CP-SAT +0.74); Q1 late = share of the tightest-quartile jobs that are late.

| arm | update | makespan | carbon | tardiness | score | Spearman | Q1 late |
|---|---|---|---|---|---|---|---|
| V2-M makespan spec, v2 | 1000 | 441.1 | 2182.0 | 398.2 | 0.8370 | +0.320 | 91.1% |
| V2-M makespan spec, v2 | 3000 | 431.7 | 2295.5 | 451.6 | 0.8733 | +0.200 | 88.6% |
| V2-S scratch, v2 | 1000 | 446.0 | 2205.3 | 385.9 | 0.8397 | +0.251 | 81.3% |
| V2-S scratch, v2 | 3000 | 433.8 | 2231.3 | 391.1 | 0.8390 | +0.265 | 86.2% |
| V1-CONT C1 continued, v1 | 1000 | 449.8 | 2432.6 | 329.2 | 0.8556 | -0.006 | 84.6% |
| V1-CONT C1 continued, v1 | 3000 | 435.6 | 2490.1 | 323.5 | 0.8534 | -0.118 | 84.6% |
| V2-T v2 tardiness spec, v2 | 1000 | 443.9 | 2281.7 | 357.3 | 0.8393 | +0.358 | 79.7% |
| V2-T v2 tardiness spec, v2 | 3000 | 448.2 | 2199.3 | 470.2 | 0.8766 | +0.268 | 82.9% |
| v1 Step 4 C1 (reference) | 1000 | 449.8 | 2432.6 | 329.2 | 0.8556 | (V1-CONT u1000) | (V1-CONT u1000) |
| CP-SAT 60 s best | - | 375.8 | 2010.3 | 54.6 | 0.6144 | +0.716 | 30.9% |

### Paired comparisons (relative gain of X over Y = (mean Y - mean X) / mean Y; positive = X better)

| rule | X | Y | gain | 95% CI | criterion | met |
|---|---|---|---|---|---|---|
| Q1 | V2-M@1000 | v1 C1 (Step 4)@1000 | +2.17% | [-1.20, +5.38] | gain >= 2%, CI > 0 | no |
| Q1 | V2-M@3000 | V1-CONT@3000 | -2.34% | [-5.74, +0.90] | gain >= 2%, CI > 0 | no |
| Q2 | V2-M@3000 | V2-M@1000 | -4.33% | [-7.32, -1.73] | gain >= 1%, CI > 0 | no |
| Q2 (report) | V1-CONT@3000 | V1-CONT@1000 | +0.26% | [-2.68, +3.06] | gain >= 1%, CI > 0 | no |
| Q3 | V2-M@250 | V2-S@250 | -0.11% | [-2.42, +2.18] | gain >= 2%, CI > 0 | no |
| Q3 | V2-M@500 | V2-S@500 | -8.30% | [-11.61, -5.18] | gain >= 2%, CI > 0 | no |
| Q3 (catch-up, report) | V2-S@3000 | V2-M@3000 | +3.93% | [+1.76, +5.93] | gain >= 0%, CI > 0 | yes |
| Q4 | V2-T@1000 | V2-M@1000 | -0.27% | [-3.00, +2.33] | gain >= 1%, CI > 0 | no |
| Q4 | V2-T@3000 | V2-M@3000 | -0.38% | [-2.90, +2.05] | gain >= 1%, CI > 0 | no |

### Cost (GPU wall-clock of training chunks; evaluation excluded)

| arm | updates trained | hours | s/update |
|---|---|---|---|
| V2-M makespan spec, v2 | 3000 | 0.63 | 0.753 |
| V2-S scratch, v2 | 3000 | 0.65 | 0.777 |
| V1-CONT C1 continued, v1 | 2000 | 0.42 | 0.765 |
| V2-T v2 tardiness spec, v2 | 3000 | 0.62 | 0.748 |

### Verdicts (exploratory, provisional until the 4-seed confirmation)

* **Q1: NOT LIFTED**
* **Q2: PLATEAU**
* **Q2 (V1-CONT, reported): PLATEAU**
* **Q3: NO**
* **Q3 catch-up (V2-S@3000 within +-1% of V2-M@3000, CI includes 0): NOT CAUGHT UP**
* **Q3 first V2-S checkpoint <= V2-M@1000: u500**
* **Q4 @1000: TIED**
* **Q4 @3000: TIED**

![score vs updates](v2_rerun_seed111/score_vs_updates.png)

*Solid: this study (mean over seeds). Dashed: the existing v1 Step 4 curves (0-1,000) for reference. Black: CP-SAT best (ceiling_check, 60 s).*
