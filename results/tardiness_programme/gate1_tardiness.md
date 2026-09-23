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
