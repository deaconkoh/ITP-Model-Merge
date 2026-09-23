# Step 4 -- validation screen on the FROZEN due dates

Training-free greedy dispatcher, shared rule family (tools/validate_frozen_due_dates.py), 100
train_vali instances per size, due dates read from the hash-checked manifest (800424d).
Advantage = how much each objective's own specialist beats the makespan specialist on it.

| objective | 10x5 (k125s060) | 20x10 (k095s060) |
|---|---|---|
| carbon (orthogonal reference) | +41.4% | +59.1% |
| **tardiness (frozen due dates)** | **+10.2%** | **+7.3%** |
| priority, per-operation weights (retired) | +0.0% | +0.0% |
| priority, per-JOB weights (before/after, recorded only) | +4.8% | +0.0% |

makespan<->tardiness correlation across blended schedules: r = +0.023 (10x5), +0.499 (20x10).

Verdict: not near-degenerate, so the stop condition does not trigger. Caution recorded: the
tardiness advantage SHRINKS with size (10.2% -> 7.3%) and the correlation rises, the opposite
direction to carbon. Per-job priority weights recover a small separation at 10x5 that vanishes at
20x10 -- consistent with the augmentation finding, but not acted on.
