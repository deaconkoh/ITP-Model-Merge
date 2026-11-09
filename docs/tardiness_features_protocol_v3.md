# Does absolute lateness information help the tardiness specialist? Benchmark v3 (protocol, fixed before the run)

## Question

`results/tardiness_diagnosis_v3.md` found that, at 10x5, most of DANIEL-T's gap to CP-SAT comes from learning, not from the no-waiting action mask:
- DANIEL-T greedy reaches 247.7;
- the best schedule found without waiting reaches 50.5;
- CP-SAT reaches 16.1.

The tardiness reward is already dense. The network's two due-date operation features, however, are z-scored per instance at every step. So the network can tell which job has the least slack, but not whether any job is already late, and it never sees the current time.

This test adds absolute lateness information to the inputs and changes nothing else.

## The change: `--lateness_inputs on` (default `off`)

**What is added.** Five per-job features, repeated on every operation of the job, are appended to the operation features. All are in the environment's scaled time, raw / 99, the benchmark's fixed time scale. None is rescaled per step (`fjsp_env_same_op_nums.py: lateness_op_features`).

| # | Feature | Definition |
|---|---|---|
| 13 | current time | t = the decision time (`next_schedule_time`), the same for every job |
| 14 | time to due date | d_j − t, signed; negative means overdue |
| 15 | slack, absolute | d_j − EST_j − W_j: the existing slack before its z-score (EST_j = earliest start of the job's next operation, W_j = the job's remaining minimum work) |
| 16 | already late | 1 if t > d_j and the job is unfinished, else 0 |
| 17 | will be late | 1 if EST_j + W_j > d_j (earliest possible completion after the due date; for a finished job, its completion time), else 0 |

**What stays the same.** The 12 existing operation features are unchanged and are still z-scored, as before. The five new columns are appended after that z-score.

**Where they enter the network.** The operation features feed:
- the first operation attention block (4 heads, `op_attention_blocks.0.attention_*.W`);
- the edge input of the first machine attention block (`mch_attention_blocks.0.attention_*.W_edge`).

Those 8 matrices grow from 12 × 32 to 17 × 32: 1,280 more parameters, 29,602 → 30,882. Nothing else changes shape. The schema becomes `benchmark_v3_f17_m9_p10_late`, with operation input 12 → 17.

**The default leaves everything bit-for-bit unchanged.** A 5-update DANIEL-T run gives identical weights and validation values before the change, after it, and with `--lateness_inputs off`. This was checked against the pre-change commit, and the unit tests cover the features and the identity. The run identity records `lateness_inputs` only when it is on.

## Run

**What is trained:** the tardiness specialist from scratch with `--lateness_inputs on`, 10x5, seed 111. It uses exactly the committed DANIEL-T s111's command (`scripts/run_baselines_v3.sh`, `train_one`):
- config `configs/benchmark_v3/t.json`;
- generator training stream;
- validation every 50 updates;
- stopping rule: window 1,000, tolerance 0.5%, at least 1,000 updates, cap 5,000;
- chunks of 500 updates;
- CUDA, training pinned to CPUs 0–3 with 4 threads.

**Differences from the committed run, both stated in the report:**
- The first-layer shapes differ, so the random initialisation of every layer differs as well. The training-instance stream is the same.
- The committed run shared the PC with CP-SAT on CPUs 4–11, whereas this run is alone. Wall-clock is therefore comparable only approximately.

**Checkpoints** go to `checkpoints/tardiness_features_v3/`, with SHA-256 hashes in `SHA256SUMS.json`:
- the start network: the update-0 snapshot, saved on the first chunk only;
- the best-validation checkpoint.

**Evaluation:**
- the evaluation split (100 instances), greedy and best of 100 samples, sample seed 50, as for the committed specialists;
- the same 20 instances as the diagnosis, read from the same files.

## Report (`results/tardiness_features_v3.md`)

Plain names: "Tardiness specialist, current" (committed DANIEL-T, s111; the 3-seed mean is also shown) and "Tardiness specialist, with lateness inputs".

**Evaluation (100 instances):** for greedy and for best of 100 samples:
- mean tardiness;
- tardiness as a share of the MWKR+ECT reference rule's tardiness;
- the share of jobs that are late.

**Diagnosis instances (the first 20):** the new result is added to the gap-split table (CP-SAT 16.1, best without waiting 50.5, current DANIEL-T 247.7). The share of the learning gap closed is

(current − new) / (current − 50.5)

with current = 247.7, the 3-seed mean, as specified. It is also shown with current = s111 (238.2).

**Training:** the update of the best checkpoint, the stop update and status, and the training wall-clock, for both specialists.

**Merging and distillation:** with a different operation-input size, the new specialist cannot be weight-merged with, or distilled alongside, the current makespan and carbon specialists. The report says what adoption would need.

**Single seed:** numbers only, no significance claims.

## Rules

- The evaluation split only; the reserved final-test sets are never loaded.
- The committed specialists, results and frozen specs are read, never changed.
- Commit locally; never push.
