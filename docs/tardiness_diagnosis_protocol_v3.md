# Why is DANIEL-T so far from CP-SAT? Tardiness diagnosis on benchmark v3 (protocol, fixed before the run)

## Question

On the evaluation split, total tardiness for the committed tardiness specialist (DANIEL-T, greedy, mean of seeds 111/222/333) against CP-SAT's single-objective optimum (`results/baselines_v3.md`):

| Size | DANIEL-T | CP-SAT |
|---|---|---|
| 10x5 | 244.9 | 17.0 |
| 20x10 | 1206.9 | 89.5 |

Is the gap caused by a sparse reward, or by something else? The specialists are those committed in `checkpoints/specialists_v3/`, with their committed evaluations in `results/baselines_v3/eval/`.

## Part A (no training): `tools/tardiness_diagnosis_v3.py`

1. **Reward.** Quote the code that computes the per-step tardiness reward and the makespan reward, and say whether each is sparse or dense.
   - **Measured:** along DANIEL-T s111's greedy schedules on the subset below, replayed through the environment with goal `t`:
     - the share of steps with a non-zero tardiness reward;
     - the share of steps where a late job completes, which is when a sparse reward would fire;
     - the same measures for the makespan potential;
     - the share of each return collected in the first half of the episode.
2. **Inputs.** List every due-date feature the network sees and how each is normalised, quoted from `fjsp_env_same_op_nums.py`.
3. **The gap in plain units.** All 100 evaluation instances per size. For DANIEL-T (greedy and best of 100 samples, per seed and mean) and CP-SAT (B1 tardiness point), report:
   - absolute tardiness;
   - tardiness as a share of the benchmark's MWKR+ECT reference rule;
   - the share of jobs that are late;
   - the number of instances with zero tardiness.
4. **Gap split.** The first 20 evaluation instances per size (`10x5+mix_001` to `_020`, `20x10+mix_001` to `_020`). Every one of them has a CP-SAT single-objective tardiness result.
   - **a) CP-SAT tardiness:** the B1 `tard` point, as committed. Its status is reported: optimal or best-known.
   - **b) Best tardiness without waiting:** the best schedule found that DANIEL's decision rule can produce. The environment's action mask allows only the pairs that can start at the earliest possible start time (non-delay). The search:
     - **Method:** random-key genetic search. A key is given to every (operation, machine) pair, and at each step the allowed pair with the smallest key is chosen. Every non-delay schedule is reachable this way.
     - **Settings:** population 200 per instance, elite 20, uniform crossover per operation, mutation of 5% of keys by N(0, 0.1), 5% random newcomers, generator seed 2026.
     - **Limit:** a fixed number of generations, 2,000 at 10x5 and 1,000 at 20x10 (about 5 and 50 minutes on this PC).
     - **Seeds in the initial population:** the three rules in (c); CP-SAT's own schedule order decoded through the mask (also reported on its own); and DANIEL-T's committed greedy and best-of-100 schedules for all three seeds.
     - **What the result is:** an upper bound on the true best non-delay tardiness, not a proof of it.
   - **c) Rules under the same mask:**
     - EDD+ECT;
     - MWKR+ECT;
     - LST+ECT (least slack, the network's own slack quantity).

     How the rules choose: the job by the rule among jobs with an allowed pair, ties to the lowest index; then the machine by earliest completion among the allowed machines. For context, the same rules with waiting allowed are also reported (`tools/objective_validation/schedules.py`, active-style).
   - **d) DANIEL-T:** greedy and best of 100 sampled schedules, from the committed evaluations. These are per seed and mean.

**Implementation and checks:**
- The search uses a lean batched copy of the mask (`NonDelay`). It is tested to give the environment's `dynamic_pair_mask` exactly, at every step.
- Every reported schedule is replayed through the real environment and re-scored by `benchmark_v3.schedule_objectives`. The run stops on any disagreement.

## Part B: only if the reward is sparse

Part B (a dense tardiness reward option and one retraining at 10x5, seed 111) runs only if step A1 finds the reward sparse. The reward is already a per-step change in a lower-bound tardiness estimate (`objectives.tardiness_potential`), so Part B is expected to be skipped. The report says so.

## Rules

- No training.
- The evaluation split only; the reserved final-test sets are never loaded.
- The committed results and specialists are read, never changed.
- Single seed for the search; numbers only, no significance claims.
- Report: `results/tardiness_diagnosis_v3.md`; raw outputs: `results/tardiness_diagnosis_v3/`.
