# Decoding strategies and the non-delay ceiling (benchmark v3, 10x5): protocol

Fixed before any decoding or constrained solve was run. Inference and CP-SAT only. No training; no change to network
weights, input features, action masks, scheduling constraints, the benchmark, or committed results. The reserved
final-test sets are never loaded.

## Questions (kept separate)

- **A.** Can better action selection improve the trained DANIEL policies without retraining?
- **B.** What is the best performance achievable under DANIEL's existing scheduling rule (the action mask), and is the
  gap to unrestricted CP-SAT due to that rule or to too little search inside it?

## Common settings

- **Checkpoints:** the three final multi-objective checkpoints of the CP-SAT audit and plateau diagnosis:
  - World-model base mix (`pd_wm_u2250`);
  - Random shared base mix (`pd_ctrl_u3650`);
  - From scratch (`pd_fs_u4550`).

  All are in `checkpoints/plateau_diagnosis_v3/part_a/`.
- **Instances:**
  - **Evaluation:** the first 20 instances of the 10x5 evaluation split, the instances with a CP-SAT B2 front, as in the audit.
  - **Validation (for choosing settings):** all 18 instances of the 10x5 validation split.
- **Preferences:** the 66-point simplex grid (step 0.1). Every method also runs the equal-weight vector, as the
  evaluation does, so that batches and sampling streams match the committed evaluations; equal weights are not in
  the metrics.
- **Selection score:** every complete schedule is ranked by the benchmark's preference-weighted normalised score,
  w_m MK/MKref + w_c C/Cref + w_t T/Tref, on the simulator's exact objectives. This is the score best-of-16 uses.
- **Metrics:**
  - hypervolume with the frozen normalisations (evaluation: `results/baselines_v3/normalisation_points.json`;
    validation: `results/warmstart_v3/norm/validation_normalisation_points_10x5.json`), reference point 1.1;
  - IGD+ against the B2 front (evaluation only);
  - unique non-dominated solutions;
  - mean preference-weighted regret: normalised space, against every schedule found for that instance by any method
    in this study, plus the CP-SAT front on evaluation.
- **Cost, per instance-preference pair:**
  - network state evaluations (rows passed through the network);
  - completed schedules;
  - wall-clock time (batched total) and single-pair latency (mean and worst) on a fixed subset of 20 pairs: instance
    i with grid preference (7 i) mod 66.

## Part A: decoders

1. **Greedy:** argmax action, ties to the lowest action index, as `evaluate_mo_v3`.
2. **Best-of-K sampling:**
   - K = 2, 4, 8, 16 are the first K of the 16 samples drawn exactly as `evaluate_mo_v3 --sample 16 --sample_seed 50`,
     so best-of-16 reproduces the committed evaluation.
   - Budget-matched extensions K = 32, 64, 128 are the first K of a separate stream of 128 (seed 51).
3. **Beam search, widths 2, 4, 8:**
   - Partial schedules are ranked by accumulated log-probability.
   - Every legal action of every kept node is a candidate.
   - Children that reach an identical partial schedule (same operations on the same machines at the same start times,
     in any order) are merged, keeping the higher log-probability.
   - Ties go to the lower parent and action index.
   - At completion, the kept schedules are ranked by the selection score.
   - Widths 16 and 32 are budget-matched extensions.
   - Each branch has its own simulator state, copied row-wise.
   - Width 1 must reproduce greedy exactly.
4. **Limited lookahead (rollout action selection):**
   - At every step with more than one legal action, take DANIEL's top-k legal actions by probability.
   - Execute each on a copy of the simulator and complete the schedule with the frozen policy.
   - Execute only the candidate with the best selection score; ties go to the higher-probability action.
   - The greedy continuation of the greedy action is the incumbent schedule and is reused, not recomputed.
   - The result is the executed schedule.
   - Variants:
     - **LA-G2:** top 2, greedy continuation.
     - **LA-G3:** top 3, greedy continuation.
     - **LA-S2:** top 2, greedy plus 3 sampled continuations per candidate (seed 52); the candidate's value is its best continuation.
   - No learned value function, no new model.

**Choosing settings on validation, not on evaluation:**
- Every method above runs on validation first.
- The **selected improved decoder** for Part B3 is the method with the highest validation hypervolume, averaged over the
  three checkpoints, among methods that use no more state evaluations per pair than best-of-128 (6,400).
- A second, budget-matched selection is the best method within the best-of-16 budget (800 state evaluations per pair).
- All methods are also reported on evaluation, but evaluation numbers are not used for any choice.

## Part B: the non-delay ceiling

- **B1. The exact action space.** From `fjsp_env_same_op_nums.py`, documented in the report.
  - The environment's mask is checked against an exact integer computation of the same rule along every step of every
    pool and Part A schedule, because the environment computes times in scaled floating point.
- **B2. CP-SAT restricted to DANIEL's rule.**
  - Model: the CP-SAT B1/B2 model of `tools/cpsat_baselines_v3.py`, plus one constraint for every machine m, every
    operation o eligible on m, and every pair of consecutive operations (a, b) on m (including the start and the end
    of m's sequence): m may not be idle during [r_o, s_o), where r_o is the completion of o's job predecessor.
    Consecutive operations are encoded with a circuit constraint per machine.
  - **Equivalence checks before any result is interpreted:**
    1. pool schedules, all made by DANIEL's rule, must be feasible when fixed;
    2. left-shifted CP-SAT schedules, which are not non-delay, must be infeasible when fixed;
    3. every returned schedule must be reproduced exactly by DANIEL's decision rule (replay through the environment)
       and re-scored by the independent scorer.
  - **Solves:** one per evaluation instance and grid preference (20 x 66). The objective is the selection score, with
    integer coefficients rounded down so that a proven bound stays valid. The hint is the best known non-delay
    schedule for that pair.
  - **Time limit:** set from a pilot on 2 validation instances.
  - **Proven bounds:**
    - Each solve's best bound gives a proven lower bound on the selection score of any schedule DANIEL can produce, for
      that preference.
    - The 66 bounds of an instance give a proven upper bound on the hypervolume of any set of DANIEL schedules for that
      instance: the volume, within the reference box, of the region allowed by all 66 bounds, computed on a grid that
      over-counts.
  - Proven values are reported separately from best-known values. A best-known value is never called a ceiling unless
    every solve is proven optimal.
- **B3.** The four-way comparison:
  - greedy;
  - the selected improved decoder;
  - best known non-delay, including the new B2 schedules;
  - unrestricted CP-SAT.

  Reported overall and by preference group (makespan-, carbon-, tardiness-heavy: weight >= 0.8; mixed).

Single seed per method: numbers only, no significance claims.

## Amendment (2026-11-13, after the validation pilot, before any evaluation solve)

The pilot of the restricted model (validation instances 0 and 1, preferences 0, 10, 30, 45 and 65, limits 10, 30 and
60 s, 6 workers, no hint) proved no optimum within 60 s. Its schedules were often worse than DANIEL's and most bounds
were weak. Solving all 1,320 evaluation pairs exactly is therefore out of reach. Part B2 becomes:

1. **Proven bounds on a representative subset:**
   - All 20 evaluation instances x 10 grid preferences: the three corners; the three edge midpoints (0.5, 0.5, 0),
     (0.5, 0, 0.5), (0, 0.5, 0.5); and (0.6, 0.2, 0.2), (0.2, 0.6, 0.2), (0.2, 0.2, 0.6), (0.4, 0.3, 0.3).
   - For each pair:
     - **Restricted (non-delay) solve:** 60 s, 6 workers, hinted with the best known DANIEL schedule.
     - **Unrestricted solve:** the same objective, 60 s, 6 workers, hinted with the best CP-SAT front point.
   - A proven restricted lower bound above the proven unrestricted optimum proves that DANIEL's rule costs at least
     the difference for that preference.
   - The DANIEL rule masks some exact ties, so DANIEL's schedules are a subset of the exact non-delay schedules. The
     check found 871 of 62,000 states affected; the float rule never allows a pair the exact rule forbids. The
     restricted bound is therefore also a bound for DANIEL.
2. **Empirical best-known front for all 66 preferences:**
   - A random-key genetic search over schedules decoded by DANIEL's own rule (the float arithmetic of the environment,
     so every result is DANIEL-reachable). The decoder is that of `results/tardiness_diagnosis_v3.md`.
   - Fitness: the selection score of each preference.
   - Seeds: the 20 best known schedules for that instance and preference (CP-SAT audit pool and every Part A decoder).
   - Settings: population 100, elite 10, 300 generations, the earlier mutation and crossover settings, seed 2026.
   - New schedules count as best-known values only.
3. Schedules returned by the restricted CP-SAT model are replayed through DANIEL's rule. Those it cannot reproduce
   (an exact tie the float rule masks) are reported, and they are not counted as DANIEL-reachable.
