# From-scratch DANIEL trained with waiting allowed (benchmark v3, 10x5, seed 111): protocol

## Control

The 0.754 no-delay cell of `results/decoding_ceiling_v3_hv_table.md`:
- **Checkpoint:** `pd_fs_u4550`, the final state of From scratch.
- **Phase 1:** `10x5+mix+bv3_cl_CLB_s111`, multi-objective contrastive training from a random start, 2,600 updates, no
  stopping rule.
- **Phase 2:** `10x5+mix+bv3_cc_CCB_s111`, continued from phase 1's saved state with the convergence rule (window
  1000, tolerance 0.005, minimum 1000, maximum 8000). It stopped CONVERGED at update 4,550, and the final state was used.
- **Evaluation:** best-of-128 sampling (seed 51) on the first 20 evaluation instances x the 10 subset preferences, HV
  with `results/baselines_v3/normalisation_points.json`, reference point 1.1, mean over instances.
- **Rescored from the saved schedules:** 0.7541.

## The only change

`--dispatch_rule semi_active` (new option; the default `nondelay` leaves everything unchanged):
- The environment drops the one mask line that forbids pairs that cannot start at the earliest possible time
  (`fjsp_env_same_op_nums.py`, `unavailable_pairs`).
- Every eligible pair of every job's next operation is legal, and it starts at max(job ready, machine free).
- Machine eligibility, job precedence and machine capacity are kept; a machine may wait.
- The reachable schedules are the semi-active ones.
- The rule applies in training, in in-training validation and in evaluation.

## Matched

- **Command line:** rebuilt from each control phase's saved `run.json` (the original launch script is not in the
  repository). The new run differs only in the run name and the dispatch rule.
- **Check before launch:** run with the no-delay rule, the rebuilt command reproduces the control's first 100 updates
  exactly (losses and validation HV), from the identical start network (SHA-256 097807a8...).
- **Matched settings:** architecture, input features, objectives, reward, generator training data, optimiser,
  hyperparameters, seed 111, the two-phase budget and the final-state selection.
- Phase 2's stopping rule depends on validation, so the new run may stop at a different update. The procedure is
  matched; the length is not fixed.
- The control ran three jobs in parallel; the new run runs alone. This affects wall-clock time only.

## Names

- Training phases: `10x5+mix+bv3_wt_W1_s111` and `10x5+mix+bv3_wt_W2_s111`.
- Results: `results/waiting_training_v3/`.
- Checkpoint: `checkpoints/waiting_training_v3/`.

Single seed: numbers only. Evaluation split only; the reserved final-test sets are never loaded.
