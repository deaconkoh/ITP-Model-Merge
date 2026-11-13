# No-delay vs unrestricted hypervolume on the 200-pair solver subset (10x5 evaluation)

The same 20 evaluation instances and 10 preference vectors as the subset solves in `results/decoding_ceiling_v3.md`
(Part B2). HV is computed per instance from that cell's 10 schedules, with the evaluation normalisation
(`results/baselines_v3/normalisation_points.json`) and reference point 1.1, then averaged over the 20 instances.
Computed by `tools/waiting_decoder_v3.py`; numbers in `results/decoding_ceiling_v3/hv_table.json`.

| Method | No-delay (HV ↑) | Unrestricted (HV ↑) |
|---|---:|---:|
| CP-SAT (60 s, 6 workers per solve) | 1.016 | 1.155 |
| From-scratch DANIEL, best-of-128 | 0.754 | unavailable (see note) |

**Solver optimality counts** (out of 200 solves each):
- no-delay: 52 OPTIMAL, 148 FEASIBLE;
- unrestricted: 171 OPTIMAL, 29 FEASIBLE.

**Notes**
- **CP-SAT, no-delay.** These are the restricted model's schedules: exact non-delay, hinted with the best known DANIEL
  schedule. 37 of them use an exact start-time tie that DANIEL's floating-point mask loses. Using only the schedules
  DANIEL's rule reproduces, HV is 0.989 (the 37 pairs are then missing points).
- **CP-SAT, unrestricted.** The unrestricted model's schedules, hinted with the best CP-SAT front point.
- **From-scratch DANIEL.** Checkpoint `pd_fs_u4550`, best-of-128 sampling (seed 51), 6,400 state evaluations per pair
  in both columns.
- **The waiting-allowed decoder** (`WaitingEnv`): the environment with only the no-delay mask line removed
  (`fjsp_env_same_op_nums.py:640`).
  - Any eligible pair of a job's next operation may be chosen, and it starts at max(job ready, machine free).
  - So machine eligibility, job precedence and machine capacity are kept, and a machine may stay idle while other
    work is ready.
  - Operations are appended, not inserted into earlier gaps. The reachable schedules are the semi-active ones, which
    include an optimal schedule for every regular objective.
  - **Small-example check** (evaluation instance 0, 8 random legal decodings): all 8 schedules are valid under the
    independent scorer and contain deliberate waiting. None is reproduced by the original no-delay rule.
- **Why the From-scratch unrestricted cell is unavailable.**
  - The relaxed decoder runs and gives valid schedules: HV 0.553 on this subset, and every schedule contains
    deliberate waiting.
  - But the checkpoint was trained only with the no-delay mask. The logits of the newly opened pairs were always set
    to minus infinity during training, so no gradient ever shaped them.
  - The 0.553 therefore measures an untrained part of the policy, not what waiting can achieve with DANIEL. A valid
    comparison needs a policy trained with waiting allowed.
