# Contrastive fine-tuning to convergence on benchmark v3: protocol (fixed before any run)

## Question

The contrastive arms were first run for fixed budgets: the update count at which each PPO counterpart stopped, or arm E's count for the merge arms. Most of them were still improving when the budget ran out.

This study judges every contrastive arm at convergence, under the same stopping rule as the PPO warm-start runs. The contrastive settings, validation, checkpoint selection and evaluation are unchanged.

## Arms

All arms are 10x5, seed 111.

| Arm | Fixed-budget source run |
|---|---|
| From scratch | `results/contrastive_v3`, 10x5+mix+bv3_cl_CLB_s111 (2,600 updates) |
| Distillation | `results/contrastive_v3`, 10x5+mix+bv3_cl_CLD_s111 (1,000 updates) |
| Naive weight average (independent specialists) | `results/contrastive_v3`, 10x5+mix+bv3_cl_CLE_s111 (3,350 updates) |
| Model Soup (shared base) | `results/merge_categories_v3`, 10x5+mix+bv3_mc_SBAVG_s111 (3,350 updates) |
| Task Arithmetic | `results/merge_categories_v3`, 10x5+mix+bv3_mc_TA_s111 (3,350 updates) |
| TIES | `results/merge_categories_v3`, 10x5+mix+bv3_mc_TIES_s111 (3,350 updates) |

## Stopping rule

This is `daniel/convergence.py`, the PPO warm-start rule. It is checked at every validation, which happens every 50 updates.
- **Stop** once the best validation HV has improved by less than 0.5% (relative) over the last 1,000 updates.
- **Minimum:** at least 1,000 updates before the rule can stop a run.
- **Hard cap:** 8,000 updates. A run that hits the cap is reported as **capped**, not converged.

## Step 1: continue or rerun (decided before any run, frozen in `configs/contrastive_converged_v3.json`)

The rule was replayed on each fixed-budget run's logged validation history.

| Arm | Rule stops the logged history at | Mode |
|---|---|---|
| From scratch | never (0–2,600) | continue from the saved final state |
| Distillation | never (0–1,000) | continue |
| Naive weight average | never (0–3,350) | continue |
| Task Arithmetic | never (0–3,350) | continue |
| Model Soup (shared base) | update 1,800 | rerun from the saved starting network |
| TIES | update 2,500 | rerun from the saved starting network |

**A continued run resumes from its source's saved state.** That state holds the weights, optimiser, RNG states (Python, numpy, torch, CUDA and the preference stream), logs, best checkpoint and generator counters.

The new option `train.py --resume_from` loads such a state:
- **Allowed differences:** the run may differ from its source only in the model name and the stopping rule (`converge_window`, `converge_tol`, `min_updates`, `max_updates`). Any other difference is refused.
- **The exactness check:** the resumed policy must reproduce the last logged validation HV exactly, within 1e-9, with RNG states untouched. Only then does training continue under the new rule.
- **When the check fails:** an arm that fails it exits with code 3 and reruns from its starting network instead. The report says so.

**Two of the continued runs end between two instance resamplings.** Naive weight average and Task Arithmetic stop at update 3,350, and instances are resampled every 20 updates. Their resume states predate saving the instance batch.
- **The fix:** the current batch is rebuilt by replaying the instance generator from the seed. It is accepted only if the replay ends in exactly the saved numpy state and generator counters.
- **Tested:** a run extended this way reproduces the uninterrupted run bit for bit (`tests/test_contrastive_converged_v3.py`).
- **New resume states:** they now also save the batch.

**A rerun starts from its source run's starting network** (hashed in the manifest), with the same seed and the same contrastive settings, under the rule. Its validation curve is compared with the fixed-budget run's over the updates both have.

## Step 2: report (`results/contrastive_converged_v3.md`, `tools/analyze_contrastive_converged_v3.py`)

Arms are shown with plain names. For each arm:
- the update of the best checkpoint, the stop update and the stop reason (converged or capped);
- the best validation HV;
- evaluation HV, IGD+ (CP-SAT B2, the same 20 instances), unique non-dominated count and mean regret;
- environment steps and wall-clock, both to the best checkpoint and to the stop.

Continued runs include their fixed-budget part in steps and wall-clock; they ran on the same machine with `--jobs 3`.

**Setup costs are listed separately:**
- Distillation: the student's distillation.
- Model Soup, Task Arithmetic and TIES: the 0.77 h shared base and its fine-tunes.
- Distillation and Naive weight average: the independent specialists they use.

**Comparison:** the earlier fixed-budget results are shown alongside and are not changed. Regret is computed over both sets together.

**Single seed:** numbers only, no significance claims.

## Saved checkpoints

`checkpoints/contrastive_converged_v3/` holds, with SHA-256 hashes in `SHA256SUMS.json`:
- each arm's best checkpoint;
- in `sources/`, the source files actually used: the resume states of continued arms and the starting networks of reruns.

## Run

```
WS_V1_STRICT=1 bash scripts/ec2_run_all.sh --arms CC-B,CC-D,CC-E,CC-SBAVG,CC-TA,CC-TIES --seeds 111 --jobs 3
```

- Runs locally and commits locally, never pushing.
- `--smoke` runs 5 updates per arm, with separate outputs.
- The reserved final-test sets are never loaded.
