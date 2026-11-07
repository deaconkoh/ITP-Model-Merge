# Merge categories on benchmark v3: protocol (fixed before any run)

## Question

Arm E merged three independently trained specialists by a plain weight average. It was a poor start: its update-0 validation HV was 0.287, against 0.494 for the distilled student. This study tests two other merge categories, Task Arithmetic (TA; Ilharco et al., ICLR 2023) and TIES-Merging (Yadav et al., NeurIPS 2023), together with a shared-base plain average (SB-AVG) as the control.

TA and TIES merge task vectors (specialist − shared base), so all three merges are built from **one shared base and the same three specialists**. Each merge is evaluated twice, untrained and after fine-tuning. The fine-tuning is **contrastive only**, using the existing CL option unchanged. No arm is fine-tuned with PPO.

## Step 1: the specialists

- **The committed v3 specialists cannot be used.** `checkpoints/specialists_v3/` was trained independently: each specialist starts from its own random initialisation (`init_from: ""`, PPO, convergence rule), so they have no common base. The seed-111 runs stopped at different points: M at update 2,300, C at 1,550 and T at 1,800.
- **The BB arms create no specialists.** They transfer one specialist's feature extractor into a fresh preference-conditioned network.
- **Shared-base specialists come from the pre-registered E-shared path,** `tools/ec2_runner.py: u_shared_base` (`docs/warmstart_protocol_v3.md`). It has never been run and is reused unchanged:
  - **Base:** an unconditioned DANIEL with the specialists' architecture and schema, trained with PPO on the equal-weight `mct` objective for 500 updates. The best-validation checkpoint is kept.
  - **M, C and T:** each fine-tuned from that base with the specialist settings (PPO, `configs/benchmark_v3/{m,c,t}.json`, seed 111, generator stream) for a fixed 1,000 updates. The best-validation checkpoints are kept.
  - This is specialist training, which is PPO by definition. It is not an arm.
- **Why this base:**
  - It has exactly the specialists' architecture.
  - It sits at the equal-weight middle of the three objectives, so each τ_x = θ_x − θ_base points from the middle toward one corner.
  - A random-initialisation base would make the "task vectors" whole networks, which defeats the point of task arithmetic.

## Step 2: merges

| Merge | Definition |
|---|---|
| SB-AVG (control) | (θ_M + θ_C + θ_T) / 3, from `merge_specialists_v3.py --shared_base`. It equals TA with λ = 1/3. |
| TA | θ_base + λ Σ_x τ_x |
| TIES | Trim each τ_x to its top 20% of entries by absolute value, over the whole flattened vector, as in the official implementation; the rest become 0. Elect a sign per entry: the sign of the sum of the trimmed values. Take the disjoint mean over the tasks whose kept value is non-zero and has the elected sign (0 if none). Result: θ_base + λ τ_TIES. |

**λ selection:**
- **Grid:** {0.3, 0.5, 0.7, 1.0}, for TA and TIES separately.
- **Score:** the merged network's update-0 validation HV. This uses the CL arms' 15-preference validation grid and frozen validation normalisation points.
- **Rule:** the highest HV wins; a tie goes to the smaller λ.
- **Evaluation data is never used.** The scores and choices are written to `results/merge_categories_v3/lambda_selection.json` before any fine-tuning.

**Preference input and critic are handled as in arm E:**
- Every tensor (extractor, actor, critic) is merged by the same rule into an unconditioned v3 bundle.
- `--init_mode full` copies that bundle into the preference-conditioned network. The three preference columns of the first actor and critic layers are set to zero, so the update-0 network ignores w.
- The contrastive objective never uses or trains the critic.

## Step 3: contrastive fine-tuning

- **Method:** `--finetune_objective contrastive`, unchanged. The constants are those of `configs/contrastive_v3.json`:
  - K = 8, β = 0.1, margin 0.01, the same pairing rules;
  - gradient clip 1.0, learning rate 3e-4, 20 environments, resampled every 20 updates, Dirichlet(1).
- **Setting:** 10x5, seed 111.
- **Budget:** 3,350 updates, which is arm E's final update. There is no early stopping.
- **Validation and selection:** validation at update 0 and every 50 updates; the best-validation-HV checkpoint is kept, exactly as for the CL arms.
- **Zero-training values:** a 0-update contrastive run from each merge validates at update 0 and saves the network. That network is evaluated like the fine-tuned arms.
  - E's zero-training merge is regenerated from the committed s111 specialists.
  - Check: its update-0 validation HV must equal CL-E's, 0.2871.

## Step 4: report (`results/merge_categories_v3.md`, `tools/analyze_merge_categories_v3.py`)

- **For SB-AVG, TA and TIES:**
  - validation HV at update 0, best and final;
  - for both the zero-training merge and the fine-tuned model: evaluation HV, IGD+ (CP-SAT B2, the same 20 instances), unique non-dominated count, mean regret and best-of-16.
- **Comparison:** the seed-111 contrastive arms CL-B, CL-D and CL-E, plus E's zero-training merge. For each: updates, environment steps to best and wall-clock to best. All of these ran on the same machine.
- **Also reported:** the λ table, and the shared-base cost (PPO updates and hours).
- **Single seed:** numbers only, no significance claims.

## Committed networks

`checkpoints/merge_categories_v3/` holds:
- the base;
- the three fine-tunes;
- the SB-AVG merge and E's merge;
- every TA and TIES candidate;
- the selected starts;
- each arm's best checkpoint.

SHA-256 hashes are in `SHA256SUMS.json`, and `tests/test_merge_categories_v3.py` checks them.

## Run

```
WS_V1_STRICT=1 bash scripts/ec2_run_all.sh --arms SB-AVG,TA,TIES --seeds 111 --jobs 3
```

- **Machine:** runs locally (RTX 5060 Ti, 12 logical CPUs) and commits locally, never pushing.
- **Inputs:** frozen in `configs/merge_categories_v3.json` and verified before every run.
- **Smoke test:** `--smoke` runs base and fine-tunes of 20 updates, 2 λ values and 5 CL updates, on the development split, with separate outputs.
- **Data:** the reserved final-test sets are never loaded.

## Known limitation

DANIEL's action mask admits only non-delay schedules (`results/diagnose_v3.md`), so part of every gap to CP-SAT is a gap of the action space.
