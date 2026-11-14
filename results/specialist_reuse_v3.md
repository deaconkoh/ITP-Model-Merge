# Specialist reuse vs from scratch, waiting allowed (10x5, seed 111)

Protocol: `docs/specialist_reuse_protocol_v3.md`; frozen settings `configs/specialist_reuse_v3.json`. Waiting option `--dispatch_rule semi_active` in specialist training, distillation, multi-objective training, validation, the TIES screen and evaluation. Single seed: numbers only.

Evaluation: 100 evaluation instances x 66 preferences; HV with the frozen normalisation, reference point 1.1, per-instance mean; IGD+ on the 20 instances with a CP-SAT B2 front; best-of-128 = `sample128` (seed 51).

| Method | Greedy HV | Δ vs scratch | Best-of-128 HV | Δ vs scratch | IGD+ greedy / best-of-128 | Updates to scratch's best validation HV | Time to it (h) | Specialist training (h) | Additional training (h) | Total training (h) | Inference s/pair greedy / best-of-128 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| From scratch | 1.018 | +0.000 | 1.040 | +0.000 | 0.107 / 0.096 | 3400 | 1.58 | - | 2.00 | 2.00 | 0.0019 / 0.2229 |
| Distillation | 1.013 | -0.005 | 1.092 | +0.052 | 0.106 / 0.069 | not reached | not reached | 8.02 | 1.16 | 9.18 | 0.0019 / 0.2221 |
| Rewarded Soups | 1.014 | -0.004 | 1.085 | +0.045 | 0.108 / 0.075 | not reached | not reached | 8.02 | 1.74 | 9.76 | 0.0019 / 0.2334 |
| Preference-weighted TIES | 0.994 | -0.024 | 1.076 | +0.036 | 0.113 / 0.076 | not reached | not reached | 8.02 | 1.45 | 9.47 | 0.0020 / 0.2345 |

- **Target** for time to target: From scratch's best validation HV, 0.9567 (greedy, 18 validation instances x 15 preferences, every 50 updates).
- **Updates and time to target** count the method's own multi-objective training. Distillation's time also includes its stage 1.
- **Specialist training** is the sum of the three specialists, 8.02 h. They ran in parallel, 4.59 h wall-clock.
- **Cost of a run when the specialists already exist:** the Additional column, which is Distillation stage 1 plus fine-tuning, or merge fine-tuning.
- **Concurrency:** runs shared 3 slots, which inflates wall-clock times. Inference was timed with the GPU alone and the same batching for every method.

Full runs: From scratch CONVERGED at update 4400 (best validation HV 0.9567); Distillation CONVERGED at update 1800 (best validation HV 0.9535); Rewarded Soups CONVERGED at update 2150 (best validation HV 0.9377); Preference-weighted TIES CONVERGED at update 1500 (best validation HV 0.9245).

## Tuning (validation; equal budget per method)

Each method: 6 trials x 1000 updates over lr x beta. Score: best validation HV in the trial. Ties go to the setting closest to the recipe.

| Method | lr | beta | Best validation HV | Trial time (h) | Selected |
|---|---|---|---|---|---|
| From scratch | 0.0001 | 0.1 | 0.4884 | 0.64 |  |
| From scratch | 0.0001 | 0.5 | 0.2877 | 0.64 |  |
| From scratch | 0.0003 | 0.1 | 0.5926 | 0.64 |  |
| From scratch | 0.0003 | 0.5 | 0.5005 | 0.64 |  |
| From scratch | 0.001 | 0.1 | 0.7969 | 0.64 | **yes** |
| From scratch | 0.001 | 0.5 | 0.6927 | 0.60 |  |
| Distillation | 0.0001 | 0.1 | 0.9312 | 0.63 |  |
| Distillation | 0.0001 | 0.5 | 0.9271 | 0.62 |  |
| Distillation | 0.0003 | 0.1 | 0.9534 | 0.62 | **yes** |
| Distillation | 0.0003 | 0.5 | 0.9248 | 0.63 |  |
| Distillation | 0.001 | 0.1 | 0.9516 | 0.65 |  |
| Distillation | 0.001 | 0.5 | 0.9203 | 0.62 |  |
| Rewarded Soups | 0.0001 | 0.1 | 0.9142 | 0.80 |  |
| Rewarded Soups | 0.0001 | 0.5 | 0.9031 | 0.80 |  |
| Rewarded Soups | 0.0003 | 0.1 | 0.9264 | 0.83 |  |
| Rewarded Soups | 0.0003 | 0.5 | 0.8999 | 0.82 |  |
| Rewarded Soups | 0.001 | 0.1 | 0.9330 | 0.80 | **yes** |
| Rewarded Soups | 0.001 | 0.5 | 0.9026 | 0.81 |  |
| Preference-weighted TIES | 0.0001 | 0.1 | 0.9245 | 0.95 | **yes** |
| Preference-weighted TIES | 0.0001 | 0.5 | 0.8898 | 0.97 |  |
| Preference-weighted TIES | 0.0003 | 0.1 | 0.9177 | 0.96 |  |
| Preference-weighted TIES | 0.0003 | 0.5 | 0.8980 | 0.96 |  |
| Preference-weighted TIES | 0.001 | 0.1 | 0.9156 | 0.96 |  |
| Preference-weighted TIES | 0.001 | 0.5 | 0.9025 | 0.97 |  |

TIES merge screen (merged start, no training; validation HV at update 0):

| keep | lambda | Validation HV | Selected |
|---|---|---|---|
| 0.2 | 0.8 | 0.6885 |  |
| 0.2 | 1.0 | 0.6920 |  |
| 0.2 | 1.5 | 0.7198 |  |
| 0.5 | 0.8 | 0.7870 |  |
| 0.5 | 1.0 | 0.8358 |  |
| 0.5 | 1.5 | 0.8799 | **yes** |
| 0.8 | 0.8 | 0.8138 |  |
| 0.8 | 1.0 | 0.8662 |  |
| 0.8 | 1.5 | 0.8399 |  |

Tuning cost (h): From scratch 3.81; Distillation 3.76; Rewarded Soups 4.85; Preference-weighted TIES 5.79. TIES includes its screen. Distillation stage 1 (0.03 h) is shared by the trials and the full run, and is counted in the Additional column.

## Checks before the multi-objective stages (real specialists)

| Merge | Preference change changes the weights (max difference) | Corners reproduce the specialist: weights / max probability difference (M, C, T) |
|---|---|---|
| rewarded_soups | yes (0.18) | yes / 4.2e-07, yes / 4.8e-07, yes / 3.0e-07 |
| ties_keep0.2_lambda0.8 | yes (0.216) | yes / 2.4e-07, yes / 6.6e-07, yes / 3.0e-07 |
| ties_keep0.2_lambda1.0 | yes (0.27) | yes / 2.4e-07, yes / 6.6e-07, yes / 3.0e-07 |
| ties_keep0.2_lambda1.5 | yes (0.406) | yes / 2.4e-07, yes / 6.6e-07, yes / 3.0e-07 |
| ties_keep0.5_lambda0.8 | yes (0.224) | yes / 2.4e-07, yes / 6.6e-07, yes / 3.0e-07 |
| ties_keep0.5_lambda1.0 | yes (0.279) | yes / 2.4e-07, yes / 6.6e-07, yes / 3.0e-07 |
| ties_keep0.5_lambda1.5 | yes (0.419) | yes / 2.4e-07, yes / 6.6e-07, yes / 3.0e-07 |
| ties_keep0.8_lambda0.8 | yes (0.224) | yes / 2.4e-07, yes / 6.6e-07, yes / 3.0e-07 |
| ties_keep0.8_lambda1.0 | yes (0.279) | yes / 2.4e-07, yes / 6.6e-07, yes / 3.0e-07 |
| ties_keep0.8_lambda1.5 | yes (0.419) | yes / 2.4e-07, yes / 6.6e-07, yes / 3.0e-07 |

Corner outputs were compared on 30 validation states visited under the waiting rule.

## Specialists

- m: CONVERGED at update 3550, best own-objective validation score 0.9298 @ 3050, 2.23 h
- c: CONVERGED at update 1950, best own-objective validation score 0.8009 @ 1950, 1.22 h
- t: CONVERGED at update 7900, best own-objective validation score 0.1768 @ 6900, 4.58 h
