# Joint latent-state + measurable-outcome world-model pilot (10x5, waiting allowed, seed 111)

Single-seed pilot. Settings frozen in `results/world_model_joint_pilot_v3/settings.json` before collection (sha256 7033c8218b1b…).

## Architecture

`S_t -> DANIEL encoder (DualAttentionNetwork, feature_exact keys) -> action-conditioned latent predictor -> predicted latent of S_t+1 -> outcome head (+ continuation-rule one-hot) -> 10 job completions, 5 machine busy times, 5 machine energies`.
- Latent predictor: the latent pilot's per-token residual MLP + one self-attention layer; input = encoder tokens of S_t, the chosen operation/machine tokens, globals, the pair's 10 observed DANIEL pair features, per-token flags.
- Target: EMA copy of the online encoder (tau 0.996), stop-gradient, applied to the actual S_t+1.
- Outcome head (D): per job, mean+max pooling of the predicted tokens of that job's operations + predicted globals + rule; per machine, the predicted machine token + predicted globals + rule. It never sees S_t, raw action features or S_t+1 (checked: perturbing every S_t+1 array leaves all predictions bit-identical; both loss terms give non-zero encoder gradients; the predicted latent is not detached).
- D loss: L_lat/L_lat0 + 1.0 x variance hinge + lambda_out x L_out/L_out0 (L0 = each term at initialisation on a fixed ~1,024-transition batch). lambda_out calibrated once after epoch 1 if the encoder gradient-norm ratio left [1/3, 3]: {"median_ratio_epoch1": 0.5949280858039856, "rescaled": false, "lam_out": 1.0}.
- B: the earlier PairWorldModel (encoder + action readout + pair features), abs + diff Huber, retrained here. C: latent only. A: the random start encoder (a predictor trained on it for latent metrics).
- Multi-step readiness: the predictor maps tokens -> tokens and can be rolled; on imagined states the observed pair features would be unavailable (limitation; not exercised here).

## Dataset

- 1350 generator factories (948 train / 201 validation / 201 test) (seed 4111); split by factory; semi-active simulator (waiting allowed).
- 53,723 states, 265,532 transitions; up to 5 branches per state; final outcomes of every branch under MWKR+ECT, MWKR+MINE, EDD+ECT (decoder continuation, never waits).
- Generation: episodes+branching 32 s, continuation rollouts 73 s; 1.39 GB; peak RAM 4.4 GB.
- Checks: replay of S_t+1, decoder = environment after prefix + action, branch independence (all passed).

| split, source | states | branches/state | same-op group | other-op same-machine | waiting | early/mid/late |
|---|---|---|---|---|---|---|
| train, MWKR+ECT | 7573 | 4.96 | 0.502 | 0.650 | 0.639 | 0.34/0.33/0.33 |
| train, MWKR+MINE | 7571 | 4.96 | 0.510 | 0.647 | 0.634 | 0.35/0.32/0.33 |
| train, EDD+ECT | 7543 | 4.94 | 0.556 | 0.629 | 0.642 | 0.35/0.34/0.32 |
| train, RANDOM | 7499 | 4.91 | 0.520 | 0.618 | 0.726 | 0.34/0.34/0.32 |
| train, DANIEL | 7549 | 4.95 | 0.518 | 0.633 | 0.722 | 0.35/0.34/0.31 |

## Learning-rate check (validation factories)

| config | lr 0.0001 | lr 0.0003 | lr 0.001 | chosen |
|---|---|---|---|---|
| B (val_out) | 0.003176 | 0.00291 | 0.002714 | 0.001 |
| C (val_change_explained) | 0.6672 | 0.8121 | 0.8693 | 0.001 |
| D (val_norm_mean) | 0.01432 | 0.01261 | 0.01206 | 0.001 |

## Training and convergence

| config | lr | epochs | seconds | peak GPU GB | verdict | still improving |
|---|---|---|---|---|---|---|
| B | 0.001 | 60 | 2126 | 0.108 | would benefit from more training | val_out |
| C | 0.001 | 60 | 1855 | 0.118 | would benefit from more training | val_change_explained, val_lat, val_var |
| D | 0.001 | 60 | 2747 | 0.215 | would benefit from more training | val_disc_same_op, val_lat, val_norm_mean, val_out, val_var |
| A_latent | 0.001 | 60 | 1329 | 0.109 | would benefit from more training | val_lat |

D gradient balance on the encoder (median over monitor batches): epoch 1 ratio latent/outcome 0.59, cosine 0.12; final epoch ratio 0.22, cosine 0.19.

## Results on the test factories

### Latent prediction (own target space; scale-free metrics)

| config | change explained (changed tokens) | shuffle worse share | discrimination overall | same op / other machine | other op / same machine | immediate vs waiting |
|---|---|---|---|---|---|---|
| A_latent | 0.814 | 0.995 | 0.881 | 0.586 | 0.953 | 0.934 |
| C | 0.889 | 0.997 | 0.909 | 0.533 | 0.993 | 0.981 |
| D | 0.899 | 0.995 | 0.965 | 0.818 | 0.991 | 0.995 |

Copy-current-latent baseline = change explained 0 by definition; action shuffle = the next branch's action.

### Outcome prediction (mean over 3 rules and 3 objectives)

| model | ranking accuracy | same op / other machine | other op / same machine | difference MAE | zero-difference MAE | absolute MAE |
|---|---|---|---|---|---|---|
| B native | 0.577 | 0.618 | 0.592 | 72.367 | 64.352 | 110.000 |
| D native (from predicted latent) | 0.569 | 0.602 | 0.582 | 72.831 | 64.352 | 116.354 |
| probe on frozen A | 0.554 | 0.593 | 0.555 | 95.475 | 64.352 | 144.074 |
| probe on frozen B | 0.574 | 0.615 | 0.582 | 74.775 | 64.352 | 112.281 |
| probe on frozen C | 0.553 | 0.598 | 0.558 | 105.060 | 64.352 | 156.556 |
| probe on frozen D | 0.567 | 0.606 | 0.573 | 78.859 | 64.352 | 118.796 |
| probe on frozen raw | 0.557 | 0.586 | 0.566 | 88.470 | 64.352 | 128.308 |

Simple pair rule (shorter processing time / lower energy) on same-operation pairs: makespan 0.611, carbon 0.682, tardiness 0.500.

Per objective (frozen probes, same-operation accuracy / ranking accuracy):

| encoder | makespan | carbon | tardiness |
|---|---|---|---|
| A | 0.593 / 0.557 | 0.576 / 0.533 | 0.612 / 0.571 |
| B | 0.629 / 0.582 | 0.558 / 0.534 | 0.657 / 0.605 |
| C | 0.597 / 0.555 | 0.583 / 0.537 | 0.612 / 0.567 |
| D | 0.619 / 0.576 | 0.543 / 0.524 | 0.656 / 0.599 |
| raw | 0.588 / 0.559 | 0.551 / 0.532 | 0.618 / 0.579 |

### Machine availability readouts (linear, from the representation of the actual next state)

| representation | absolute R^2 | absolute MAE (min) | same-op difference sign accuracy | difference MAE (min) | zero-baseline MAE |
|---|---|---|---|---|---|
| A | 0.001 | 39.211 | 0.519 | 66.084 | 66.324 |
| B | 0.469 | 26.109 | 0.923 | 39.267 | 66.324 |
| C | 0.000 | 39.226 | 0.512 | 66.102 | 66.324 |
| D | 0.391 | 28.392 | 0.938 | 38.917 | 66.324 |
| raw | 0.594 | 21.825 | 0.954 | 33.913 | 66.324 |

### Collapse diagnostics (target encoder on test next states)

| config | effective rank (start) | effective rank (end) | mean std (start -> end) | collapsed |
|---|---|---|---|---|
| C | 5.04 | 5.98 | 0.4219 -> 0.3016 | False |
| D | 5.04 | 5.98 | 0.4219 -> 0.2435 | False |

## Pre-registered decision

D justifies specialist training only if, on the TEST factories: (1) in the frozen-encoder probes D's mean ranking accuracy AND mean same-operation accuracy each beat both B and C by >= 0.03, and D's mean difference MAE is below the zero-difference baseline; (2) D's latent change explained (changed tokens) and overall discrimination are within 0.02 of C's or better; (3) D's availability readouts (absolute R^2 and same-operation difference sign accuracy) are at least C's; (4) D did not collapse.

```
{
 "1_frozen_probe_ranking": {
  "rank_acc_D_minus_B": -0.007099880435536998,
  "rank_acc_D_minus_C": 0.013605985694437606,
  "same_op_D_minus_B": -0.008698633693830504,
  "same_op_D_minus_C": 0.008606534827583112,
  "D_diff_mae": 78.85879792107477,
  "zero_baseline": 64.3515518506368,
  "pass": false
 },
 "2_latent_vs_C": {
  "change_explained_D_minus_C": 0.009672263447366491,
  "discrimination_D_minus_C": 0.05528397020808273,
  "pass": true
 },
 "3_availability_vs_C": {
  "abs_r2_D": 0.39138861952290993,
  "abs_r2_C": 0.00011425835238432214,
  "diff_sign_D": 0.9382160707420223,
  "diff_sign_C": 0.5115340253748558,
  "pass": true
 },
 "4_no_collapse": {
  "collapsed": false,
  "pass": true
 },
 "justifies_specialist_training": false
}
```

**Verdict: does not yet justify specialist training.**

## Limitations

- One seed, pilot scale; differences of a few hundredths are within run-to-run noise not measured here.
- Final outcomes depend on the continuation rule (three fixed non-waiting rules); they are not properties of S_t+1 alone.
- Latent metrics live in each model's own target space; only scale-free metrics are compared across configs.
- The predictor uses the observed pair features of S_t; multi-step rollouts on imagined states would need them predicted.
- Resources: see train_*.json (peak GPU memory, RAM) and coverage.json.
