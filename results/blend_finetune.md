# Fine-tuning the three-specialist blend vs fine-tuning one specialist, at equal PPO updates

## Pre-registration

Written and committed BEFORE any training in this study (including smoke tests). Never edited afterwards;
changes go under "Deviations". This is a new experiment and does not overturn the zero-training verdict
(results/blend_zero_training.md, NOT MEANINGFUL).

### Question
Once fine-tuned under the same PPO budget, does recombining three specialists beat reusing one?

### Setup (identical to Step 4 except where an arm says otherwise)
* 10x5 only; seeds 111/222/333/444; each seed uses only its own specialists `m_s<seed>`, `c_s<seed>`,
  `t_k125s060_s<seed>` (and, for C3, its own Step 4 merge composition).
* Reward, preference and PPO exactly as Step 4 (`tools/run_arms_t.py`): goal `mct`, preference (1/3,1/3,1/3),
  carbon_reward_weight 0.002384, tardiness_reward_weight 5.815362; the `params.py` defaults Step 4 used (lr 3e-4,
  eps_clip 0.2, k_epochs 4, minibatch 1024, entloss_coef 0.01, vloss_coef 0.5, ploss_coef 1, gamma 1,
  gae_lambda 0.98, tau 0, num_envs 20, reset_env_timestep 20, validate_timestep 10); training instances sampled
  from the 10x5 train split as in Step 4; due dates from the frozen manifest k125s060; 1000 updates; budget
  snapshots at 0, 50, 100, 250, 500, 1000. No hyperparameter is tuned for any arm.
* Implementation: Step 4's own `train.Trainer` drives every run (instance sampling, environments, rollouts from
  `policy_old`, GAE, validation, snapshots). Only the policy module and the PPO update step are replaced; the
  update step is a copy of `PPO.update` plus the critic warm-up. `Trainer.train()` reseeds with the run's seed
  before any sampling, so any extra initialisation done beforehand does not change the random streams.
* Evaluation: the 50 HELD-OUT instances of blend_zero_training (same IDs, `results/blend_zero_training/split.json`)
  at every snapshot, with the Step 4 score (per-seed centroid reference over all 100 instances), greedy
  decoding, due dates from the manifest, batching only among instances sharing the same maximum processing time.
  Blend policies are decoded with the same arithmetic as blend_zero_training's B3 (weighted log-probabilities over
  valid actions, log_softmax, argmax); single networks as in the canonical evaluator.
* Required checks at update 0 (the study stops if any fails): F1@0 and F2@0 reproduce B3 exactly on HELD-OUT;
  C2@0 reproduces the makespan specialist; C3@0 reproduces the Step 4 merge at update 0.

### Fresh critic and warm-up (arms F1, F2, C2, C3)
* Critic architecture = DANIEL's critic: MLP with 3 layers, hidden size 64, tanh (`Critic(3, in, 64, 1)`), freshly
  initialised with a fixed per-seed seed.
* F1, F2: input = the concatenation of the three specialists' pooled state embeddings (3 x 16 = 48), with
  stop-gradient, so the value loss never reaches the specialists.
* C2, C3: input = the network's own pooled state embedding (16). DANIEL's critic module is re-initialised in
  place and keeps Step 4's gradient path into the shared feature extractor; the arm differs from its Step 4
  counterpart ONLY in the critic treatment (fresh weights + warm-up), as specified.
* Warm-up: updates 1-50 train ONLY the critic (value loss only; actor and feature extractors frozen), with an
  Adam optimiser over the critic parameters at lr 3e-4. From update 51 one Adam optimiser covers all trainable
  parameters of the arm, as Step 4 does. Warm-up updates count toward the 1000 and appear on the x-axis.

### Arms (1000 PPO updates each, 4 seeds)
* **F1 Blend, joint fine-tune.** log pi = sum_i alpha_i log p_i over valid actions, renormalised; alpha fixed at the
  seed's B3 weights (permille m/c/t: 111 100/100/800, 222 300/200/500, 333 600/100/300, 444 200/0/800). All three
  specialists' feature extractors and actor heads are trainable; gradients flow through the blend.
* **F2 Blend + residual.** The three specialists are frozen. Policy logits = the blend's weighted log-probabilities
  + r_theta(s, a). r_theta is an MLP on each candidate action's three specialist action embeddings (the 41-dim
  vector each specialist's actor scores; stop-gradient) plus the three log-probabilities of that action (126
  inputs), with 2 hidden layers of 64 and tanh; its last layer is zero-initialised, so update 0 is exactly the
  blend. The trainable parameter count is reported.
* **C1 Makespan specialist, Step 4 as-is.** The existing Step 4 snapshots; no new training.
* **C2 Makespan specialist + fresh critic + warm-up.**
* **C3 Merge + fresh critic + warm-up.** From the seed's Step 4 merge composition.
* Run order (priority if time is short): all seeds of F1, then F2, then C2, then C3.

### Metrics
* Step 4 score on HELD-OUT at every snapshot, per seed and pooled; raw makespan, carbon and tardiness at 1000.
* Paired comparisons (relative gain of X over Y = (mean Y - mean X)/mean Y over the 50 held-out instances,
  positive = X better; paired bootstrap 95% CI from 10,000 instance resamples, seed 20260927; pooled = the same
  resampled instances applied to all seeds):
  F1 and F2 at 1000 vs the BEST SINGLE-SPECIALIST CONTROL at 1000 (C1 or C2, whichever has the lower pooled
  score at 1000; stated); C3 at 1000 vs Step 4 merge at 1000.
* Updates-to-threshold: the first snapshot at which an arm's score is <= the best control's score at 1000,
  pooled and per seed (per seed, against that seed's control score).
* Wall-clock and GPU time per arm.

### Decision rule
A seed "counts" when the relative gain is >= 1% AND the lower end of its 95% CI is > 0.
* **WIN:** F1 or F2 beats the best single-specialist control at 1000 at >= 3 of 4 seeds; OR the arm's per-seed
  score reaches that seed's control score at 1000 by update 500 at >= 3 of 4 seeds.
* **LOSE:** the best single-specialist control beats BOTH F1 and F2 at 1000 (both at the same seed) at >= 3 of 4
  seeds.
* **TIE:** neither WIN nor LOSE.
* **Secondary (merge stall):** "critic caused the stall" if C3 beats Step 4 merge at 1000 at >= 3 of 4 seeds;
  otherwise "not supported".

## Step 4 critic audit (Step 0a, from the code)

Step 4 (`tools/run_arms_t.py` -> `daniel/train.py`) started every warm-start arm with `--init_from`, and
`Trainer.load_initial_checkpoint` loads the WHOLE checkpoint (`policy.load_state_dict`) into both `policy` and
`policy_old`: feature extractor, actor AND critic.
* **Makespan / carbon / tardiness specialist arms:** each kept the specialist's OWN critic, trained to predict
  that specialist's single-objective return, not the three-objective return it was now fine-tuned on.
* **Merge arm:** the WEIGHT-AVERAGED critic. `tools/merge_weight_soup.py` averages every tensor in the state
  dict, critic included, so the merge started with a weighted average of three critics trained on different
  rewards.
* **Scratch arm:** a freshly initialised DANIEL, critic included (random initialisation under the run's seed).
* **Warm-up:** none. From update 1, one Adam optimiser (lr 3e-4, fresh state) trained all parameters jointly on
  `vloss_coef * value loss + policy loss + entloss_coef * entropy loss`. The critic shares DANIEL's feature
  extractor, so value-loss gradients also move the policy's features.
* The results Step 4 reports are the fixed-update snapshots (`@u<N>`), not validation-selected checkpoints.

## Deviations

*(none yet -- filled in when the study runs)*
