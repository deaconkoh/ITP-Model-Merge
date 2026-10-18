# Diagnosis of two v3 specialist failures

Symptoms, from `results/baselines_v3.md`:

- **A.** DANIEL-C, trained on carbon alone, scores about the same carbon as DANIEL-M, which never saw a carbon reward:
  - 10x5: 153.5 for C against 154.8 for M;
  - 20x10: 481.8 for C against 482.5 for M;
  - the unrestricted optimum is 118.8 at 10x5 and 328.4 at 20x10.
- **B.** At 20x10, DANIEL-T's validation tardiness improves and then degrades. It ends worse than DANIEL-C on tardiness.

How the diagnosis was run:
- Read-only use of the existing code. The scripts are in `tools/diagnose_v3/` and the raw outputs in `results/diagnose_v3/*.json`.
- No PPO training. The reserved final-test sets were never touched.
- Data used:
  - Test 1 uses the evaluation split, as requested, so its numbers sit next to the baselines table.
  - Tests 2–4 and 6 use the validation split. Test 2 tops this up from the development split, because validation has only 18 (10x5) and 13 (20x10) instances.
  - Test 5 trains on the v3 train pool.
- Ratios are against the manifest references (MWKR+ECT schedules); lower is better. Tardiness ratios are T / Tref.

## Test 1: rule baselines (evaluation split, 100 instances per size)

These are unrestricted dispatching rules (`tools/objective_validation/schedules.py`), re-scored by `benchmark_v3.schedule_objectives`.

| 10x5 | mk ratio | carbon ratio | T / Tref | tardy share |
|---|---|---|---|---|
| MWKR+ECT (reference) | 1.000 | 1.000 | 1.023 | 0.60 |
| **EDD+ECT** | 1.287 | 0.948 | **0.377** | 0.32 |
| MWKR+MINE (min-energy machine) | 1.490 | **0.794** | 2.509 | 0.74 |
| EDD+MINE | 1.613 | **0.794** | 1.735 | 0.64 |
| DANIEL-M / C / T, greedy (from baselines report) | 1.00 / 1.10 / 1.14 | 1.04 / 1.03 / 1.08 | 0.74 / 0.65 / 0.44 | |

| 20x10 | mk ratio | carbon ratio | T / Tref | tardy share |
|---|---|---|---|---|
| MWKR+ECT (reference) | 1.000 | 1.000 | 1.033 | 0.80 |
| **EDD+ECT** | 1.386 | 0.946 | **0.699** | 0.38 |
| MWKR+MINE | 1.776 | **0.748** | 4.208 | 0.93 |
| EDD+MINE | 1.872 | **0.748** | 3.610 | 0.83 |
| DANIEL-M / C / T, greedy | 1.03 / 1.18 / 1.11 | 1.10 / 1.10 / 1.13 | 0.93 / 0.78 / 0.83 | |

- **Carbon.** The minimum-energy machine rule reaches the carbon optimum (0.794 and 0.748), because carbon depends only on the machine assignment. Every DANIEL model is 30–47% above it.
- **Tardiness.** EDD+ECT beats DANIEL-T at both sizes in raw tardiness: 210 against 245 at 10x5, and 951 against 1207 at 20x10. At 20x10 it also beats DANIEL-C (1059).

## Test 2: the action mask (A1, B5)

**Which pairs are allowed** (`daniel/fjsp_env_same_op_nums.py`):
- The pair (job j's next operation, machine m) must be eligible: `candidate_process_relation`, lines 268 and 551–554.
- Its earliest start max(r_j, F_m) must also equal `next_schedule_time`, the minimum earliest start over all eligible candidate pairs (lines 558–566 and 613–617).
- `daniel/model/main_model.py:174` sets every other logit to −inf.

So every decision starts an operation at the earliest time anything can start, and no machine can be held idle.

**Exact consequence.** Every schedule the environment can produce is *non-delay*: no machine is ever idle while an operation it can process is ready and waiting.
- Proof: suppose machine m′ were idle at time t while operation o was ready (r_o ≤ t < s_o). Then, until o or m′ is used, the pair (o, m′) has earliest start ≤ t, so `next_schedule_time` ≤ t. No operation starting after t could be dispatched, which contradicts o starting at s_o > t.
- Check: all 200 environment schedules checked in this test (the greedy carbon rule and the four replays, 20 instances, both sizes) have zero violations.

**Lowest-energy ALLOWED pair, applied greedily** (ties broken by earliest completion), on 20 instances per size:

| | 10x5 | 20x10 |
|---|---|---|
| Steps where an unfinished job's next operation has its lowest-energy machine masked (and no lowest-energy pair allowed for that operation) | 91.6% | 98.1% |
| Steps where no allowed pair puts any operation on its lowest-energy machine | 20.0% | 29.4% |
| Operations ending off their lowest-energy machine | 35.2% | 49.9% |
| Carbon reached (ratio) | **1.094** | **1.298** |
| Unrestricted minimum-energy assignment (ratio) | 0.798 | 0.752 |
| Excess over the minimum | +38% | +73% |

**Target schedules replayed through the mask.** Each target is an unrestricted schedule: EDD+ECT, EDD+MINE, or CP-SAT solved on the same instances with 20 s for tardiness and 10 s for makespan.

| target (10x5 / 20x10) | target value | value after projection through the mask | targets that are non-delay (of 20) |
|---|---|---|---|
| EDD+ECT, T / Tref | 0.400 / 0.768 | 0.705 / 2.032 | 0 / 0 |
| EDD+MINE, carbon ratio | 0.798 / 0.752 | 1.179 / 1.451 | 0 / 0 |
| CP-SAT min tardiness, T / Tref | 0.052 / 0.231 | 0.712 / 2.066 | 0 / 0 |
| CP-SAT min makespan, mk ratio | 0.793 / 0.862 | 1.038 / 1.229 | 0 / 0 |

- **None of the 160 target schedules is non-delay**, so by the result above, none of them can be produced by DANIEL's environment.
  - On average, 14–22 operations per 10x5 target and 70–97 per 20x10 target wait while an eligible machine is idle.
- The projection column is a greedy approximation of each target inside the mask, not the best non-delay schedule.
- EDD applied inside the mask is very poor: tardiness ratio 1.21 at 10x5 and 4.30 at 20x10. The mask forces the most urgent job onto whichever machine happens to be free, however slow.
- CP-SAT runs are time-limited with 8 workers, so their targets vary slightly between runs.

**Example from Test 4's state dump** (20x10, validation instance 0, step 100):
- Job 0's next operation is masked on machine 8: 25 time units, 1.36 kg.
- It is allowed only on machine 2: 84 time units, 7.56 kg, which is 5.6× the carbon.

## Test 3: what the specialists choose (A4)

These are greedy rollouts on the whole validation split. Each cell is the share of decisions where the chosen machine meets the condition.

| | lowest-energy eligible | fastest eligible | earliest-completing eligible | lowest-energy among the op's allowed | lowest-carbon allowed pair overall | allowed machines for the chosen op |
|---|---|---|---|---|---|---|
| 10x5 C (3 seeds) | 0.68–0.71 | 0.72–0.76 | 0.85–0.89 | 0.98–0.99 | **0.77–0.81** | 1.26–1.32 |
| 10x5 M | 0.69–0.70 | 0.70–0.74 | 0.85–0.87 | 0.99–1.00 | 0.63–0.70 | 1.17–1.21 |
| 10x5 T | 0.64–0.68 | 0.67–0.71 | 0.82–0.83 | 0.98–0.99 | 0.65–0.72 | 1.24–1.25 |
| 20x10 C | 0.58–0.60 | 0.62–0.66 | 0.80–0.82 | 0.98 | **0.65–0.72** | 1.36–1.41 |
| 20x10 M | 0.58–0.60 | 0.64–0.65 | 0.81–0.83 | 0.98–0.99 | 0.59–0.69 | 1.21–1.30 |
| 20x10 T | 0.56–0.58 | 0.60–0.61 | 0.79–0.80 | 0.98–0.99 | 0.58–0.62 | 1.25–1.32 |

- **The premise of A4 holds.** The fastest eligible machine is also the lowest-energy one for 84.3% of operations at 10x5 and 81.8% at 20x10.
- **Its prediction fails.** The policies choose the fastest machine only 60–76% of the time and the lowest-energy machine 56–71% of the time.
- **The machine is set by the mask, not by the policy.** The chosen operation has only 1.2–1.4 allowed machines, and every model, M included, takes the lowest-energy one among them 98–100% of the time. The policy's real freedom is which job to dispatch.
- **DANIEL-C does prefer low carbon where it has a choice.** It picks the lowest-carbon allowed pair more often than M and T, at both sizes.

## Test 4: observations and reward (A2, A3, B3)

- **Feature dump:** 3 states per size, every candidate pair, in `t4_obs_reward.json` → `dump`. Every feature matches its raw source.
  - Carbon feature = raw carbon / 13.365. For example, 4.347 kg gives 0.325.
  - Power feature = P_m / 15, then standardised across machines.
  - Lateness feature = max(0, pair start + p + remaining min-work − due) / 99.
  - Slack and time-to-due are standardised across operations.
- **Trace to the actor:**
  - A forward hook on `actor` shows that the last 10 columns of its input exactly equal the pair features, for all 18 checkpoints. So carbon and lateness reach the actor directly (`main_model.py:167-170`).
  - Power enters the first machine-attention layer, whose weights have input width 9.
- **Ablation.** On each model's own greedy trajectory, replacing a column by its per-state mean changes this share of decisions:
  - carbon: 2–5% for 10x5 C (and also 2–4% for M), 10–13% for 20x10 C (4–5% for M);
  - processing time: 4–14%;
  - due-date operation features: 5–20%.
  - The inputs are used. Machine-level inputs matter little, because the mask fixes the machine.
- **Reward check.** Per-step rewards were summed over 50 uniform-random episodes per size and goal. The sum equals −objective/ref to within 5.3e-15 for all six (size, goal) cases.
  - Random-policy returns: m −1.53 / −2.04, c −1.48 / −1.99, t −2.48 / **−6.53** (10x5 / 20x10). The 20x10 tardiness returns are about 3× larger in magnitude than any other run's.
- **Feature ranges** (raw, allowed pairs along DANIEL-C greedy trajectories, whole validation split):

| | 10x5 p50 / p99 / max | 20x10 p50 / p99 / max |
|---|---|---|
| lateness / 99 (pair) | 0.00 / 2.73 / 5.06 | 0.00 / 2.24 / 4.03 |
| slack / 99 (raw, before standardisation) | 1.11 / 3.51 / 4.14 | 1.27 / 3.45 / 4.24 |
| time to due / 99 (raw) | 2.11 / 5.13 / 6.13 | 2.56 / 5.23 / 5.83 |
| carbon / 13.365 | 0.31 / 0.93 / 1.00 | 0.29 / 0.91 / 0.98 |
| wait time (pair) | 0.00 / 2.59 / 4.68 | 0.04 / 1.53 / 2.74 |

The due-date features are not larger at 20x10. The TWK due dates scale with work content, so dividing by 99 keeps them in the same range.

## Test 5: learnability by behaviour cloning (A2 vs A5)

- **Setup:**
  - Model: a fresh DANIEL with the same architecture and v3 inputs.
  - Loss: cross-entropy against the Test 2 rule (lowest-energy allowed pair).
  - Training data: 37,800 states from rule rollouts on the 63-instance train pool, with ε = 0.2 random actions for diversity.
  - Held-out data: the 900 rule-trajectory states of the validation split.
  - Run: 40 epochs in 2.7 GPU-minutes, against a 15-minute budget.
- **Accuracy:** 94.7% top-1 after one epoch. The peak is 98.4% top-1 and 98.9% "carbon-equivalent" (the chosen pair has the rule's carbon), at epoch 30. At the end it is 97.7% top-1.
- **Greedy rollouts on validation:**

| | carbon ratio | mk ratio | T / Tref |
|---|---|---|---|
| cloned policy | 1.076 | 1.337 | 0.931 |
| the rule itself | 1.089 | 1.294 | 0.888 |
| DANIEL-C (seeds 111 / 222 / 333) | 1.021 / 1.008 / 0.993 | 1.13 / 1.15 / 1.08 | 0.82 / 0.60 / 0.63 |

- The network learns the carbon rule almost perfectly from its inputs, so the information is present and usable.
- **The rule is itself worse on carbon than DANIEL-C.** Inside the mask, the obvious carbon heuristic is beaten by the RL policy. RL did find better carbon than greedy carbon-chasing; the room it can search is just small.

## Test 6: training logs (A5, B1, B2, B4)

**What was logged.** Per update: training mean return (`reward_<run>.json`), mean total PPO loss, training makespan and carbon (stdout). Every 50 updates: validation objectives (`valiquality_<run>.json`).

**What was NOT logged:** entropy, approximate KL, clip fraction, value loss on its own, explained variance and gradient norm. `PPO.update` (`daniel/model/PPO.py:154-213`) returns only the total and value losses, and train.py prints only the total. There is also **no gradient clipping** (`PPO.py:204-208`).

**Reconstructed without training.** Each run kept two policies: the selected "best" checkpoint and the "final" policy in its resume state. The measurements below come from sampled validation episodes with the run's own reward, plus Adam's second-moment estimate at the last update.

| per-seed ranges | EV best → final | normalised entropy best → final | KL(best ‖ final), nats/state | critic RMS grad ×1e3 | loss, first 50 updates | episode-return std across instances | val best → last | train return best 100-update window → last 200 |
|---|---|---|---|---|---|---|---|---|
| 10x5 M | 0.67–0.75 → 0.74–0.78 | 0.14–0.18 → 0.12–0.16 | 0.21–0.59 | 2.2–2.5 | 0.025 | 0.08–0.10 | 0.97–0.98 → 1.00 | −0.99 → −1.00 |
| 10x5 C | 0.94–0.95 → 0.94–0.95 | 0.12–0.14 → 0.09–0.14 | 0.43–0.70 | 1.6–1.8 | 0.045 | 0.06–0.08 | 0.99–1.02 → 1.00–1.05 | −1.03 → −1.04 |
| 10x5 T | 0.13–0.27 → 0.26–0.33 | 0.13–0.24 → 0.09–0.15 | 0.73–1.20 | 5.2–5.9 | 0.24 | 0.39–0.42 | 0.37–0.42 → 0.43–0.56 | −0.38 → −0.46 |
| 20x10 M | 0.85–0.88 → 0.85–0.91 | 0.13–0.16 → 0.13–0.15 | 1.08–2.13 | 1.8–2.3 | 0.00 | 0.05–0.06 | 0.99–1.01 → 1.02–1.07 | −1.01 → −1.04 |
| 20x10 C | 0.98 → 0.97–0.98 | 0.09–0.10 → 0.08–0.09 | 0.33–1.51 | 1.7–1.8 | 0.03 | 0.05–0.06 | 1.08–1.12 → 1.10–1.13 | −1.10 → −1.10 |
| **20x10 T** | 0.60–0.70 → 0.78–0.80 | **0.22–0.27 → 0.16–0.18** | 1.12–2.12 | **8.8–10.0** | **1.07–1.37** | **0.32–0.38** | **0.84–1.01 → 1.29–1.71** | **−1.15 to −1.36 → −1.61 to −1.65** |

**Curves for 20x10 T** (per 50 updates, in `t6_logs.json` → `curve`), the same on all three seeds:
- Training return improves from about −4.7 to about −1.2 within about 150 updates.
- It plateaus noisily, then drifts worse from about update 1,000–1,300 to the end.
- Validation follows the same path.
- The total loss is large only in the first 50 updates (about 1.1–1.4) and shows no spike when the degradation starts.

**The C runs are healthy:**
- monotone training curves;
- the critic's explained variance is 0.94–0.98, the best of all runs;
- entropy is moderate;
- training and validation agree.

## Verdicts

| hypothesis | verdict | evidence |
|---|---|---|
| **A1** Action space excludes the lowest-energy machine | **SUPPORTED: the primary cause of A** | Allowed pairs must start at the earliest possible time, so every reachable schedule is non-delay (proof plus 200/200 checks). 0/40 minimum-energy (EDD+MINE) target schedules are non-delay. A lowest-energy machine is masked at 92% / 98% of steps. The best greedy carbon rule inside the mask reaches 1.09 / 1.30 against an unrestricted 0.80 / 0.75. Only 1.2–1.4 machines are allowed for the chosen operation, so the machine is largely forced for C and M alike, which is why their carbon matches. |
| A2 Carbon or power missing, mis-scaled, or not reaching the actor | NOT SUPPORTED | Features equal their raw sources with fixed scales. The actor's input contains the carbon column exactly. A fresh network clones the carbon rule to 98% accuracy in under 3 minutes. |
| A3 Carbon reward does not sum to −carbon/ref | NOT SUPPORTED | The sum equals −objective/ref to within 5e-15 for every goal and size. |
| A4 Policy keys on time, not energy | NOT SUPPORTED | Its prediction (good carbon, since fastest = lowest-energy for 82–84% of operations) fails because the machine is set by the mask. Where it has a choice, C prefers the lowest-carbon allowed pair more than M (0.77–0.81 against 0.63–0.70 at 10x5). Carbon ablation changes more of C's decisions than M's at 20x10. |
| A5 RL training problem for C | NOT SUPPORTED | C runs have smooth curves, a critic explained variance of 0.94–0.98, and moderate entropy. C beats the in-mask greedy carbon rule (0.99–1.02 against 1.09). |
| B1 PPO instability (KL, clip fraction, entropy) | **UNCLEAR: not logged** | KL, clip fraction and per-update entropy were never recorded. Indirect evidence: entropy falls from best to final (normalised 0.22–0.27 → 0.16–0.18), and the policy drifts by 1.1–2.1 nats/state. But the healthy 20x10 M runs drift as much (1.1–2.1) without collapsing. There is no loss spike at onset. The degradation is gradual and appears on the **training** objective itself, which is consistent with an optimisation problem, but the mechanism is unidentified. |
| B2 Return scale: the critic fails to track large returns | NOT SUPPORTED as stated | The returns *are* unusually large and variable: random-policy return −6.5 against −2.0 to −2.5; return std across instances 0.32–0.38 against 0.05–0.06 for M and C; first-50-update loss 1.1–1.4 against ≤ 0.05; critic gradient RMS 4–5× higher. But the critic tracks them: explained variance 0.60–0.70 at best and 0.78–0.80 at the end, better than 10x5 T (0.13–0.33), which trained fine. Whether the large scale drives the drift without breaking the critic is untested. |
| B3 Due-date features too large at 20x10 | NOT SUPPORTED | Lateness, slack and time-to-due have the same ranges at both sizes (p99 lateness 2.73 at 10x5 against 2.24 at 20x10). |
| B4 Training return improves while validation degrades | NOT SUPPORTED | Training return degrades too (best window −1.15 to −1.36 → last 200 updates −1.61 to −1.65), on fresh generator instances. |
| **B5** Low-tardiness schedules unreachable through the mask | **SUPPORTED for the ceiling, not for the degradation** | 0/40 EDD+ECT and 0/40 CP-SAT minimum-tardiness targets are non-delay, so none is reachable. This is why every DANIEL model loses to EDD+ECT and CP-SAT. But it does not explain the degradation, nor DANIEL-T ending worse than DANIEL-C: C reaches lower tardiness inside the same mask (best-of-100 0.53 against 0.74 for T, evaluation split), so better non-delay schedules exist and T's training lost them. |

## Proposed fixes (not implemented)

Changing the action space or the PPO recipe changes the DANIEL baseline. Each fix would need recording as a protocol deviation and retraining all affected specialists.

1. **Widen the action space (A1, B5). This is the main fix.**
   - Allow pairs whose start is later than the global earliest start. For example, use a Giffler–Thompson conflict set (pairs that can start before the earliest completion time c*), or allow any eligible pair, so the policy can wait for a better machine.
   - Before retraining, rerun Test 2's replay and non-delay check on the new mask to confirm that the CP-SAT and minimum-energy targets become reachable.
   - The carbon optimum needs operations to wait for their lowest-energy machine, so the new mask must allow that kind of delay.
2. **Report the mask's ceiling separately.** Until the mask changes, read any gap to CP-SAT as partly a gap of the action space. A "best non-delay" reference, such as CP-SAT with non-delay constraints, would separate the action-space gap from the learning gap.
3. **Instrument PPO before changing it (B1).** Log entropy, approximate KL, clip fraction, value loss, explained variance and gradient norm per update. Rerun one 20x10 T seed with only that change to see what happens at the onset of degradation.
4. **Then test stabilisers one at a time on 20x10 T (B1, B2):**
   - gradient-norm clipping (currently none);
   - per-update KL early stopping;
   - a lower or decaying learning rate (currently a constant 3e-4);
   - a fixed per-size scale on the tardiness reward so returns are about 1, like the other goals. This keeps the score definition and only rescales PPO's reward.
5. **Keep best-checkpoint selection on validation.** It already protected the reported 20x10 T checkpoints from the late collapse.

## Plain-English summary

Both failures trace mainly to the environment rather than to the network or the reward. DANIEL only lets the policy start an operation on a machine that is free at the earliest possible moment, so it can never wait for a cleaner or faster machine. The minimum-carbon schedules, the EDD schedules and the CP-SAT schedules are all impossible to produce, and that is why DANIEL-C cannot get anywhere near the carbon optimum and every DANIEL model loses to EDD on tardiness. The carbon inputs and the reward are correct, and a fresh network learns a carbon rule from those inputs almost perfectly, so DANIEL-C is not blind to carbon; it simply has almost no machine choice left to make. Separately, the 20x10 tardiness training gets worse on its own training objective after about 1,000 updates, and finding out why needs PPO statistics that were not logged; the next step is to log them on one rerun before changing anything.
