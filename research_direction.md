# Research Overview

* **Problem:** Multi-objective scheduling policies must serve operational preferences that change
  over time. Training a fresh RL policy for each preference is expensive, and the cost is paid in
  environment interaction rather than memory.
* **Central Theme / Study:** **Initialisation quality and its effect on optimisation cost.** How
  cheaply can a good starting point for a multi-objective scheduling policy be obtained, and how
  much optimisation does it save?
* **Position of model merging:** merging is the **tool**, not the object of study. It is a
  zero-cost way to compose independently trained single-objective specialists into a sub-optimal
  but well-placed initialisation. The claims below are about what that initialisation buys, not
  about merging as an end in itself.

**Core claim.** Three independently trained single-objective specialists (makespan, carbon,
operation-priority) can be composed at zero cost into a starting point from which modest
fine-tuning approaches the Pareto front substantially faster than training from scratch on a
three-objective reward.

---

### Research Questions

* **RQ1 — Budget.** At what optimisation budget, measured in PPO updates / environment interaction
  steps, does a composed initialisation reach a given quality, and how does that compare with
  training from scratch on the three-objective reward?
* **RQ2 — Is it composition, or just warm-starting?** Does composing multiple specialists help
  specifically, or does any single pretrained specialist serve as an equally good initialisation?
* **RQ3 — Persistence.** Does the choice of starting composition leave a lasting imprint on the
  final policy, or do different starting points converge to equivalent performance given enough
  budget?
* **RQ4 — The per-instance bound.** Can composition be selected per scheduling instance to beat a
  single static composition? *(Closed — see Established Findings. Retained because the negative
  result bounds what instance-aware adaptation can ever contribute.)*

---

### Hypotheses

**Overall Hypothesis.**
The dominant cost in adapting an RL scheduler to a new multi-objective preference is optimisation,
not representation. A composed initialisation removes most of that cost because it starts inside
the region of parameter space that the objective requires, rather than because it is itself
near-optimal.

* **H1 — Initialisation Efficiency.**
  Fine-tuning from a composed initialisation reaches a given quality at a substantially smaller
  budget than training from scratch on the three-objective reward, with the advantage largest at
  low budgets and narrowing as budget grows. The scarce resource is environment interaction: the
  policy has ~29k parameters, so parameter-efficient methods (LoRA/QLoRA) do not apply.

* **H2 — Composition Specificity.**
  A composed initialisation outperforms any single specialist used as the starting point. If it
  does not, the finding reduces to "warm-starting helps", which is weaker and largely known.

* **H3 — Convergence over Persistence.**
  Different starting compositions converge to equivalent final performance as budget grows; the
  initialisation determines the *path length*, not the *destination*. Lasting path dependence
  would be the surprising result.

* **H4 — Bounded Instance Adaptivity.**
  Per-instance composition selection cannot be exploited: the optimal composition for an instance
  is a property of the specialists that were trained, not of the instance.

---

### Established Findings (not to be re-derived)

**Initialisation (supports H1).**
* Merge-initialised fine-tuning dominates from-scratch training at **every** budget tested
  (10x5, two-objective preference w=0.5, 3 seeds).
* The merge alone captures **92.4%** of the achievable improvement at zero training cost.
* From-scratch at 1000 updates is still worse than merge-initialised at 500 updates — the
  initialisation is worth roughly 500+ updates of optimisation.

**The per-instance bound (RQ4 / H4 — closed).**
* Static instance features do not predict per-instance composition response: pre-registered test
  failed all three criteria at both sizes, at **86% power** for |r| = 0.4.
* Instance x seed interaction accounts for **52-62%** of the variance in composition response.
* Cross-seed transfer of the per-instance optimal composition is **negative** — applying one
  seed's per-instance optimum to another seed's specialists is worse than a single fixed choice.
* Specialists trained on the **same** objective from different seeds are no more
  permutation-related than two **untrained random** networks (symmetry share 29.6% vs a 27.6%
  random-network null), and are further apart in parameter space (46.5) than two random
  initialisations are from each other (42.2).

**Objective-space geometry: operation-priority is structurally time-like (independent result).**

Measured from the three canonical specialists (4 seeds each, 100-instance pools), distances in
objective space after normalising each objective to [0,1] across the three specialists:

| | makespan ↔ carbon | carbon ↔ priority | **makespan ↔ priority** | ratio |
|---|---|---|---|---|
| 10x5 | 1.665 | 1.651 | **0.169** | m↔p is **9.8x** smaller |
| 20x10 | 1.701 | 1.666 | **0.128** | m↔p is **13.3x** smaller |

The priority specialist differs from the makespan specialist by only +5.3% makespan / -0.8%
carbon / -6.2% priority at 10x5, and +9.9% / +0.3% / -4.4% at 20x10 — at 20x10 its carbon is
*fractionally worse* than the makespan specialist's.

The cause is structural, not a training artifact. Operation-priority-weighted completion is
`sum(q_o * C_o) / sum(q_o)` — a **weighted completion time**. Optimising it pulls in
substantially the same direction as minimising makespan, because both are time-like
quantities. Carbon is the only genuinely orthogonal axis among the three.

**Consequence, and why it generalises beyond this project:** multi-objective FJSP work that
uses makespan + energy/carbon + priority-weighted-completion is operating in an effectively
**two-dimensional** objective space (time-like vs carbon) while presenting it as
three-dimensional. Two of the three axes are near-duplicates. The effect **grows with instance
size** (9.8x -> 13.3x), so it worsens exactly where deployment-realistic problems live. Any
claim of three-objective Pareto coverage in this setting should be checked against the actual
separation of the objectives rather than assumed from their names.

**Third objective: operation-priority RETIRED (20 Sep 2026), replaced by total tardiness.**

Operation-priority-weighted completion is retired as an experimental axis. It is *kept as a
result* -- the collinearity measurement above is reportable and is not superseded by this
decision. The makespan and carbon specialists are unchanged; only the third vertex moves.

Reasons, in the order they were established:
1. **Structural.** `sum(q_o C_o)/sum(q_o)` is a weighted COMPLETION TIME, i.e. a re-weighting of
   the makespan axis rather than an independent objective.
2. **Gate 1 (trained specialists).** The makespan<->priority distance is 9.8x smaller than either
   is to carbon at 10x5 and 13.3x smaller at 20x10 -- the degeneracy gets WORSE with size.
3. **Gate 2 (264 coarse-simplex evaluations, 4 seeds, 10x5).** The priority specialist's weight at
   the pooled optimum is 42.5% at the equal preference but 15.0% at a carbon-leaning one, with
   seed 333 collapsing onto the makespan-carbon edge. The interior-vs-edge advantage is +0.39%
   (4/4 seeds, p = 0.041, n = 4) -- and each seed individually is not significant (p = 0.20-0.86)
   even though the comparison is biased in the interior's favour, both compositions being
   selected maxima.
4. **Literature.** Makespan + energy/carbon + total tardiness is an established objective triple
   in multi-objective FJSP, and the 2022 critical review of FJSP objective functions names
   delivery-tardiness among the top objectives after makespan. The replacement is not an
   idiosyncratic choice.

**Gate 2 results in full (10x5, 67 compositions x 4 seeds, reference = simplex centroid):**

| preference | pooled optimum per seed (m/c/p) | priority weight | edge distance | interior advantage |
|---|---|---|---|---|
| (1/3,1/3,1/3) | 20/10/70, 60/20/20, 50/10/40, 40/20/40 | 42.5% | 10-20 pp | +0.08 to +0.62% |
| (0.25,0.5,0.25) | 50/20/30, 50/30/20, **70/30/0**, 60/30/10 | 15.0% | 0-20 pp | -0.24 to +0.52% |

The two preference points DISAGREE: interior at all four seeds under the equal preference, on the
makespan-carbon edge at seed 333 under the carbon-leaning one.

**Calibration weights (Step 2, 10x5), recorded here because they existed only in checkpoint
filenames:** `carbon_reward_weight = 0.002215`, `priority_reward_weight = 1.158337`, derived from
the closed-form anchors and confirmed by a verification sweep (the anchor did not win; the
empirical optimum was the half-anchor on priority, paired win rate 62%, t = 2.25, p = 0.026).
Reference composition metrics: makespan 470.2, carbon 2144.2, priority 203.0.

**A screen that reproduces the failure it is meant to prevent.** A training-free greedy
dispatcher, scoring every objective on identical schedules, recovers priority's degeneracy from
the instance data alone: the best priority schedule IS the best makespan schedule (0.0%
advantage), and the makespan<->priority distance ratio measured this way is 11.8x at 10x5 and
12.0x at 20x10, against the 9.8x and 13.3x measured from trained specialists. The screen costs
minutes of CPU; the trained measurement cost roughly six GPU-hours. Candidate objectives are
screened this way BEFORE training from now on.

**Why tardiness, and what the screen is for.** Total tardiness was chosen because it is
established in the literature (makespan + energy + total tardiness is a standard multi-objective
FJSP triple) and because it is operationally meaningful (on-time delivery). It was NOT chosen for
its separation numbers. The orthogonality screen is a validity check on how the objective's data
is GENERATED: if it shows degeneracy, the generation is fixed, not the objective.

**One root cause, three objectives.** Three objectives have now degenerated for the same reason:
each was generated in a way that removed the structure the objective depends on.
* **i.i.d. per-operation priority** -- a mean over 50-200 independent draws concentrates with
  size, so every instance looks alike and the priority-weighted completion collapses onto a
  re-weighted completion time (Gate 1: 9.8x -> 13.3x collinear with makespan).
* **i.i.d. per-pair carbon** -- independent per (operation, machine) draws likewise concentrate
  with instance size, erasing machine-level carbon character (the motivation for the Track B
  structured generator).
* **Uniform-tightness due dates** -- TWK due dates with a single k give every job identical
  relative tightness, so no job is more urgent than another; the best tardiness schedule is the
  best makespan schedule (screen: exactly 0.0% specialist advantage at every k and both sizes).

The general lesson: a synthetic annotation must carry the heterogeneity its objective trades on.
Independent, identically distributed or uniform annotations average that heterogeneity away,
and the damage is invisible from the objective's name.

**Tardiness due-date generation (frozen 2026-09-23, manifest `daniel/data/due_dates/due_date_manifest.json`).**
* Rule: `d_j = k * u_j * P_j`, `u_j ~ U[1 - s/2, 1 + s/2]`, `P_j` = sum of mean eligible
  processing times of job j's operations, release times zero (static instances).
* **s = 0.6**, the midpoint of the relative-range-of-due-dates values RDD in {0.2, 0.4, 0.6, 0.8,
  1.0} used to generate the OR-Library weighted-tardiness benchmarks (Crauwels, Potts & Van
  Wassenhove 1998, *INFORMS Journal on Computing* 10, 341-350; verified on the OR-Library
  `wtinfo` page). There the range is RDD times the processing-time scale; here it is s times each
  job's own due-date scale.
* **k is PER SIZE: 10x5 k = 1.25 (tag `k125s060`), 20x10 k = 0.95 (tag `k095s060`).** Chosen by the
  pre-stated rule (smallest k with seed-mean tardy fraction in [0.2, 0.6] and <=10% zero-tardiness
  instances) applied to the job completion times of the four TRAINED makespan specialists. No
  single k qualified at both sizes (10x5 qualifies on 1.25-1.65, 20x10 on 0.95-1.20), so per-size
  k by the same rule, as pre-approved. Every tardiness checkpoint and result carries its tag.
* Trained makespan specialists leave MORE jobs tardy than the earliest-completion heuristic
  (at k = 1.0: 82% vs 56% of jobs at 10x5, 49% vs 38% at 20x10), despite a 24-38% better
  makespan: a makespan-optimal policy only cares about the last job, so individual jobs finish
  late. Per-job completion under a trained policy is therefore not a relabelled makespan.
* Step 4 screen on the frozen due dates: tardiness specialist advantage +10.2% (10x5) and +7.3%
  (20x10), against carbon +41.4% / +59.1% and retired priority +0.0% / +0.0%. A real but modest
  axis whose separation SHRINKS with size -- the opposite direction to carbon.
* **Gate 1 (trained tardiness specialists): passes at 10x5, FAILS at 20x10.** At 10x5 every
  specialist beats its seed's makespan specialist on tardiness (+18.2% mean, 4/4 seeds, p = 0.001).
  At 20x10 two of four LOSE to the seed-matched makespan specialist (mean +0.7%, p = 0.86) while
  paying 6-9% in makespan, and the makespan<->tardiness distance (0.085, 20.4x) is smaller than
  retired priority's (0.128, 13.3x). The CPU screen foreshadowed the size trend. Full record:
  `results/tardiness_programme/gate1_tardiness.md`. Programme stopped before Step 6.

**Methodological constraints carried forward.**
* Coarse composition grids are inadequate: the optimum region sits around 0.2-0.4 and a 5-point
  grid missed it entirely, understating headroom by roughly 60%. Composition sweeps must be run at
  fine resolution, or coarse-to-fine with a genuinely fine refinement.
* Seed-averaging suppresses composition-response headroom by 58-65%; report per-seed.
* Any comparison between merging and fine-tuning must match the **operating point**, not just the
  objective score: the fine-tuning reward must be calibrated to the same preference the merge
  targets, or the comparison is apples-to-oranges.

---

### Thesis (Provisional)

> Adapting a reinforcement-learning scheduler to a new multi-objective preference is primarily an
> optimisation-cost problem, and most of that cost can be removed by initialisation rather than by
> better optimisation or by adaptivity. Independently trained single-objective specialists can be
> composed at zero cost into a starting point from which modest fine-tuning approaches the Pareto
> front far faster than training from scratch. The benefit is a property of where the composition
> places the policy in parameter space, not of the instance being scheduled: per-instance
> composition selection is bounded near zero and does not generalise across training runs.
