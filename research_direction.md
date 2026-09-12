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
