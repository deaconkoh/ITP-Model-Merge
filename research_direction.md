# Research Overview

**Direction (2026-09-26): recombining the DECISIONS of frozen objective specialists.**

* **Theme.** Do independently learned objective specialists contain complementary scheduling behaviours
  that can be RECOMBINED, rather than relearned, to solve multi-objective FJSP efficiently? "Efficiently"
  means training cost -- environment interactions AND GPU-hours -- to reach a target hypervolume.
* **Model merging is no longer the method.** It is prior work and a baseline. The specialists' WEIGHTS are
  unrelated (the permutation-alignment test: same-objective specialists are no more alignable than random
  networks), and weight merging adds nothing over the best single specialist (Step 4: the tardiness
  specialist alone matches the merge; the makespan specialist overtakes it). But at every step all
  specialists score the SAME set of valid operation-machine pairs, so their DECISIONS are directly
  comparable even though their weights are not.
* **The specialists stay FROZEN.** The eventual method is a small learned router that combines their
  decisions at every step, and it must be STATE-DEPENDENT:

      alpha_t = g(s_t, w, pi_M(.|s_t), pi_C(.|s_t), ...)

  so two scheduling states with the same preference w can receive different expert combinations. A router
  that outputs alpha = w, or samples an expert with probability w, is a BASELINE, not the method.
* **Not claimed as contributions** (prior work / baselines): model merging, mixture-of-experts / expert
  gating, preference-conditioned policies, specialist reuse (Rewarded Soups, MAPEX, CoMEx, DCAN), and
  preference-proportional expert selection (e.g. Wu et al. 2025).

---

### Research Questions

* **RQ1 -- Complementarity.** Do independently trained objective specialists make usefully different
  decisions at different points in a schedule, enough that STATE-DEPENDENT switching could beat every
  fixed combination?
* **RQ2 -- Efficiency.** Can a learned state-dependent router over frozen specialists reach good
  Pareto-front quality with less training than a preference-conditioned DANIEL trained from scratch?
* **RQ3 -- Mechanism.** Does a router using candidate-level disagreement beat a plain gate that weights
  whole specialists?

Current scope: 10x5, two objectives (makespan, carbon), the canonical makespan and carbon specialists at
seeds 111/222/333/444, frozen. Tardiness is out of scope for now. RQ1 is addressed first by an
evaluation-only pilot (`results/complementarity_pilot.md`).

---

### Established Findings from the merging programme (prior work; not to be re-derived)

The hypotheses H1-H4 referred to below belong to the earlier initialisation-quality framing, where
merging was the method: H1 initialisation efficiency, H2 composition specificity, H3 convergence over
persistence, H4 bounded instance adaptivity.

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
* **Gate 2 (10x5, tardiness triple): FAILS by its letter at both preferences, but not the way priority
  did.** The optima sit on an edge (equal preference: 2 of 4 seeds; carbon-leaning: 3 of 4), and the
  interior advantage is absent (-0.17%, p = 0.73). But the specialist at zero weight is usually
  MAKESPAN (5 of 7 edge optima); tardiness carries 55-60% of the weight on average and is the most
  valuable specialist at equal preference (dropping it costs +2.48%, against +0.52% for priority).
  The tardiness specialist largely covers the makespan direction, so the best compositions are
  carbon + tardiness. Seed 333 disagrees with the rest at both preferences. Full record:
  `results/tardiness_programme/gate2_tardiness.md`.
* **Calibration (10x5):** the closed-form anchor under-weights tardiness by at least 2x; 2x, 4x and
  8x the anchor are indistinguishable (a plateau). Record: `results/tardiness_programme/step6_calibration.md`.

**Finding: Gate 2 tests whether the best merge uses every specialist, not whether an objective is
genuine.** The two tests gave OPPOSITE answers across the two triples (10x5):
* Retired **priority** is not a genuine separate axis (collinear with makespan at Gate 1), yet it PASSED
  Gate 2 at the equal preference: interior optimum at all four seeds, interior advantage +0.38%
  (4/4 seeds, p = 0.04). It failed only at the carbon-leaning preference, where one seed's optimum
  (seed 333) sat on the makespan-carbon edge.
* **Tardiness** at 10x5 IS a genuine axis (its specialists win their own objective by 14-21% at every
  seed), yet it FAILED Gate 2: interior advantage -0.17% / -0.35% (p = 0.72 / 0.60).

What the two triples share is that their two TIME-LIKE specialists act as near-substitutes for
composition. In the priority triple either one can be dropped for about 1% or less (makespan +0.78% /
+1.08%, priority +0.52% / +0.25%). In the tardiness triple the substitution runs one way: makespan can
be dropped cheaply (+0.54% / +0.83%, and adding it back costs nothing up to 20-30% weight), but
tardiness cannot (+2.48% / +1.72%) -- the tardiness specialist largely covers the makespan direction.

Stated precisely, because the shorthand overstates it: the tardiness optimum is ON the
tardiness-carbon edge in 3 of 8 (seed, preference) cases and within 20% makespan in 6 of 8, but seed 333
sits on the makespan-carbon edge at both preferences. Priority's optimum was interior at every seed at
the equal preference; it did not collapse onto the makespan-carbon edge. The shared finding is
redundancy between the time-like specialists, not a common edge.

Consequence for the programme: an edge optimum is not evidence against the objective, and an interior
optimum is not evidence for it. Objective genuineness is judged by Gate 1 (per-specialist) and the
separation screen; Gate 2 answers the composition question only. Records:
`results/tardiness_programme/gate2_full_report.md`.

**Gate 2's role changed (2026-09-24): it is no longer a go/no-go.** The central question is whether a
merged checkpoint is a better starting point for fine-tuning than training from scratch, and that
does not require the merge to use every specialist. Gate 2 disagreed with the genuineness tests in
BOTH triples -- priority (not genuine) passed it at the equal preference, tardiness at 10x5 (genuine)
failed it -- so it cannot be read as evidence about the objective, and whether the best merge is
interior is not what the fine-tuning comparison depends on. Gate 2 is kept as a descriptive report
of where the best compositions sit. Steps 4 onward proceed at 10x5 with the tardiness triple.

**20x10 k sensitivity (CPU screen only, pre-registered k unchanged).** At the loosest k in the
pre-stated band (1.20 instead of 0.95) the screen's tardiness advantage rises from +7.3% to +30.6%
(trained-makespan tardy fraction 0.57 -> 0.24), larger than 10x5's +10.2%. Much of the 20x10
degeneracy may come from the smallest-k rule rather than from size. Heuristic screen only; the trained
20x10 Gate 1 failure at k = 0.95 stands. Record: `results/tardiness_programme/k_sensitivity_20x10.md`.

**Step 4 at 10x5 (tardiness triple, 2026-09-24/25).** Five arms, 4 seeds, budgets 0-1000
(`results/step4_and_20x10_sensitivity.md`). Fine-tuning from the validation-selected merge beats
training from scratch at every budget up to 500 updates (4/4 seeds, p <= 0.009; 60% better at budget 0)
and converges with it by 1000 -- **H1 supported at low and middle budgets**. But warm-starting from the
tardiness specialist ALONE is indistinguishable from the merge at every budget, and the makespan
specialist overtakes the merge from about 100-250 updates and ends significantly better (-3.1%, 4/4
seeds, p = 0.001) -- **H2 (composition specificity) not supported**: the benefit comes from starting at
a relevant specialist, not from merging several. On validation, the best composition was a single
specialist at 2 of 4 seeds. All arms converge to the same operating point by 1000 updates (supports H3).
The merge itself does not improve under fine-tuning (0.888 -> 0.889).

**20x10 sensitivity at k = 1.2 (loosest in the pre-stated band), SENSITIVITY EVIDENCE ONLY.** Trained
tardiness specialists do worse, not better: -10.4% against their makespan counterparts on tardiness (2/4
significantly worse, 2 tied), against +0.7% at the pre-registered k = 0.95. The CPU screen had predicted
+30.6%. So the smallest-k rule did not cause the 20x10 failure, and the screen's k-sensitivity does not
transfer to trained policies at this size. The pre-registered 20x10 Gate 1 failure stands.

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

> Independently trained objective specialists score the same candidate decisions at every scheduling step,
> so their behaviours can be recombined at the level of decisions even when their weights cannot be
> merged. If they disagree in structured, state-dependent ways, a small router over frozen specialists can
> reach a good Pareto front for a fraction of the training cost of a preference-conditioned policy trained
> from scratch. Whether that precondition holds is RQ1, and it is tested before any router is built.
