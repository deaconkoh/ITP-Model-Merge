# Pre-registration: static-feature route to H3

**Status:** APPROVED 2026-09-10 (deviation in section 0 signed off), EXECUTED 2026-09-10.
Outcome recorded in section 9 below; sections 0-8 are unchanged from the approved plan.
**Date:** 2026-09-10
**Scope:** Tests whether a cheap, statically-measurable instance characteristic can predict
per-instance merge composition (lambda), which is the bridge H3 depends on.

---

## 0. Deviation from the originally-requested plan (requires sign-off)

The originally-requested primary hypothesis was `carb_pt_corr_global` vs the seed-averaged
**carbon-only** lambda-response slope. Before writing this document I computed the *headroom*
of per-instance lambda selection and found the carbon-only target is **degenerate for
selection purposes**:

- Mean carbon falls monotonically across the whole lambda grid at both sizes
  (10x5: 2532 -> 1962; 20x10: 10135 -> 6052 from lambda=0 to lambda=1).
- Therefore "the lambda that minimises carbon" is always the grid endpoint (the carbon
  specialist). Oracle gain from per-instance lambda selection under a carbon-only objective:
  **2.12% at 10x5, 0.15% at 20x10** — and exactly **0.00% for 3 of 4 seeds at 20x10**.

A perfect oracle buys ~nothing at 20x10 under that target, so a feature that predicted it
perfectly would still be worthless. The carbon-only slope remains a valid *sensitivity*
measure (the existing variance decomposition stands as a statement about carbon sensitivity),
but it cannot support a composition rule.

**This pre-registration therefore re-points the target at a fixed multi-objective preference**,
which is what H2/H3 actually claim ("for a fixed multi-objective preference, the best
composition varies across instances"). Under a scalarized preference the per-instance optima
become non-degenerate (2-5 distinct optima per seed) and real headroom exists.

Prior established results (instance main effect ~32%/~24.6%; the static-feature null) were all
computed against the carbon-only slope. **They do not automatically transfer to the
preference-scalarized target and will be recomputed here.**

---

## 1. Headroom (computed in advance; this is the ceiling, not the hypothesis)

Scalarized objective, per instance, dimensionless and referenced to that instance's own
lambda=0.5 result:

    obj(lambda) = w * carbon(lambda)/carbon(0.5) + (1-w) * makespan(lambda)/makespan(0.5)

Oracle gain = % improvement of per-instance-best lambda over the single best static lambda,
averaged over the 4 specialist seeds:

| w_carbon | 10x5 oracle gain | 20x10 oracle gain |
|---|---|---|
| 0.25 | 1.70% (sd 0.48) | 0.73% (sd 0.32) |
| 0.50 | 2.54% (sd 0.56) | 0.82% (sd 0.56) |
| 0.75 | 1.97% (sd 0.36) | 1.50% (sd 0.53) |

**No feature-based rule can exceed these numbers.** They bound everything below.

---

## 2. Primary hypothesis (one, pre-specified)

**H-static:** `carb_pt_corr_global` — the within-instance Pearson correlation between carbon
cost and processing time across all compatible operation-machine pairs — predicts the
seed-averaged per-instance lambda-response under the fixed preference w_carbon=0.5, with
**positive sign at both problem sizes**.

**Mechanism:** when the low-carbon machine also tends to be the fast machine (high
`carb_pt_corr_global`), the two objectives do not conflict on that instance, so shifting lambda
changes little and the instance tolerates a low lambda. When the clean machine is slow (low or
negative correlation), there is genuine tension and lambda matters more.

Sign convention is fixed in advance: positive r between `carb_pt_corr_global` and the
(negative-valued) response slope = weaker response for less-conflicted instances.

**No other feature is primary.** Everything else below is secondary or exploratory and will be
labelled as such in the report.

## 3. Secondary family (one, tested with correction)

The other two carbon-processing-time relationship features, tested as a family of 2 with
Holm-Bonferroni correction within the family:
- `carb_pt_corr_withinop_mean`
- `clean_is_slow_frac`

Expected signs, fixed in advance: same direction as primary for
`carb_pt_corr_withinop_mean` (positive), opposite for `clean_is_slow_frac` (negative), since a
high "clean machine is slow" fraction means more conflict.

## 4. Everything else = exploratory

All remaining features (priority moments, carbon moments, flexibility, processing-time moments)
are exploratory. Any correlation among them will be reported with the explicit label
"exploratory, not pre-registered, requires independent replication before being believed."

---

## 5. Data and analysis, fixed in advance

- **Evaluation pool:** `data_train_vali/SD2/{10x5,20x10}` — 100 instances per size. Hash-verified
  (structural SHA-256) disjoint from `data_train`, `data_validation`, `data_development`, the
  reserved `data_final_test`, and the legacy pool. Never seen by any of the 8 specialists
  (confirmed from run manifests: training used `./data/data_train/SD2/...` only).
  The 10x5 pool requires carbon+priority annotation via the existing deterministic generator.
- **Neither reserved final-test set is touched.**
- **Models:** the existing 12 merged checkpoints (lambda in {0.3,0.5,0.7} x 4 seeds) plus the 8
  specialists (lambda=0 and lambda=1). No retraining. No new checkpoints.
- **Repeats:** 1 greedy repeat, not 5. Verified empirically that the 5 repeats are bit-identical
  on all metric columns (max abs diff 0.0); only the timing column varies.
- **Target:** per-instance lambda-response under the w_carbon=0.5 preference, seed-averaged
  across the 4 seeds in **output space** (never weight space — seed weight-averaging is unsound
  here, each seed has a distinct random init).
- **Statistics:** Pearson and Spearman, Fisher-z 95% CIs, two-sided. Primary is tested
  uncorrected (that is the point of pre-registering a single hypothesis); the secondary family
  gets Holm-Bonferroni within-family. Achieved power at the realised n will be reported.
- **Out-of-sample:** leave-one-out R^2 for the primary feature, plus a nested LOO where feature
  selection happens inside each fold.
- **Seed transfer:** correlations recomputed per-seed to confirm the relationship is not an
  artifact of one specialist seed.

---

## 6. Decision rule (committed before seeing results)

The static-feature route **supports H3** only if ALL of the following hold:

1. **Sign and significance:** primary feature r is positive at BOTH sizes, and p < 0.05 at both.
2. **Out-of-sample validity:** LOO R^2 >= 0.10 at both sizes (i.e. it actually predicts held-out
   instances better than the mean does).
3. **Practical floor — realized benefit:** a lambda-selection rule driven by the primary feature,
   evaluated leave-one-out, recovers **>= 50% of the oracle gain** over the best static lambda,
   at both sizes.

Criterion 3 is the one that matters. Given the headroom in section 1, >= 50% of oracle
corresponds to roughly **>= 1.3% absolute** improvement at 10x5 and **>= 0.4% absolute** at
20x10 (w_carbon=0.5). Those absolute numbers are small; if they are not worth engineering
for, then H3-as-static-rule is not worth building **even if criteria 1 and 2 pass**, and the
report will say so.

**Anti-gaming note, recorded in advance:** the currently observed point estimates for
`carb_pt_corr_global` against the *old* carbon-only target were r = +0.378 (10x5) and +0.524
(20x10). If the true effect against the new target is of similar magnitude, criterion 2 will
likely fail (LOO R^2 was -0.087 / +0.041 on the old target) and criterion 3 almost certainly
will. **The pre-registered verdict in that case is "not supported."** This floor is deliberately
set where the current evidence would fail it.

## 7. What a null result means

A null here means: instance identity affects lambda-response (already established), but that
effect is not accessible from cheap static observables, and in any case the achievable benefit
is bounded at 1-2.5%. That would retire H3-as-static-rule and redirect to the online-probe
mechanism (scoped separately), or to reconsidering the instance generator.

---

## 8. Estimated cost

Measured per-instance greedy time: 0.595 s (10x5), 1.682 s (20x10).

| Scope | 10x5 | 20x10 | total |
|---|---|---|---|
| 12 merged checkpoints x 100 instances | 11.9 min | 33.6 min | ~46 min |
| + 8 specialists (20 models) | 19.8 min | 56.1 min | ~76 min |

Both within the 2-hour budget. A denser lambda grid (0.1..0.9, 9 points x 4 seeds = 36 models)
would cost ~36 min (10x5) + ~101 min (20x10) = **~2.3 hours** and would need separate approval;
it is NOT part of this pre-registration unless explicitly added.

---

## 9. OUTCOME (recorded after execution; plan above unchanged)

Executed on the `data_train_vali` pools, n=100 per size, 1 greedy repeat, 20 checkpoints
per size. Analysis: `tools/analyze_prereg.py`. Raw evaluation: `tools/eval_prereg_pool.py`.

### Verdict: **NOT SUPPORTED** — all three criteria fail at both sizes.

| Criterion | 10x5 | 20x10 |
|---|---|---|
| C1 positive sign + p<0.05 | FAIL (r = -0.107, p = 0.287 — sign is *wrong*) | FAIL (r = +0.154, p = 0.126) |
| C2 LOO R^2 >= 0.10 | FAIL (-0.034) | FAIL (-0.016) |
| C3 >= 50% of oracle gain | FAIL (-11.9%) | FAIL (-23.4%) |

The primary feature's sign **flips between sizes**, so the pre-registered mechanism is not
merely unsupported, it is not directionally consistent. The feature-driven lambda rule
performs *worse than the best static lambda* at both sizes (negative recovery).

**This is now a well-powered null.** At n=100 the achieved power is 86.2% for |r|=0.3 and
98.7% for |r|=0.4. The earlier "underpowered null" caveat no longer applies: if a true
effect of |r| >= 0.3 existed, it would have been detected with ~86% probability.

**Secondary family.** `clean_is_slow_frac` reaches within-family significance at 20x10
(r = -0.235, p_holm = 0.037) with the predicted sign — but has the *opposite* sign at 10x5
(r = +0.174). Sign reversal across sizes is the signature of a chance finding, and the
cross-size consistency requirement was fixed in advance precisely to catch this. Not promoted.

**Exploratory (not pre-registered, requires independent replication).** `pt_var` at 20x10
is the only feature in the entire study with a positive out-of-sample R^2 (nested LOO
R^2 = +0.078, selected in 100/100 folds; r = +0.334, p = 0.0007). It does not replicate at
10x5. Recorded, not promoted.

### Numbers that do NOT transfer from the carbon-only target

| Quantity | Carbon-only (old) | Preference target (new) | Transfers? |
|---|---|---|---|
| Instance main effect, 10x5 | 32.7% | 47.7% (p=1.5e-11) | Yes — larger |
| Instance main effect, 20x10 | 24.6% | 37.3% (p=7.0e-05) | Yes — larger |
| Seed main effect, 10x5 | 35.3% | 0.5% (p=0.385, n.s.) | **No — vanishes** |
| Seed main effect, 20x10 | 57.1% | 0.9% (p=0.249, n.s.) | **No — vanishes** |
| Instance x seed interaction, 10x5 | 32.0% | 51.8% | Worse |
| Instance x seed interaction, 20x10 | 18.3% | 61.8% | **Much worse** |
| Slope-sign agreement, 10x5 | 79% | 45.0% | **No** |
| Slope-sign agreement, 20x10 | 100% | 49.0% | **No** |
| Argmin agreement | 26% / 28.6% | 8.0% / 8.0% | Worse |
| Target reliability | 0.674 / 0.753 | 0.638 / 0.448 | Degrades at 20x10 |

Two explanations, both mechanical:
* The **seed main effect vanishes** because the preference objective is normalised within
  each (instance, seed) cell to that cell's own lambda=0.5 result, which removes seed-level
  scale differences. The old finding "seed is the largest variance component" was largely an
  artifact of carbon *level* differences between seeds, not of differences in how seeds
  *respond* to lambda.
* The **direction-stability collapse** happens because carbon decreases monotonically in
  lambda for nearly every instance, making sign agreement trivially high. Under a genuine
  two-objective preference the response direction is close to a coin flip.

### Implication

H2 is *strengthened*: the instance main effect is larger under the preference target and
highly significant at both sizes. But the instance x seed interaction now **dominates**
(52-62%), so which instance prefers which composition is mostly specific to the training
run — and the static-feature bridge H3 needs does not exist at a level this study can
detect with 86% power. Oracle headroom on these pools is 2.34% (10x5) and 1.17% (20x10).

---

## 10. CORRECTION (2026-09-10, after the oracle-ceiling verification)

The headroom figures in section 1 were computed on a **5-point lambda grid**
{0, 0.3, 0.5, 0.7, 1.0} and are now known to **understate** the ceiling. Re-running on an
11-point grid {0, 0.1, ..., 0.9, 1.0} using the same specialists, same pools, evaluation only:

| | section 1 (5-point) | corrected (11-point) | factor |
|---|---|---|---|
| 10x5, w=0.5 | 2.34% | **3.72%** (LOO baseline) | 1.6x |
| 10x5, across w=0.1..0.9 | 1.58-3.31% | **3.42-4.90%** | ~1.6x |
| 20x10, w=0.5 | 1.17% | **2.07%** | 1.8x |
| 20x10, across w=0.1..0.9 | 0.53-1.53% | **1.95-2.45%** | ~1.9x |

Cause: the best fixed lambda sits at 0.2-0.5, and 0.2/0.4 were absent from the coarse grid,
so the oracle was denied the interior points it needed.

Two further corrections that bear on the section 9 verdict:

* **Seed-averaging suppresses headroom by 58-65%** (10x5: 2.34% per-seed vs 0.99% seed-averaged;
  20x10: 1.17% vs 0.41%). The section 1 numbers were per-seed and therefore unaffected, but the
  Track A *target* was seed-averaged, so that analysis ran against a heavily attenuated signal.
* **The per-instance gain distribution is highly skewed.** At 11-point resolution the median
  instance has 2.83% (10x5) / 1.03% (20x10) available, and the top ~15-33% of instances hold
  47-72% of all available gain. The mean was concealing this.

**Status of the section 9 verdict:** the pre-registered decision rule was applied correctly to
the data as it stood, and criteria 1-2 (sign, significance, out-of-sample R^2) concerned
feature predictiveness and are unaffected by the ceiling. However, criterion 3 was scored
against an oracle gain now known to be ~1.6-1.9x too small, and the target was both
seed-averaged and coarse-grid-derived. **The static-feature null should be re-run against the
corrected target before being treated as settled.**

---

## 11. CORRECTION TO THE CORRECTION (2026-09-10, transfer test)

Section 10 reported that the oracle ceiling roughly doubles on a denser lambda grid, and
implied this reopens the instance-aware direction. **That implication was wrong.** A 21-point
grid (0.05 resolution) plus a cross-seed transfer test shows the raised ceiling is largely
maximum-selection bias, not exploitable headroom.

**Convergence** (LOO baseline, w=0.5): the unrestricted ceiling does not converge —

| grid | 10x5 | 20x10 |
|---|---|---|
| 5-point | 2.34% | 1.17% |
| 11-point | 3.72% (+58.9%) | 2.07% (+76.8%) |
| 21-point | 4.61% (+23.9%) | 3.29% (+59.0%) |

**Why it does not converge:** the oracle takes a min over increasingly many near-duplicate
policies. Adjacent lambda values differ by weight perturbations that chaotically flip individual
dispatch decisions. More candidates => more opportunity to pick a lucky one => a higher
apparent ceiling, without any gain in generalisable structure.

**The decisive test.** Apply seed A's per-instance optimal lambda to seed B:

| | own-seed oracle | cross-seed transferred | survives |
|---|---|---|---|
| 10x5, 21-point | 4.61% | **-0.72%** | -16% |
| 20x10, 21-point | 3.29% | **-1.14%** | -35% |

Transferred selection is **worse than a single fixed lambda**. Cross-seed correlation of the
per-instance optimum is +0.07 to +0.19, i.e. approximately zero.

**Conclusion.** The per-instance optimal composition is a property of the specific specialist
set, not of the scheduling instance. This is a stronger and more mechanistic version of the
section 9 null: Track A found that cheap static features do not predict the response; the
transfer test shows there is no stable instance-level target to predict in the first place.

Practical ceiling for anything that generalises: **at or below zero.** For comparison, Track C
measured fine-tuning from the merged checkpoint at **+2.57%** (10x5, 1000 updates), which is
achievable without hindsight.

**Do not run 0.01 resolution.** It would raise the hindsight number further while leaving
transferable headroom unchanged.
