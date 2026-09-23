"""Three-objective scalarised preference, and closed-form reward-weight anchors.

PREFERENCE (what we score everything by)
----------------------------------------
For target preference weights (w_m, w_c, w_p) summing to 1, and per-instance reference
values taken from a fixed reference composition (the simplex centroid):

    obj_i(theta) = w_m * makespan_i/makespan_ref_i
                 + w_c * carbon_i  /carbon_ref_i
                 + w_p * priority_i/priority_ref_i

Per-instance normalisation makes every instance contribute comparably regardless of its
absolute scale, and obj = 1 at the reference composition by construction. Lower is better.

REWARD-WEIGHT ANCHORS (what fine-tuning must optimise)
-----------------------------------------------------
From objectives.py the training selection metric for goal 'mcp' is

    L(theta) = makespan/time_scale + w_c_train * carbon + w_p_train * priority/time_scale

Both L and obj are linear in (makespan, carbon, priority), so they induce the same
preference exactly when their coefficient RATIOS match. Taking makespan as the reference
coordinate:

    carbon/makespan  ratio in L   : w_c_train * time_scale
    carbon/makespan  ratio in obj : (w_c/w_m) * (makespan_ref/carbon_ref)
        =>  w_c_train = (w_c/w_m) * (makespan_ref/carbon_ref) / time_scale

    priority/makespan ratio in L  : w_p_train
    priority/makespan ratio in obj: (w_p/w_m) * (makespan_ref/priority_ref)
        =>  w_p_train = (w_p/w_m) * (makespan_ref/priority_ref)

Note w_p_train carries no time_scale factor: the selection metric divides BOTH makespan and
priority by time_scale, so it cancels. w_c_train does carry it, because carbon is unscaled.

These are anchors, not answers. The policy optimises expected return, not the metric
directly, so Step 2 verifies them with a short sweep around each anchor.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

TIME_SCALE = 99.0          # env.pt_upper_bound for SD2 (processing times in [1,99]); verify per pool
CENTROID = (1 / 3, 1 / 3, 1 / 3)


@dataclass(frozen=True)
class Preference:
    """w_p is the weight of the THIRD objective: operation-priority (retired) or, with
    third='t', total tardiness. Field name kept so every priority result stays reproducible."""
    w_m: float
    w_c: float
    w_p: float
    third: str = "p"

    def __post_init__(self):
        s = self.w_m + self.w_c + self.w_p
        if abs(s - 1.0) > 1e-9:
            raise ValueError(f"preference weights must sum to 1, got {s}")
        if min(self.w_m, self.w_c, self.w_p) < 0:
            raise ValueError("preference weights must be non-negative")

    @property
    def tag(self):
        return (f"m{round(100*self.w_m):02d}c{round(100*self.w_c):02d}"
                f"{self.third}{round(100*self.w_p):02d}")


def scalarise(metrics, ref, pref: Preference, pooled_third=False):
    """metrics/ref: arrays (n_inst, 3) ordered [makespan, carbon, third]. Lower is better.

    pooled_third: normalise the third objective by its POOL-MEAN reference value instead of the
    per-instance one. Required for tardiness (fixed before any tardiness data existed): zero is a
    legitimate per-instance tardiness, so the per-instance ratio is undefined at zero and
    explodes near it. Makespan and carbon keep the per-instance normalisation.
    """
    m, c, p = metrics[..., 0], metrics[..., 1], metrics[..., 2]
    rm, rc, rp = ref[..., 0], ref[..., 1], ref[..., 2]
    if pooled_third:
        rp = float(np.mean(ref[..., 2]))
        if rp <= 0:
            raise ValueError("pooled reference for the third objective is zero; cannot normalise")
    return pref.w_m * (m / rm) + pref.w_c * (c / rc) + pref.w_p * (p / rp)


def reward_anchors(pref: Preference, ref_mean, time_scale=TIME_SCALE):
    """Closed-form (carbon_reward_weight, priority_reward_weight) for a target preference.

    ref_mean: pool-mean [makespan, carbon, priority] at the reference composition.
    """
    if pref.w_m <= 0:
        raise ValueError("anchor derivation uses makespan as the reference coordinate; "
                         "w_m must be > 0. For w_m = 0 rescale against another coordinate.")
    m_ref, c_ref, p_ref = map(float, ref_mean)
    w_c_train = (pref.w_c / pref.w_m) * (m_ref / c_ref) / time_scale
    w_p_train = (pref.w_p / pref.w_m) * (m_ref / p_ref)
    return w_c_train, w_p_train


def sweep_around(anchor, factors=(0.25, 0.5, 1.0, 2.0, 4.0)):
    """Verification sweep points around an anchor (log-spaced)."""
    return [anchor * f for f in factors]


if __name__ == "__main__":
    # illustrative only -- real reference values come from evaluating the centroid merge
    demo_ref = np.array([536.2, 1958.2, 210.0])
    print("closed-form anchors (illustrative reference values, NOT measured):")
    print(f"  reference composition metrics: makespan={demo_ref[0]}, carbon={demo_ref[1]}, "
          f"priority={demo_ref[2]}   time_scale={TIME_SCALE}")
    for pref in [Preference(1 / 3, 1 / 3, 1 / 3), Preference(0.5, 0.25, 0.25),
                 Preference(0.25, 0.5, 0.25), Preference(0.25, 0.25, 0.5)]:
        wc, wp = reward_anchors(pref, demo_ref)
        print(f"  {pref.tag}:  carbon_reward_weight={wc:.6f}   priority_reward_weight={wp:.6f}")
    print("\nsanity: the two-objective case should reproduce Track C's calibrated 0.002766")
    two = Preference(0.5, 0.5, 0.0)
    wc, wp = reward_anchors(two, np.array([536.2, 1958.2, 1.0]))
    print(f"  m50c50p00 -> carbon_reward_weight={wc:.6f}  (Track C measured 0.002766)")
