"""Multi-objective metrics of the v3 MO stage (docs/evaluation_protocol_v3.md, section 6 and Addendum A).

All objectives are minimised. Per instance, objectives are normalised with frozen ideal and nadir points,
f~ = (f - ideal) / (nadir - ideal); an objective whose nadir equals its ideal is dropped for that instance.
Hypervolume uses the reference point 1.1 in every kept objective; only points that strictly dominate the reference
point contribute. IGD+ (Ishibuchi et al., 2015) is measured against a reference front in the same normalised space.
"""
from __future__ import annotations
import numpy as np

OBJECTIVES = ("makespan", "carbon", "tardiness")
REF = 1.1


def simplex_grid(step: float) -> np.ndarray:
    """All (w_m, w_c, w_t) >= 0 summing to 1 on a grid of the given step: 0.25 -> 15 vectors, 0.1 -> 66."""
    n = int(round(1 / step))
    if abs(n * step - 1) > 1e-9:
        raise ValueError(f"step {step} does not divide 1")
    return np.array([(i / n, j / n, (n - i - j) / n) for i in range(n + 1) for j in range(n + 1 - i)])


def nondominated(points: np.ndarray) -> np.ndarray:
    """Boolean mask of the non-dominated rows (duplicates: the first copy is kept)."""
    p = np.asarray(points, dtype=float)
    keep = np.ones(len(p), dtype=bool)
    for i in range(len(p)):
        if not keep[i]:
            continue
        dom = np.all(p <= p[i], 1) & np.any(p < p[i], 1)
        dup = np.all(p == p[i], 1) & (np.arange(len(p)) < i)
        if dom.any() or dup.any():
            keep[i] = False
    return keep


def normalise(points, ideal, nadir):
    """points [n, 3] raw; ideal/nadir [3]. Returns the normalised points restricted to the kept objectives, and the
    kept-objective mask."""
    ideal = np.asarray(ideal, dtype=float); nadir = np.asarray(nadir, dtype=float)
    keep = nadir > ideal
    p = (np.asarray(points, dtype=float)[:, keep] - ideal[keep]) / (nadir[keep] - ideal[keep])
    return p, keep


def _hv2(p, ref):
    """2-D hypervolume of points p [n, 2] (all strictly below ref)."""
    if len(p) == 0:
        return 0.0
    p = p[np.lexsort((p[:, 1], p[:, 0]))]
    vol, best_y = 0.0, ref[1]
    for x, y in p:
        if y < best_y:
            vol += (ref[0] - x) * (best_y - y)
            best_y = y
    return vol


def hypervolume(points, ref=REF) -> float:
    """Exact hypervolume of minimisation points [n, d], d in {1, 2, 3}, w.r.t. the reference point `ref` (scalar or
    [d]). Points that do not strictly dominate the reference point contribute nothing."""
    p = np.asarray(points, dtype=float)
    if p.ndim != 2:
        raise ValueError("points must be [n, d]")
    d = p.shape[1]
    r = np.full(d, ref, dtype=float) if np.isscalar(ref) else np.asarray(ref, dtype=float)
    p = p[np.all(p < r, 1)]
    if len(p) == 0 or d == 0:
        return 0.0
    if d == 1:
        return float(r[0] - p[:, 0].min())
    if d == 2:
        return float(_hv2(p, r))
    if d != 3:
        raise ValueError("hypervolume implemented for up to 3 objectives")
    p = p[np.argsort(p[:, 0], kind="stable")]
    vol = 0.0
    for i in range(len(p)):
        x_next = p[i + 1, 0] if i + 1 < len(p) else r[0]
        if x_next > p[i, 0]:
            vol += (x_next - p[i, 0]) * _hv2(p[:i + 1, 1:], r[1:])
    return float(vol)


def igd_plus(approx, reference) -> float:
    """IGD+: mean over reference points z of min over approximation points a of ||max(a - z, 0)||_2."""
    a = np.asarray(approx, dtype=float); z = np.asarray(reference, dtype=float)
    if len(z) == 0:
        return float("nan")
    if len(a) == 0:
        return float("inf")
    d = np.sqrt((np.maximum(a[None, :, :] - z[:, None, :], 0.0) ** 2).sum(-1))
    return float(d.min(1).mean())


def instance_hv(raw_points, ideal, nadir, ref=REF) -> float:
    """Normalised hypervolume of the non-dominated set of one instance's raw objective triples."""
    p, _ = normalise(raw_points, ideal, nadir)
    p = p[nondominated(p)] if len(p) else p
    return hypervolume(p, ref)


def instance_igd_plus(raw_points, raw_front, ideal, nadir) -> float:
    p, _ = normalise(raw_points, ideal, nadir)
    z, _ = normalise(raw_front, ideal, nadir)
    return igd_plus(p[nondominated(p)], z[nondominated(z)])


def ideal_nadir(rec):
    """(ideal [3], nadir [3]) from a normalisation-points record {'ideal': {...}, 'nadir': {...}}."""
    return (np.array([rec["ideal"][k] for k in OBJECTIVES], dtype=float),
            np.array([rec["nadir"][k] for k in OBJECTIVES], dtype=float))


# --------------------------------------------------------------------- warm-start analysis (docs/warmstart_protocol_v3.md)

def preference_regret(f_hat, candidates, lam) -> float:
    """R(lambda) = lambda . f_hat - min over candidate points z of lambda . z, all in normalised space (kept objectives).
    f_hat [d]: the method's point for preference lambda; candidates [n, d]: every arm's and reference's points."""
    lam = np.asarray(lam, dtype=float)
    return float(lam @ np.asarray(f_hat, dtype=float) - (np.asarray(candidates, dtype=float) @ lam).min())


def curve_at(updates, values, u) -> float:
    """Validation value at update u: the last validation at or before u (last value carried forward)."""
    updates = np.asarray(updates); values = np.asarray(values, dtype=float)
    k = np.searchsorted(updates, u, side="right") - 1
    return float(values[max(k, 0)]) if len(values) else float("nan")


def auc(updates, values, horizon=2000) -> float:
    """Area under the validation-HV curve on [0, horizon] (step function, last value carried forward) / horizon."""
    updates = np.asarray(updates); values = np.asarray(values, dtype=float)
    edges = [u for u in updates if u < horizon] + [horizon]
    total = 0.0
    for a, b in zip(edges[:-1], edges[1:]):
        total += curve_at(updates, values, a) * (b - a)
    return total / horizon


def time_to(updates, values, seconds, target):
    """(update, training seconds) of the first validation whose value reaches `target`; (None, None) if never."""
    for u, v, s in zip(updates, values, seconds):
        if v >= target:
            return int(u), float(s)
    return None, None
