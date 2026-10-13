"""Convergence rule of the v3 baselines (docs/evaluation_protocol_v3.md). Lower validation values are better.

CONVERGED at a validation step u when u >= max(min_updates, window) and the best validation value improved by less
than `tol` (relative) over the last `window` updates:
    (best over validations at updates <= u - window  -  best over all validations <= u) / best_then < tol
"""
from __future__ import annotations


def relative_improvement(val_log, window: int):
    """val_log: [(update, value), ...] in update order. None if no validation is at least `window` updates old."""
    u = val_log[-1][0]
    old = [v for x, v in val_log if x <= u - window]
    if not old:
        return None
    best_then, best_now = min(old), min(v for _, v in val_log)
    return (best_then - best_now) / abs(best_then) if best_then != 0 else float("inf")


def is_converged(val_log, window: int, tol: float, min_updates: int) -> tuple[bool, float | None]:
    if window <= 0 or not val_log:
        return False, None
    rel = relative_improvement(val_log, window)
    u = val_log[-1][0]
    return (rel is not None and u >= max(min_updates, window) and rel < tol), rel
