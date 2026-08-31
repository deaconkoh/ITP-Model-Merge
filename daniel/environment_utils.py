"""Shared scientific invariants used by both environment batch adapters."""

from __future__ import annotations

import numpy as np
import numpy.ma as ma


def normalize_processing_times(raw_processing_times):
    raw = np.asarray(raw_processing_times, dtype=np.float64)
    relation = raw > 0
    if not np.any(relation):
        raise ValueError("Every environment batch must contain a feasible operation-machine pair")
    positive = raw[relation]
    lower = float(np.min(positive))
    upper = float(np.max(positive))
    normalized = np.zeros_like(raw)
    normalized[relation] = raw[relation] / max(upper, 1e-8)
    return normalized, relation, ~relation, lower, upper


def machine_min_processing_time(normalized_processing_times, process_relation):
    masked = ma.array(normalized_processing_times, mask=~np.asarray(process_relation, dtype=bool))
    return np.min(masked, axis=1).filled(0)


def candidate_incompatibility(reverse_process_relation, candidate):
    return np.array([
        reverse_process_relation[env_idx][candidate[env_idx]]
        for env_idx in range(len(candidate))
    ])
