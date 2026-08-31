"""Canonical inherited objectives and evaluation metrics.

This module intentionally contains no model or environment dependencies.  The
same functions are used by training, validation, and standalone evaluation so
that a goal cannot silently acquire an unrelated reward or selection metric.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np


VALID_GOALS = ("m", "c", "p", "mc", "mp", "mcp")


@dataclass(frozen=True)
class ObjectiveSpec:
    goal: str
    carbon_weight: float = 0.01
    priority_weight: float = 1.0

    def __post_init__(self) -> None:
        if self.goal not in VALID_GOALS:
            raise ValueError(f"Unsupported goal {self.goal!r}; expected one of {VALID_GOALS}")
        if self.carbon_weight < 0 or self.priority_weight < 0:
            raise ValueError("Objective weights must be non-negative")

    @property
    def uses_makespan(self) -> bool:
        return "m" in self.goal

    @property
    def uses_carbon(self) -> bool:
        return "c" in self.goal

    @property
    def uses_priority(self) -> bool:
        return "p" in self.goal

    def step_reward(
        self,
        makespan_delta: np.ndarray,
        chosen_carbon: np.ndarray,
        priority_delta: np.ndarray,
    ) -> np.ndarray:
        """Compose only the reward terms named by ``goal``."""
        reward = np.zeros_like(np.asarray(makespan_delta), dtype=np.float64)
        if self.uses_makespan:
            reward = reward + makespan_delta
        if self.uses_carbon:
            reward = reward - self.carbon_weight * chosen_carbon
        if self.uses_priority:
            reward = reward + self.priority_weight * priority_delta
        return reward

    def selection_metric(self, metrics: Mapping[str, float]) -> float:
        """Validation loss mathematically matching the accumulated objective."""
        # The inherited environment optimizes time potentials after dividing
        # processing times by the batch maximum, while carbon remains unscaled.
        # Preserve that relative weighting for carbon combinations.
        time_scale = float(metrics.get("time_scale", 1.0))
        if time_scale <= 0:
            raise ValueError("Validation time_scale must be positive")
        value = 0.0
        if self.uses_makespan:
            value += float(metrics["makespan"]) / time_scale
        if self.uses_carbon:
            value += self.carbon_weight * float(metrics["carbon"])
        if self.uses_priority:
            value += self.priority_weight * float(metrics["priority"]) / time_scale
        return value


def operation_priority_weighted_completion(
    operation_completion_times: np.ndarray,
    operation_priorities: np.ndarray,
    valid_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Return sum(q_o C_o) / sum(q_o) for each environment."""
    completion = np.asarray(operation_completion_times, dtype=np.float64)
    priorities = np.asarray(operation_priorities, dtype=np.float64)
    if completion.shape != priorities.shape:
        raise ValueError(
            "Operation completion and priority arrays must have identical shapes; "
            f"got {completion.shape} and {priorities.shape}"
        )
    if completion.ndim == 1:
        completion = completion[np.newaxis, :]
        priorities = priorities[np.newaxis, :]
        squeeze = True
    elif completion.ndim == 2:
        squeeze = False
    else:
        raise ValueError("Operation metric expects a one- or two-dimensional array")

    if valid_mask is not None:
        mask = np.asarray(valid_mask, dtype=bool)
        if mask.ndim == 1 and completion.shape[0] == 1:
            mask = mask[np.newaxis, :]
        if mask.shape != completion.shape:
            raise ValueError("Operation valid mask must match completion-time shape")
        priorities = np.where(mask, priorities, 0.0)

    denominator = np.sum(priorities, axis=1)
    if np.any(denominator <= 0):
        raise ValueError("Operation priorities must have a positive sum in every environment")
    result = np.sum(completion * priorities, axis=1) / denominator
    return result[0] if squeeze else result


def environment_metrics(env) -> dict[str, float]:
    """Compute canonical terminal metrics from an environment batch."""
    if getattr(env, "priority_scope", "operation") == "legacy_job":
        denominator = np.sum(env.job_priorities, axis=1)
        if np.any(denominator <= 0):
            raise ValueError("Legacy job priorities must have a positive sum")
        priority = np.sum(env.true_candidate_free_time * env.job_priorities, axis=1) / denominator
    else:
        valid_mask = getattr(env, "valid_operation_mask", None)
        priority = operation_priority_weighted_completion(
            env.true_op_ct,
            env.op_priorities,
            valid_mask=valid_mask,
        )
    return {
        "makespan": float(np.mean(env.current_makespan)),
        "carbon": float(np.mean(env.total_carbon)),
        "priority": float(np.mean(priority)),
        "time_scale": float(env.pt_upper_bound),
    }
