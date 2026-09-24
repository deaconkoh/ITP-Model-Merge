# SENSITIVITY ANALYSIS -- due-date tightness k at 20x10 (CPU screen only)

This does NOT replace the pre-registered k. The frozen manifest (k = 0.95 at 20x10) is unchanged, nothing
was retrained, and no checkpoint or composition uses the k tested here.

The pre-stated rule qualified 20x10 on k in [0.95, 1.20] and selected the smallest value. This reruns the
Step 4 separation screen at the LOOSEST value in that band, k = 1.20. Due dates are the frozen ones rescaled
by 1.20 / 0.95; since d_j = k u_j P_j is linear in k, this is exactly what the rule would have produced at
k = 1.20 (same per-job draws u_j, same s = 0.6). The k = 0.95 row was rerun through the same code path as a
check and reproduces the Step 4 result exactly.

| 20x10 | k = 0.95 (pre-registered) | k = 1.20 (loosest in band) |
|---|---|---|
| tardy fraction, trained makespan specialists (Step 2 table) | 0.572 | 0.240 |
| tardy fraction, ECT heuristic | 0.412 | 0.229 |
| tardiness specialist advantage over the makespan specialist (screen) | **+7.3%** | **+30.6%** |
| makespan <-> tardiness correlation across blended schedules | +0.499 | +0.471 |
| carbon reference advantage | +59.1% | +59.1% |
| retired priority (per-operation) advantage | +0.0% | +0.0% |

Reading: on the heuristic screen, loosening k from the tightest to the loosest value the rule allowed
multiplies tardiness's separation from makespan by about four (7.3% -> 30.6%). That is larger than the 10x5
screen at its frozen k (+10.2%), so the "shrinks with size" trend seen in Step 4 does not survive a change of
k within the pre-stated band. The correlation barely moves (0.50 -> 0.47).

This supports the recorded caveat: at 20x10 the smallest-k rule, by leaving 57% of jobs late, pushed total
tardiness toward a completion-time objective, and much of the 20x10 degeneracy may come from that choice
rather than from instance size. Two limits: this is the heuristic screen, not trained specialists (the Gate 1
failure at 20x10 was measured on trained policies at k = 0.95 and is untouched by this analysis), and 10x5 was
not rerun at its own loosest in-band k (1.65), so the two sizes are not compared like for like here.
